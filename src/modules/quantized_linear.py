import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
# import sys, os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from quant_utils.quantization import AsymmetricQuantization, NVFP4Quantization, NVFP4ActivationQuantization
# from quant_utils.rotation import make_rotation
# from quant_utils.permutation import make_permutation

from src.quant_utils.quantization import AsymmetricQuantization, NVFP4Quantization, NVFP4ActivationQuantization
from src.quant_utils.rotation import make_rotation
from src.quant_utils.permutation import make_permutation


class QuantizedLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        self.bitW = kwargs.pop('bitW', 8)
        self.bitA = kwargs.pop('bitA', 8)
        self.bitG = kwargs.pop('bitG', 8)
        self.parameter_quantization_axis = kwargs.pop('parameter_quantization_axis', -1) # Default: Per-column (out channel) quantization for parameters
        self.layer_prefix = kwargs.pop('layer_prefix', None)
        super().__init__(*args, **kwargs)
        # print(self.weight.shape) # num_col (out_channel) x num_row (in_channel)
        if self.parameter_quantization_axis is None:
            self.parameter_left_threshold = nn.Parameter(torch.zeros())
            self.parameter_right_threshold = nn.Parameter(torch.ones())
        else:
            self.parameter_left_threshold = nn.Parameter(torch.zeros(self.weight.shape[-3 - self.parameter_quantization_axis]))
            self.parameter_right_threshold = nn.Parameter(torch.ones(self.weight.shape[-3 - self.parameter_quantization_axis]))
        # print(f"Initialize quantizated parameter threshold with shape: {self.parameter_left_threshold.shape}")
        self.activation_left_threshold = nn.Parameter(torch.tensor(-1.0))
        self.activation_right_threshold = nn.Parameter(torch.tensor(1.0))
        # print(f"Initialize quantized activation threshold with shape: {self.activation_left_threshold.shape}")

    def forward(self, input, quantization_error_info=None):

        # if quantization_error_info is None and QuantizedLinear._CTX is not None:
        #     quantization_error_info = QuantizedLinear._CTX

        pref = (self.layer_prefix + '.') if self.layer_prefix else ''

        quantized_input = AsymmetricQuantization.apply(
            input, self.bitA,
            self.activation_left_threshold, self.activation_right_threshold,
            None, quantization_error_info, pref + 'input'
        )

        quantized_weight = AsymmetricQuantization.apply(
            self.weight, self.bitW,
            self.parameter_left_threshold, self.parameter_right_threshold,
            self.parameter_quantization_axis, quantization_error_info, pref + 'weight'
        )
        return F.linear(quantized_input, quantized_weight, self.bias)


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 (two-level scaled 4-bit) weight quantization.

    Optionally applies a rotation and/or a permutation on the contraction
    (feature) dim before quantization. The SAME transform is applied to the
    activation, so the (un-quantized) output is mathematically identical to a
    plain ``nn.Linear``.

    Args:
        rotation:     RotationBase instance or None.
        permutation:  PermutationBase instance or None.
        quantize:     if False, the transform is still applied but the weight is
                      NOT quantized (useful for consistency testing).
        block_size:   NVFP4 block size (default 16).
    """

    def __init__(
        self, in_features, out_features, bias=True, 
        rotation=None, permutation=None, permute_weight=True, 
        quantize=True, block_size=16, layer_prefix=None
    ):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        self.rotation = make_rotation(rotation, in_features)
        self.permutation = make_permutation(permutation)
        self.quantize = quantize
        self.layer_prefix = layer_prefix
        self.permute_weight = permute_weight
        if self.permute_weight and self.permutation is not None:
            self.permutation.fit(self.weight)
        self.x_eff = None
        self.W_eff = None
        self.x_quant = None
        self.W_quant = None
        self.output = None

    def _effective_weight(self):
        # rotation first, then permutation
        W = self.weight
        if self.rotation is not None:
            W = self.rotation.rotate_weight(W)
        if self.permutation is not None:
            W = self.permutation.transform_weight(W)
        return W

    def _effective_activation(self, x):
        if self.rotation is not None:
            x = self.rotation.rotate_activation(x)
        if not self.permute_weight and self.permutation is not None:
            self.permutation.fit(x)
        if self.permutation is not None:
            x = self.permutation.transform_activation(x)
        return x

    def forward(self, x, quantization_error_info=None):
        self.x_eff = self._effective_activation(x)
        self.W_eff = self._effective_weight()
        if self.quantize:
            # quantize WEIGHT in the rotated+permuted basis
            self.W_quant = NVFP4Quantization.apply(
                self.W_eff, self.block_size,
                quantization_error_info,
                f"{self.layer_prefix}.weight" if self.layer_prefix else "weight",
            )
            # quantize ACTIVATION in the SAME transformed basis (R·P applied first,
            # then FP4), so the block layout aligns with the weight's contraction
            # dimension. NVFP4ActivationQuantization expects 3D [bs, n_seq, dim];
            # for higher-rank activations we fuse all leading dims into (bs, n_seq)
            # and restore the original shape afterwards (tokens are quantized
            # independently along the last/feature dim, so ordering is irrelevant).
            act_prefix = f"{self.layer_prefix}.input" if self.layer_prefix else "input"
            orig_shape = self.x_eff.shape
            xq3d = NVFP4ActivationQuantization.apply(
                self.x_eff.reshape(orig_shape[0], -1, orig_shape[-1]),
                self.block_size,
                quantization_error_info, act_prefix,
            )
            self.x_quant = xq3d.reshape(orig_shape)
        else:
            self.W_quant = self.W_eff
            self.x_quant = self.x_eff
        self.output = F.linear(self.x_quant, self.W_quant, self.bias)
        return self.output

    def get_differentiable_quantization_error(self, loss_fn):
        return loss_fn(self.x_quant.detach(), self.x_eff), loss_fn(self.W_quant.detach(), self.W_eff)



if __name__ == "__main__":
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    batch_size = 10
    seq_len = 20
    hidden_dim_1 = 30
    hidden_dim_2 = 40
    x = torch.randn(batch_size, seq_len, hidden_dim_1)
    """
    # Test QuantizedLinear
    linear = QuantizedLinear(hidden_dim_1, hidden_dim_2, bitW=8, bitA=8)
    y = linear(x)
    # print(y.shape)
    loss = torch.nn.MSELoss()(y, torch.randn(batch_size, hidden_dim_2))
    loss.backward()
    # print(loss)
    print(linear.parameter_left_threshold.grad.shape)
    print(linear.parameter_right_threshold.grad.shape)
    # print(linear.activation_left_threshold.grad.shape)
    # print(linear.activation_right_threshold.grad.shape)
    """
    # ---
    # Test NVFP4Linear
    # ---
    with torch.no_grad():
        linear_ori = nn.Linear(hidden_dim_1, hidden_dim_2, bias=True)
        y_ori = linear_ori(x)
    ## Test Rotation and Permutation
    # rots = ["identity", "random", "hadamard", "cayley"]
    # perms = ["identity", "random", "mag"]
    # for rot in rots:
    #     for perm in perms:
    #         for quantize in [False]:
    #             quantization_error_info = {}
    #             linear_rpq = NVFP4Linear(hidden_dim_1, hidden_dim_2, rotation=rot, permutation=perm, quantize=quantize, bias=True, block_size=16)
    #             linear_rpq.load_state_dict(linear_ori.state_dict())
    #             y_rpq = linear_rpq(x, quantization_error_info)
    #             print(f"{rot}-{perm}-{quantize}: {torch.mean(torch.abs(y_rpq - y_ori))}") 