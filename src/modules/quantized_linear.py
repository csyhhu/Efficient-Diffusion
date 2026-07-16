import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
try:
    from ..quant_utils.quantization import (
        AsymmetricQuantization, NVFP4Quantization, NVFP4ActivationQuantization)
except ImportError:
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from quant_utils.quantization import AsymmetricQuantization, NVFP4Quantization


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

    def __init__(self, in_features, out_features, bias=True, block_size=16,
                 rotation=None, permutation=None, quantize=True, layer_prefix=None):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        self.rotation = rotation
        self.permutation = permutation
        self.quantize = quantize
        self.layer_prefix = layer_prefix
        # The permutation is fitted on the ROTATED weight, so that a
        # magnitude-based sort groups the columns of the final (to-be-quantized)
        # representation. Order is therefore always: rotation first, then
        # permutation (see _effective_weight / _effective_activation).
        w_for_fit = self.rotation.rotate_weight(self.weight.data) if self.rotation is not None else self.weight.data
        if self.permutation is not None:
            self.permutation.fit(w_for_fit)

    def _effective_weight(self):
        # rotation first, then permutation
        W = self.weight
        if self.rotation is not None:
            W = self.rotation.rotate_weight(W)
        if self.permutation is not None:
            W = self.permutation.transform_weight(W)
        return W

    def _effective_activation(self, x):
        # rotation first, then permutation
        if self.rotation is not None:
            x = self.rotation.rotate_activation(x)
        if self.permutation is not None:
            x = self.permutation.transform_activation(x)
        return x

    def fit_permutation(self):
        """(Re)fit the optional permutation on the CURRENT weight.

        The permutation is always fitted on the ROTATED weight, matching the
        ordering used in forward (rotation first, then permutation). Call this
        AFTER the real weights have been loaded (e.g. via ``_copy_weights``).
        """
        if self.permutation is None:
            return
        w_for_fit = (self.rotation.rotate_weight(self.weight.data)
                     if self.rotation is not None else self.weight.data)
        self.permutation.fit(w_for_fit)

    def forward(self, x, quantization_error_info=None):
        W_eff = self._effective_weight()
        x_eff = self._effective_activation(x)
        if self.quantize:
            # quantize WEIGHT in the rotated+permuted basis
            Wq = NVFP4Quantization.apply(
                W_eff, self.block_size,
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
            orig_shape = x_eff.shape
            xq3d = NVFP4ActivationQuantization.apply(
                x_eff.reshape(orig_shape[0], -1, orig_shape[-1]),
                self.block_size,
                quantization_error_info, act_prefix,
            )
            xq = xq3d.reshape(orig_shape)
        else:
            Wq = W_eff
            xq = x_eff
        return F.linear(xq, Wq, self.bias)



if __name__ == "__main__":
    
    import torch

    batch_size = 10
    hidden_dim_1 = 30
    hidden_dim_2 = 40
    x = torch.randn(batch_size, hidden_dim_1)

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