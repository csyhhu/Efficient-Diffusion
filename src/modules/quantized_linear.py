import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
try:
    from ..quant_utils.quantization import AsymmetricQuantization
except ImportError:
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from quant_utils.quantization import AsymmetricQuantization


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