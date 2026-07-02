import torch

def asymmetric_quantization(x_, num_bits_=8, left_threshold_=0.0, right_threshold_=1.0, axis_=None):
    """Asymmetric quantization: quantize input to [0, 2^num_bits - 1] and then dequantize back to float.

    Args:
        x_: input tensor
        num_bits_: quantization bit-width
        left_threshold_: left boundary (min value) of the quantization range
        right_threshold_: right boundary (max value) of the quantization range

    Returns:
        simulated quantized tensor (quantize + dequantize)
    """

    q_max = 2 ** num_bits_ - 1
    scale = (right_threshold_ - left_threshold_) / q_max

    # clamp input to quantization range
    x_clamped = torch.clamp(x_, left_threshold_, right_threshold_)

    # quantize: map to [0, q_max]
    q = torch.round((x_clamped - left_threshold_) / scale)
    q = torch.clamp(q, 0, q_max)

    # dequantize
    x_hat = q * scale + left_threshold_

    return x_hat


class AsymmetricQuantization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_, num_bits_, left_threshold_, right_threshold_, axis_=None, error_info_=None, module_prefix_=None):
        
        if axis_ is None:
            if len(left_threshold_.shape) != 0:
                raise ValueError(f"None axis requires scalar threshold, but got shape: {left_threshold_.shape}")
        else:
            if left_threshold_.shape != (x_.shape[-3 - axis_], ):
                raise ValueError(f"Axis {axis_} requires shape: {x_.shape[-3 - axis_]}, but got shape: {left_threshold_.shape}")
        
        if axis_ is not None:
            left_threshold_ = left_threshold_.unsqueeze(axis_)
            right_threshold_ = right_threshold_.unsqueeze(axis_)
        # print(x_.shape, left_threshold_.shape)
        
        quantized_x_ = asymmetric_quantization(x_, num_bits_, left_threshold_, right_threshold_, axis_)
        ctx.save_for_backward(x_, quantized_x_, left_threshold_, right_threshold_)
        ctx.num_bits_ = num_bits_
        ctx.axis_ = axis_
        ctx.error_info_ = error_info_
        ctx.module_prefix_ = module_prefix_
        
        if num_bits_ == 32:
            return x_
        else:
            return quantized_x_

    @staticmethod
    def loss_fn(error_mask_, method="mse", axis_=None):
        if method == "mse":
            return (error_mask_ ** 2).mean(axis_ if axis_ is not None else None)
        elif method == "mae":
            return error_mask_.abs().mean(axis_ if axis_ is not None else None)
        else:
            raise ValueError(f"Unknown loss method: {method}")

    @staticmethod
    def backward(ctx, grad_output):
        x_, quantized_x_, left_threshold_, right_threshold_ = ctx.saved_tensors
        num_bits_ = ctx.num_bits_
        axis_ = ctx.axis_
        error_info_ = ctx.error_info_
        module_prefix_ = ctx.module_prefix_

        in_domain_mask_ = (x_ >= left_threshold_) & (x_ <= right_threshold_)
        left_domain_mask_ = (x_ < left_threshold_)
        right_domain_mask_ = (x_ > right_threshold_)
        quantization_error_mask_ = x_ - quantized_x_
        # print(quantization_error_mask_.shape)
        in_domain_error = AsymmetricQuantization.loss_fn(quantization_error_mask_ * in_domain_mask_, axis_=axis_)
        left_domain_error = AsymmetricQuantization.loss_fn(quantization_error_mask_ * left_domain_mask_, axis_=axis_)
        right_domain_error = AsymmetricQuantization.loss_fn(quantization_error_mask_ * right_domain_mask_, axis_=axis_)

        left_grad = in_domain_error + left_domain_error
        right_grad = in_domain_error + right_domain_error

        if error_info_ is not None:
            error_info_[f"{module_prefix_+"." if module_prefix_ is not None else ""}in_domain_error"] = in_domain_error.data
            error_info_[f"{module_prefix_+"." if module_prefix_ is not None else ""}left_domain_error"] = left_domain_error.data
            error_info_[f"{module_prefix_+"." if module_prefix_ is not None else ""}right_domain_error"] = right_domain_error.data

        # STE: pass gradient through for in-domain values
        if num_bits_ == 32:
            grad_input = grad_output
        else:
            grad_input = grad_output * in_domain_mask_.float()

        return grad_input, None, left_grad, right_grad, None, None, None



def quantization_check(quantized_x_, num_bits_):

    assert len(torch.unique(quantized_x_)) <= 2**num_bits_


if __name__ == "__main__":
    
    import torch

    batch_size = 10
    hidden_dim_1 = 30
    hidden_dim_2 = 40
    bit = 2

    # x = torch.randn(hidden_dim_1, hidden_dim_2)
    x = torch.randn(batch_size, hidden_dim_1)
    
    # quantized_x = asymmetric_quantization(x, bit, 0.0, 1.0)
    # quantization_check(quantized_x, bit)

    error_info = {}

    left_threshold = torch.nn.Parameter(torch.zeros(hidden_dim_1))
    right_threshold = torch.nn.Parameter(torch.ones(hidden_dim_1))
    quantized_x = AsymmetricQuantization.apply(x, bit, left_threshold, right_threshold, -2, error_info)
    # left_threshold = torch.nn.Parameter(torch.zeros(hidden_dim_2))
    # right_threshold = torch.nn.Parameter(torch.ones(hidden_dim_2))
    # quantized_x = AsymmetricQuantization.apply(x, bit, left_threshold, right_threshold, -1)
    # left_threshold = torch.nn.Parameter(torch.tensor(0.0))
    # right_threshold = torch.nn.Parameter(torch.tensor(1.0))
    # quantized_x = AsymmetricQuantization.apply(x, bit, left_threshold, right_threshold, None)
    loss = torch.nn.MSELoss()(x, quantized_x)
    loss.backward()
    print(loss)
    # print(left_threshold.grad.shape)
    # print(right_threshold.grad.shape)
    print(error_info)
