import torch

def asymmetric_quantize(_x, _bit, _alpha, _beta):
    """
    Quantize _x to 2^_bit, given min/max threshold _alpha/_beta
    """
    # Calculate number of quantization levels
    num_levels = 2 ** _bit

    # Ensure _alpha and _beta are tensors for broadcasting
    _alpha = torch.as_tensor(_alpha, dtype=_x.dtype, device=_x.device)
    _beta = torch.as_tensor(_beta, dtype=_x.dtype, device=_x.device)
    
    # Clip input to [alpha, beta] range (alpha is min, beta is max)
    x_clamped = torch.clamp(_x, _alpha, _beta)
    
    # Normalize to [0, 1]
    x_normalized = (x_clamped - _alpha) / (_beta - _alpha)
    
    # Quantize to discrete levels
    x_quantized = torch.round(x_normalized * (num_levels - 1))
    
    # Dequantize back to original range
    x_dequantized = x_quantized / (num_levels - 1) * (_beta - _alpha) + _alpha
    
    return x_dequantized


def symmetric_quantize(_x, _bit, _alpha):
    """
    Quantize _x to 2^_bit, given min/max threshold -_alpha/_alpha
    """
    # Calculate number of quantization levels
    num_levels = 2 ** _bit

    # Ensure _alpha and _beta are tensors for broadcasting
    _alpha = torch.as_tensor(_alpha, dtype=_x.dtype, device=_x.device)
    
    # Clip input to [-alpha, alpha] range
    x_clamped = torch.clamp(_x, -_alpha, _alpha)
    
    # Normalize to [-1, 1]
    x_normalized = x_clamped / _alpha
    
    # Quantize to discrete levels (preserve zero-point alignment)
    x_quantized = torch.round(x_normalized * (num_levels // 2))
    
    # Dequantize back to original range
    x_dequantized = x_quantized / (num_levels // 2) * _alpha
    
    return x_dequantized