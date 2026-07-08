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



# ============================================================================
# NVFP4 Format Constants
# ============================================================================

# FP4 (E2M1) format: 1 sign, 2 exponent, 1 mantissa (bias=1)
# Representable positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
FP4_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_MAX = 6.0

# E4M3 format (block scale): 1 sign, 4 exponent, 3 mantissa (bias=7)
FP8_MAX = 448.0


def _generate_e4m3_positive_values(max_val=FP8_MAX):
    """Generate all positive representable values in E4M3 format (FP8).
    
    E4M3: 1 sign, 4 exponent, 3 mantissa, bias=7.
    Subnormals: e=0, value = (m/8) * 2^{-6}
    Normals: e=1..15, value = 2^{e-7} * (1 + m/8)
    """
    values = {0.0}
    # Subnormals: e = 0
    for m in range(8):
        v = (m / 8.0) * (2.0 ** -6)
        if v <= max_val:
            values.add(v)
    # Normals: e = 1 to 15
    for e in range(1, 16):
        for m in range(8):
            v = (2.0 ** (e - 7)) * (1.0 + m / 8.0)
            if v <= max_val:
                values.add(v)
    return torch.tensor(sorted(values), dtype=torch.float32)


E4M3_VALUES = _generate_e4m3_positive_values()


def _nearest_quantize(x, ref_values):
    """Quantize x to nearest value in ref_values (sorted, 1D tensor).
    
    Args:
        x: input tensor of any shape
        ref_values: sorted 1D tensor of representable values
    
    Returns:
        quantized tensor with same shape as x
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1)
    # Ensure x is on the same device as ref_values
    ref_values = ref_values.to(x.device)
    # Clamp x to valid range before searchsorted
    x_flat = x_flat.clamp(min=ref_values[0], max=ref_values[-1])
    # Binary search to find insertion positions
    idx = torch.searchsorted(ref_values, x_flat)
    idx = idx.clamp(1, len(ref_values) - 1)
    left = ref_values[idx - 1]
    right = ref_values[idx]
    diff_left = x_flat - left
    diff_right = right - x_flat
    nearest = torch.where(diff_left <= diff_right, left, right)
    return nearest.reshape(orig_shape)


def quantize_to_e4m3(x):
    """Quantize input to nearest E4M3 (FP8 block scale) representable value.
    
    Args:
        x: tensor of any shape (assumed non-negative for scale factors)
    
    Returns:
        quantized tensor with nearest E4M3 values
    """
    return _nearest_quantize(x, E4M3_VALUES)


def quantize_to_fp4(x):
    """Quantize input to nearest FP4 (E2M1) value.
    
    Handles signed input by quantizing absolute values and restoring sign.
    Values outside [-6, 6] are clipped to the boundary.
    
    Args:
        x: tensor of any shape
    
    Returns:
        quantized tensor of FP4 values
    """
    # Clip to valid FP4 range
    x_clipped = x.clamp(-FP4_MAX, FP4_MAX)
    x_abs = x_clipped.abs()
    x_sign = torch.sign(x_clipped)
    # Handle zero sign correctly (keep positive)
    x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
    # Quantize absolute values
    nearest_abs = _nearest_quantize(x_abs, FP4_POS_VALUES)
    return x_sign * nearest_abs


# ============================================================================
# Core NVFP4 quantization function
# ============================================================================

def _nvfp4_quantize_core(x_groups, block_size=16):
    """Core NVFP4 quantize + dequantize on partitioned blocks.

    This is the shared quantization engine for both tensor-wise and token-wise
    NVFP4 quantization. Each group (dimension 0) is quantized independently
    with its own global scale.

    Args:
        x_groups: float tensor of shape [G, N, block_size] where
                  G = number of independent quantization groups,
                  N = number of blocks per group,
                  block_size = elements per block.
        block_size: elements per block (default 16).

    Returns:
        x_deq_groups: dequantized tensor of shape [G, N, block_size].
        s_global: per-group FP32 global scale, shape [G].
        s_block: per-group per-block E4M3 scale, shape [G, N].
    """
    device = x_groups.device
    G, N, B = x_groups.shape

    # --- Step 1: Per-group per-block amax: [G, N] ---
    block_amax = x_groups.abs().max(dim=-1).values  # [G, N]

    # --- Step 2: Per-group global amax: [G] ---
    global_amax = block_amax.max(dim=-1).values  # [G]

    # Handle all-zero groups: use a dummy scale to avoid NaN, restore later
    zero_mask = (global_amax == 0)  # [G]
    global_amax_safe = global_amax.clamp(min=1e-10)

    # --- Step 3: Per-group s_global (FP32): [G] ---
    s_global = global_amax_safe / (FP8_MAX * FP4_MAX)  # [G]

    # --- Step 4: Per-group per-block s_block (E4M3): [G, N] ---
    s_block_ideal = (block_amax / FP4_MAX) / s_global.unsqueeze(-1)  # [G, N]
    s_block = quantize_to_e4m3(s_block_ideal)
    s_block = s_block.clamp(min=1e-10)

    # --- Step 5: Build per-element scale: [G, N, B] ---
    s_global_exp = s_global.unsqueeze(-1).unsqueeze(-1)  # [G, 1, 1]
    s_block_exp = s_block.unsqueeze(-1)                   # [G, N, 1]
    s_elem = s_global_exp * s_block_exp                   # [G, N, B]

    # --- Step 6: Quantize and dequantize ---
    x_scaled = x_groups / s_elem
    x_fp4 = quantize_to_fp4(x_scaled)
    x_deq_groups = x_fp4 * s_elem

    # --- Step 7: Restore all-zero groups ---
    if zero_mask.any():
        x_deq_groups = torch.where(
            zero_mask.unsqueeze(-1).unsqueeze(-1),
            x_groups,
            x_deq_groups,
        )
        s_global = torch.where(zero_mask, torch.zeros_like(s_global), s_global)

    return x_deq_groups, s_global, s_block


def _to_block_padded(x_1d, block_size):
    """Pad a 1D tensor to a multiple of block_size and reshape to 2D blocks.

    Args:
        x_1d: 1D tensor of shape [total_elements].
        block_size: block size.

    Returns:
        x_blocks: [num_blocks, block_size], the padded block view.
        pad_size: number of zero-pad elements appended (0 if no pad needed).
    """
    total = x_1d.numel()
    pad_size = (block_size - total % block_size) % block_size
    if pad_size > 0:
        x_1d = torch.cat([x_1d, torch.zeros(pad_size, device=x_1d.device, dtype=x_1d.dtype)])
    x_blocks = x_1d.reshape(-1, block_size)
    return x_blocks, pad_size


def _from_block_padded(x_deq_blocks, pad_size):
    """Reverse _to_block_padded: flatten blocks back to 1D and remove padding.

    Args:
        x_deq_blocks: [num_blocks, block_size].
        pad_size: number of padded elements to strip.

    Returns:
        1D tensor without padding.
    """
    x_1d = x_deq_blocks.reshape(-1)
    if pad_size > 0:
        x_1d = x_1d[:-pad_size]
    return x_1d


# ============================================================================
# NVFP4 Tensor-wise Quantization (single global scale)
# ============================================================================

class NVFP4Quantization(torch.autograd.Function):
    """NVFP4 4-bit floating-point weight quantization with two-level scaling.

    Weight matrices ``[out_features, in_features]`` are blocked along the
    contraction dimension (``in_features``, dim=-1). Each output channel
    (row) is quantized independently with its own per-channel FP32 global
    scale and per-block E4M3 block scales. This per-channel + per-in-feature-
    block layout is designed for efficient FP4 matrix multiplication: the
    block boundaries of weight and activation align along the contraction
    dimension, allowing block-wise FP4 multiply-accumulate with scale
    multiplication deferred to the end of each block.

    Implements the NVFP4 format introduced in NVIDIA Blackwell architecture:
    - Element format: E2M1 (1 sign, 2 exponent, 1 mantissa) = 4-bit
    - Block size: block_size elements per scaling block along in_features
    - Block scale: E4M3 (1 sign, 4 exponent, 3 mantissa) = 8-bit per block
    - Global scale: FP32 per output channel = 32-bit per channel

    Quantization formula:
        s_global[o] = amax_global[o] / (FP8_MAX × FP4_MAX)
        s_block[o, k] = quantize_e4m3(amax_block[o, k] / FP4_MAX / s_global[o])
        x_fp4[o, k, :] = quantize_fp4(x[o, k, :] / (s_global[o] × s_block[o, k]))

    Reference:
        NVIDIA. "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery." 2026.
    """

    @staticmethod
    def forward(ctx, x_, block_size_=16, error_info_=None, module_prefix_=None):
        """Forward pass: per-in-feature-block NVFP4 weight quantization.

        Args:
            x_: weight matrix of shape ``[out_features, in_features]``.
            block_size_: elements per scaling block along in_features (default: 16).
            error_info_: optional dict to store error statistics.
            module_prefix_: optional prefix for error_info_ keys.

        Returns:
            dequantized weight tensor with same shape as input.
        """
        if x_.dim() != 2:
            raise ValueError(
                f"NVFP4Quantization expects 2D weight [out_features, in_features], "
                f"but got {x_.dim()}D: {x_.shape}"
            )

        out_features, in_features = x_.shape
        device = x_.device

        # Pad in_features to multiple of block_size (pad along dim=-1)
        pad_dim = (block_size_ - in_features % block_size_) % block_size_
        if pad_dim > 0:
            x_pad = torch.nn.functional.pad(x_.float(), (0, pad_dim))
        else:
            x_pad = x_.float()
        in_features_padded = in_features + pad_dim
        num_blocks = in_features_padded // block_size_

        # [out_features, num_blocks, block_size]  →  G = out_features
        x_groups = x_pad.reshape(out_features, num_blocks, block_size_)

        x_deq_groups, s_global, _ = _nvfp4_quantize_core(x_groups, block_size_)

        # Reshape back: [out_features, num_blocks, block_size] → [out_features, in_features_padded]
        x_deq_pad = x_deq_groups.reshape(out_features, in_features_padded)

        # Remove padding
        if pad_dim > 0:
            x_deq = x_deq_pad[:, :in_features]
        else:
            x_deq = x_deq_pad

        # Cast back to original dtype
        x_deq = x_deq.to(x_.dtype)

        # Store for backward
        ctx.save_for_backward(x_)
        ctx.error_info_ = error_info_
        ctx.module_prefix_ = module_prefix_

        # Compute quantization error info
        if error_info_ is not None:
            prefix = f"{module_prefix_}." if module_prefix_ is not None else ""
            quantization_error = (x_ - x_deq).float()
            error_info_[f"{prefix}nvfp4_error_mse"] = (quantization_error ** 2).mean().item()
            error_info_[f"{prefix}nvfp4_error_mae"] = quantization_error.abs().mean().item()
            error_info_[f"{prefix}nvfp4_s_global_mean"] = s_global.mean().item()

        return x_deq

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: Straight-Through Estimator (STE).

        Gradients pass through unchanged since the quantization function
        is treated as identity in the backward pass.
        """
        return grad_output, None, None, None


# ============================================================================
# NVFP4 Token-wise Activation Quantization (per-token global scale)
# ============================================================================

class NVFP4ActivationQuantization(torch.autograd.Function):
    """NVFP4 activation quantization with per-token (token-wise) two-level scaling.

    For input of shape ``[bs, n_seq, dim]``, each token is quantized
    independently along the feature dimension (``dim``, last axis) with its
    own FP32 global scale and per-block E4M3 scales. The blocking direction
    matches ``NVFP4Quantization`` (both block along the contraction / feature
    dimension), enabling efficient FP4 matrix multiplication where block
    boundaries align between weight and activation.

    Scale layout:
        - s_global[tok]: FP32, one per token
        - s_block[tok, block_k]: E4M3, one per block per token
        - x_fp4[tok, block_k, :]: FP4 (E2M1) data elements

    This differs from NVFP4Quantization (weight) in that:
        - NVFP4Quantization: one s_global PER OUTPUT CHANNEL.
        - NVFP4ActivationQuantization: one s_global PER TOKEN.
    """

    @staticmethod
    def forward(ctx, x_, block_size_=16, error_info_=None, module_prefix_=None):
        """Forward pass: per-token per-in-feature-block NVFP4 activation quantization.

        Args:
            x_: input tensor of shape ``[bs, n_seq, dim]``.
            block_size_: elements per scaling block along the feature dim
                (default: 16).
            error_info_: optional dict to store error statistics.
            module_prefix_: optional prefix for error_info_ keys.

        Returns:
            dequantized tensor with shape ``[bs, n_seq, dim]``.
        """
        if x_.dim() != 3:
            raise ValueError(
                f"NVFP4ActivationQuantization expects 3D input [bs, n_seq, dim], "
                f"but got {x_.dim()}D: {x_.shape}"
            )

        bs, n_seq, dim = x_.shape
        device = x_.device

        # Merge batch and sequence dimensions: [num_tokens, dim]
        num_tokens = bs * n_seq
        x_2d = x_.reshape(num_tokens, dim).float()

        # Pad last dim and reshape to blocks: [num_tokens, num_blocks, block_size]
        pad_dim = (block_size_ - dim % block_size_) % block_size_
        if pad_dim > 0:
            x_pad = torch.nn.functional.pad(x_2d, (0, pad_dim))
        else:
            x_pad = x_2d
        dim_padded = dim + pad_dim
        num_blocks = dim_padded // block_size_

        x_groups = x_pad.reshape(num_tokens, num_blocks, block_size_)  # [G, N, B]

        # Core quantize: G = num_tokens, each token has its own global + block scales
        x_deq_groups, s_global, _ = _nvfp4_quantize_core(x_groups, block_size_)

        # Reshape back: [num_tokens, num_blocks, block_size] → [num_tokens, dim_padded]
        x_deq_pad = x_deq_groups.reshape(num_tokens, dim_padded)

        # Remove padding → [num_tokens, dim]
        if pad_dim > 0:
            x_deq_2d = x_deq_pad[:, :dim]
        else:
            x_deq_2d = x_deq_pad

        # Restore original shape
        x_deq = x_deq_2d.reshape(bs, n_seq, dim)

        # Store for backward
        ctx.save_for_backward(x_)
        ctx.error_info_ = error_info_
        ctx.module_prefix_ = module_prefix_

        # Compute quantization error info
        if error_info_ is not None:
            prefix = f"{module_prefix_}." if module_prefix_ is not None else ""
            quantization_error = (x_ - x_deq).float()
            error_info_[f"{prefix}nvfp4_act_error_mse"] = (quantization_error ** 2).mean().item()
            error_info_[f"{prefix}nvfp4_act_error_mae"] = quantization_error.abs().mean().item()

        return x_deq

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: Straight-Through Estimator (STE)."""
        return grad_output, None, None, None


def quantization_check(quantized_x_, num_bits_):

    assert len(torch.unique(quantized_x_)) <= 2**num_bits_


if __name__ == "__main__":
    
    import torch

    batch_size = 10
    n_seq = 5
    hidden_dim_1 = 3
    hidden_dim_2 = 4
    bit = 4

    # x = torch.randn(hidden_dim_1, hidden_dim_2)
    x = torch.randn(batch_size, n_seq, hidden_dim_1)
    w = torch.randn(hidden_dim_1, hidden_dim_2)
    output = x @ w
    
    # quantized_x = asymmetric_quantization(x, bit, 0.0, 1.0)
    # quantization_check(quantized_x, bit)

    error_info = {}
    # """
    # Learnable Threshold for INT Quantization
    activateion_left_threshold = torch.nn.Parameter(torch.zeros([]))
    activateion_right_threshold = torch.nn.Parameter(torch.ones([]))
    quantized_x = AsymmetricQuantization.apply(x, bit, activateion_left_threshold, activateion_right_threshold, None, error_info, "activation")
    paramater_left_threshold = torch.nn.Parameter(torch.zeros(hidden_dim_2))
    paramater_right_threshold = torch.nn.Parameter(torch.ones(hidden_dim_2))
    quantized_w = AsymmetricQuantization.apply(w, bit, paramater_left_threshold, paramater_right_threshold, -2, error_info, "parameter")
    quantized_output = quantized_x @ quantized_w
    print(torch.mean(torch.abs(output - quantized_output)))
    # loss = torch.nn.MSELoss()(x, quantized_x)
    # loss.backward()
    # print(loss)
    # print(error_info)
    # """
    # Statistic for FP4 Quantization
    quantized_w = NVFP4Quantization.apply(w, 16, error_info, "parameter")
    quantized_x = NVFP4ActivationQuantization.apply(x, 16, error_info, "activation")
    quantized_output = quantized_x @ quantized_w
    # print(x)
    # print(quantized_x)
    print(torch.mean(torch.abs(output - quantized_output)))
