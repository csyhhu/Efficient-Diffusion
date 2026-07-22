"""
NVFP4-quantized Sana Transformer — exact replica of diffusers SanaTransformer2DModel.

Key differences from previous (buggy) implementation:
  - Self-attention uses LINEAR attention (ReLU + matmul reordering) per
    ``SanaLinearAttnProcessor2_0``, NOT full quadratic attention.
  - Cross-attention uses standard SDPA per ``SanaAttnProcessor2_0``.
  - Text projection uses GELU(tanh), not SiLU.
  - Matches diffusers forward path exactly (reshape, return types, etc.).

All ``nn.Linear`` layers can be toggled between NVFP4-quantized and standard
mode via the ``use_nvfp4`` flag.

Usage::

    from src.models.nvfp4_quantized_Sana import NVFP4QuantizedSana

    # Non-quantized (standard nn.Linear everywhere):
    model = NVFP4QuantizedSana.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        use_nvfp4=False,
    )

    # NVFP4-quantized:
    model = NVFP4QuantizedSana.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        block_size=16,
    )

    # NVFP4 + Hadamard rotation + magnitude-sort permutation (each layer gets
    # its own permutation, fitted on the real weights after from_pretrained):
    model = NVFP4QuantizedSana.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        block_size=16,
        rotation="hadamard",     # "none" | "hadamard" | "random" | "cayley" | RotationBase
        permutation="mag",       # "none" | "identity" | "random" | "mag" | PermutationBase
    )

    # Or pass instances directly for full control (e.g. a pre-fitted Cayley):
    from src.quant_utils.rotation import HadamardRotation
    from src.quant_utils.permutation import MagnitudeSortPermutation
    model = NVFP4QuantizedSana.from_pretrained(
        "...", rotation=HadamardRotation(block_size=16),
        permutation=MagnitudeSortPermutation(block_size=16))
"""

import math
import os
import json
from typing import Optional, List, Any
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quant_utils.rotation import (RotationBase, IdentityRotation, HadamardRotation, RandomRotation, CayleyRotation, make_rotation)
from src.quant_utils.permutation import (PermutationBase, IdentityPermutation, RandomPermutation, MagnitudeSortPermutation, make_permutation)
from src.modules.quantized_linear import NVFP4Linear
    

def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings that preserve input dtype.
    
    Same as diffusers' get_timestep_embedding but without the .float() conversion.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=timesteps.dtype, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None] * emb[None, :]

    emb = scale * emb

    if flip_sin_to_cos:
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
    else:
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class _Timesteps(nn.Module):
    """Timesteps embedding layer that preserves input dtype.
    
    Same as diffusers' Timesteps but uses the custom _get_timestep_embedding.
    """
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


# ===========================================================================
# Timestep & position embedding helpers (exact diffusers replicas)
# ===========================================================================

def _get_1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal encoding.  pos: (M,) -> (M, D)."""
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim))
    out = torch.outer(pos.to(torch.float64), omega)
    return torch.cat([out.sin(), out.cos()], dim=-1).to(pos.dtype)


def _get_2d_sincos(embed_dim: int, h: int, w: int,
                   base_size: int = 16, interpolation_scale: float = None,
                   device: torch.device = None) -> torch.Tensor:
    """2D sinusoidal position encoding. Returns (h*w, embed_dim)."""
    grid_h = torch.arange(h, dtype=torch.float32, device=device) / (h / base_size)
    grid_w = torch.arange(w, dtype=torch.float32, device=device) / (w / base_size)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_h = _get_1d_sincos(embed_dim // 2, grid_h.reshape(-1))
    emb_w = _get_1d_sincos(embed_dim // 2, grid_w.reshape(-1))
    out = torch.cat([emb_h, emb_w], dim=-1)
    if interpolation_scale is not None:
        out = out * interpolation_scale
    return out


# ===========================================================================
# NVFP4 Linear layers (reuse src.modules.quantized_linear.NVFP4Linear)
# ===========================================================================

def _make_linear(in_features: int, out_features: int, bias: bool = True,
                 block_size: int = 16, layer_prefix: str = None,
                 use_nvfp4: bool = True, rotation=None,
                 permutation=None) -> nn.Module:
    """Factory: returns an ``NVFP4Linear`` with ``quantize=use_nvfp4``.

    ``rotation`` is a per-layer factory ``(in_features) -> RotationBase | None``
    (as produced by ``make_rotation``) — a fresh rotation is created for this
    layer so they are not shared across layers. ``permutation`` is also a
    per-layer factory ``(in_features) -> PermutationBase | None`` (as produced
    by ``make_permutation``) — a fresh permutation is created for this layer.
    """
    return NVFP4Linear(
        in_features, out_features, bias=bias, block_size=block_size,
        rotation=rotation, permutation=permutation,
        layer_prefix=layer_prefix, quantize=use_nvfp4,
    )


# ===========================================================================
# NVFP4 Attention — mirrors diffusers Attention + inlined processors
# ===========================================================================

class NVFP4Attention(nn.Module):
    """Mirrors ``diffusers.models.attention_processor.Attention`` with NVFP4 support.

    Self-attention (attn_type="linear") uses the ``SanaLinearAttnProcessor2_0``
    algorithm:  φ(x)=ReLU(x), computes φ(V) φ(K)^T φ(Q) in O(N·d²).

    Cross-attention (attn_type="sdpa") uses standard
    ``F.scaled_dot_product_attention`` per ``SanaAttnProcessor2_0``.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        qk_norm: Optional[str] = None,
        rescale_output_factor: float = 1.0,
        attn_type: str = "sdpa",  # "linear" or "sdpa"
        block_size: int = 16,
        layer_prefix: str = None,
        use_nvfp4: bool = True,
        rotation=None,
        permutation=None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.rescale_output_factor = rescale_output_factor
        self.attn_type = attn_type
        self.layer_prefix = layer_prefix

        # Q/K/V projections
        kv_input_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_q = _make_linear(query_dim, self.inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.to_q",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)
        self.to_k = _make_linear(kv_input_dim, self.inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.to_k",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)
        self.to_v = _make_linear(kv_input_dim, self.inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.to_v",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)

        # QK norm (None for Sana Sprint 0.6B)
        self.norm_q = None
        self.norm_k = None
        if qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=1e-5)
            self.norm_k = nn.LayerNorm(dim_head, eps=1e-5)
        elif qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
            self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)

        # Output projection
        self.to_out = nn.ModuleList([
            _make_linear(self.inner_dim, query_dim, bias=out_bias,
                         block_size=block_size,
                         layer_prefix=f"{layer_prefix}.to_out.0",
                         use_nvfp4=use_nvfp4,
                         rotation=rotation, permutation=permutation),
            nn.Dropout(dropout),
        ])

    # ------------------------------------------------------------------
    # Linear attention (self-attention) — SanaLinearAttnProcessor2_0
    # ------------------------------------------------------------------
    def _linear_attention_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        quantization_error_info: dict = None,
    ) -> torch.Tensor:
        """Exact replica of ``SanaLinearAttnProcessor2_0.__call__``."""
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states, quantization_error_info)
        key = self.to_k(encoder_hidden_states, quantization_error_info)
        value = self.to_v(encoder_hidden_states, quantization_error_info)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Reshape: (B, N, D) -> (B, heads, head_dim, N)
        query = query.transpose(1, 2).unflatten(1, (self.heads, -1))
        # key: (B, N, D) -> (B, heads, N, head_dim) -> transpose to (B, heads, head_dim, N)
        key = key.transpose(1, 2).unflatten(1, (self.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (self.heads, -1))

        # φ(x) = ReLU(x)
        query = F.relu(query)
        key = F.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        # Padding trick for normalization
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        # (B, heads, head_dim+1, N) x (B, heads, N, head_dim) -> (B, heads, head_dim+1, head_dim)
        scores = torch.matmul(value, key)
        # (B, heads, head_dim+1, head_dim) x (B, heads, head_dim, N) -> (B, heads, head_dim+1, N)
        hidden_states = torch.matmul(scores, query)

        # Normalize
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        # Reshape back: (B, heads, head_dim, N) -> (B, N, inner_dim)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.to_out[0](hidden_states, quantization_error_info)
        hidden_states = self.to_out[1](hidden_states)

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    # ------------------------------------------------------------------
    # Standard SDPA (cross-attention) — SanaAttnProcessor2_0
    # ------------------------------------------------------------------
    def _sdpa_attention_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        quantization_error_info: dict = None,
    ) -> torch.Tensor:
        """Exact replica of ``SanaAttnProcessor2_0.__call__``."""
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # Mirror diffusers ``Attention.prepare_attention_mask`` for SDPA:
        #   input:  (B, 1, key_tokens) -> output: (B, heads, 1, key_tokens)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(2)  # (B, 1, 1, key_tokens)
            attention_mask = attention_mask.expand(-1, self.heads, -1, -1)  # (B, heads, 1, key_tokens)

        query = self.to_q(hidden_states, quantization_error_info)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states, quantization_error_info)
        value = self.to_v(encoder_hidden_states, quantization_error_info)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states, quantization_error_info)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        quantization_error_info: dict = None,
    ) -> torch.Tensor:
        if self.attn_type == "linear":
            return self._linear_attention_forward(
                hidden_states, encoder_hidden_states, quantization_error_info)
        else:
            return self._sdpa_attention_forward(
                hidden_states, encoder_hidden_states, attention_mask, quantization_error_info)


# ===========================================================================
# Embedding modules (exact diffusers replicas with NVFP4 support)
# ===========================================================================

class Timesteps(nn.Module):
    """Maps integer timesteps -> sinusoidal features. Matches diffusers ``Timesteps``."""

    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True,
                 downscale_freq_shift: float = 0.0, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Maps sine-cosine features -> embedding via 2-layer MLP.
    Matches diffusers ``TimestepEmbedding``."""

    def __init__(self, in_channels: int, time_embed_dim: int,
                 out_dim: Optional[int] = None,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.linear_1 = _make_linear(in_channels, time_embed_dim, bias=True,
                                     block_size=block_size,
                                     layer_prefix=f"{layer_prefix}.linear_1",
                                     use_nvfp4=use_nvfp4,
                                     rotation=rotation, permutation=permutation)
        self.act = nn.SiLU()
        out_dim = out_dim or time_embed_dim
        self.linear_2 = _make_linear(time_embed_dim, out_dim, bias=True,
                                     block_size=block_size,
                                     layer_prefix=f"{layer_prefix}.linear_2",
                                     use_nvfp4=use_nvfp4,
                                     rotation=rotation, permutation=permutation)

    def forward(self, sample: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        sample = sample.to(self.linear_1.weight.dtype)
        h = self.act(self.linear_1(sample, quantization_error_info))
        return self.linear_2(h, quantization_error_info)


class PatchEmbed(nn.Module):
    """2D latent -> patch tokens with sin-cos position embedding.
    Matches diffusers ``PatchEmbed`` for Sana."""

    def __init__(self, height: int, width: int, patch_size: int, in_channels: int,
                 embed_dim: int, interpolation_scale: Optional[float] = None):
        super().__init__()
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.base_size = self.height
        self.interpolation_scale = interpolation_scale

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)
        self.pos_embed = None
        if interpolation_scale is not None:
            pos_embed = _get_2d_sincos(embed_dim, self.base_size, self.base_size,
                                       interpolation_scale=interpolation_scale)
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

    def _interpolate_pos_encoding(self, h: int, w: int) -> torch.Tensor:
        pe = self.pos_embed.reshape(1, self.base_size, self.base_size, -1).permute(0, 3, 1, 2)
        orig_dtype = pe.dtype
        pe = F.interpolate(pe.float(), size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return pe.to(orig_dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(self.proj.weight.dtype)
        B, C, H, W = latent.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.proj(latent).flatten(2).transpose(1, 2)  # BCHW -> BNC

        if self.pos_embed is None:
            return x.to(x.dtype)

        if h == self.base_size and w == self.base_size:
            pos_embed = self.pos_embed[:, :h * w, :]
        else:
            pos_embed = self._interpolate_pos_encoding(h, w)
        return (x + pos_embed).to(x.dtype)


class PixArtAlphaTextProjection(nn.Module):
    """Projects caption embeddings. Matches diffusers ``PixArtAlphaTextProjection``.
    Uses GELU(tanh) activation (NOT SiLU)."""

    def __init__(self, in_features: int, hidden_size: int,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.linear_1 = _make_linear(in_features, hidden_size, bias=True,
                                     block_size=block_size,
                                     layer_prefix=f"{layer_prefix}.linear_1",
                                     use_nvfp4=use_nvfp4,
                                     rotation=rotation, permutation=permutation)
        self.act_1 = nn.GELU(approximate="tanh")  # <- this is the correct activation
        self.linear_2 = _make_linear(hidden_size, hidden_size, bias=True,
                                     block_size=block_size,
                                     layer_prefix=f"{layer_prefix}.linear_2",
                                     use_nvfp4=use_nvfp4,
                                     rotation=rotation, permutation=permutation)

    def forward(self, caption: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        hidden_states = self.linear_1(caption, quantization_error_info)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states, quantization_error_info)
        return hidden_states


# ===========================================================================
# GLUMBConv (Conv2d-based FF, no quantizable Linear — exact diffusers copy)
# ===========================================================================

class GLUMBConv(nn.Module):
    """Gated Linear Unit with Masked Benes Convolution — Sana feed-forward."""

    def __init__(self, in_channels: int, out_channels: int,
                 expand_ratio: float = 4.0, norm_type: Optional[str] = None,
                 residual_connection: bool = True):
        super().__init__()
        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1,
                                    groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)
        self.norm = None
        if norm_type == "rms_norm":
            self.norm = nn.RMSNorm(out_channels, eps=1e-5, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states
        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)
        if self.norm_type == "rms_norm":
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


# ===========================================================================
# Normalization / modulation (exact diffusers copies)
# ===========================================================================

class SanaModulatedNorm(nn.Module):
    """Sana modulation norm. Matches diffusers ``SanaModulatedNorm``."""

    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor,
                scale_shift_table: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (
            scale_shift_table[None] + temb[:, None].to(scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class SanaCombinedTimestepGuidanceEmbeddings(nn.Module):
    """Matches diffusers ``SanaCombinedTimestepGuidanceEmbeddings``.

    Structure::

        time_proj (Timesteps)  -> timestep_embedder  (256->D)  -\
        guidance_proj (Timesteps) -> guidance_embedder (256->D) --+ -> silu -> linear(D->6D)

    Returns ``(modulation_6D, conditioning_D)``.
    """

    def __init__(self, embedding_dim: int, block_size: int = 16,
                 layer_prefix: str = None, use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True,
                                   downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim,
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.timestep_embedder",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation)
        self.guidance_condition_proj = Timesteps(num_channels=256,
                                                  flip_sin_to_cos=True,
                                                  downscale_freq_shift=0)
        self.guidance_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim,
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.guidance_embedder",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation)

        self.silu = nn.SiLU()
        self.linear = _make_linear(embedding_dim, 6 * embedding_dim, bias=True,
                                   block_size=block_size,
                                   layer_prefix=f"{layer_prefix}.linear",
                                   use_nvfp4=use_nvfp4,
                                   rotation=rotation, permutation=permutation)

    def forward(self, timestep: torch.Tensor,
                guidance: torch.Tensor = None,
                hidden_dtype: torch.dtype = None,
                quantization_error_info: dict = None,
                batch_size: int = None):
        """Returns ``(modulation_6D, conditioning_D)``."""
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=timestep.dtype), quantization_error_info)

        guidance_proj = self.guidance_condition_proj(guidance)
        guidance_emb = self.guidance_embedder(
            guidance_proj.to(dtype=timestep.dtype), quantization_error_info)

        conditioning = timesteps_emb + guidance_emb
        modulation = self.linear(self.silu(conditioning), quantization_error_info)
        return modulation, conditioning


# ===========================================================================
# Sana Transformer Block (exact diffusers replica)
# ===========================================================================

class SanaTransformerBlock(nn.Module):
    """Sana DiT block. Matches diffusers ``SanaTransformerBlock``.

    self-attn (linear) -> cross-attn (SDPA) -> GLUMBConv FF.
    """

    def __init__(self, dim: int,
                 num_attention_heads: int, attention_head_dim: int,
                 num_cross_attention_heads: int, cross_attention_head_dim: int,
                 cross_attention_dim: int,
                 attention_bias: bool = True, attention_out_bias: bool = True,
                 mlp_ratio: float = 2.5,
                 norm_elementwise_affine: bool = False, norm_eps: float = 1e-6,
                 qk_norm: Optional[str] = None,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix

        # Self-attention (LINEAR)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = NVFP4Attention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=0.0,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rescale_output_factor=1.0,
            attn_type="linear",
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.attn1",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation,
        )

        # Cross-attention (SDPA)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = NVFP4Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_cross_attention_heads,
            dim_head=cross_attention_head_dim,
            dropout=0.0,
            bias=True,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rescale_output_factor=1.0,
            attn_type="sdpa",
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.attn2",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation,
        )

        # FF — GLUMBConv
        self.ff = GLUMBConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        height: int = None,
        width: int = None,
        quantization_error_info: dict = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # Self-attention (LINEAR)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        attn_output = self.attn1(norm_hidden_states,
                                 quantization_error_info=quantization_error_info)
        hidden_states = hidden_states + gate_msa * attn_output

        # Cross-attention (SDPA)
        attn_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            quantization_error_info=quantization_error_info,
        )
        hidden_states = attn_output + hidden_states

        # Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


# ===========================================================================
# NVFP4 Quantized Sana (main model — exact diffusers SanaTransformer2DModel)
# ===========================================================================

class NVFP4QuantizedSana(nn.Module):
    """Sana Transformer with optional NVFP4 quantization.

    Exact architecture replica of ``diffusers.SanaTransformer2DModel``.
    Set ``use_nvfp4=False`` for standard ``nn.Linear`` (correctness testing).

    SANA-Sprint 0.6B defaults::

        model = NVFP4QuantizedSana(
            sample_size=32, patch_size=1, in_channels=32,
            num_layers=20, attention_head_dim=64, num_attention_heads=24,
            num_cross_attention_heads=24, cross_attention_head_dim=64,
            cross_attention_dim=1536, caption_channels=2304,
            out_channels=32, mlp_ratio=2.5,
        )
    """

    def __init__(
        self,
        sample_size: int = 32,
        patch_size: int = 1,
        in_channels: int = 32,
        out_channels: Optional[int] = None,
        num_layers: int = 20,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_cross_attention_heads: int = 24,
        cross_attention_head_dim: int = 64,
        cross_attention_dim: int = 1536,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[float] = None,
        guidance_embeds: bool = True,
        guidance_embeds_scale: float = 0.1,
        qk_norm: Optional[str] = None,
        block_size: int = 16,
        use_nvfp4: bool = True,
        rotation="identity",
        permutation="identity",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.use_nvfp4 = use_nvfp4

        # ---- rotation / permutation ----
        self.rotation = rotation
        self.permutation = permutation

        # Pipeline-compatible config namespace
        self.config = SimpleNamespace(
            sample_size=sample_size,
            in_channels=in_channels,
            patch_size=patch_size,
            guidance_embeds_scale=guidance_embeds_scale,
        )

        # Patch embedding
        self.patch_embed = PatchEmbed(
            height=sample_size, width=sample_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        # Timestep + guidance embedding
        if guidance_embeds:
            self.time_embed = SanaCombinedTimestepGuidanceEmbeddings(
                embedding_dim=inner_dim,
                block_size=block_size,
                layer_prefix="time_embed",
                use_nvfp4=use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
            )
        else:
            raise NotImplementedError("Only guidance_embeds=True is supported")

        # Text projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=inner_dim,
            block_size=block_size,
            layer_prefix="caption_projection",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation,
            permutation=self.permutation,
        )
        self.caption_norm = nn.RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SanaTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                num_cross_attention_heads=num_cross_attention_heads,
                cross_attention_head_dim=cross_attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                attention_bias=attention_bias,
                attention_out_bias=True,
                mlp_ratio=mlp_ratio,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                block_size=block_size,
                layer_prefix=f"block.{i}",
                use_nvfp4=use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
            )
            for i in range(num_layers)
        ])

        # Output
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = _make_linear(
            inner_dim, patch_size * patch_size * out_channels,
            bias=True, block_size=block_size, layer_prefix="proj_out",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation,
            permutation=self.permutation
        )

        self.gradient_checkpointing = False
        self.quantization_error_info: dict = {}
        print(f"Initialize Sana Transformer {'Quantized' if use_nvfp4 else 'Unquantized'} mode, block_size={block_size}, "
              f"rotation={self.rotation}, permutation={self.permutation})")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # Permutation fitting
    # ------------------------------------------------------------------

    # def fit_all_permutations(self):
    #     """Fit every layer's permutation on the CURRENT (real) weights.

    #     Call this after weights are loaded (e.g. ``from_pretrained`` or
    #     ``_copy_weights``). Rotations are data-free (or pre-fitted by the
    #     caller) and need no action here.
    #     """
    #     for m in self.modules():
    #         if isinstance(m, NVFP4Linear):
    #             m.fit_permutation()

    # ------------------------------------------------------------------
    # from_pretrained
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_checkpoint_path(cls, pretrained_model_name_or_path: str,
                                 download_source: str = "modelscope",
                                 cache_dir: str = None,
                                 **kwargs):
        """Return a local checkpoint directory, using the cache first and only downloading when needed."""
        if os.path.isdir(pretrained_model_name_or_path):
            return pretrained_model_name_or_path

        if cache_dir:
            candidate = os.path.join(cache_dir, pretrained_model_name_or_path.replace("/", os.sep))
            if os.path.isdir(candidate):
                print(f"Using local cache directory: {candidate}")
                return candidate

        if download_source == "modelscope":
            try:
                from modelscope import snapshot_download
                local_path = snapshot_download(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir or ".cache/modelscope",
                    local_files_only=True,
                )
                print(f"  ModelScope cache: {local_path}")
                return local_path
            except Exception as exc:
                print(f"  Local ModelScope cache not found; attempting download: {exc}")
                from modelscope import snapshot_download
                local_path = snapshot_download(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir or ".cache/modelscope",
                )
                print(f"  ModelScope download: {local_path}")
                return local_path
        else:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            hf_kwargs = {"local_files_only": True}
            if cache_dir:
                hf_kwargs["cache_dir"] = cache_dir
            try:
                local_path = hf_snapshot_download(
                    pretrained_model_name_or_path, **hf_kwargs)
            except Exception:
                print("  (first download: pulling from HuggingFace ...)")
                hf_kwargs["local_files_only"] = False
                local_path = hf_snapshot_download(
                    pretrained_model_name_or_path, **hf_kwargs)
            return local_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str,
                        subfolder: str = "transformer",
                        download_source: str = None,
                        cache_dir: str = None,
                        torch_dtype: torch.dtype = torch.float32,
                        block_size: int = 16,
                        use_nvfp4: bool = True,
                        rotation=None,
                        permutation=None,
                        **kwargs):
        """Build NVFP4QuantizedSana / unquantized Sana from a pretrained checkpoint."""
        from diffusers import SanaTransformer2DModel

        download_source = (download_source
                           or os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower())

        # Step 1: resolve / download checkpoint
        local_path = cls._resolve_checkpoint_path(
            pretrained_model_name_or_path,
            download_source=download_source,
            cache_dir=cache_dir,
            **kwargs,
        )

        # Step 2: read config.json
        config_file = os.path.join(local_path, subfolder, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        inner_dim = cfg.get("num_attention_heads", 0) * cfg.get("attention_head_dim", 0)
        print(f"Config: layers={cfg.get('num_layers')}, "
              f"heads={cfg.get('num_attention_heads')}, "
              f"head_dim={cfg.get('attention_head_dim')}, "
              f"inner_dim={inner_dim}")

        # Step 3: build model with config dimensions
        model = cls(
            sample_size=cfg.get("sample_size", 32),
            patch_size=cfg.get("patch_size", 1),
            in_channels=cfg.get("in_channels", 32),
            out_channels=cfg.get("out_channels") or cfg.get("in_channels", 32),
            num_layers=cfg["num_layers"],
            attention_head_dim=cfg["attention_head_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            num_cross_attention_heads=cfg.get("num_cross_attention_heads")
                or cfg["num_attention_heads"],
            cross_attention_head_dim=cfg.get("cross_attention_head_dim")
                or cfg["attention_head_dim"],
            cross_attention_dim=cfg.get("cross_attention_dim")
                or inner_dim,
            caption_channels=cfg.get("caption_channels", 2304),
            mlp_ratio=cfg.get("mlp_ratio", 2.5),
            attention_bias=cfg.get("attention_bias", True),
            norm_elementwise_affine=cfg.get("norm_elementwise_affine", False),
            norm_eps=cfg.get("norm_eps", 1e-6),
            interpolation_scale=cfg.get("interpolation_scale", None),
            guidance_embeds=cfg.get("guidance_embeds", True),
            guidance_embeds_scale=cfg.get("guidance_embeds_scale", 0.1),
            qk_norm=cfg.get("qk_norm", None),
            block_size=block_size,
            use_nvfp4=use_nvfp4,
            rotation=rotation,
            permutation=permutation,
        )

        # Step 4: load reference model & copy weights
        ref = SanaTransformer2DModel.from_pretrained(
            local_path, subfolder=subfolder, local_files_only=True, 
            torch_dtype=torch_dtype,
        )
        model._copy_weights(ref)
        del ref

        return model

    # ------------------------------------------------------------------
    # Weight copy
    # ------------------------------------------------------------------

    def _copy_weights(self, ref):
        """Copy weights from HuggingFace SanaTransformer2DModel to this model."""
        # patch_embed
        self.patch_embed.proj.weight.data.copy_(ref.patch_embed.proj.weight.data)
        self.patch_embed.proj.bias.data.copy_(ref.patch_embed.proj.bias.data)
        if ref.patch_embed.pos_embed is not None and self.patch_embed.pos_embed is not None:
            self.patch_embed.pos_embed.data.copy_(ref.patch_embed.pos_embed.data)

        # time_embed: SanaCombinedTimestepGuidanceEmbeddings
        my_te = self.time_embed
        ref_te = ref.time_embed
        my_te.timestep_embedder.linear_1.weight.data.copy_(
            ref_te.timestep_embedder.linear_1.weight.data)
        my_te.timestep_embedder.linear_1.bias.data.copy_(
            ref_te.timestep_embedder.linear_1.bias.data)
        my_te.timestep_embedder.linear_2.weight.data.copy_(
            ref_te.timestep_embedder.linear_2.weight.data)
        my_te.timestep_embedder.linear_2.bias.data.copy_(
            ref_te.timestep_embedder.linear_2.bias.data)
        my_te.guidance_embedder.linear_1.weight.data.copy_(
            ref_te.guidance_embedder.linear_1.weight.data)
        my_te.guidance_embedder.linear_1.bias.data.copy_(
            ref_te.guidance_embedder.linear_1.bias.data)
        my_te.guidance_embedder.linear_2.weight.data.copy_(
            ref_te.guidance_embedder.linear_2.weight.data)
        my_te.guidance_embedder.linear_2.bias.data.copy_(
            ref_te.guidance_embedder.linear_2.bias.data)
        my_te.linear.weight.data.copy_(ref_te.linear.weight.data)
        my_te.linear.bias.data.copy_(ref_te.linear.bias.data)

        # caption_projection
        self.caption_projection.linear_1.weight.data.copy_(
            ref.caption_projection.linear_1.weight.data)
        self.caption_projection.linear_1.bias.data.copy_(
            ref.caption_projection.linear_1.bias.data)
        self.caption_projection.linear_2.weight.data.copy_(
            ref.caption_projection.linear_2.weight.data)
        self.caption_projection.linear_2.bias.data.copy_(
            ref.caption_projection.linear_2.bias.data)

        # caption_norm
        self.caption_norm.weight.data.copy_(ref.caption_norm.weight.data)

        # transformer_blocks
        for i, (my_block, ref_block) in enumerate(
                zip(self.transformer_blocks, ref.transformer_blocks)):
            self._copy_block_weights(my_block, ref_block)

        # scale_shift_table
        self.scale_shift_table.data.copy_(ref.scale_shift_table.data)

        # proj_out
        self.proj_out.weight.data.copy_(ref.proj_out.weight.data)
        self.proj_out.bias.data.copy_(ref.proj_out.bias.data)

    def _copy_block_weights(self, my, ref):
        """Copy weights for one SanaTransformerBlock."""
        # Self-attention
        my.attn1.to_q.weight.data.copy_(ref.attn1.to_q.weight.data)
        my.attn1.to_k.weight.data.copy_(ref.attn1.to_k.weight.data)
        my.attn1.to_v.weight.data.copy_(ref.attn1.to_v.weight.data)
        my.attn1.to_out[0].weight.data.copy_(ref.attn1.to_out[0].weight.data)
        if ref.attn1.to_q.bias is not None:
            my.attn1.to_q.bias.data.copy_(ref.attn1.to_q.bias.data)
            my.attn1.to_k.bias.data.copy_(ref.attn1.to_k.bias.data)
            my.attn1.to_v.bias.data.copy_(ref.attn1.to_v.bias.data)
        if ref.attn1.to_out[0].bias is not None:
            my.attn1.to_out[0].bias.data.copy_(ref.attn1.to_out[0].bias.data)

        # Cross-attention
        my.attn2.to_q.weight.data.copy_(ref.attn2.to_q.weight.data)
        my.attn2.to_k.weight.data.copy_(ref.attn2.to_k.weight.data)
        my.attn2.to_v.weight.data.copy_(ref.attn2.to_v.weight.data)
        my.attn2.to_out[0].weight.data.copy_(ref.attn2.to_out[0].weight.data)
        if ref.attn2.to_q.bias is not None:
            my.attn2.to_q.bias.data.copy_(ref.attn2.to_q.bias.data)
            my.attn2.to_k.bias.data.copy_(ref.attn2.to_k.bias.data)
            my.attn2.to_v.bias.data.copy_(ref.attn2.to_v.bias.data)
        if ref.attn2.to_out[0].bias is not None:
            my.attn2.to_out[0].bias.data.copy_(ref.attn2.to_out[0].bias.data)

        # FF (GLUMBConv — Conv2d)
        for my_m, ref_m in zip(my.ff.modules(), ref.ff.modules()):
            if isinstance(my_m, nn.Conv2d) and isinstance(ref_m, nn.Conv2d):
                my_m.weight.data.copy_(ref_m.weight.data)
                if ref_m.bias is not None and my_m.bias is not None:
                    my_m.bias.data.copy_(ref_m.bias.data)
            elif isinstance(my_m, nn.RMSNorm) and isinstance(ref_m, nn.RMSNorm):
                my_m.weight.data.copy_(ref_m.weight.data)

        # scale_shift_table
        my.scale_shift_table.data.copy_(ref.scale_shift_table.data)

    # ------------------------------------------------------------------
    # Forward (exact diffusers replica)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        controlnet_block_samples: Optional[tuple] = None,
        return_dict: bool = False,
        quantization_error_info: dict = None,
    ):
        """Matches ``SanaTransformer2DModel.forward`` exactly.

        Args:
            hidden_states: (B, C, H, W) latent.
            encoder_hidden_states: (B, N_txt, caption_channels) text embeddings.
            timestep: (B,).
            encoder_attention_mask: (B, N_txt) — optional.
            guidance: (B,) — classifier-free guidance scale.
            attention_kwargs: forwarded for pipeline compatibility (ignored).
            return_dict: returns tuple when False (default for pipeline).

        Returns:
            ``(output,)`` tuple.
        """
        # Convert masks to bias format
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height, post_patch_width = height // p, width // p
        hidden_states = self.patch_embed(hidden_states)

        # Timestep + guidance embedding
        # print(timestep.dtype, guidance.dtype) # torch.bfloat16 torch.bfloat16
        if guidance is not None:
            timestep_emb, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype,
                quantization_error_info=quantization_error_info,
            )
        else:
            timestep_emb, embedded_timestep = self.time_embed(
                timestep, guidance=torch.zeros_like(timestep).type(hidden_states.dtype),
                hidden_dtype=hidden_states.dtype,
                quantization_error_info=quantization_error_info,
            )

        # Text projection + reshape (matches diffusers exactly)
        encoder_hidden_states = self.caption_projection(
            encoder_hidden_states.to(dtype=hidden_states.dtype),
            quantization_error_info)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                def _forward(blk, *args):
                    return blk(*args, quantization_error_info=quantization_error_info)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    _forward, block, hidden_states, attention_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    timestep_emb, post_patch_height, post_patch_width,
                    use_reentrant=False,
                )
        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_emb,
                    post_patch_height,
                    post_patch_width,
                    quantization_error_info=quantization_error_info,
                )

        # 3. Output normalization
        hidden_states = self.norm_out(
            hidden_states, embedded_timestep, self.scale_shift_table)

        # 4. Output projection
        hidden_states = self.proj_out(hidden_states, quantization_error_info)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width,
            p, p, self.out_channels,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(
            batch_size, self.out_channels, post_patch_height * p, post_patch_width * p)

        if not return_dict:
            return (output,)

        # Return dict-compatible for diffusers
        return type("Transformer2DModelOutput", (), {"sample": output})()


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":

    from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
    
    bs = 2
    in_channels = 32
    width = 16
    height = 16
    n_txt = 32
    caption_channels = 512
    # dtype = torch.float32
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")

    hidden_states = torch.randn(bs, in_channels, height, width, dtype=dtype).to(device)
    encoder_hidden_states = torch.randn(bs, n_txt, caption_channels, dtype=dtype).to(device)
    timestep_embs = torch.randn(bs, dtype=dtype).to(device)
    guidance = torch.ones(bs, dtype=dtype).to(device) * 1.0

    inner_dim = 128
    num_attention_heads = 4
    attention_head_dim = inner_dim // num_attention_heads
    
    model_ori = SanaTransformer2DModel(
        sample_size=16,
        patch_size=1,
        in_channels=in_channels,
        out_channels=in_channels,
        num_layers=2,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        num_cross_attention_heads=num_attention_heads,
        cross_attention_head_dim=attention_head_dim,
        cross_attention_dim=inner_dim,
        caption_channels=caption_channels,
        mlp_ratio=2.0,
        guidance_embeds=True,
    ).to(device, dtype=dtype)
    y_ori = model_ori(hidden_states, encoder_hidden_states, timestep_embs, guidance=guidance, return_dict=True).sample
    # print(f"Original model output shape: {y_ori.shape}")
    # rots = ["identity", "random", "hadamard", "cayley"]
    # perms = ["identity", "random", "mag"]
    # rots = [None]
    # perms = [None]
    rots = [None, "identity"]
    perms = [None, "identity"]
    for rot in rots:
        for perm in perms:
            for quantize in [False]: 
                model_rpq = NVFP4QuantizedSana(
                    sample_size=16,
                    patch_size=1,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_layers=2,
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads,
                    num_cross_attention_heads=num_attention_heads,
                    cross_attention_head_dim=attention_head_dim,
                    cross_attention_dim=inner_dim,
                    caption_channels=caption_channels,
                    mlp_ratio=2.0,
                    attention_bias=False,
                    use_nvfp4=quantize,
                    rotation=rot,
                    permutation=perm,
                    # rotation="identity",
                    # permutation="identity",
                    # rotation=None,
                    # permutation=None,
                ).to(device, dtype=dtype)
                model_rpq.load_state_dict(model_ori.state_dict())
                y_rpq = model_rpq(hidden_states, encoder_hidden_states, timestep_embs, guidance=guidance, return_dict=True).sample
                # print(f"Quantized model output shape: {y_rpq.shape}")
                
                diff = torch.mean(torch.abs(y_rpq - y_ori))
                print(f"[{rot}-{perm}-{quantize}] MSE between original and quantized model: {diff}")
