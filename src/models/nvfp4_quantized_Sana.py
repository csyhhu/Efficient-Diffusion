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

# supports both package import and direct script execution
try:
    from ..modules.quantized_linear import NVFP4Linear
except ImportError:
    import sys
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from modules.quantized_linear import NVFP4Linear

try:
    from ..quant_utils.rotation import (
        RotationBase, IdentityRotation, HadamardRotation, RandomRotation, CayleyRotation)
    from ..quant_utils.permutation import (
        PermutationBase, IdentityPermutation, RandomPermutation,
        MagnitudeSortPermutation)
except ImportError:
    from quant_utils.rotation import (
        RotationBase, IdentityRotation, HadamardRotation, RandomRotation, CayleyRotation)
    from quant_utils.permutation import (
        PermutationBase, IdentityPermutation, RandomPermutation,
        MagnitudeSortPermutation)

import copy


def make_rotation(name, block_size=16, seed=None):
    """Build a rotation instance from a name / instance / None.

    A single (shared) rotation instance can be reused across all layers
    because ``RotationBase`` caches its matrix per ``(n, device, dtype)``.
    """
    if name is None or name == "none":
        return IdentityRotation(block_size=block_size)
    if isinstance(name, RotationBase):
        return name
    if name == "hadamard":
        return HadamardRotation(block_size=block_size)
    if name == "random":
        return RandomRotation(block_size=block_size, seed=seed)
    if name == "cayley":
        # learnable: caller must .fit() on calibration weights before use
        return CayleyRotation(block_size=block_size, seed=seed)
    raise ValueError(f"unknown rotation: {name}")


def make_permutation(name, block_size=16, seed=None):
    """Build a PER-LAYER permutation FACTORY from a name / instance / None.

    Returns a callable ``factory(in_features) -> PermutationBase | None``.
    A fresh permutation instance is created for every layer (permutations are
    layer-specific and must NOT be shared), and is later fitted on that
    layer's real weight via ``NVFP4Linear.fit_permutation``.
    """
    if name is None or name == "none":
        return lambda in_features: IdentityPermutation(block_size=block_size)
    if isinstance(name, PermutationBase):
        return lambda in_features: copy.deepcopy(name)
    if name == "identity":
        return lambda in_features: IdentityPermutation(block_size=block_size)
    if name == "random":
        return lambda in_features: RandomPermutation(block_size=block_size, seed=seed)
    if name == "mag":
        return lambda in_features: MagnitudeSortPermutation(block_size=block_size)
    raise ValueError(f"unknown permutation: {name}")


# ===========================================================================
# Timestep & position embedding helpers (exact diffusers replicas)
# ===========================================================================

def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Exact replica of ``diffusers.models.embeddings.get_timestep_embedding``."""
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def _get_1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal encoding.  pos: (M,) -> (M, D)."""
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim))
    out = torch.outer(pos.to(torch.float64), omega)
    return torch.cat([out.sin(), out.cos()], dim=-1).float()


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

    ``rotation`` is a (shared) ``RotationBase`` instance or None. ``permutation``
    is a per-layer factory ``(in_features) -> PermutationBase | None`` (as
    produced by ``make_permutation``) — a fresh permutation is created for this
    layer so they are not shared across layers.
    """
    perm = permutation(in_features) if callable(permutation) else permutation
    return NVFP4Linear(
        in_features, out_features, bias=bias, block_size=block_size,
        rotation=rotation, permutation=perm,
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
            # diffusers uses sincos when interpolation_scale is set
            pos_embed = _get_2d_sincos(embed_dim, self.base_size, self.base_size,
                                       interpolation_scale=interpolation_scale)
            self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0))

    def _interpolate_pos_encoding(self, h: int, w: int) -> torch.Tensor:
        pe = self.pos_embed.reshape(1, self.base_size, self.base_size, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe.float(), size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return pe

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
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
                quantization_error_info: dict = None):
        """Returns ``(modulation_6D, conditioning_D)``."""
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype), quantization_error_info)

        guidance_proj = self.guidance_condition_proj(guidance)
        guidance_emb = self.guidance_embedder(
            guidance_proj.to(dtype=hidden_dtype), quantization_error_info)

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
        rotation=None,
        permutation=None,
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
        # rotation: shared RotationBase instance or None (caches per dim).
        # permutation: normalized into a per-layer factory so each NVFP4Linear
        # gets its OWN permutation, fitted on its real weight after copy.
        self.rotation = (rotation if isinstance(rotation, (RotationBase, type(None)))
                         else make_rotation(rotation, block_size))
        self.permutation_factory = (
            permutation if callable(permutation)
            else make_permutation(permutation, block_size))

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
                permutation=self.permutation_factory,
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
            permutation=self.permutation_factory,
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
                permutation=self.permutation_factory,
            )
            for i in range(num_layers)
        ])

        # Output
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim ** 0.5)
        self.norm_out = SanaModulatedNorm(inner_dim,
                                          elementwise_affine=False, eps=1e-6)
        self.proj_out = _make_linear(
            inner_dim, patch_size * patch_size * out_channels,
            bias=True, block_size=block_size, layer_prefix="proj_out",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation,
            permutation=self.permutation_factory)

        self.gradient_checkpointing = False
        self.quantization_error_info: dict = {}
        # Fit permutations on the (current) weights; for from_pretrained this is
        # redone after _copy_weights loads the real weights.
        self.fit_all_permutations()
        mode = "NVFP4" if use_nvfp4 else "unquantized"
        rot_name = type(self.rotation).__name__ if self.rotation is not None else "none"
        perm_name = (permutation if isinstance(permutation, str)
                     else type(self.permutation_factory(16)).__name__
                     if self.permutation_factory(16) is not None else "none")
        print(f"Initialize Sana Transformer ({mode} mode, block_size={block_size}, "
              f"rotation={rot_name}, permutation={perm_name})")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # Permutation fitting
    # ------------------------------------------------------------------

    def fit_all_permutations(self):
        """Fit every layer's permutation on the CURRENT (real) weights.

        Call this after weights are loaded (e.g. ``from_pretrained`` or
        ``_copy_weights``). Rotations are data-free (or pre-fitted by the
        caller) and need no action here.
        """
        for m in self.modules():
            if isinstance(m, NVFP4Linear):
                m.fit_permutation()

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
                print(f"  Using local cache directory: {candidate}")
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
        print(f"  Config: layers={cfg.get('num_layers')}, "
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
        )
        model._copy_weights(ref)
        del ref

        # Step 5: fit per-layer permutations on the real weights
        model.fit_all_permutations()

        # If a learnable (Cayley) rotation was requested, it must be fitted on
        # calibration data by the caller BEFORE construction; here we just
        # invalidate any cached matrix so it is rebuilt on the correct device.
        if model.rotation is not None:
            model.rotation.invalidate()

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
    
    import os

    model_id = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    download_source = os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["MODEL_CACHE_DIR"] = "G://models"
    cache_dir = os.environ.get("MODEL_CACHE_DIR", None)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        print(f"Model cache dir set to: {cache_dir}")
    
    """
    from PIL import ImageDraw

    block_size = 16

    # Toggle: False = standard nn.Linear (no quantization)
    USE_NVFP4 = True

    print("=" * 60)
    mode_str = "NVFP4" if USE_NVFP4 else "unquantized"
    print(f"Test: NVFP4QuantizedSana ({mode_str} mode) via from_pretrained")

    print(f"Loading from {model_id} (source: {download_source}) ...")
    model = NVFP4QuantizedSana.from_pretrained(
        model_id, download_source=download_source, cache_dir=cache_dir,
        block_size=block_size, use_nvfp4=USE_NVFP4
    )
    model.to(device=device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  NVFP4 mode: {USE_NVFP4}")

    if USE_NVFP4:
        print("  NVFP4 quantization active [OK]")
    else:
        print("  Unquantized mode -- using standard nn.Linear [OK]")
    # ====================================================================
    # Inference
    # ====================================================================
    print("  Loading SanaSprintPipeline ...")
    from diffusers import SanaSprintPipeline

    pipe_load_kwargs = dict(torch_dtype=torch.bfloat16)
    if cache_dir:
        pipe_load_kwargs["cache_dir"] = cache_dir

    if download_source == "modelscope":
        from modelscope import snapshot_download
        pipe_local = snapshot_download(
            model_id, cache_dir=cache_dir or ".cache/modelscope",
        )
        pipe = SanaSprintPipeline.from_pretrained(
            pipe_local, local_files_only=True, **pipe_load_kwargs,
        )
    else:
        pipe = SanaSprintPipeline.from_pretrained(
            model_id, local_files_only=True, **pipe_load_kwargs,
        )

    # Discard unused transformer and slot in ours
    _unused = pipe.transformer
    pipe.transformer = None
    del _unused
    if device == "cuda":
        torch.cuda.empty_cache()

    pipe.transformer = model
    print("  Swapped NVFP4 transformer into pipeline [OK]")

    if device == "cuda":
        pipe.to(device)

    # ---- Generate image ----
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    print(f"  Generating image with prompt: {prompt}")
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=2,
            guidance_scale=4.5,
            height=1024, width=1024,
        ).images[0]

    # ---- Save ----
    output_dir = "G:/Outputs/Efficient-Diffusion/images"
    os.makedirs(output_dir, exist_ok=True)

    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), prompt)
    text_height = text_bbox[3] - text_bbox[1]
    padding = 8
    draw.rectangle(
        [(0, 0), (image.width, text_height + padding * 2)],
        fill=(0, 0, 0, 180),
    )
    draw.text((padding, padding), prompt, fill=(255, 255, 255))

    save_path = os.path.join(output_dir, "sana_nvfp4_sample_bs16.png")
    image.save(save_path)
    print(f"  Image saved to: {save_path}")
    """
    # ====================================================================
    # NVFP4 Correctness Verification: three-way forward output comparison
    #   (1) diffusers original vs (2) NVFP4QuantizedSana(unquantized) vs (3) NVFP4QuantizedSana(NVFP4)
    # ====================================================================
    """
    print("\n\n" + "=" * 70)
    print("NVFP4 Correctness Verification: Three-way forward comparison")
    print("=" * 70)
    block_size = 16

    print("\n[Loading original diffusers model for reference ...]")
    from diffusers import SanaTransformer2DModel

    local_path = NVFP4QuantizedSana._resolve_checkpoint_path(
        model_id,
        download_source=os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower(),
        cache_dir=cache_dir,
    )
    ref_model = SanaTransformer2DModel.from_pretrained(
        local_path, subfolder="transformer", local_files_only=True,
    ).to(device).eval()
    print(f"  diffusers SanaTransformer2DModel loaded [OK]")

    print("\n[Loading NVFP4QuantizedSana (use_nvfp4=False, unquantized) ...]")
    model_uq = NVFP4QuantizedSana.from_pretrained(
        model_id, use_nvfp4=False,
        download_source=os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower(),
        cache_dir=cache_dir,
    ).to(device).eval()
    print(f"  model_uq loaded [OK]")

    print("\n[Loading NVFP4QuantizedSana (use_nvfp4=True, block_size=256) ...]")
    model_q = NVFP4QuantizedSana.from_pretrained(
        model_id, block_size=block_size, use_nvfp4=True, rotation="hadamard", permutation="mag",
        download_source=os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower(),
        cache_dir=cache_dir,
    ).to(device).eval()
    print(f"  model_q loaded [OK]")

    # Construct identical inputs
    B, C, H, W = 1, 32, 32, 32
    torch.manual_seed(1234)
    hs = torch.randn(B, C, H, W, device=device)
    txt = torch.randn(B, 77, 2304, device=device)
    t = torch.tensor([500], device=device)
    g = torch.tensor([1.0], device=device)
    print(f"\n  Input: latent={tuple(hs.shape)}, text={tuple(txt.shape)}, t={t.item()}")

    # Forward
    with torch.no_grad():
        out_ref = ref_model(
            hidden_states=hs, encoder_hidden_states=txt,
            timestep=t, guidance=g,
        ).sample
        out_uq = model_uq(
            hs.clone(), txt.clone(), t.clone(), guidance=g.clone(),
        )[0]
        error_info = {}
        out_q = model_q(
            hs.clone(), txt.clone(), t.clone(), guidance=g.clone(),
            quantization_error_info=error_info,
        )[0]

    def _cmp(a, b, label_a, label_b):
        a_f, b_f = a.float().reshape(-1), b.float().reshape(-1)
        cos = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
        mse = F.mse_loss(a_f, b_f).item()
        l1 = F.l1_loss(a_f, b_f).item()
        rel = (a_f - b_f).abs().sum() / (a_f.abs().sum() + 1e-12)
        max_e = (a_f - b_f).abs().max().item()
        print(f"\n  [{label_a}] vs [{label_b}]:")
        print(f"    Cosine similarity: {cos:.10f}", "✓" if cos > 0.999999 else "✗ BUG?")
        print(f"    MSE:              {mse:.10f}")
        print(f"    L1:               {l1:.10f}")
        print(f"    Relative error:   {rel:.6%}")
        print(f"    Max absolute err: {max_e:.6f}")
        print(f"    a range: [{a_f.min().item():.4f}, {a_f.max().item():.4f}]")
        print(f"    b range: [{b_f.min().item():.4f}, {b_f.max().item():.4f}]")

    print("\n" + "-" * 70)
    print("Step 1: Architecture correctness (diffusers original vs our unquantized)")
    print("        -> Cosine sim should be ~1.0000000000, otherwise there is a bug")
    _cmp(out_ref, out_uq, "diffusers(original)", "our(unquantized)")

    print("\n" + "-" * 70)
    print("Step 2: FP4 quantization error (unquantized vs NVFP4)")
    print("        -> Cosine sim reflects FP4 quantization precision loss")
    _cmp(out_uq, out_q, "our(unquantized)", "our(NVFP4)")

    print("\n" + "-" * 70)
    print("Step 3: End-to-end (diffusers original vs NVFP4)")
    print("        -> Combined architecture + quantization error")
    _cmp(out_ref, out_q, "diffusers(original)", "our(NVFP4)")

    # Per-layer quantization error top-10
    if error_info:
        layer_errors = [(k, v) for k, v in error_info.items() if 'nvfp4_error' in k]
        layer_errors.sort(key=lambda x: -x[1])
        print(f"\nPer-layer NVFP4 errors (top 10 of {len(layer_errors)}):")
        for name, val in layer_errors[:10]:
            print(f"  {name:70s} = {val:.6e}")

    del ref_model, model_uq, model_q
    if device == "cuda":
        torch.cuda.empty_cache()
    """
