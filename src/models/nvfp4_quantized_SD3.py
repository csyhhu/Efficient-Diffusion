"""
NVFP4-quantized SD3 (Stable Diffusion 3) Transformer — exact replica of
diffusers ``SD3Transformer2DModel``.

Architecture mirrors HuggingFace ``SD3Transformer2DModel`` (MMDiT joint
image-text attention) so that ``from_pretrained`` can load official SD3
checkpoints. All ``nn.Linear`` layers can be toggled between NVFP4-quantized
and standard mode via the ``use_nvfp4`` flag.

Usage::

    from src.models.nvfp4_quantized_SD3 import NVFP4QuantizedSD3

    # Non-quantized (standard nn.Linear everywhere):
    model = NVFP4QuantizedSD3.from_pretrained(
        "AI-ModelScope/stable-diffusion-3-medium-diffusers",
        use_nvfp4=False,
    )

    # NVFP4-quantized:
    model = NVFP4QuantizedSD3.from_pretrained(
        "AI-ModelScope/stable-diffusion-3-medium-diffusers",
        block_size=16,
    )

    # NVFP4 + Hadamard rotation + magnitude-sort permutation:
    model = NVFP4QuantizedSD3.from_pretrained(
        "AI-ModelScope/stable-diffusion-3-medium-diffusers",
        block_size=16,
        rotation="hadamard",
        permutation="mag",
    )
"""

import math
import os
import gc
import json
from typing import Optional, List, Any
from types import SimpleNamespace
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quant_utils.rotation import (
    RotationBase, IdentityRotation, HadamardRotation, RandomRotation,
    CayleyRotation, make_rotation,
)
from src.quant_utils.permutation import (
    PermutationBase, IdentityPermutation, RandomPermutation,
    MagnitudeSortPermutation, make_permutation,
)
from src.modules.quantized_linear import NVFP4Linear

from src.utils import compute_computation_diff


# ===========================================================================
# Sinusoidal position-encoding helpers (exact diffusers replicas)
# ===========================================================================

def _get_1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal encoding.  pos: (M,) -> (M, D)."""
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim))
    out = torch.outer(pos.to(torch.float64), omega)
    return torch.cat([out.sin(), out.cos()], dim=-1).to(pos.dtype)


def _get_2d_sincos(embed_dim: int, h: int, w: int,
                   base_size: int = 16,
                   device: torch.device = None) -> torch.Tensor:
    """2D sinusoidal position encoding. Returns (h*w, embed_dim)."""
    grid_h = torch.arange(h, dtype=torch.float32, device=device) / (h / base_size)
    grid_w = torch.arange(w, dtype=torch.float32, device=device) / (w / base_size)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_h = _get_1d_sincos(embed_dim // 2, grid_h.reshape(-1))
    emb_w = _get_1d_sincos(embed_dim // 2, grid_w.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=-1)


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings — matches diffusers' get_timestep_embedding.

    Computes in float32 (like diffusers) to avoid bfloat16 precision loss
    in the sinusoidal computation, then casts back to the input dtype.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    if flip_sin_to_cos:
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
    else:
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb.to(timesteps.dtype)


# ===========================================================================
# NVFP4 Linear factory
# ===========================================================================

def _make_linear(in_features: int, out_features: int, bias: bool = True,
                 block_size: int = 16, layer_prefix: str = None,
                 use_nvfp4: bool = True, rotation=None,
                 permutation=None) -> nn.Module:
    """Factory: returns an ``NVFP4Linear`` with ``quantize=use_nvfp4``.

    ``rotation`` / ``permutation`` are passed as strings (or None /
    instances) to ``NVFP4Linear``, which calls ``make_rotation`` /
    ``make_permutation`` internally to create a fresh per-layer instance.
    """
    return NVFP4Linear(
        in_features, out_features, bias=bias, block_size=block_size,
        rotation=rotation, permutation=permutation,
        layer_prefix=layer_prefix, quantize=use_nvfp4,
    )


# ===========================================================================
# Timestep & text embedding modules (with NVFP4 support)
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


class PixArtAlphaTextProjection(nn.Module):
    """Projects pooled text embedding -> conditioning dimension.
    Matches diffusers ``PixArtAlphaTextProjection``. Uses SiLU activation."""

    def __init__(self, in_features: int, hidden_size: int,
                 out_features: Optional[int] = None,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        out_features = out_features or hidden_size
        self.linear_1 = _make_linear(in_features, hidden_size, bias=True,
                                     block_size=block_size,
                                     layer_prefix=f"{layer_prefix}.linear_1",
                                     use_nvfp4=use_nvfp4,
                                     rotation=rotation, permutation=permutation)
        self.act = nn.SiLU()
        self.linear_2 = _make_linear(hidden_size, out_features, bias=True,
                                     block_size=block_size,
                                     layer_prefix=f"{layer_prefix}.linear_2",
                                     use_nvfp4=use_nvfp4,
                                     rotation=rotation, permutation=permutation)

    def forward(self, caption: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        hidden_states = self.linear_1(caption, quantization_error_info)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states, quantization_error_info)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Fuse timestep and pooled text into a single conditioning vector.
    Matches diffusers ``CombinedTimestepTextProjEmbeddings``.

    Structure::

        time_proj (Timesteps)  -> timestep_embedder (256->D)  -\\
        text_embedder (pooled_dim -> D)                       --+ -> sum
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int,
                 block_size: int = 16,
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
        self.text_embedder = PixArtAlphaTextProjection(
            in_features=pooled_projection_dim, hidden_size=embedding_dim,
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.text_embedder",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=pooled_projection.dtype),
            quantization_error_info)
        pooled_projections = self.text_embedder(
            pooled_projection, quantization_error_info)
        return timesteps_emb + pooled_projections


# ===========================================================================
# Patch embedding (SD3 uses pos_embed_max_size with center cropping)
# ===========================================================================

class PatchEmbed(nn.Module):
    """2D latent -> patch tokens with sin-cos position embedding.
    Matches diffusers ``PatchEmbed`` for SD3 (center-crop pos_embed)."""

    def __init__(self, height: int, width: int, patch_size: int,
                 in_channels: int, embed_dim: int,
                 pos_embed_max_size: Optional[int] = None):
        super().__init__()
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

        grid = pos_embed_max_size if pos_embed_max_size is not None else max(self.height, self.width)
        pos_embed = _get_2d_sincos(embed_dim, grid, grid)
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0))

    def _crop_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """Crop pre-computed position embedding to the given height/width (center)."""
        max_size = self.pos_embed_max_size
        top = (max_size - h) // 2
        left = (max_size - w) // 2
        pe = self.pos_embed.reshape(1, max_size, max_size, -1)
        pe = pe[:, top:top + h, left:left + w, :]
        return pe.reshape(1, h * w, -1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(self.proj.weight.dtype)
        B, C, H, W = latent.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.proj(latent).flatten(2).transpose(1, 2)  # BCHW -> BNC

        if self.pos_embed_max_size is not None:
            pos_embed = self._crop_pos_embed(h, w)
        else:
            pos_embed = self.pos_embed[:, :h * w, :]
        return (x + pos_embed).to(x.dtype)


# ===========================================================================
# Adaptive normalization layers (with NVFP4 support)
# ===========================================================================

class AdaLayerNormZero(nn.Module):
    """adaLN-Zero: layer norm + 6-dim scale/shift/gate modulation.
    Matches diffusers ``AdaLayerNormZero``."""

    def __init__(self, embedding_dim: int, bias: bool = True,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.silu = nn.SiLU()
        self.linear = _make_linear(embedding_dim, 6 * embedding_dim, bias=bias,
                                   block_size=block_size,
                                   layer_prefix=f"{layer_prefix}.linear",
                                   use_nvfp4=use_nvfp4,
                                   rotation=rotation, permutation=permutation)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                quantization_error_info: dict = None):
        emb = self.linear(self.silu(emb), quantization_error_info)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SD35AdaLayerNormZeroX(nn.Module):
    """SD3.5 adaLN-Zero variant for dual attention blocks.
    Outputs 9*dim: 6 for attn1+FF + 3 for attn2 (shift_msa2, scale_msa2, gate_msa2).
    Matches diffusers ``SD35AdaLayerNormZeroX``."""

    def __init__(self, embedding_dim: int, bias: bool = True,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.silu = nn.SiLU()
        self.linear = _make_linear(embedding_dim, 9 * embedding_dim, bias=bias,
                                   block_size=block_size,
                                   layer_prefix=f"{layer_prefix}.linear",
                                   use_nvfp4=use_nvfp4,
                                   rotation=rotation, permutation=permutation)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                quantization_error_info: dict = None):
        emb = self.linear(self.silu(emb), quantization_error_info)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, \
            shift_msa2, scale_msa2, gate_msa2 = emb.chunk(9, dim=1)
        norm_hidden_states = self.norm(x)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, \
            norm_hidden_states2, gate_msa2


class AdaLayerNormContinuous(nn.Module):
    """Continuous adaLN: layer norm + 2-dim scale/shift modulation.
    Matches diffusers ``AdaLayerNormContinuous``."""

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int,
                 elementwise_affine: bool = False, eps: float = 1e-6,
                 bias: bool = True,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.silu = nn.SiLU()
        self.linear = _make_linear(conditioning_embedding_dim, embedding_dim * 2,
                                   bias=bias, block_size=block_size,
                                   layer_prefix=f"{layer_prefix}.linear",
                                   use_nvfp4=use_nvfp4,
                                   rotation=rotation, permutation=permutation)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps,
                                 elementwise_affine=elementwise_affine, bias=bias)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype),
                          quantization_error_info)
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ===========================================================================
# GELU activation wrapper (matches diffusers GELU module)
# ===========================================================================

class GELUActivation(nn.Module):
    """Linear + GELU activation. Matches diffusers ``GELU``.

    ``proj: Linear(dim, inner_dim)`` then ``GELU(proj(x))``.
    No gating (unlike GEGLU).  SD3 uses this activation in FeedForward.
    """

    def __init__(self, dim: int, inner_dim: int, bias: bool = True,
                 block_size: int = 16, layer_prefix: str = None,
                 use_nvfp4: bool = True, rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.proj = _make_linear(dim, inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.proj",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)

    def forward(self, hidden_states: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        hidden_states = self.proj(hidden_states, quantization_error_info)
        return F.gelu(hidden_states, approximate="tanh")


# ===========================================================================
# FeedForward (with NVFP4 support — matches diffusers FeedForward with GELU)
# ===========================================================================

class FeedForward(nn.Module):
    """GELU-based feed-forward layer. Matches diffusers ``FeedForward``.

    Structure (same as diffusers)::
        net.0: GELU(dim, inner_dim)    -> Linear(dim, inner_dim) + GELU
        net.1: Dropout(0.0)
        net.2: Linear(inner_dim, dim_out)
    """

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 bias: bool = True,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.net = nn.Sequential(
            GELUActivation(
                dim, inner_dim, bias=bias,
                block_size=block_size,
                layer_prefix=f"{layer_prefix}.net.0",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation),
            nn.Dropout(0.0),
            _make_linear(inner_dim, dim_out, bias=bias,
                         block_size=block_size,
                         layer_prefix=f"{layer_prefix}.net.2",
                         use_nvfp4=use_nvfp4,
                         rotation=rotation, permutation=permutation),
        )

    def forward(self, hidden_states: torch.Tensor,
                quantization_error_info: dict = None,
                computation_diff_dict: dict = None) -> torch.Tensor:
        """Forward pass with optional change capture.

        Args:
            return_computation_diff: when True, write FF-module before/after tensors and
                per-token rel_l2 into ``self.computation_diff_dict``. Tensors are stored
                detached + on CPU (``.detach().cpu()``) so the dict can be
                inspected after forward without holding the autograd graph.
        """
        if computation_diff_dict is not None:
            computation_diff_dict[f"{self.layer_prefix}.before"] = hidden_states.detach().cpu()
        # net[0] is GELUActivation (needs quantization_error_info)
        # net[1] is Dropout (no args)
        # net[2] is NVFP4Linear (needs quantization_error_info)
        hidden_states = self.net[0](hidden_states, quantization_error_info)
        hidden_states = self.net[1](hidden_states)
        hidden_states = self.net[2](hidden_states, quantization_error_info)
        if computation_diff_dict is not None:
            computation_diff_dict[f"{self.layer_prefix}.after"] = hidden_states.detach().cpu()
        return hidden_states


# ===========================================================================
# Joint Attention (MMDiT — with NVFP4 support)
# ===========================================================================

class JointAttention(nn.Module):
    """Joint image-text attention used in SD3 MMDiT blocks.
    Matches diffusers ``JointAttention``. Uses ``F.scaled_dot_product_attention``.

    Supports ``context_pre_only`` (last block skips text output).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 context_pre_only: bool = False,
                 qk_norm: Optional[str] = None, bias: bool = True,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None,
                 has_added_kv: bool = True):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.has_added_kv = has_added_kv
        # attn2 (pure self-attention) has no text stream, so it never
        # produces a context output.
        self.context_pre_only = context_pre_only if has_added_kv else True

        # Image QKV
        self.to_q = _make_linear(dim, self.inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.to_q",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)
        self.to_k = _make_linear(dim, self.inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.to_k",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)
        self.to_v = _make_linear(dim, self.inner_dim, bias=bias,
                                 block_size=block_size,
                                 layer_prefix=f"{layer_prefix}.to_v",
                                 use_nvfp4=use_nvfp4,
                                 rotation=rotation, permutation=permutation)

        # Text (added) KV — only for joint attention (attn1).
        # attn2 in SD3.5 dual-attention blocks is a pure self-attention
        # without added KV projections.
        if has_added_kv:
            self.add_k_proj = _make_linear(dim, self.inner_dim, bias=bias,
                                           block_size=block_size,
                                           layer_prefix=f"{layer_prefix}.add_k_proj",
                                           use_nvfp4=use_nvfp4,
                                           rotation=rotation, permutation=permutation)
            self.add_v_proj = _make_linear(dim, self.inner_dim, bias=bias,
                                           block_size=block_size,
                                           layer_prefix=f"{layer_prefix}.add_v_proj",
                                           use_nvfp4=use_nvfp4,
                                           rotation=rotation, permutation=permutation)
            self.added_proj_bias = True
            # Note: diffusers always creates add_q_proj regardless of context_pre_only;
            # only to_add_out (text output projection) is conditional.
            self.add_q_proj = _make_linear(dim, self.inner_dim, bias=bias,
                                           block_size=block_size,
                                           layer_prefix=f"{layer_prefix}.add_q_proj",
                                           use_nvfp4=use_nvfp4,
                                           rotation=rotation, permutation=permutation)
        else:
            self.add_k_proj = None
            self.add_v_proj = None
            self.add_q_proj = None
            self.added_proj_bias = False

        # Output
        self.to_out = nn.ModuleList([
            _make_linear(self.inner_dim, dim, bias=bias,
                         block_size=block_size,
                         layer_prefix=f"{layer_prefix}.to_out.0",
                         use_nvfp4=use_nvfp4,
                         rotation=rotation, permutation=permutation),
            nn.Dropout(0.0),
        ])
        if not self.context_pre_only:
            self.to_add_out = _make_linear(self.inner_dim, dim, bias=bias,
                                           block_size=block_size,
                                           layer_prefix=f"{layer_prefix}.to_add_out",
                                           use_nvfp4=use_nvfp4,
                                           rotation=rotation, permutation=permutation)
        else:
            self.to_add_out = None

        # QK norm (optional)
        self.qk_norm = qk_norm
        if qk_norm == "rms_norm":
            from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
            self.norm_q = DiffusersRMSNorm(head_dim, eps=1e-6)
            self.norm_k = DiffusersRMSNorm(head_dim, eps=1e-6)
            if has_added_kv:
                self.norm_added_q = DiffusersRMSNorm(head_dim, eps=1e-6)
                self.norm_added_k = DiffusersRMSNorm(head_dim, eps=1e-6)
            else:
                self.norm_added_q = None
                self.norm_added_k = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, inner_dim) -> (B, num_heads, N, head_dim)."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_heads, N, head_dim) -> (B, N, inner_dim)."""
        B, _, N, _ = x.shape
        return x.transpose(1, 2).reshape(B, N, -1)

    def forward(
        self, hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        quantization_error_info: dict = None,
        computation_diff_dict: dict = None,
        # skip_plan: dict = None # [seq_len, num_head]
    ):
        """Args:
            hidden_states: (B, N_img, D) image tokens.
            encoder_hidden_states: (B, N_txt, D) text tokens (ignored when
            ``has_added_kv=False``).
            computation_diff_dict: capture before/after tensors and attention
                weights into ``self.computation_diff_dict``. Tensors are stored detached +
                on CPU (``.detach().cpu()``) so the dict can be inspected after
                forward without holding the autograd graph.
        Returns:
            attn_img: (B, N_img, D), attn_txt: (B, N_txt, D) or None.
        """
        # Each module owns its own self.computation_diff_dict. Clear at the START of each
        # forward so repeated calls don't accumulate stale entries.
        # print(f"Processing {self.layer_prefix} ...")
        if computation_diff_dict is not None:
            img_before = hidden_states
            txt_before = encoder_hidden_states

        batch_size = hidden_states.shape[0]
        n_img = hidden_states.shape[1]

        # Image Q, K, V
        query = self.to_q(hidden_states, quantization_error_info)
        key = self.to_k(hidden_states, quantization_error_info)
        value = self.to_v(hidden_states, quantization_error_info)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # QK norm on image Q/K BEFORE concat (matches diffusers)
        if self.qk_norm == "rms_norm":
            query = self.norm_q(query)
            key = self.norm_k(key)

        # Pure self-attention path (attn2 in SD3.5 dual-attention blocks):
        if not self.has_added_kv:
            # Always use fused SDPA for the actual output — guarantees the
            # instrumented path produces byte-identical output to baseline.
            attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            """
            if computation_diff_dict is not None:
                # As a separate BYPRODUCT compute softmax(QK^T/√d) for weight
                # recovery.  This duplicates the Q·Kᵀ matmul when diff is
                # requested, but keeps the SDPA path identical across runs.
                scale = 1.0 / math.sqrt(self.head_dim)
                with torch.no_grad():
                    attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
                    attn_weights = F.softmax(attn_scores, dim=-1).to(query.dtype)
                # computation_diff_dict[f"{self.layer_prefix}.attn_weights"] = attn_weights.detach().cpu()
                # [bs, n_heads, n_img, dim] => [n_heads, n_img]
                computation_diff_dict[f"{self.layer_prefix}.attn"] = compute_computation_diff(value, attn_output, dim=[0, -1])
            """
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
            attn_output = attn_output.to(query.dtype)
            attn_output = self.to_out[0](attn_output, quantization_error_info)
            attn_output = self.to_out[1](attn_output)
            return attn_output, None

        # Text (context) Q, K, V
        encoder_query = self.add_q_proj(encoder_hidden_states, quantization_error_info)
        encoder_key = self.add_k_proj(encoder_hidden_states, quantization_error_info)
        encoder_value = self.add_v_proj(encoder_hidden_states, quantization_error_info)

        encoder_query = encoder_query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        encoder_key = encoder_key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        encoder_value = encoder_value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # QK norm on text Q/K BEFORE concat (matches diffusers)
        if self.qk_norm == "rms_norm":
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

        # Concatenate image + text, single SDPA call (matches diffusers)
        query = torch.cat([query, encoder_query], dim=2)
        key = torch.cat([key, encoder_key], dim=2)
        value = torch.cat([value, encoder_value], dim=2)

        # Always use fused SDPA for actual output (identical regardless of
        # return_computation_diff). Weight recovery is an orthogonal byproduct computed
        # below when needed — duplicates Q·Kᵀ matmul, keeps output identical.
        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        """
        if computation_diff_dict is not None:
            scale = 1.0 / math.sqrt(self.head_dim)
            with torch.no_grad():
                attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_scores, dim=-1).to(query.dtype)
            # computation_diff_dict[f"{self.layer_prefix}.attn_weights"] = attn_weights.detach().cpu()
            # computation_diff_dict[f"{self.layer_prefix}.attn"]= {"before": value.detach().cpu()}   
            # computation_diff_dict[f"{self.layer_prefix}.attn"]["after"] = attn_output.detach().cpu()
            computation_diff_dict[f"{self.layer_prefix}.attn"]= compute_computation_diff(value, attn_output, dim=[0, -1])
        """
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        attn_output = attn_output.to(query.dtype)

        # Split attention outputs
        attn_img, attn_txt = attn_output[:, :n_img], attn_output[:, n_img:]

        attn_img = self.to_out[0](attn_img, quantization_error_info)
        attn_img = self.to_out[1](attn_img)

        if self.context_pre_only:
            if computation_diff_dict is not None:
                computation_diff_dict[f"{self.layer_prefix}.img"] = compute_computation_diff(img_before, attn_img, dim=[0, -1])
            return attn_img, None

        attn_txt = self.to_add_out(attn_txt, quantization_error_info)
        # Scale A: joint-attn final outputs (both streams, after output projection)
        if computation_diff_dict is not None:
            computation_diff_dict[f"{self.layer_prefix}.txt"] = compute_computation_diff(txt_before, attn_txt, dim=[0, -1])
        return attn_img, attn_txt


# ===========================================================================
# JointTransformerBlock — MMDiT block (with NVFP4 support)
# ===========================================================================

class JointTransformerBlock(nn.Module):
    """MMDiT block used in SD3 — joint image+text attention + dual FeedForward.
    Matches diffusers ``JointTransformerBlock``."""

    def __init__(self, dim: int, num_attention_heads: int,
                 attention_head_dim: int,
                 context_pre_only: bool = False,
                 qk_norm: Optional[str] = None,
                 use_dual_attention: bool = False,
                 block_size: int = 16,
                 layer_prefix: str = None,
                 use_nvfp4: bool = True,
                 rotation=None, permutation=None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        # Image-stream norm
        # SD3.5 dual-attention blocks use SD35AdaLayerNormZeroX (9*dim output:
        # 6 for attn1+FF + 3 for attn2). Other blocks use AdaLayerNormZero (6*dim).
        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(
                dim, block_size=block_size,
                layer_prefix=f"{layer_prefix}.norm1",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation)
        else:
            self.norm1 = AdaLayerNormZero(
                dim, block_size=block_size,
                layer_prefix=f"{layer_prefix}.norm1",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation)

        # Text-stream norm
        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True,
                block_size=block_size,
                layer_prefix=f"{layer_prefix}.norm1_context",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation)
        else:
            self.norm1_context = AdaLayerNormZero(
                dim, block_size=block_size,
                layer_prefix=f"{layer_prefix}.norm1_context",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation)

        # Joint attention
        self.attn = JointAttention(
            dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim,
            context_pre_only=context_pre_only, qk_norm=qk_norm, bias=True,
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.attn",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation)

        # Second self-attention (SD3.5 dual attention)
        # attn2 is a pure self-attention (no added KV / text projections).
        if use_dual_attention:
            self.attn2 = JointAttention(
                dim=dim, num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=context_pre_only, qk_norm=qk_norm, bias=True,
                block_size=block_size,
                layer_prefix=f"{layer_prefix}.attn2",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation,
                has_added_kv=False)
        else:
            self.attn2 = None

        # Image-stream FF
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, dim_out=dim,
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.ff",
            use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation)

        # Text-stream FF
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(
                dim=dim, dim_out=dim,
                block_size=block_size,
                layer_prefix=f"{layer_prefix}.ff_context",
                use_nvfp4=use_nvfp4,
                rotation=rotation, permutation=permutation)
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor,
                skip_plan: dict or float = None,
                quantization_error_info: dict = None,
                computation_diff_dict: dict = None) -> tuple:

        if computation_diff_dict is not None: 
            img_before = hidden_states
            txt_before = encoder_hidden_states

        n_latent_seq = hidden_states.shape[1]
        n_txt_seq = encoder_hidden_states.shape[1]
        # print(skip_plan.keys())
        if isinstance(skip_plan, dict):
            token_skip_plan = skip_plan.get(f"{self.layer_prefix}", None)
            # print(len(token_skip_plan))
            if token_skip_plan is not None:
                latent_skip_token = [t for t in token_skip_plan if t < n_latent_seq]
                text_skip_token = [t - n_latent_seq for t in token_skip_plan if t >= n_latent_seq and t < n_txt_seq + n_latent_seq]
                # print(len(latent_skip_token), len(text_skip_token))
            else:
                latent_skip_token = None
                text_skip_token = None
        elif isinstance(skip_plan, float):
            latent_skip_token = torch.randperm(n_latent_seq)[:int(skip_plan * n_latent_seq)]
            text_skip_token = torch.randperm(n_txt_seq)[:int(skip_plan * n_txt_seq)]
        else:
            latent_skip_token = None
            text_skip_token = None

        if latent_skip_token is not None and len(latent_skip_token) > 0:
            full_hidden_states = hidden_states
            latent_skip_token = torch.tensor(latent_skip_token)
            latent_skip_mask = torch.zeros(n_latent_seq, dtype=torch.bool)
            # print(latent_skip_token)
            latent_skip_mask[latent_skip_token] = True
            latent_keep_mask = ~latent_skip_mask
            hidden_states = full_hidden_states[:, latent_keep_mask, :]
            # print(f"[{self.layer_prefix}] skip img tokens: {latent_skip_token} | {full_hidden_states.shape} -> {hidden_states.shape}")
        
        if text_skip_token is not None and len(text_skip_token) > 0:
            full_encoder_hidden_states = encoder_hidden_states
            text_skip_token = torch.tensor(text_skip_token)
            text_skip_mask = torch.zeros(n_txt_seq, dtype=torch.bool)
            text_skip_mask[text_skip_token] = True
            text_keep_mask = ~text_skip_mask
            encoder_hidden_states = full_encoder_hidden_states[:, text_keep_mask, :]
            # print(f"[{self.layer_prefix}] skip txt tokens: {text_skip_token} | {full_encoder_hidden_states.shape} -> {encoder_hidden_states.shape}")

        # ---- Image norm ----
        # Dual-attention blocks use SD35AdaLayerNormZeroX which returns 7
        # values (the extra norm_hidden_states2 / gate_msa2 feed attn2).
        if self.use_dual_attention:
            (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp,
             norm_hidden_states2, gate_msa2) = \
                self.norm1(hidden_states, emb=temb,
                           quantization_error_info=quantization_error_info)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.norm1(hidden_states, emb=temb,
                           quantization_error_info=quantization_error_info)

        # ---- Text norm ----
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(
                encoder_hidden_states, temb,
                quantization_error_info=quantization_error_info)
        else:
            (norm_encoder_hidden_states, c_gate_msa,
             c_shift_mlp, c_scale_mlp, c_gate_mlp) = \
                self.norm1_context(encoder_hidden_states, emb=temb,
                                   quantization_error_info=quantization_error_info)

        # ---- Joint attention ----
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            quantization_error_info=quantization_error_info,
            computation_diff_dict=computation_diff_dict,
        )

        # Image: residual + gate
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # ---- Second self-attention (SD3.5 dual attention) ----
        if self.use_dual_attention:
            attn_output2, _ = self.attn2(
                hidden_states=norm_hidden_states2,
                encoder_hidden_states=norm_encoder_hidden_states,
                quantization_error_info=quantization_error_info,
                computation_diff_dict=computation_diff_dict,
            )
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        # Image: FF
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(
            norm_hidden_states,
            quantization_error_info=quantization_error_info,
            # computation_diff_dict=computation_diff_dict
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # ---- Text path ----
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(
                norm_encoder_hidden_states,
                quantization_error_info=quantization_error_info,
                # computation_diff_dict=computation_diff_dict,
            )
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if latent_skip_token is not None and len(latent_skip_token) > 0:
            full_hidden_states[:, latent_keep_mask, :] = hidden_states
            hidden_states = full_hidden_states

        if text_skip_token is not None and len(text_skip_token) > 0:
            full_encoder_hidden_states[:, text_keep_mask, :] = encoder_hidden_states
            encoder_hidden_states = full_encoder_hidden_states

        # block outputs (after full attn + FF + residual + gate)
        # NOTE: capture diff AFTER skip restoration so img_after/txt_after have
        # the same full shape as img_before/txt_before (captured at block entry).
        if computation_diff_dict is not None:
            img_after = hidden_states
            # [bs, n_img, dim] => [n_img]
            computation_diff_dict[f"{self.layer_prefix}.img"] = compute_computation_diff(img_before, img_after, dim=[0, -1])
            if encoder_hidden_states is not None:
                txt_after = encoder_hidden_states
                computation_diff_dict[f"{self.layer_prefix}.txt"] = compute_computation_diff(txt_before[:, :77, :], txt_after[:, :77, :], dim=[0, -1])

        return encoder_hidden_states, hidden_states


# ===========================================================================
# NVFP4 Quantized SD3 (main model — exact diffusers SD3Transformer2DModel)
# ===========================================================================

class NVFP4QuantizedSD3(nn.Module):
    """SD3 Transformer with optional NVFP4 quantization.

    Exact architecture replica of ``diffusers.SD3Transformer2DModel``.
    Set ``use_nvfp4=False`` for standard ``nn.Linear`` (correctness testing).

    SD3 medium defaults::

        model = NVFP4QuantizedSD3(
            sample_size=128, patch_size=2, in_channels=16,
            num_layers=24, attention_head_dim=64, num_attention_heads=24,
            joint_attention_dim=4096, caption_projection_dim=1536,
            pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=96,
            dual_attention_layers=(), qk_norm="rms_norm",
        )
    """

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple = (),
        qk_norm: Optional[str] = "rms_norm",
        block_size: int = 16,
        use_nvfp4: bool = True,
        rotation="identity",
        permutation="identity"
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
        )

        # SD3 architecture constraint: text and image streams share the same dim
        if caption_projection_dim != inner_dim:
            raise ValueError(
                f"caption_projection_dim ({caption_projection_dim}) must equal "
                f"inner_dim ({inner_dim}) for SD3 MMDiT architecture."
            )

        # Position embedding (Conv2d + sin-cos PE with center crop)
        self.pos_embed = PatchEmbed(
            height=sample_size, width=sample_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        # Time + pooled text -> conditioning
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            block_size=block_size,
            layer_prefix="time_text_embed",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation,
            permutation=self.permutation,
        )

        # Context projection: [clip + t5] hidden -> caption_projection_dim
        self.context_embedder = _make_linear(
            joint_attention_dim, caption_projection_dim, bias=True,
            block_size=block_size, layer_prefix="context_embedder",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation, permutation=self.permutation,
        )

        # Transformer blocks (MMDiT)
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                context_pre_only=(i == num_layers - 1),
                qk_norm=qk_norm,
                use_dual_attention=(i in dual_attention_layers),
                block_size=block_size,
                layer_prefix=f"block.{i}",
                use_nvfp4=use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim,
            elementwise_affine=False, eps=1e-6, bias=True,
            block_size=block_size,
            layer_prefix="norm_out",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation, permutation=self.permutation,
        )
        self.proj_out = _make_linear(
            inner_dim, patch_size * patch_size * out_channels,
            bias=True, block_size=block_size, layer_prefix="proj_out",
            use_nvfp4=use_nvfp4,
            rotation=self.rotation, permutation=self.permutation,
        )

        self.gradient_checkpointing = False
        self.quantization_error_info: dict = {}
        # Aggregated per-module diff records (populated after forward when
        # return_computation_diff=True by collecting each sub-module's self.computation_diff_dict)
        # self.computation_diff_dict = {}
        print(f"Initialize SD3 Transformer {'[Quantized]' if use_nvfp4 else '[Unquantized]'} mode, "
              f"block_size={block_size}, rotation={self.rotation}, permutation={self.permutation})")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # from_pretrained
    # ------------------------------------------------------------------
    @classmethod
    def _resolve_checkpoint_path(cls, pretrained_model_name_or_path: str,
                                 download_source: str = "modelscope",
                                 cache_dir: str = None,
                                 **kwargs):
        """Return a local checkpoint directory, using the cache first."""
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
                    ignore_patterns=[
                        "**/text_encoder_3/**",
                        "**/tokenizer_3/**",
                    ],
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
                        ref_model=None,
                        **kwargs):
        """Build NVFP4QuantizedSD3 from a pretrained checkpoint.

        Args:
            ref_model: Optional pre-loaded SD3Transformer2DModel to copy weights
                       from. If None, loads weights directly from the checkpoint.
        """
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

        # Step 3: build model
        # Create on meta device: zero memory allocation.
        # assign=True replaces meta tensors with checkpoint tensors.
        # This avoids allocating ~5GB of random weights that are
        # immediately discarded.
        _t0 = time.time()
        with torch.device('meta'):
            model = cls(
                sample_size=cfg.get("sample_size", 128),
                patch_size=cfg.get("patch_size", 2),
                in_channels=cfg.get("in_channels", 16),
                out_channels=cfg.get("out_channels") or cfg.get("in_channels", 16),
                num_layers=cfg["num_layers"],
                attention_head_dim=cfg["attention_head_dim"],
                num_attention_heads=cfg["num_attention_heads"],
                joint_attention_dim=cfg.get("joint_attention_dim", 4096),
                caption_projection_dim=cfg.get("caption_projection_dim", inner_dim),
                pooled_projection_dim=cfg.get("pooled_projection_dim", 2048),
                pos_embed_max_size=cfg.get("pos_embed_max_size", 96),
                dual_attention_layers=cfg.get("dual_attention_layers", ()),
                qk_norm=cfg.get("qk_norm", "rms_norm"),
                block_size=block_size,
                use_nvfp4=use_nvfp4,
                rotation=rotation,
                permutation=permutation,
            )
            # print(f"  [timing] model creation (meta): {time.time() - _t0:.2f}s")

        # Step 4: load weights
        # Load checkpoint directly. Our model is a structural replica of
        # SD3Transformer2DModel so checkpoint keys match; only extra
        # rotation/permutation buffers are absent (strict=False).
        import glob
        from safetensors.torch import load_file as load_safetensors

        _t2 = time.time()
        ckpt_dir = os.path.join(local_path, subfolder)
        state_dict = {}
        safetensors_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
        if safetensors_files:
            for f in safetensors_files:
                state_dict.update(load_safetensors(f))
        else:
            bin_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.bin")))
            if not bin_files:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
            for f in bin_files:
                state_dict.update(torch.load(f, map_location="cpu"))
        # print(f"  [timing] safetensors load: {time.time() - _t2:.2f}s")

        _t3 = time.time()
            # assign=True replaces meta tensors with loaded tensors (no copy).
            # ~1000x faster than copy_() for large models.
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)    
        # print(f"  [timing] load_state_dict: {time.time() - _t3:.2f}s")

        real_missing = [k for k in missing if not any(s in k for s in ('_rotation', '_R_init', 'permutation'))]
        if real_missing:
            print(f"Warning: {len(real_missing)} missing keys, first 5: {real_missing[:5]}")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys, first 5: {unexpected[:5]}")
        # Check for remaining meta tensors (rotation/permutation buffers
        # not present in checkpoint)
        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        meta_buffers = [n for n, b in model.named_buffers() if b.is_meta]
        if meta_params or meta_buffers:
            print(f"  [info] {len(meta_params) + len(meta_buffers)} meta tensors "
                    f"remaining, materializing rotation/permutation...")
            for module in model.modules():
                if isinstance(module, NVFP4Linear):
                    if module.rotation is not None and hasattr(module.rotation, 'fit'):
                        module.rotation.fit()
                    if module.permutation is not None and hasattr(module.permutation, 'fit'):
                        module.permutation.fit(module.weight)
        # Convert to target dtype (no-op if already matching)
        model = model.to(dtype=torch_dtype)

        del state_dict
        gc.collect()

        return model

    # ------------------------------------------------------------------
    # Weight copy
    # ------------------------------------------------------------------

    def _copy_weights(self, ref):
        """Copy weights from HuggingFace SD3Transformer2DModel to this model."""
        # pos_embed: PatchEmbed.proj (Conv2d)
        self.pos_embed.proj.weight.data.copy_(ref.pos_embed.proj.weight.data)
        self.pos_embed.proj.bias.data.copy_(ref.pos_embed.proj.bias.data)
        # pos_embed buffer
        self.pos_embed.pos_embed.data.copy_(ref.pos_embed.pos_embed.data)

        # time_text_embed
        my_tte = self.time_text_embed
        ref_tte = ref.time_text_embed
        my_tte.timestep_embedder.linear_1.weight.data.copy_(
            ref_tte.timestep_embedder.linear_1.weight.data)
        my_tte.timestep_embedder.linear_1.bias.data.copy_(
            ref_tte.timestep_embedder.linear_1.bias.data)
        my_tte.timestep_embedder.linear_2.weight.data.copy_(
            ref_tte.timestep_embedder.linear_2.weight.data)
        my_tte.timestep_embedder.linear_2.bias.data.copy_(
            ref_tte.timestep_embedder.linear_2.bias.data)
        my_tte.text_embedder.linear_1.weight.data.copy_(
            ref_tte.text_embedder.linear_1.weight.data)
        my_tte.text_embedder.linear_1.bias.data.copy_(
            ref_tte.text_embedder.linear_1.bias.data)
        my_tte.text_embedder.linear_2.weight.data.copy_(
            ref_tte.text_embedder.linear_2.weight.data)
        my_tte.text_embedder.linear_2.bias.data.copy_(
            ref_tte.text_embedder.linear_2.bias.data)

        # context_embedder
        self.context_embedder.weight.data.copy_(ref.context_embedder.weight.data)
        if ref.context_embedder.bias is not None:
            self.context_embedder.bias.data.copy_(ref.context_embedder.bias.data)

        # transformer_blocks
        for my_block, ref_block in zip(self.transformer_blocks, ref.transformer_blocks):
            self._copy_block_weights(my_block, ref_block)

        # norm_out
        self.norm_out.linear.weight.data.copy_(ref.norm_out.linear.weight.data)
        self.norm_out.linear.bias.data.copy_(ref.norm_out.linear.bias.data)

        # proj_out
        self.proj_out.weight.data.copy_(ref.proj_out.weight.data)
        self.proj_out.bias.data.copy_(ref.proj_out.bias.data)

    def _copy_block_weights(self, my, ref):
        """Copy weights for one JointTransformerBlock."""
        # norm1 (AdaLayerNormZero)
        my.norm1.linear.weight.data.copy_(ref.norm1.linear.weight.data)
        my.norm1.linear.bias.data.copy_(ref.norm1.linear.bias.data)

        # norm1_context
        my.norm1_context.linear.weight.data.copy_(ref.norm1_context.linear.weight.data)
        my.norm1_context.linear.bias.data.copy_(ref.norm1_context.linear.bias.data)

        # JointAttention
        self._copy_attn_weights(my.attn, ref.attn)

        # attn2 (dual attention)
        if my.attn2 is not None and hasattr(ref, 'attn2') and ref.attn2 is not None:
            self._copy_attn_weights(my.attn2, ref.attn2)

        # FF (image stream)
        self._copy_ff_weights(my.ff, ref.ff)

        # FF (text stream)
        if my.ff_context is not None and hasattr(ref, 'ff_context') and ref.ff_context is not None:
            self._copy_ff_weights(my.ff_context, ref.ff_context)

    def _copy_attn_weights(self, my, ref):
        """Copy all weights from diffusers Attention to JointAttention.

        Handles ``has_added_kv=False`` (attn2 in SD3.5 dual-attention blocks)
        where added KV / text output projections are absent.
        """
        my.to_q.weight.data.copy_(ref.to_q.weight.data)
        my.to_k.weight.data.copy_(ref.to_k.weight.data)
        my.to_v.weight.data.copy_(ref.to_v.weight.data)
        if ref.to_q.bias is not None:
            my.to_q.bias.data.copy_(ref.to_q.bias.data)
            my.to_k.bias.data.copy_(ref.to_k.bias.data)
            my.to_v.bias.data.copy_(ref.to_v.bias.data)

        # Only copy added KV projections if both have them
        if getattr(my, 'add_k_proj', None) is not None and \
                getattr(ref, 'add_k_proj', None) is not None:
            my.add_k_proj.weight.data.copy_(ref.add_k_proj.weight.data)
            my.add_v_proj.weight.data.copy_(ref.add_v_proj.weight.data)
            if ref.add_k_proj.bias is not None:
                my.add_k_proj.bias.data.copy_(ref.add_k_proj.bias.data)
            if ref.add_v_proj.bias is not None:
                my.add_v_proj.bias.data.copy_(ref.add_v_proj.bias.data)
            if hasattr(ref, 'add_q_proj') and hasattr(my, 'add_q_proj') and \
                    ref.add_q_proj is not None:
                my.add_q_proj.weight.data.copy_(ref.add_q_proj.weight.data)
                if ref.add_q_proj.bias is not None:
                    my.add_q_proj.bias.data.copy_(ref.add_q_proj.bias.data)

        my.to_out[0].weight.data.copy_(ref.to_out[0].weight.data)
        if ref.to_out[0].bias is not None:
            my.to_out[0].bias.data.copy_(ref.to_out[0].bias.data)

        # Copy to_add_out if both have it
        if getattr(my, 'to_add_out', None) is not None and \
                getattr(ref, 'to_add_out', None) is not None:
            my.to_add_out.weight.data.copy_(ref.to_add_out.weight.data)
            if ref.to_add_out.bias is not None:
                my.to_add_out.bias.data.copy_(ref.to_add_out.bias.data)

        # Copy QK norms if both have them
        if hasattr(my, 'norm_q') and hasattr(ref, 'norm_q') and \
                my.norm_q is not None and ref.norm_q is not None:
            my.norm_q.weight.data.copy_(ref.norm_q.weight.data)
            my.norm_k.weight.data.copy_(ref.norm_k.weight.data)
        if getattr(my, 'norm_added_q', None) is not None and \
                getattr(ref, 'norm_added_q', None) is not None:
            my.norm_added_q.weight.data.copy_(ref.norm_added_q.weight.data)
            my.norm_added_k.weight.data.copy_(ref.norm_added_k.weight.data)

    def _copy_ff_weights(self, my, ref):
        """Copy weights from diffusers FeedForward to our FeedForward.

        Handles GEGLU structure: net.0 is a GEGLU whose ``proj`` is a Linear,
        net.2 is a plain Linear.  We recurse into submodules to find all
        Linear / NVFP4Linear layers in forward order.
        """
        def _collect_linears(module):
            """Recursively collect all Linear/NVFP4Linear in forward order."""
            linears = []
            for m in module.modules():
                if isinstance(m, (nn.Linear, NVFP4Linear)):
                    linears.append(m)
            return linears

        my_linears = _collect_linears(my.net)
        ref_linears = _collect_linears(ref.net)

        if len(my_linears) != len(ref_linears):
            raise RuntimeError(
                f"FeedForward structure mismatch: our model has {len(my_linears)} "
                f"Linear layers, but the checkpoint has {len(ref_linears)}."
            )

        for ml, rl in zip(my_linears, ref_linears):
            ml.weight.data.copy_(rl.weight.data)
            if rl.bias is not None and ml.bias is not None:
                ml.bias.data.copy_(rl.bias.data)

    # ------------------------------------------------------------------
    # Forward (exact diffusers replica)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
        return_dict: bool = True,
        quantization_error_info: dict = None,
        encoder_attention_mask=None,
        guidance=None,
        block_controlnet_hidden_states=None,
        joint_attention_kwargs=None,
        skip_layers=None,
        return_computation_diff: bool = False,
        skip_plan: dict or float = None,
    ):
        """Matches ``SD3Transformer2DModel.forward``.

        Args:
            hidden_states: (B, C, H, W) latent.
            encoder_hidden_states: (B, N_txt, joint_attention_dim) T5/CLIP embeddings.
            pooled_projections: (B, pooled_projection_dim).
            timestep: (B,).
            return_dict: returns tuple when False.
            encoder_attention_mask: (B, N_txt) or (B, 2*N_txt) — accepted and
                forwarded to joint attention for explicit masking; ignored
                when None.  SD3 typically pads the text embedding instead of
                using this mask, but we still accept it so external callers
                can pass it without error.
        Returns:
            ``(output,)`` tuple or dict-like with ``.sample``.
        """
        if return_computation_diff:
            computation_diff_dict = {}
        else:
            computation_diff_dict = None

        height, width = hidden_states.shape[-2:]

        # 1. Patch embed + position
        hidden_states = self.pos_embed(hidden_states)  # (B, N, D)

        # 2. Conditioning
        temb = self.time_text_embed(
            timestep, pooled_projections,
            quantization_error_info=quantization_error_info)  # (B, D)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states,
            quantization_error_info=quantization_error_info)

        # 3. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                def _forward(blk, *args):
                    return blk(*args, quantization_error_info=quantization_error_info)
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _forward, block, hidden_states, encoder_hidden_states, temb,
                    use_reentrant=False,
                )
        else:
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    quantization_error_info=quantization_error_info,
                    computation_diff_dict=computation_diff_dict,
                    skip_plan=skip_plan,
                )

        # 4. Output normalization
        hidden_states = self.norm_out(
            hidden_states, temb,
            quantization_error_info=quantization_error_info)

        # 5. Output projection
        hidden_states = self.proj_out(
            hidden_states, quantization_error_info=quantization_error_info)

        # 6. Unpatchify
        patch_size = self.patch_size
        h, w = height // patch_size, width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], h, w, patch_size, patch_size, self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, height, width,
        )

        if return_computation_diff:
            output = output, computation_diff_dict

        if not return_dict:
            return (output,)

        return type("Transformer2DModelOutput", (), {"sample": output})()


if __name__ == "__main__":

    """
    python -m src.models.nvfp4_quantized_SD3
    """
    import time
    from diffusers import SD3Transformer2DModel

    bs = 1 
    in_channels = 16
    height = 16
    width = 16
    n_txt = 32
    joint_attention_dim = 4096      # SD3.5-medium text embedding dim
    pooled_projection_dim = 2048    # SD3.5-medium pooled projection dim
    # dtype = torch.float32
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")

    hidden_states = torch.randn(bs, in_channels, height, width, dtype=dtype).to(device)
    encoder_hidden_states = torch.randn(bs, n_txt, joint_attention_dim, dtype=dtype).to(device)
    pooled_projections = torch.randn(bs, pooled_projection_dim, dtype=dtype).to(device)
    timestep = torch.randn(bs, dtype=dtype).to(device)

    inner_dim = 128
    num_attention_heads = 4
    attention_head_dim = inner_dim // num_attention_heads  # 32
    caption_projection_dim = inner_dim  # must equal inner_dim for SD3

    """
    cur = time.time()
    model = NVFP4QuantizedSD3.from_pretrained(
        "G://models//stabilityai//stable-diffusion-3.5-medium",
        cache_dir="G://models",
        local_files_only=True,
        torch_dtype=dtype,
        rotation=None,
        permutation=None,
        use_nvfp4=False,
    ).to(device, dtype=dtype)
    
    load_time = time.time() - cur
    print(f"Model loading time: {load_time:.4f} seconds")
    y = model(
        hidden_states, encoder_hidden_states, pooled_projections, timestep,
        return_dict=True
        , return_computation_diff=True
    ).sample
    # print(f"Model output shape: {y.shape}")

    # gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
    # print(f"Peak GPU memory: {gpu_mem:.2f} GB")
    # import psutil
    # rss = psutil.Process().memory_info().rss / 1024**3
    # print(f"Process RSS: {rss:.2f} GB")
    """
    
    """
    # ---- Original diffusers SD3Transformer2DModel ----
    # model_ori = SD3Transformer2DModel(
    #     sample_size=16,
    #     patch_size=2,
    #     in_channels=in_channels,
    #     out_channels=in_channels,
    #     num_layers=2,
    #     attention_head_dim=attention_head_dim,
    #     num_attention_heads=num_attention_heads,
    #     joint_attention_dim=joint_attention_dim,
    #     caption_projection_dim=caption_projection_dim,
    #     pooled_projection_dim=pooled_projection_dim,
    #     pos_embed_max_size=16,
    #     dual_attention_layers=(),
    #     qk_norm="rms_norm",
    # ).to(device, dtype=dtype)

    cur = time.time()
    model_ori = SD3Transformer2DModel.from_pretrained(
        "G://models//stabilityai//stable-diffusion-3.5-medium",
        subfolder="transformer",
        cache_dir="G://models",
        local_files_only=True,
        torch_dtype=dtype
    ).to(device, dtype=dtype)

    y_ori = model_ori(
        hidden_states, encoder_hidden_states, pooled_projections, timestep,
        return_dict=True,
    ).sample
    # print(f"Original model output shape: {y_ori.shape}")
    # load_time = time.time() - cur
    # print(f"Origin model loading+forward time: {load_time:.4f} seconds")
    
    print(torch.mean(torch.abs(y_ori - y)))
    """
    """
    # Free origin model to save memory for NVFP4 model
    del model_ori
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Compare with NVFP4QuantizedSD3 ----
    for rot in ["identity"]:
        for perm in ["identity"]:
            for quantize in [False]:
                cur = time.time()
                model_rpq = NVFP4QuantizedSD3.from_pretrained(
                    "G://models//stabilityai//stable-diffusion-3.5-medium",
                    cache_dir="G://models",
                    local_files_only=True,
                    torch_dtype=dtype,
                    use_nvfp4=quantize,
                    rotation=rot,
                    permutation=perm
                ).to(device, dtype=dtype)
                print(f"rpq model loading time: {time.time() - cur:.4f}")
                y_rpq = model_rpq(
                    hidden_states, encoder_hidden_states, pooled_projections,
                    timestep, return_dict=True,
                ).sample
                diff = torch.mean(torch.abs(y_rpq - y_ori))
                print(f"[rot={rot}-perm={perm}-quantize={quantize}] "
                      f"MAE between original and quantized model: {diff:.6e}")
                del model_rpq
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    """
    """
    # Test return diff 
    model = NVFP4QuantizedSD3.from_pretrained(
        "G://models//stabilityai//stable-diffusion-3.5-medium",
        cache_dir="G://models",
        local_files_only=True,
        torch_dtype=dtype,
        use_nvfp4=False,
        rotation=None,
        permutation=None,
        nvfp4_quantize=False,
    ).to(device, dtype=dtype)
    y, computation_diff_dict = model(
        hidden_states, encoder_hidden_states, pooled_projections,
        timestep, 
        # return_dict=True,
        return_computation_diff=True
    ).sample
    for key, value in computation_diff_dict.items():
        if isinstance(value, dict):
            for subKey, subValue in value.items():
                print(f"{key}.{subKey}: {subValue.shape}, {torch.mean(subValue)}")
        else:
            print(f"{key}: {value.shape}, {torch.mean(value)}")
    """
    # Test token skip plan
    # n_skip = 16
    # skip_plan = {"block.0.img": torch.randperm(in_channels)[:n_skip].to(device)}
    with open("G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K//token_skip_plan.json", "r") as f:
        skip_plan = json.load(f)
    model = NVFP4QuantizedSD3.from_pretrained(
        "G://models//stabilityai//stable-diffusion-3.5-medium",
        cache_dir="G://models",
        local_files_only=True,
        torch_dtype=dtype,
        use_nvfp4=False,
        rotation=None,
        permutation=None,
        nvfp4_quantize=False,
    ).to(device, dtype=dtype)
    y = model(
        hidden_states, encoder_hidden_states, 
        pooled_projections,
        timestep, 
        skip_plan=skip_plan.get("0")
        # skip_plan=0.5
    ).sample
    print(y.shape)
    
    