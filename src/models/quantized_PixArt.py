"""
Quantized PixArt-Σ — Self-contained implementation.

Architecture replicates HuggingFace ``PixArtTransformer2DModel`` so that
``from_pretrained`` can load official PixArt-Σ / PixArt-α checkpoints.

PixArt-Σ: ~0.6B parameters, ~3 GB download, NO authentication required.
HuggingFace ID: ``PixArt-alpha/PixArt-Sigma-XL-2-512-MS``

All ``nn.Linear`` layers have been replaced with ``QuantizedLinear``;
each quantized module receives a ``layer_prefix`` for error bookkeeping.

Usage::

    from src.models.quantized_PixArt import QuantizedPixArt

    model = QuantizedPixArt()
    model.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-512-MS", subfolder="transformer")
    out = model(hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, height, width)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
try:
    from ..modules.quantized_linear import QuantizedLinear
except ImportError:
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from modules.quantized_linear import QuantizedLinear


# ---------------------------------------------------------------------------
# Helpers — sinusoidal position / timestep encoding
# ---------------------------------------------------------------------------

def _get_1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal encoding.  pos: (M,) → (M, D)."""
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim))
    out = torch.outer(pos.to(torch.float64), omega)
    return torch.cat([out.sin(), out.cos()], dim=-1).float()


def _get_2d_sincos(embed_dim: int, h: int, w: int, base_size: int = 16) -> torch.Tensor:
    """2D sinusoidal position encoding.  Returns (h*w, embed_dim)."""
    grid_h = torch.arange(h, dtype=torch.float32) / (h / base_size)
    grid_w = torch.arange(w, dtype=torch.float32) / (w / base_size)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_h = _get_1d_sincos(embed_dim // 2, grid_h.reshape(-1))
    emb_w = _get_1d_sincos(embed_dim // 2, grid_w.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=-1)


# ---------------------------------------------------------------------------
# Timestep / Resolution embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Maps sine-cosine features → embedding via small MLP (quantized)."""

    def __init__(self, in_channels: int, time_embed_dim: int,
                 out_dim: Optional[int] = None,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        out_dim = out_dim or time_embed_dim
        self.layer_prefix = layer_prefix
        self.linear_1 = QuantizedLinear(in_channels, time_embed_dim, bias=True,
                                        bitW=bitW, bitA=bitA, bitG=bitG,
                                        layer_prefix=f"{layer_prefix}.linear_1")
        self.act = nn.SiLU()
        self.linear_2 = QuantizedLinear(time_embed_dim, out_dim, bias=True,
                                        bitW=bitW, bitA=bitA, bitG=bitG,
                                        layer_prefix=f"{layer_prefix}.linear_2")

    def forward(self, sample: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        h = self.act(self.linear_1(sample, quantization_error_info))
        return self.linear_2(h, quantization_error_info)


class Timesteps(nn.Module):
    """Converts integer timesteps → sinusoidal features (no learnable params)."""

    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True,
                 downscale_freq_shift: float = 0.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = _get_1d_sincos(self.num_channels, timesteps.float())
        if self.flip_sin_to_cos:
            t_emb = torch.cat([t_emb[:, self.num_channels // 2:],
                               t_emb[:, :self.num_channels // 2]], dim=-1)
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Combine timestep + resolution (H, W) → base conditioning embedding.

    Structure matches HuggingFace ``PixArtAlphaCombinedTimestepSizeEmbeddings``:

    - ``.time_proj``: sin-cos (no params)
    - ``.timestep_embedder``: MLP (256→size_emb_dim)
    - ``.size_embedder``: MLP (256→size_emb_dim)
    - ``.emb``: MLP (2*size_emb_dim → embedding_dim)  →  base embedding  (B, D)
    - ``.linear``: QuantizedLinear(embedding_dim, 6*embedding_dim)  →  modulation  (B, 6*D)
    """

    def __init__(self, embedding_dim: int, size_emb_dim: int = 256,
                 use_additional_conditions: bool = False,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        self.use_additional_conditions = use_additional_conditions
        self.layer_prefix = layer_prefix

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True,
                                   downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=size_emb_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.timestep_embedder")
        self.size_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=size_emb_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.size_embedder")
        self.emb = TimestepEmbedding(
            in_channels=size_emb_dim * 2, time_embed_dim=embedding_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.emb")
        # PixArt modulation: base embedding → 6 modulation values
        self.linear = QuantizedLinear(
            embedding_dim, 6 * embedding_dim, bias=True,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.linear")

        if use_additional_conditions:
            self.use_additional_condition_as = nn.Sequential(
                nn.SiLU(),
                QuantizedLinear(embedding_dim, embedding_dim * 2, bias=True,
                                bitW=bitW, bitA=bitA, bitG=bitG,
                                layer_prefix=f"{layer_prefix}.additional_condition"),
            )

    def forward(self, timestep: torch.Tensor, height: torch.Tensor,
                width: torch.Tensor, batch_size: int,
                hidden_dtype: torch.dtype,
                quantization_error_info: dict = None) -> torch.Tensor:
        """Returns base conditioning: (B, embedding_dim)."""
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype), quantization_error_info)

        h_emb = _get_1d_sincos(256 // 2, height.float())
        w_emb = _get_1d_sincos(256 // 2, width.float())
        size_emb = self.size_embedder(
            torch.cat([h_emb, w_emb], dim=-1).to(dtype=hidden_dtype),
            quantization_error_info)

        conditioning = torch.cat([timesteps_emb, size_emb], dim=-1)
        return self.emb(conditioning, quantization_error_info)  # (B, D)

    def get_modulation(self, base_emb: torch.Tensor,
                       quantization_error_info: dict = None) -> torch.Tensor:
        """Apply SiLU + linear to produce (B, 6*embedding_dim) modulation."""
        return self.linear(F.silu(base_emb), quantization_error_info)  # (B, 6*D)


# ---------------------------------------------------------------------------
# Text projection
# ---------------------------------------------------------------------------

class PixArtAlphaTextProjection(nn.Module):
    """Project pooled text embedding → conditioning dimension."""

    def __init__(self, in_features: int, hidden_size: int,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.linear_1 = QuantizedLinear(in_features, hidden_size, bias=True,
                                        bitW=bitW, bitA=bitA, bitG=bitG,
                                        layer_prefix=f"{layer_prefix}.linear_1")
        self.act = nn.SiLU()
        self.linear_2 = QuantizedLinear(hidden_size, hidden_size, bias=True,
                                        bitW=bitW, bitA=bitA, bitG=bitG,
                                        layer_prefix=f"{layer_prefix}.linear_2")

    def forward(self, caption: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        h = self.act(self.linear_1(caption, quantization_error_info))
        return self.linear_2(h, quantization_error_info)


# ---------------------------------------------------------------------------
# Patch embedding (identical to SD3 — Conv2d, no quantizable Linear)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D latent → patch tokens with sin-cos position embedding."""

    def __init__(self, height: int, width: int, patch_size: int, in_channels: int,
                 embed_dim: int, pos_embed_max_size: Optional[int] = None,
                 interpolation_scale: Optional[float] = None):
        super().__init__()
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.interpolation_scale = interpolation_scale

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

        grid = pos_embed_max_size if pos_embed_max_size is not None \
            else max(self.height, self.width)
        pos_embed = _get_2d_sincos(embed_dim, grid, grid)
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0))

    def _crop_pos_embed(self, h: int, w: int) -> torch.Tensor:
        max_size = self.pos_embed_max_size
        top = (max_size - h) // 2
        left = (max_size - w) // 2
        pe = self.pos_embed.reshape(1, max_size, max_size, -1)
        pe = pe[:, top:top + h, left:left + w, :]
        return pe.reshape(1, h * w, -1)

    def _interpolate_pos_encoding(self, h: int, w: int) -> torch.Tensor:
        """Interpolate pos embedding for arbitrary resolution (PixArt-Σ up to 4K)."""
        max_size = self.pos_embed_max_size
        if h == max_size and w == max_size:
            return self.pos_embed

        pe = self.pos_embed.reshape(1, max_size, max_size, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe.float(), size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        if self.interpolation_scale is not None:
            pe = pe * self.interpolation_scale
        return pe

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, H, W = latent.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.proj(latent)
        x = x.flatten(2).transpose(1, 2)

        if self.pos_embed_max_size is not None:
            if h == self.pos_embed_max_size and w == self.pos_embed_max_size:
                pos_embed = self.pos_embed[:, :h * w, :]
            else:
                pos_embed = self._interpolate_pos_encoding(h, w)
        else:
            pos_embed = self.pos_embed[:, :h * w, :]
        return (x + pos_embed).to(x.dtype)


# ---------------------------------------------------------------------------
# Adaptive normalization layers
# ---------------------------------------------------------------------------

class AdaLayerNormZero(nn.Module):
    """adaLN-Zero: layer norm + 6-dim scale/shift/gate modulation (quantized)."""

    def __init__(self, embedding_dim: int, bias: bool = True,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.silu = nn.SiLU()
        self.linear = QuantizedLinear(embedding_dim, 6 * embedding_dim, bias=bias,
                                      bitW=bitW, bitA=bitA, bitG=bitG,
                                      layer_prefix=f"{layer_prefix}.linear")
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                quantization_error_info: dict = None):
        emb = self.linear(self.silu(emb), quantization_error_info)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    """adaLN-Single: produces global modulation for all blocks (quantized)."""

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        self.use_additional_conditions = use_additional_conditions
        self.layer_prefix = layer_prefix
        self.silu = nn.SiLU()
        self.linear = QuantizedLinear(embedding_dim, 6 * embedding_dim, bias=True,
                                      bitW=bitW, bitA=bitA, bitG=bitG,
                                      layer_prefix=f"{layer_prefix}.linear")

    def forward(self, emb: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        return self.linear(self.silu(emb), quantization_error_info)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """GELU-approximate feed-forward layer (quantized Linear)."""

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 bias: bool = True, dropout: float = 0.0,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.layer_prefix = layer_prefix

        self.net = nn.ModuleList([
            QuantizedLinear(dim, inner_dim, bias=bias,
                            bitW=bitW, bitA=bitA, bitG=bitG,
                            layer_prefix=f"{layer_prefix}.linear_1"),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            QuantizedLinear(inner_dim, dim_out, bias=bias,
                            bitW=bitW, bitA=bitA, bitG=bitG,
                            layer_prefix=f"{layer_prefix}.linear_2"),
        ])

    def forward(self, hidden_states: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        for mod in self.net:
            if isinstance(mod, QuantizedLinear):
                hidden_states = mod(hidden_states, quantization_error_info)
            else:
                hidden_states = mod(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Attention (self-attn + cross-attn) — individual QuantizedLinear projections
# ---------------------------------------------------------------------------

class PixArtAttention(nn.Module):
    """Multi-head attention for PixArt.

    Supports both self-attn and cross-attn.
    Q/K/V/O projections are individual ``QuantizedLinear`` layers
    (PixArt attention does NOT use PyTorch's ``nn.MultiheadAttention``).
    """

    def __init__(self, query_dim: int, cross_attention_dim: Optional[int] = None,
                 num_heads: int = 8, head_dim: int = 64, bias: bool = True,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.layer_prefix = layer_prefix

        # Q projection (always from query_dim)
        self.to_q = QuantizedLinear(query_dim, self.inner_dim, bias=bias,
                                    bitW=bitW, bitA=bitA, bitG=bitG,
                                    layer_prefix=f"{layer_prefix}.to_q")

        # KV projection: from cross_attention_dim if cross-attn, else from query_dim
        kv_input_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_k = QuantizedLinear(kv_input_dim, self.inner_dim, bias=bias,
                                    bitW=bitW, bitA=bitA, bitG=bitG,
                                    layer_prefix=f"{layer_prefix}.to_k")
        self.to_v = QuantizedLinear(kv_input_dim, self.inner_dim, bias=bias,
                                    bitW=bitW, bitA=bitA, bitG=bitG,
                                    layer_prefix=f"{layer_prefix}.to_v")

        # Output projection
        self.to_out = nn.ModuleList([
            QuantizedLinear(self.inner_dim, query_dim, bias=bias,
                            bitW=bitW, bitA=bitA, bitG=bitG,
                            layer_prefix=f"{layer_prefix}.to_out"),
            nn.Dropout(0.0),
        ])

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N, _ = x.shape
        return x.transpose(1, 2).reshape(B, N, -1)

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                quantization_error_info: dict = None) -> torch.Tensor:
        q = self._split_heads(self.to_q(hidden_states, quantization_error_info))

        if encoder_hidden_states is not None:
            # Cross-attention: Q from image, K/V from text
            k = self._split_heads(self.to_k(encoder_hidden_states, quantization_error_info))
            v = self._split_heads(self.to_v(encoder_hidden_states, quantization_error_info))
        else:
            # Self-attention: QKV from image
            k = self._split_heads(self.to_k(hidden_states, quantization_error_info))
            v = self._split_heads(self.to_v(hidden_states, quantization_error_info))

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self._merge_heads(attn)
        attn = self.to_out[0](attn, quantization_error_info)
        return self.to_out[1](attn)


# ---------------------------------------------------------------------------
# PixArt Transformer Block
# ---------------------------------------------------------------------------

class PixArtTransformerBlock(nn.Module):
    """PixArt DiT block: self-attn → cross-attn → FF, with pre-computed modulation.

    Modulation values (shift/scale/gate) are computed externally by the parent model
    using ``AdaLayerNormSingle`` and ``scale_shift_table``, matching the diffusers
    PixArt architecture exactly.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 cross_attention_dim: int, dropout: float = 0.0,
                 bitW: int = 8, bitA: int = 8, bitG: int = 8,
                 layer_prefix: str = None):
        super().__init__()
        self.layer_prefix = layer_prefix

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn1 = PixArtAttention(
            query_dim=dim, cross_attention_dim=None,
            num_heads=num_attention_heads, head_dim=attention_head_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.attn1")

        # Cross-attention (no modulation)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn2 = PixArtAttention(
            query_dim=dim, cross_attention_dim=cross_attention_dim,
            num_heads=num_attention_heads, head_dim=attention_head_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.attn2")

        # FeedForward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, dropout=dropout,
                              bitW=bitW, bitA=bitA, bitG=bitG,
                              layer_prefix=f"{layer_prefix}.ff")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        quantization_error_info: dict = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N_img, D)
            encoder_hidden_states: (B, N_txt, D_x)
            shift_msa, scale_msa, gate_msa: (B, D) — self-attn modulation
            shift_mlp, scale_mlp, gate_mlp: (B, D) — FF modulation
        """
        # --- Self-attention ---
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        attn_output = self.attn1(norm_hidden, encoder_hidden_states=None,
                                 quantization_error_info=quantization_error_info)
        hidden_states = hidden_states + gate_msa[:, None, :] * attn_output

        # --- Cross-attention (no modulation) ---
        norm_hidden = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden, encoder_hidden_states=encoder_hidden_states,
                                 quantization_error_info=quantization_error_info)
        hidden_states = hidden_states + attn_output

        # --- FeedForward ---
        norm_hidden = self.norm3(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ff_output = self.ff(norm_hidden, quantization_error_info)
        hidden_states = hidden_states + gate_mlp[:, None, :] * ff_output

        return hidden_states


# ---------------------------------------------------------------------------
# Quantized PixArt model
# ---------------------------------------------------------------------------

class QuantizedPixArt(nn.Module):
    """Quantized PixArt-α / PixArt-Σ Transformer.

    Architecture mirrors ``PixArtTransformer2DModel`` from HuggingFace diffusers so
    that ``from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-512-MS", subfolder="transformer")``
    works.

    PixArt-Σ defaults (~0.6B params)::

        model = QuantizedPixArt(
            sample_size=128, patch_size=2, in_channels=4,
            num_layers=28, attention_head_dim=72, num_attention_heads=16,
            cross_attention_dim=1152, caption_channels=4096,
            out_channels=8, pos_embed_max_size=256,
            bitW=8, bitA=8, bitG=8,
        )
    """

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 4,
        num_layers: int = 28,
        attention_head_dim: int = 72,
        num_attention_heads: int = 16,
        cross_attention_dim: int = 1152,
        caption_channels: int = 4096,
        out_channels: Optional[int] = None,
        pos_embed_max_size: int = 256,
        pos_embed_interpolation_scale: Optional[float] = None,
        dropout: float = 0.0,
        bitW: int = 8,
        bitA: int = 8,
        bitG: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.num_layers = num_layers

        # Position embedding (Conv2d + sin-cos PE)
        self.pos_embed = PatchEmbed(
            height=sample_size, width=sample_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            interpolation_scale=pos_embed_interpolation_scale,
        )

        # Timestep + resolution → conditioning embedding
        self.adaln_single = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim=self.inner_dim, size_emb_dim=self.inner_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix="adaln_single",
        )

        # Per-block scale-shift offset table (learnable)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, num_layers, self.inner_dim) / (self.inner_dim ** 0.5)
        )

        # Text embedding projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=cross_attention_dim,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix="caption_projection",
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            PixArtTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                dropout=dropout,
                bitW=bitW, bitA=bitA, bitG=bitG,
                layer_prefix=f"block.{i}",
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm_out = AdaLayerNormZero(self.inner_dim,
                                         bitW=bitW, bitA=bitA, bitG=bitG,
                                         layer_prefix="norm_out")
        self.proj_out = QuantizedLinear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix="proj_out",
        )

        self.gradient_checkpointing = False
        self.quantization_error_info: dict = {}
        print(f"Initialize Quantized PixArt with {bitW}-bitW, {bitA}-bitA, {bitG}-bitG")

    # -----------------------------------------------------------------------
    # from_pretrained
    # -----------------------------------------------------------------------

    def from_pretrained(self, pretrained_model_name_or_path: str,
                        subfolder: str = "transformer", **kwargs):
        """Load weights from a HuggingFace PixArt / PixArtSigma checkpoint."""
        from diffusers import PixArtTransformer2DModel
        ref = PixArtTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, **kwargs,
        )
        self._copy_weights(ref)
        del ref
        return self

    def _copy_weights(self, ref):
        """Copy weights from HuggingFace PixArtTransformer2DModel to this model.

        Since ``QuantizedLinear`` subclasses ``nn.Linear``, ``.weight`` and ``.bias``
        attributes still exist and can be copied as usual.
        """
        # pos_embed: Conv2d + buffer
        self.pos_embed.proj.weight.data.copy_(ref.pos_embed.proj.weight.data)
        self.pos_embed.proj.bias.data.copy_(ref.pos_embed.proj.bias.data)
        self.pos_embed.pos_embed.data.copy_(ref.pos_embed.pos_embed.data)

        # adaln_single
        for name, param in ref.adaln_single.named_parameters():
            target_name = {
                "time_proj.num_channels": None,
                "timestep_embedder.linear_1.weight": "adaln_single.timestep_embedder.linear_1.weight",
                "timestep_embedder.linear_1.bias": "adaln_single.timestep_embedder.linear_1.bias",
                "timestep_embedder.linear_2.weight": "adaln_single.timestep_embedder.linear_2.weight",
                "timestep_embedder.linear_2.bias": "adaln_single.timestep_embedder.linear_2.bias",
                "size_embedder.linear_1.weight": "adaln_single.size_embedder.linear_1.weight",
                "size_embedder.linear_1.bias": "adaln_single.size_embedder.linear_1.bias",
                "size_embedder.linear_2.weight": "adaln_single.size_embedder.linear_2.weight",
                "size_embedder.linear_2.bias": "adaln_single.size_embedder.linear_2.bias",
                "emb.linear_1.weight": "adaln_single.emb.linear_1.weight",
                "emb.linear_1.bias": "adaln_single.emb.linear_1.bias",
                "emb.linear_2.weight": "adaln_single.emb.linear_2.weight",
                "emb.linear_2.bias": "adaln_single.emb.linear_2.bias",
                "linear.weight": "adaln_single.linear.weight",
                "linear.bias": "adaln_single.linear.bias",
            }.get(name, None)
            if target_name is not None:
                self.get_parameter(target_name).data.copy_(param.data)

        # scale_shift_table
        self.scale_shift_table.data.copy_(ref.scale_shift_table.data)

        # caption_projection
        for name, param in ref.caption_projection.named_parameters():
            target_name = {
                "linear_1.weight": "caption_projection.linear_1.weight",
                "linear_1.bias": "caption_projection.linear_1.bias",
                "linear_2.weight": "caption_projection.linear_2.weight",
                "linear_2.bias": "caption_projection.linear_2.bias",
            }.get(name, None)
            if target_name is not None:
                self.get_parameter(target_name).data.copy_(param.data)

        # transformer_blocks
        for i, (my_block, ref_block) in enumerate(zip(self.transformer_blocks,
                                                       ref.transformer_blocks)):
            self._copy_block_weights(my_block, ref_block)

        # norm_out
        self.norm_out.linear.weight.data.copy_(ref.norm_out.linear.weight.data)
        self.norm_out.linear.bias.data.copy_(ref.norm_out.linear.bias.data)

        # proj_out
        self.proj_out.weight.data.copy_(ref.proj_out.weight.data)
        self.proj_out.bias.data.copy_(ref.proj_out.bias.data)

    def _copy_block_weights(self, my, ref):
        """Copy weights for one PixArtTransformerBlock.

        ``QuantizedLinear.weight`` / ``.bias`` mirror the original ``nn.Linear``
        parameters, so the copy logic is unchanged.
        """
        # Self-attention
        my.attn1.to_q.weight.data.copy_(ref.attn1.to_q.weight.data)
        my.attn1.to_k.weight.data.copy_(ref.attn1.to_k.weight.data)
        my.attn1.to_v.weight.data.copy_(ref.attn1.to_v.weight.data)
        if ref.attn1.to_q.bias is not None:
            my.attn1.to_q.bias.data.copy_(ref.attn1.to_q.bias.data)
            my.attn1.to_k.bias.data.copy_(ref.attn1.to_k.bias.data)
            my.attn1.to_v.bias.data.copy_(ref.attn1.to_v.bias.data)
        my.attn1.to_out[0].weight.data.copy_(ref.attn1.to_out[0].weight.data)
        if ref.attn1.to_out[0].bias is not None:
            my.attn1.to_out[0].bias.data.copy_(ref.attn1.to_out[0].bias.data)

        # Cross-attention
        my.attn2.to_q.weight.data.copy_(ref.attn2.to_q.weight.data)
        my.attn2.to_k.weight.data.copy_(ref.attn2.to_k.weight.data)
        my.attn2.to_v.weight.data.copy_(ref.attn2.to_v.weight.data)
        if ref.attn2.to_q.bias is not None:
            my.attn2.to_q.bias.data.copy_(ref.attn2.to_q.bias.data)
            my.attn2.to_k.bias.data.copy_(ref.attn2.to_k.bias.data)
            my.attn2.to_v.bias.data.copy_(ref.attn2.to_v.bias.data)
        my.attn2.to_out[0].weight.data.copy_(ref.attn2.to_out[0].weight.data)
        if ref.attn2.to_out[0].bias is not None:
            my.attn2.to_out[0].bias.data.copy_(ref.attn2.to_out[0].bias.data)

        # FF: structure-agnostic copy (QuantizedLinear is nn.Linear, check still works)
        my_linears = [m for m in my.ff.net if isinstance(m, nn.Linear)]
        ref_linears = [m for m in ref.ff.net if isinstance(m, nn.Linear)]
        if len(my_linears) != len(ref_linears):
            raise RuntimeError(
                f"FF structure mismatch: our={len(my_linears)}, ref={len(ref_linears)}"
            )
        for ml, rl in zip(my_linears, ref_linears):
            ml.weight.data.copy_(rl.weight.data)
            if rl.bias is not None and ml.bias is not None:
                ml.bias.data.copy_(rl.bias.data)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        width: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        quantization_error_info: dict = None,
    ):
        """
        Args:
            hidden_states: (B, C, H, W) latent.
            encoder_hidden_states: (B, N_txt, caption_channels) T5 embeddings.
            timestep: (B,).
            encoder_attention_mask: (B, N_txt) — currently ignored.
            height: (B,) or scalar tensor — image height for resolution conditioning.
            width: (B,) or scalar tensor — image width for resolution conditioning.
        Returns:
            (B, C, H, W) predicted velocity/noise.
        """
        B, C, H_img, W_img = hidden_states.shape
        H_latent, W_latent = H_img // self.patch_size, W_img // self.patch_size

        # Resolution conditioning
        if height is None:
            height = torch.full((B,), H_img, device=hidden_states.device, dtype=torch.long)
        if width is None:
            width = torch.full((B,), W_img, device=hidden_states.device, dtype=torch.long)

        # Patch embed + position
        hidden_states = self.pos_embed(hidden_states)  # (B, N, D)

        # Conditioning — base embedding from timestep + resolution
        adaLN_base = self.adaln_single(
            timestep=timestep,
            height=height,
            width=width,
            batch_size=B,
            hidden_dtype=hidden_states.dtype,
            quantization_error_info=quantization_error_info,
        )  # (B, D)

        # Global modulation: SiLU + QuantizedLinear → (B, 6*D)
        modulation = self.adaln_single.get_modulation(
            adaLN_base, quantization_error_info)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation.chunk(6, dim=1)  # each (B, D)

        # Text projection
        encoder_hidden_states = self.caption_projection(
            encoder_hidden_states, quantization_error_info)
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                shift_msa=shift_msa + self.scale_shift_table[0, i],
                scale_msa=scale_msa + self.scale_shift_table[1, i],
                gate_msa=gate_msa + self.scale_shift_table[2, i],
                shift_mlp=shift_mlp + self.scale_shift_table[3, i],
                scale_mlp=scale_mlp + self.scale_shift_table[4, i],
                gate_mlp=gate_mlp + self.scale_shift_table[5, i],
                quantization_error_info=quantization_error_info,
            )

        # Output norm (AdaLayerNormZero returns tuple)
        hidden_states, _, _, _, _ = self.norm_out(
            hidden_states, emb=adaLN_base,
            quantization_error_info=quantization_error_info)

        # Output projection
        hidden_states = self.proj_out(hidden_states, quantization_error_info)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            B, H_latent, W_latent, self.patch_size, self.patch_size, self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(B, self.out_channels, H_img, W_img)

        if not return_dict:
            return output
        return output


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    bitW, bitA, bitG = 8, 8, 8

    print("=" * 60)
    print("QuantizedPixArt dry-run smoke test")

    model = QuantizedPixArt(
        sample_size=64,
        patch_size=2,
        in_channels=4,
        num_layers=4,
        attention_head_dim=16,
        num_attention_heads=4,
        cross_attention_dim=256,
        caption_channels=256,
        out_channels=4,
        pos_embed_max_size=64,
        bitW=bitW, bitA=bitA, bitG=bitG,
    ).to(device=device, dtype=dtype)

    quantization_error_info: dict = {}

    hidden_states = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 77, 256, device=device, dtype=dtype)
    timestep = torch.randint(0, 1000, (2,), device=device)
    h = torch.full((2,), 64, device=device, dtype=torch.long)
    w = torch.full((2,), 64, device=device, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            height=h, width=w,
            quantization_error_info=quantization_error_info,
        )
    print(f"  output shape: {out.shape}  (expected [2, 4, 64, 64])")

    # Forward + backward
    model.train()
    out = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        height=h, width=w,
        quantization_error_info=quantization_error_info,
    )
    loss = torch.nn.MSELoss()(out, torch.randn_like(out))
    loss.backward()
    print(f"  loss: {loss.item():.6f}")

    print(f"  quantization_error_info keys ({len(quantization_error_info)}):")
    for k in sorted(quantization_error_info.keys()):
        print(f"    {k}  {quantization_error_info[k].shape}")
