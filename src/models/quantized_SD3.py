"""
Quantized SD3 (Stable Diffusion 3) — Self-contained implementation.

Architecture replicates HuggingFace ``SD3Transformer2DModel`` so that
``from_pretrained`` can be used to load official SD3 checkpoints.

Key modules (nn.Linear) are marked with ``# <-- quantizable`` — replace these
with ``QuantizedLinear`` from ``src.modules.quantized_linear``.

Usage (after replacing quantizable layers)::

    from src.models.quantized_SD3 import QuantizedSD3

    model = QuantizedSD3()
    model.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer")
    out = model(hidden_states, encoder_hidden_states, pooled_projections, timestep)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers — sinusoidal position encoding
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
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")      # (H, W)
    emb_h = _get_1d_sincos(embed_dim // 2, grid_h.reshape(-1))          # (H*W, D/2)
    emb_w = _get_1d_sincos(embed_dim // 2, grid_w.reshape(-1))          # (H*W, D/2)
    return torch.cat([emb_h, emb_w], dim=-1)                             # (H*W, D)


# ---------------------------------------------------------------------------
# Timestep embedding (sine-cosine → small MLP)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Maps timestep scalar → sine-cosine features → embedding."""

    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or time_embed_dim
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)     # <-- quantizable
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim, bias=True)         # <-- quantizable

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class Timesteps(nn.Module):
    """Converts integer timesteps → sinusoidal features (no learnable params)."""

    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = _get_1d_sincos(self.num_channels, timesteps.float())
        if self.flip_sin_to_cos:
            t_emb = torch.cat([t_emb[:, self.num_channels // 2:], t_emb[:, :self.num_channels // 2]], dim=-1)
        return t_emb


class PixArtAlphaTextProjection(nn.Module):
    """Project pooled text embedding → conditioning dimension."""

    def __init__(self, in_features: int, hidden_size: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or hidden_size
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)        # <-- quantizable
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)       # <-- quantizable

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(caption)))


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Fuse timestep and pooled text into a single conditioning vector."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D latent → patch tokens with sin-cos position embedding."""

    def __init__(self, height: int, width: int, patch_size: int, in_channels: int,
                 embed_dim: int, pos_embed_max_size: Optional[int] = None):
        super().__init__()
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        grid = pos_embed_max_size if pos_embed_max_size is not None else max(self.height, self.width)
        pos_embed = _get_2d_sincos(embed_dim, grid, grid)                         # (grid*grid, D)
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0))          # (1, grid*grid, D)

    def _crop_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """Crop pre-computed position embedding to the given height/width."""
        max_size = self.pos_embed_max_size
        top = (max_size - h) // 2
        left = (max_size - w) // 2
        pe = self.pos_embed.reshape(1, max_size, max_size, -1)                    # (1, H, W, D)
        pe = pe[:, top:top + h, left:left + w, :]                                  # (1, h, w, D)
        return pe.reshape(1, h * w, -1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, H, W = latent.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.proj(latent)                                                      # (B, D, h, w)
        x = x.flatten(2).transpose(1, 2)                                           # (B, h*w, D)
        if self.pos_embed_max_size is not None:
            pos_embed = self._crop_pos_embed(h, w)
        else:
            pos_embed = self.pos_embed[:, :h * w, :]
        return (x + pos_embed).to(x.dtype)


# ---------------------------------------------------------------------------
# Adaptive normalization layers
# ---------------------------------------------------------------------------

class AdaLayerNormZero(nn.Module):
    """adaLN-Zero: layer norm + 6-dim scale/shift/gate modulation."""

    def __init__(self, embedding_dim: int, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)       # <-- quantizable
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    """Continuous adaLN: layer norm + 2-dim scale/shift modulation."""

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int,
                 elementwise_affine: bool = False, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)   # <-- quantizable
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine, bias=bias)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """GELU-approximate feed-forward layer."""

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=bias),                                # <-- quantizable
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out, bias=bias),                            # <-- quantizable
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


# ---------------------------------------------------------------------------
# Joint attention (native PyTorch — no diffusers Attention dependency)
# ---------------------------------------------------------------------------

class JointAttention(nn.Module):
    """
    Joint image-text attention used in SD3 MMDiT blocks.
    Supports ``context_pre_only`` (last block).  Uses ``F.scaled_dot_product_attention``.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, context_pre_only: bool = False,
                 qk_norm: Optional[str] = None, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.context_pre_only = context_pre_only

        # Image QKV
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)                      # <-- quantizable
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)                      # <-- quantizable
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)                      # <-- quantizable

        # Text (added) KV
        self.add_k_proj = nn.Linear(dim, self.inner_dim, bias=bias)                # <-- quantizable
        self.add_v_proj = nn.Linear(dim, self.inner_dim, bias=bias)                # <-- quantizable
        self.added_proj_bias = True
        if not context_pre_only:
            self.add_q_proj = nn.Linear(dim, self.inner_dim, bias=bias)            # <-- quantizable

        # Output
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim, bias=bias),                             # <-- quantizable
            nn.Dropout(0.0),
        ])
        if not context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, dim, bias=bias)            # <-- quantizable

        # QK norm (optional)
        self.qk_norm = qk_norm
        if qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, inner_dim) → (B, num_heads, N, head_dim)."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_heads, N, head_dim) → (B, N, inner_dim)."""
        B, _, N, _ = x.shape
        return x.transpose(1, 2).reshape(B, N, -1)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (B, N_img, D) image tokens.
            encoder_hidden_states: (B, N_txt, D) text tokens.
        Returns:
            attn_img: (B, N_img, D), attn_txt: (B, N_txt, D) or None if context_pre_only.
        """
        residual_img = hidden_states
        residual_txt = encoder_hidden_states

        # Image Q, text K/V → concat for joint attention
        q_img = self._split_heads(self.to_q(hidden_states))                        # (B, H, N_img, d)
        k_img = self._split_heads(self.to_k(hidden_states))                        # (B, H, N_img, d)
        v_img = self._split_heads(self.to_v(hidden_states))                        # (B, H, N_img, d)

        k_txt = self._split_heads(self.add_k_proj(encoder_hidden_states))          # (B, H, N_txt, d)
        v_txt = self._split_heads(self.add_v_proj(encoder_hidden_states))          # (B, H, N_txt, d)

        if not self.context_pre_only:
            q_txt = self._split_heads(self.add_q_proj(encoder_hidden_states))      # (B, H, N_txt, d)

        # Concatenate → image attends to [image + text]
        k = torch.cat([k_img, k_txt], dim=2)                                       # (B, H, N_img+N_txt, d)
        v = torch.cat([v_img, v_txt], dim=2)

        # QK norm
        if self.qk_norm == "rms_norm":
            q_img = self.norm_q(q_img)
            k = self.norm_k(k)

        # Joint attention: image queries
        attn_img = F.scaled_dot_product_attention(q_img, k, v)
        attn_img = self._merge_heads(attn_img)                                     # (B, N_img, D)
        attn_img = self.to_out[0](attn_img)
        attn_img = self.to_out[1](attn_img)

        if self.context_pre_only:
            return attn_img, None

        # Joint attention: text queries
        if self.qk_norm == "rms_norm":
            q_txt = self.norm_added_q(q_txt)
            k = self.norm_added_k(k)
        attn_txt = F.scaled_dot_product_attention(q_txt, k, v)
        attn_txt = self._merge_heads(attn_txt)                                     # (B, N_txt, D)
        attn_txt = self.to_add_out(attn_txt)
        return attn_img, attn_txt


# ---------------------------------------------------------------------------
# JointTransformerBlock — MMDiT block
# ---------------------------------------------------------------------------

class JointTransformerBlock(nn.Module):
    """MMDiT block used in SD3 — joint image+text attention + dual FeedForward."""

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 context_pre_only: bool = False, qk_norm: Optional[str] = None,
                 use_dual_attention: bool = False):
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        # Image-stream norm
        self.norm1 = AdaLayerNormZero(dim)

        # Text-stream norm
        if context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True)
        else:
            self.norm1_context = AdaLayerNormZero(dim)

        # Joint attention
        self.attn = JointAttention(
            dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim,
            context_pre_only=context_pre_only, qk_norm=qk_norm, bias=True,
        )

        # Second self-attention (SD3.5 dual attention)
        if use_dual_attention:
            self.attn2 = JointAttention(
                dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim,
                context_pre_only=context_pre_only, qk_norm=qk_norm, bias=True,
            )
        else:
            self.attn2 = None

        # Image-stream FF
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim)

        # Text-stream FF
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim)
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor) -> tuple:
        # ---- Image norm ----
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # ---- Text norm ----
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
                self.norm1_context(encoder_hidden_states, emb=temb)

        # ---- Joint attention ----
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Image: residual + gate
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # Image: FF
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # ---- Text path ----
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Quantized SD3 model
# ---------------------------------------------------------------------------

class QuantizedSD3(nn.Module):
    """
    Quantized Stable Diffusion 3 Transformer.

    Architecture mirrors ``SD3Transformer2DModel`` from HuggingFace diffusers so that
    ``from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer")`` works.

    Use SD3.0 defaults::
        sd3_medium = QuantizedSD3(
            sample_size=128, patch_size=2, in_channels=16,
            num_layers=24, attention_head_dim=64, num_attention_heads=24,
            joint_attention_dim=4096, caption_projection_dim=1152,
            pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=96,
            dual_attention_layers=(), qk_norm=None,
        )
    """

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple = (),
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size

        # SD3 architecture constraint: text and image streams share the same dim
        if caption_projection_dim != self.inner_dim:
            raise ValueError(
                f"caption_projection_dim ({caption_projection_dim}) must equal "
                f"inner_dim ({self.inner_dim}) for SD3 MMDiT architecture."
            )

        # Position embedding (Conv2d + sin-cos PE)
        self.pos_embed = PatchEmbed(
            height=sample_size, width=sample_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        # Time + pooled text → conditioning
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim,
        )

        # Context projection: [clip + t5] hidden → caption_projection_dim
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)  # <-- quantizable

        # Transformer blocks (MMDiT)
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                context_pre_only=(i == num_layers - 1),
                qk_norm=qk_norm,
                use_dual_attention=(i in dual_attention_layers),
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim,
                                                elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels,
                                  bias=True)                                          # <-- quantizable

        self.gradient_checkpointing = False

    # -----------------------------------------------------------------------
    # from_pretrained
    # -----------------------------------------------------------------------

    def from_pretrained(self, pretrained_model_name_or_path: str, subfolder: str = "transformer",
                        **kwargs):
        """Load weights from a HuggingFace SD3 checkpoint."""
        from diffusers import SD3Transformer2DModel
        ref = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, **kwargs,
        )
        self._copy_weights(ref)
        del ref
        return self

    def _copy_weights(self, ref):
        """Copy weights from HuggingFace SD3Transformer2DModel to this model."""
        # pos_embed: PatchEmbed.proj (Conv2d)
        self.pos_embed.proj.weight.data.copy_(ref.pos_embed.proj.weight.data)
        self.pos_embed.proj.bias.data.copy_(ref.pos_embed.proj.bias.data)
        # pos_embed buffer
        self.pos_embed.pos_embed.data.copy_(ref.pos_embed.pos_embed.data)

        # time_text_embed
        for name, param in ref.time_text_embed.named_parameters():
            target_name = {
                "time_proj.num_channels": None,   # not a Parameter
                "timestep_embedder.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
                "timestep_embedder.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
                "timestep_embedder.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
                "timestep_embedder.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
                "text_embedder.linear_1.weight": "time_text_embed.text_embedder.linear_1.weight",
                "text_embedder.linear_1.bias": "time_text_embed.text_embedder.linear_1.bias",
                "text_embedder.linear_2.weight": "time_text_embed.text_embedder.linear_2.weight",
                "text_embedder.linear_2.bias": "time_text_embed.text_embedder.linear_2.bias",
            }.get(name, None)
            if target_name is not None:
                self.get_parameter(target_name).data.copy_(param.data)

        # context_embedder
        self.context_embedder.weight.data.copy_(ref.context_embedder.weight.data)
        if ref.context_embedder.bias is not None:
            self.context_embedder.bias.data.copy_(ref.context_embedder.bias.data)

        # transformer_blocks
        for i, (my_block, ref_block) in enumerate(zip(self.transformer_blocks, ref.transformer_blocks)):
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
        if isinstance(my.norm1_context, AdaLayerNormZero):
            my.norm1_context.linear.weight.data.copy_(ref.norm1_context.linear.weight.data)
            my.norm1_context.linear.bias.data.copy_(ref.norm1_context.linear.bias.data)
        elif isinstance(my.norm1_context, AdaLayerNormContinuous):
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
        """Copy all weights from diffusers Attention to JointAttention."""
        my.to_q.weight.data.copy_(ref.to_q.weight.data)
        my.to_k.weight.data.copy_(ref.to_k.weight.data)
        my.to_v.weight.data.copy_(ref.to_v.weight.data)
        if ref.to_q.bias is not None:
            my.to_q.bias.data.copy_(ref.to_q.bias.data)
            my.to_k.bias.data.copy_(ref.to_k.bias.data)
            my.to_v.bias.data.copy_(ref.to_v.bias.data)

        my.add_k_proj.weight.data.copy_(ref.add_k_proj.weight.data)
        my.add_v_proj.weight.data.copy_(ref.add_v_proj.weight.data)
        if ref.add_k_proj.bias is not None:
            my.add_k_proj.bias.data.copy_(ref.add_k_proj.bias.data)
        if ref.add_v_proj.bias is not None:
            my.add_v_proj.bias.data.copy_(ref.add_v_proj.bias.data)

        if hasattr(ref, 'add_q_proj') and hasattr(my, 'add_q_proj'):
            my.add_q_proj.weight.data.copy_(ref.add_q_proj.weight.data)
            if ref.add_q_proj.bias is not None:
                my.add_q_proj.bias.data.copy_(ref.add_q_proj.bias.data)

        my.to_out[0].weight.data.copy_(ref.to_out[0].weight.data)
        if ref.to_out[0].bias is not None:
            my.to_out[0].bias.data.copy_(ref.to_out[0].bias.data)

        if hasattr(ref, 'to_add_out') and hasattr(my, 'to_add_out'):
            my.to_add_out.weight.data.copy_(ref.to_add_out.weight.data)
            if ref.to_add_out.bias is not None:
                my.to_add_out.bias.data.copy_(ref.to_add_out.bias.data)

        # QK norm
        if hasattr(ref, 'norm_q') and hasattr(my, 'norm_q'):
            my.norm_q.weight.data.copy_(ref.norm_q.weight.data)
            my.norm_k.weight.data.copy_(ref.norm_k.weight.data)
            my.norm_added_q.weight.data.copy_(ref.norm_added_q.weight.data)
            my.norm_added_k.weight.data.copy_(ref.norm_added_k.weight.data)

    def _copy_ff_weights(self, my, ref):
        """Copy weights from diffusers FeedForward to our FeedForward.

        Structure-agnostic: extracts all nn.Linear layers from both nets and
        copies weights/biases in order.  This works regardless of whether
        the diffusers FF uses [act_fn, Dropout, Linear] or
        [Linear, act_fn, Dropout, Linear].
        """
        my_linears = [m for m in my.net if isinstance(m, nn.Linear)]
        ref_linears = [m for m in ref.net if isinstance(m, nn.Linear)]

        if len(my_linears) != len(ref_linears):
            raise RuntimeError(
                f"FeedForward structure mismatch: our model has {len(my_linears)} "
                f"Linear layers, but the checkpoint has {len(ref_linears)}."
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
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
        return_dict: bool = False,
    ):
        """
        Args:
            hidden_states: (B, C, H, W) latent.
            encoder_hidden_states: (B, N_txt, joint_attention_dim) T5/CLIP embeddings.
            pooled_projections: (B, pooled_projection_dim).
            timestep: (B,).
        Returns:
            (B, C, H, W) predicted velocity (or noise).
        """
        height, width = hidden_states.shape[-2:]

        # Patch embed + position
        hidden_states = self.pos_embed(hidden_states)                               # (B, N, D)

        # Conditioning
        temb = self.time_text_embed(timestep, pooled_projections)                   # (B, D)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)        # (B, N_txt, caption_dim)

        # Transformer blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        # Output
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)                                # (B, N, patch*patch*C)

        # Unpatchify
        patch_size = self.patch_size
        h, w = height // patch_size, width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], h, w, patch_size, patch_size, self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, height, width,
        )

        if not return_dict:
            return output
        return output


# ---------------------------------------------------------------------------
# Smoke test: model loading + sampling
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import os
    import torch
    from PIL import ImageDraw, ImageFont

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Model cache directory (default: HuggingFace/ModelScope default cache) --
    cache_dir = os.environ.get("MODEL_CACHE_DIR", None)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        print(f"Model cache dir set to: {cache_dir}")

    # ---- Choose download source ----
    # SD3 is a gated model on HuggingFace (requires auth + license acceptance).
    # Default: ModelScope (no auth needed).
    # To use HuggingFace (requires HF_TOKEN):
    #   $env:DOWNLOAD_SOURCE="huggingface"
    #   $env:HF_TOKEN="hf_xxxxxxxxxxxx"
    DOWNLOAD_SOURCE = os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower()

    from diffusers import StableDiffusion3Pipeline

    if DOWNLOAD_SOURCE == "modelscope":
        # ==================== ModelScope (no auth required) ====================
        from modelscope import snapshot_download

        model_id = "AI-ModelScope/stable-diffusion-3-medium-diffusers"
        print(f"Downloading {model_id} via ModelScope ...")
        local_path = snapshot_download(
            model_id,
            cache_dir=cache_dir or ".cache/modelscope",
            allow_patterns=[
                "**/*fp16*",             # all fp16 safetensors weights
                "**/*.json",             # all JSON configs (model_index, config, scheduler, etc.)
                "*.json",                # root config files
                "*.txt",                 # root text files
                "**/*.txt",              # tokenizer vocab/merges in subdirs
                "**/tokenizer.json",     # tokenizer files in text_encoder dirs
                "**/tokenizer_config.json",
                "**/special_tokens_map.json",
            ],
            ignore_patterns=[
                "**/text_encoder_3/**",  # skip T5-XXL (~9GB, not required)
            ],
        )
        print(f"Model cached at: {local_path}")

        print("Loading pipeline ...")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            local_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16",
            local_files_only=True,
        )
    else:
        # ==================== HuggingFace (requires auth token) ====================
        # 1. Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
        # 2. Create token at: https://huggingface.co/settings/tokens
        # 3. Set env: $env:HF_TOKEN="YOUR_HF_TOKEN_HERE"
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token is None:
            from huggingface_hub import whoami
            try:
                whoami()
                hf_token = True  # already logged in via `huggingface-cli login`
            except Exception:
                pass

        if hf_token is None:
            raise RuntimeError(
                "SD3 is gated on HuggingFace. Authenticate via one of:\n"
                "  1. Set $env:HF_TOKEN=\"hf_xxx\"\n"
                "  2. Run: huggingface-cli login\n"
                "  3. Use ModelScope instead: $env:DOWNLOAD_SOURCE=\"modelscope\""
            )

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        print(f"Loading {model_id} via HuggingFace (mirror: {os.environ.get('HF_ENDPOINT')}) ...")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16",
            token=hf_token if isinstance(hf_token, str) else True,
            cache_dir=cache_dir,
        )

    # Use CPU offload to avoid OOM: keeps components on CPU,
    # only moves one to GPU at a time during inference.
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    print("Pipeline loaded successfully.")

    # ---- Generate image ----
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    print(f"Generating image with prompt: {prompt}")
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

    # ---- Save image with prompt text overlay ----
    output_dir = "outputs/images"
    os.makedirs(output_dir, exist_ok=True)

    draw = ImageDraw.Draw(image)
    text = prompt
    text_bbox = draw.textbbox((0, 0), text)
    text_height = text_bbox[3] - text_bbox[1]
    padding = 8
    draw.rectangle(
        [(0, 0), (image.width, text_height + padding * 2)],
        fill=(0, 0, 0, 180),
    )
    draw.text((padding, padding), text, fill=(255, 255, 255))

    save_path = os.path.join(output_dir, "sd3_sample.png")
    image.save(save_path)
    print(f"Image saved to: {save_path}")
    print(f"Prompt: {prompt}")
