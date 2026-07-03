"""
NVFP4-Quantized DiT (Diffusion Transformer) — nn.Linear → NVFP4Linear,
MultiheadAttention → NVFP4MultiHeadAttention.

Weights are quantized with NVFP4Quantization (tensor-wise, single global scale).
Activations are quantized with NVFP4ActivationQuantization (token-wise, per-token
global scale) when input is 3D [bs, n_seq, dim], or NVFP4Quantization (tensor-wise)
when input is 2D [bs, dim] (e.g. time embedding MLP).

NVFP4 format (NVIDIA Blackwell):
  - Element: E2M1  (4-bit, max=6)
  - Block scale: E4M3 (8-bit per 16 elements)
  - Global scale: FP32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
try:
    from ..quant_utils.quantization import NVFP4Quantization, NVFP4ActivationQuantization
except ImportError:
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from quant_utils.quantization import NVFP4Quantization, NVFP4ActivationQuantization


# ===========================================================================
# NVFP4 Module Helpers
# ===========================================================================

class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 quantization on both weight and input activation.

    - Weight:      quantized with NVFP4Quantization  (tensor-wise).
    - Activation:  quantized with NVFP4ActivationQuantization (token-wise)
                   for 3D inputs [bs, n_seq, dim], or NVFP4Quantization
                   (tensor-wise) for 2D inputs [bs, dim].

    Args:
        in_features, out_features: standard nn.Linear args.
        bias: whether to use bias.
        block_size: elements per NVFP4 scaling block (default 16).
        layer_prefix: optional string prefix for error_info_ keys.
    """

    def __init__(self, in_features, out_features, bias=True,
                 block_size=16, layer_prefix=None):
        super().__init__(in_features, out_features, bias=bias)
        self.block_size = block_size
        self.layer_prefix = layer_prefix

    def forward(self, x, quantization_error_info=None):
        prefix = f"{self.layer_prefix}." if self.layer_prefix else ""

        # --- Quantize weight (tensor-wise) ---
        quantized_weight = NVFP4Quantization.apply(
            self.weight, self.block_size,
            quantization_error_info, prefix + 'weight'
        )

        # --- Quantize activation ---
        if x.dim() == 3:
            # Token-wise quantization for transformer activations
            quantized_x = NVFP4ActivationQuantization.apply(
                x, self.block_size,
                quantization_error_info, prefix + 'input'
            )
        else:
            # Tensor-wise quantization (e.g. time embedding, 2D input)
            quantized_x = NVFP4Quantization.apply(
                x, self.block_size,
                quantization_error_info, prefix + 'input'
            )

        return F.linear(quantized_x, quantized_weight, self.bias)


class NVFP4MultiHeadAttention(nn.MultiheadAttention):
    """MultiHeadAttention with NVFP4 quantization on Q/K/V projections and
    attention output.

    - Q/K/V/out_proj weights: NVFP4Quantization (tensor-wise).
    - Q/K/V inputs and attention output: NVFP4ActivationQuantization (token-wise).

    Args:
        embed_dim, num_heads, ...: standard nn.MultiheadAttention args.
        block_size: elements per NVFP4 scaling block (default 16).
        layer_prefix: optional string prefix for error_info_ keys.
    """

    def __init__(self, embed_dim, num_heads, block_size=16,
                 layer_prefix=None, **kwargs):
        self.block_size = block_size
        self.layer_prefix = layer_prefix
        super().__init__(embed_dim, num_heads, **kwargs)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False,
                quantization_error_info=None):

        # --- Replicate parent's batch_first handling → [seq, batch, embed] ---
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0)
                                     for x in (query, key, value))

        pref = f"{self.layer_prefix}." if self.layer_prefix else ""

        # --- 1. Quantize Q / K / V inputs (token-wise) ---
        query = NVFP4ActivationQuantization.apply(
            query, self.block_size,
            quantization_error_info, pref + 'query.input'
        )
        key = NVFP4ActivationQuantization.apply(
            key, self.block_size,
            quantization_error_info, pref + 'key.input'
        )
        value = NVFP4ActivationQuantization.apply(
            value, self.block_size,
            quantization_error_info, pref + 'value.input'
        )

        # --- 2. Quantize Q / K / V / out projection weights (tensor-wise) ---
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads

        q_weight, k_weight, v_weight = self.in_proj_weight.split(
            embed_dim, dim=0)
        q_weight = NVFP4Quantization.apply(
            q_weight, self.block_size,
            quantization_error_info, pref + 'query.weight'
        )
        k_weight = NVFP4Quantization.apply(
            k_weight, self.block_size,
            quantization_error_info, pref + 'key.weight'
        )
        v_weight = NVFP4Quantization.apply(
            v_weight, self.block_size,
            quantization_error_info, pref + 'value.weight'
        )
        out_proj_weight = NVFP4Quantization.apply(
            self.out_proj.weight, self.block_size,
            quantization_error_info, pref + 'output.weight'
        )

        # --- 3. Q / K / V projection ---
        q_bias = k_bias = v_bias = None
        if self.in_proj_bias is not None:
            q_bias, k_bias, v_bias = self.in_proj_bias.split(embed_dim, dim=0)

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)

        tgt_len, bsz = q.shape[0], q.shape[1]
        src_len = k.shape[0]

        # reshape → [batch, num_heads, seq, head_dim]
        q = q.view(tgt_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)

        # --- add_bias_kv ---
        bias_k = self.bias_k if hasattr(self, 'bias_k') and self.bias_k is not None else None
        bias_v = self.bias_v if hasattr(self, 'bias_v') and self.bias_v is not None else None
        if bias_k is not None and bias_v is not None:
            if bias_k.dim() == 5:
                bias_k = bias_k.reshape(1, num_heads, 1, head_dim).expand(bsz, -1, 1, -1)
                bias_v = bias_v.reshape(1, num_heads, 1, head_dim).expand(bsz, -1, 1, -1)
            elif bias_k.dim() == 3:
                bias_k = bias_k.unsqueeze(0).expand(bsz, -1, -1, -1)
                bias_v = bias_v.unsqueeze(0).expand(bsz, -1, -1, -1)
            k = torch.cat([k, bias_k], dim=2)
            v = torch.cat([v, bias_v], dim=2)
            src_len = src_len + 1
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 3:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 4:
                    attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # --- add_zero_attn ---
        if self.add_zero_attn:
            zero_attn = torch.zeros(bsz, num_heads, 1, head_dim,
                                    dtype=k.dtype, device=k.device)
            k = torch.cat([k, zero_attn], dim=2)
            v = torch.cat([v, zero_attn], dim=2)
            src_len = src_len + 1
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 3:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 4:
                    attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # --- merge key_padding_mask into attn_mask ---
        if key_padding_mask is not None:
            kpm = torch.zeros(bsz, 1, 1, src_len, dtype=q.dtype, device=q.device)
            kpm.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_mask = kpm if attn_mask is None else attn_mask + kpm

        # --- 4. Attention computation ---
        dropout_p = self.dropout if self.training else 0.0
        if need_weights:
            scale = head_dim ** -0.5
            attn_weights_out = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_weights_out = attn_weights_out + attn_mask
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(tgt_len, src_len, device=q.device, dtype=torch.bool),
                    diagonal=1)
                attn_weights_out.masked_fill_(causal_mask, float('-inf'))
            attn_weights_out = F.softmax(attn_weights_out, dim=-1)
            if dropout_p > 0:
                attn_weights_out = F.dropout(attn_weights_out, p=dropout_p,
                                             training=self.training)
            attn_output = torch.matmul(attn_weights_out, v)
            if average_attn_weights:
                attn_weights_for_return = attn_weights_out.detach().mean(dim=1)
            else:
                attn_weights_for_return = attn_weights_out.detach()
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal)
            attn_weights_for_return = None

        # reshape back: [batch, num_heads, tgt, head_dim] → [tgt, batch, embed_dim]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(
            tgt_len, bsz, embed_dim)

        # --- 5. Quantize attention output BEFORE out_proj (token-wise) ---
        attn_output = NVFP4ActivationQuantization.apply(
            attn_output, self.block_size,
            quantization_error_info, pref + 'attn.output'
        )

        # --- 6. Output projection ---
        attn_output = F.linear(attn_output, out_proj_weight, self.out_proj.bias)

        # transpose back for batch_first
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            return attn_output, attn_weights_for_return
        else:
            return (attn_output,)


# ===========================================================================
# Sinusoidal time embedding — NVFP4-quantized MLP
# ===========================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding → small NVFP4-quantized MLP."""

    def __init__(self, dim: int, max_period: int = 10000,
                 block_size: int = 16, layer_prefix: str = None):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            NVFP4Linear(dim, dim * 4, block_size=block_size,
                        layer_prefix=f"{layer_prefix}.mlp.0"),
            nn.SiLU(),
            NVFP4Linear(dim * 4, dim, block_size=block_size,
                        layer_prefix=f"{layer_prefix}.mlp.1"),
        )
        self.layer_prefix = layer_prefix

    def forward(self, t: torch.Tensor,
                quantization_error_dict: dict = None) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        for module in self.mlp:
            if isinstance(module, NVFP4Linear):
                emb = module(emb, quantization_error_dict)
            else:
                emb = module(emb)
        return emb


# ===========================================================================
# adaLN-Zero modulation
# ===========================================================================

class AdaLNZeo(nn.Module):
    """Adaptive LayerNorm with zero-initialized residual scaling (NVFP4)."""

    def __init__(self, time_dim: int, hidden_dim: int,
                 block_size: int = 16, layer_prefix: str = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Sequential(
            nn.SiLU(),
            NVFP4Linear(time_dim, hidden_dim * 3, block_size=block_size,
                        layer_prefix=f"{layer_prefix}.head"),
        )
        # Zero-init the output bias so that at t=0 the residual is identity
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                quantization_error_info: dict) -> tuple:
        for block in self.head:
            if isinstance(block, NVFP4Linear):
                t_emb = block(t_emb, quantization_error_info)
            else:
                t_emb = block(t_emb)
        shift, scale, gate = t_emb.chunk(3, dim=-1)  # each (B, C)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift, gate


# ===========================================================================
# Transformer block
# ===========================================================================

class DiTBlock(nn.Module):
    """One transformer block with adaLN-Zero conditioning (NVFP4)."""

    def __init__(self, hidden_dim: int, num_heads: int, time_dim: int,
                 mlp_ratio: float = 4.0, block_size: int = 16,
                 layer_prefix: str = None):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        self.adaln1 = AdaLNZeo(time_dim, hidden_dim, block_size=block_size,
                               layer_prefix=f"{layer_prefix}.adaln1")
        self.attn = NVFP4MultiHeadAttention(
            hidden_dim, num_heads, batch_first=True,
            block_size=block_size,
            layer_prefix=f"{layer_prefix}.attn"
        )
        self.adaln2 = AdaLNZeo(time_dim, hidden_dim, block_size=block_size,
                               layer_prefix=f"{layer_prefix}.adaln2")
        self.mlp = nn.Sequential(
            NVFP4Linear(hidden_dim, mlp_dim, block_size=block_size,
                        layer_prefix=f"{layer_prefix}.mlp.0"),
            nn.GELU(),
            NVFP4Linear(mlp_dim, hidden_dim, block_size=block_size,
                        layer_prefix=f"{layer_prefix}.mlp.1"),
        )
        self.layer_prefix = layer_prefix

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                quantization_error_info: dict) -> torch.Tensor:
        # Self-attention with adaLN
        modulated, gate = self.adaln1(x, t_emb, quantization_error_info)
        attn_out, _ = self.attn(modulated, modulated, modulated,
                                quantization_error_info=quantization_error_info)
        x = x + gate * attn_out

        # MLP with adaLN
        modulated, gate = self.adaln2(x, t_emb, quantization_error_info)
        for module in self.mlp:
            if isinstance(module, NVFP4Linear):
                modulated = module(modulated, quantization_error_info)
            else:
                modulated = module(modulated)
        x = x + gate * modulated

        return x


# ===========================================================================
# NVFP4-Quantized DiT model
# ===========================================================================

class NVFP4DiT(nn.Module):
    """NVFP4-quantized Diffusion Transformer for image generation.

    All linear weights are quantized with NVFP4Quantization (tensor-wise).
    All transformer activations [bs, n_seq, dim] are quantized with
    NVFP4ActivationQuantization (token-wise).  2D activations (e.g. time
    embedding MLP) use NVFP4Quantization (tensor-wise).

    Args:
        in_channels:   number of input image channels (1 for MNIST).
        image_size:    spatial size of the input image (28 for MNIST).
        patch_size:    size of each square patch.
        hidden_dim:    transformer hidden dimension.
        depth:         number of transformer blocks.
        num_heads:     attention heads.
        time_dim:      dimension of the sinusoidal time embedding.
        mlp_ratio:     MLP hidden / transformer hidden ratio.
        block_size:    elements per NVFP4 scaling block (default 16).
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        patch_size: int = 4,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
        block_size: int = 16,
        **kwargs,  # accept (and ignore) bitW/bitA/bitG etc. for config compatibility
    ):
        super().__init__()
        assert image_size % patch_size == 0, \
            "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size  # 16 for MNIST

        # Patch embedding: (B, C, H, W) → (B, num_patches, hidden_dim)
        self.patch_embed = NVFP4Linear(
            patch_dim, hidden_dim, block_size=block_size,
            layer_prefix="patch_embed"
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(
            time_dim, block_size=block_size, layer_prefix="time_embed")

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_dim, mlp_ratio,
                     block_size=block_size,
                     layer_prefix=f"block.{idx}")
            for idx in range(depth)
        ])

        # Final norm + output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.proj = NVFP4Linear(
            hidden_dim, patch_dim, block_size=block_size, layer_prefix="proj")

        # Init weights
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self._init_weights()

        self.quantization_error_info = {}
        print(f"Initialize NVFP4-Quantized DiT (block_size={block_size}, "
              f"hidden_dim={hidden_dim}, depth={depth}, heads={num_heads})")

    def _init_weights(self):
        # Small init for final proj
        nn.init.zeros_(self.proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.mlp[0].weight)  # MLP first layer
            nn.init.zeros_(block.mlp[0].bias)
            nn.init.xavier_uniform_(block.mlp[-1].weight)  # MLP last layer
            nn.init.zeros_(block.mlp[-1].bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches: (B, C, H, W) → (B, N, patch_dim)."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert patches back to image: (B, N, patch_dim) → (B, C, H, W)."""
        p = self.patch_size
        h, w = H // p, W // p
        C = x.shape[-1] // (p * p)
        x = x.reshape(x.shape[0], h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        x = x.reshape(x.shape[0], C, H, W)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                quantization_error_info: dict = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image.
            t: (B,) integer timesteps.
        Returns:
            (B, C, H, W) predicted noise.
        """
        B, C, H, W = x.shape

        # Patchify
        tokens = self.patchify(x)                        # (B, N, patch_dim)
        tokens = self.patch_embed(tokens, quantization_error_info)  # (B, N, hidden_dim)
        tokens = tokens + self.pos_embed                  # add position

        # Time conditioning
        t_emb = self.time_embed(t, quantization_error_info)  # (B, time_dim)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb, self.quantization_error_info)

        # Output projection
        tokens = self.final_norm(tokens)
        patches = self.proj(tokens, quantization_error_info)  # (B, N, patch_dim)

        # Unpatchify back to image
        out = self.unpatchify(patches, H, W)              # (B, C, H, W)
        return out


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":

    import torch

    model = NVFP4DiT()
    batch_size = 10
    inputs = torch.rand(batch_size, 1, 28, 28)
    timestamp = torch.randint(0, 10000, (batch_size,))
    outputs = model(inputs, timestamp)
    print("output shape:", outputs.shape)
    loss = torch.nn.MSELoss()(outputs, torch.randn_like(outputs))
    loss.backward()
    print("loss:", loss.item())
    for layer_name, value in model.quantization_error_info.items():
        print(layer_name)
