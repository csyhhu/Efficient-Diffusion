"""
Quantized DiT (Diffusion Transformer) — nn.Linear → QuantizedLinear,
MultiheadAttention → QuantizedMultiHeadAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# supports both package import and direct script execution
try:
    from ..modules.quantized_linear import QuantizedLinear
    from ..modules.quantized_mha import QuantizedMultiHeadAttention
except ImportError:
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src)
    from modules.quantized_linear import QuantizedLinear
    from modules.quantized_mha import QuantizedMultiHeadAttention


# ---------------------------------------------------------------------------
# Sinusoidal time embedding — same logic, quantized Linear
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding → small MLP."""

    def __init__(self, dim: int, max_period: int = 10000, bitW: int = 8, bitA: int = 8, bitG: int = 8, layer_prefix: str = None):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            QuantizedLinear(dim, dim * 4, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.mlp.0"),
            nn.SiLU(),
            QuantizedLinear(dim * 4, dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.mlp.1"),
        )
        self.layer_prefix = layer_prefix

    def forward(self, t: torch.Tensor, quantization_error_dict: dict =None) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        # return self.mlp(emb)
        mlp_idx = 0
        for module in self.mlp:
            if isinstance(module, QuantizedLinear):
                emb = module(emb, quantization_error_dict)
                mlp_idx += 1
            else:
                emb = module(emb)
        return emb


# ---------------------------------------------------------------------------
# adaLN-Zero modulation
# ---------------------------------------------------------------------------

class AdaLNZeo(nn.Module):
    """Adaptive LayerNorm with zero-initialized residual scaling."""

    def __init__(self, time_dim: int, hidden_dim: int, bitW: int = 8, bitA: int = 8, bitG: int = 8, layer_prefix: str = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Sequential(
            nn.SiLU(),
            QuantizedLinear(time_dim, hidden_dim * 3, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.head"),
        )
        # Zero-init the output bias so that at t=0 the residual is identity
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        # self.layer_prefix = layer_prefix

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, quantization_error_info:dict) -> tuple:
        # shift, scale, gate = self.head(t_emb).chunk(3, dim=-1)  # each (B, C)
        for block in self.head:
            if isinstance(block, QuantizedLinear):
                t_emb = block(t_emb, quantization_error_info)
            else:
                t_emb = block(t_emb)
        shift, scale, gate = t_emb.chunk(3, dim=-1)  # each (B, C)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift, gate


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """One transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, time_dim: int,
                 mlp_ratio: float = 4.0, bitW: int = 8, bitA: int = 8, bitG: int = 8, layer_prefix: str = None):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        self.adaln1 = AdaLNZeo(time_dim, hidden_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.adaln1")
        self.attn = QuantizedMultiHeadAttention(
            hidden_dim, num_heads, batch_first=True,
            bitW=bitW, bitA=bitA, bitG=bitG,
            layer_prefix=f"{layer_prefix}.attn"
        )
        self.adaln2 = AdaLNZeo(time_dim, hidden_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.adaln2")
        self.mlp = nn.Sequential(
            QuantizedLinear(hidden_dim, mlp_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.mlp.0"),
            nn.GELU(),
            QuantizedLinear(mlp_dim, hidden_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"{layer_prefix}.mlp.1"),
        )
        self.layer_prefix = layer_prefix

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, quantization_error_info: dict) -> torch.Tensor:
        # Self-attention with adaLN
        modulated, gate = self.adaln1(x, t_emb, quantization_error_info)
        attn_out, _ = self.attn(modulated, modulated, modulated, quantization_error_info=quantization_error_info)
        x = x + gate * attn_out

        # MLP with adaLN
        modulated, gate = self.adaln2(x, t_emb, quantization_error_info)
        # x = x + gate * self.mlp(modulated)
        for module in self.mlp:
            if isinstance(module, QuantizedLinear):
                modulated = module(modulated, quantization_error_info)
            else:
                modulated = module(modulated)
        x = x + gate * modulated

        return x


# ---------------------------------------------------------------------------
# Quantized DiT model
# ---------------------------------------------------------------------------

class QuantizedDiT(nn.Module):
    """Quantized Diffusion Transformer for image generation.

    Args:
        in_channels:   number of input image channels (1 for MNIST).
        image_size:    spatial size of the input image (28 for MNIST).
        patch_size:    size of each square patch.
        hidden_dim:    transformer hidden dimension.
        depth:         number of transformer blocks.
        num_heads:     attention heads.
        time_dim:      dimension of the sinusoidal time embedding.
        mlp_ratio:     MLP hidden / transformer hidden ratio.
        bitW:          weight quantization bit-width.
        bitA:          activation quantization bit-width.
        bitG:          gradient quantization bit-width (reserved).
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
        bitW: int = 8,
        bitA: int = 8,
        bitG: int = 8,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size  # 16 for MNIST

        # Patch embedding: (B, C, H, W) → (B, num_patches, hidden_dim)
        self.patch_embed = QuantizedLinear(patch_dim, hidden_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"patch_embed")

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix="time_embed")

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_dim, mlp_ratio, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix=f"block.{idx}")
            for idx in range(depth)
        ])

        # Final norm + output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.proj = QuantizedLinear(hidden_dim, patch_dim, bitW=bitW, bitA=bitA, bitG=bitG, layer_prefix="proj")

        # Init weights
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self._init_weights()

        self.quantization_error_info = {}
        print(f"Initialize Quantized Simple DiT with {bitW}-bitW, {bitA}-bitA, {bitG}-bitG")

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, quantization_error_info: dict=None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image.
            t: (B,) integer timesteps.
        Returns:
            (B, C, H, W) predicted noise.
        """
        B, C, H, W = x.shape

        # Patchify
        tokens = self.patchify(x)                     # (B, N, patch_dim)
        tokens = self.patch_embed(tokens, quantization_error_info)             # (B, N, hidden_dim)
        tokens = tokens + self.pos_embed              # add position

        # Time conditioning
        t_emb = self.time_embed(t)                    # (B, time_dim)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb, self.quantization_error_info)

        # Output projection
        tokens = self.final_norm(tokens)
        patches = self.proj(tokens, quantization_error_info)                   # (B, N, patch_dim)

        # Unpatchify back to image
        out = self.unpatchify(patches, H, W)          # (B, C, H, W)
        return out


if __name__ == "__main__":

    import torch
    model = QuantizedDiT()
    batch_size = 10
    inputs = torch.rand(batch_size, 1, 28, 28)
    timestamp = torch.randint(0, 10000, (batch_size,))
    outputs = model(inputs, timestamp)
    print("output shape:", outputs.shape)
    loss = torch.nn.MSELoss()(outputs, torch.randn_like(outputs))
    loss.backward()
    print("loss:", loss.item())
    for layer_name, value in model.quantization_error_info.items():
        # print(layer_name, value.shape)
        print(layer_name)
