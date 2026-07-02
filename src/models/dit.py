"""
Lightweight DiT (Diffusion Transformer) for small images (e.g. MNIST 28×28).

Key design choices:
- Patchify 28×28 into 7×7 patches (patch_size=4) → 49 tokens
- adaLN-Zero conditioning on timestep (residual scale starts at 0)
- 4-head self-attention, 6 transformer blocks
- ~1.5M parameters — comparable to SimpleUNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (same as in unet.py)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding → small MLP."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# adaLN-Zero modulation
# ---------------------------------------------------------------------------

class AdaLNZeo(nn.Module):
    """Adaptive LayerNorm with zero-initialized residual scaling.

    For an input of shape (B, N, C):
        x = norm(x) * (1 + scale) + shift
        x = x + gate * block(x)

    The modulation parameters (shift, scale, gate) are produced by a shared
    SiLU → Linear head from the time embedding.
    """

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 3),
        )
        # Zero-init the output bias so that at t=0 the residual is identity
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple:
        """
        Returns:
            (modulated_x, gate) — gate is applied later after the transformer block.
        """
        shift, scale, gate = self.head(t_emb).chunk(3, dim=-1)  # each (B, C)
        # Add token dimension for broadcasting: (B, 1, C)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift, gate


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """One transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, time_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        self.adaln1 = AdaLNZeo(time_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.adaln2 = AdaLNZeo(time_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Self-attention with adaLN
        modulated, gate = self.adaln1(x, t_emb)
        attn_out, _ = self.attn(modulated, modulated, modulated)
        x = x + gate * attn_out

        # MLP with adaLN
        modulated, gate = self.adaln2(x, t_emb)
        x = x + gate * self.mlp(modulated)

        return x


# ---------------------------------------------------------------------------
# DiT model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer for image generation.

    Args:
        in_channels:   number of input image channels (1 for MNIST).
        image_size:    spatial size of the input image (28 for MNIST).
        patch_size:    size of each square patch.
        hidden_dim:    transformer hidden dimension.
        depth:         number of transformer blocks.
        num_heads:     attention heads.
        time_dim:      dimension of the sinusoidal time embedding.
        mlp_ratio:     MLP hidden / transformer hidden ratio.
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
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size  # 16 for MNIST

        # Patch embedding: (B, C, H, W) → (B, num_patches, hidden_dim)
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_dim, mlp_ratio)
            for _ in range(depth)
        ])

        # Final norm + output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, patch_dim)

        # Init weights
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self._init_weights()

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
        # Reshape into patches
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        tokens = self.patch_embed(tokens)             # (B, N, hidden_dim)
        tokens = tokens + self.pos_embed              # add position

        # Time conditioning
        t_emb = self.time_embed(t)                    # (B, time_dim)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb)

        # Output projection
        tokens = self.final_norm(tokens)
        patches = self.proj(tokens)                   # (B, N, patch_dim)

        # Unpatchify back to image
        out = self.unpatchify(patches, H, W)          # (B, C, H, W)
        return out


if __name__ == "__main__":

    import torch
    model = DiT()
    batch_size = 10
    inputs = torch.rand(batch_size, 1, 28, 28)
    timestamp = torch.randint(0, 10000, (batch_size,))
    outputs = model(inputs, timestamp)
    print(outputs.shape)