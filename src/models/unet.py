"""
Lightweight UNet for diffusion models on small images (e.g. MNIST 28×28).

Encoder:   28 → 14 → 7   (channels: 64 → 128 → 256)
Bottleneck: 7×7 with self-attention
Decoder:   7 → 14 → 28   (with skip connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _gnorm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Return GroupNorm with a valid number of groups that divides num_channels."""
    for g in range(max_groups, 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)


# ---------------------------------------------------------------------------
# Time-step embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps.

    Projects a scalar timestep `t` into a fixed-dimensional vector via
    sin / cos frequencies, then passes through a small MLP.
    """

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
        """
        Args:
            t: (B,) integer timesteps in [0, T-1].
        Returns:
            (B, dim) time embedding.
        """
        half = self.dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with optional down/up-sampling and time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 downsample: bool = False, upsample: bool = False):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _gnorm(in_ch)
        self.norm2 = _gnorm(out_ch)

        # Residual shortcut
        if in_ch != out_ch or downsample or upsample:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else None
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if upsample else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]  # time conditioning
        h = F.silu(self.norm2(h))
        h = self.conv2(h)

        shortcut = self.shortcut(x)

        if self.upsample:
            h = self.upsample(h)
            shortcut = self.upsample(shortcut)
        if self.downsample:
            h = self.downsample(h)
            shortcut = self.downsample(shortcut)

        return h + shortcut


class SelfAttention(nn.Module):
    """2D self-attention with a single head (sufficient for 7×7 MNIST)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _gnorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(dim=1)  # each (B, C, N)

        attn = torch.softmax((q.transpose(1, 2) @ k) / (C ** 0.5), dim=-1)  # (B, N, N)
        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Simple UNet
# ---------------------------------------------------------------------------

class SimpleUNet(nn.Module):
    """A minimal UNet backbone for MNIST (grayscale, 28×28).

    Encoder path:   28 → 14 → 7   (channels: 64 → 128 → 256)
    Bottleneck:     7×7 with self-attention
    Decoder path:   7 → 14 → 28   (with skip connections from encoder)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # --- Encoder ---
        self.enc1 = ResBlock(in_channels, base_channels, time_dim)                          # 28×28
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_dim, downsample=True)   # 14×14
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_dim, downsample=True)  # 7×7

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 4, time_dim),
            SelfAttention(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4, time_dim),
        )

        # --- Decoder ---
        self.dec3 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.dec2 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        self.dec1_res = ResBlock(base_channels + in_channels, base_channels, time_dim)
        self.dec1_tail = nn.Sequential(
            _gnorm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image.
            t: (B,) integer timesteps.
        Returns:
            (B, C, H, W) predicted noise.
        """
        t_emb = self.time_embed(t)

        # Encoder with skip connections
        e1 = self.enc1(x, t_emb)           # (B, 64,  28, 28)
        e2 = self.enc2(e1, t_emb)           # (B, 128, 14, 14)
        e3 = self.enc3(e2, t_emb)           # (B, 256,  7,  7)

        # Bottleneck
        h = self.bottleneck[0](e3, t_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb)    # (B, 256,  7,  7)

        # Decoder with skip connections (explicit upsample before concat)
        h = F.interpolate(h, scale_factor=2, mode="nearest")        # 7 → 14
        d3 = self.dec3(torch.cat([h, e2], dim=1), t_emb)            # (B, 128, 14, 14)
        d3 = F.interpolate(d3, scale_factor=2, mode="nearest")      # 14 → 28
        d2 = self.dec2(torch.cat([d3, e1], dim=1), t_emb)           # (B, 64,  28, 28)
        h2 = self.dec1_res(torch.cat([d2, x], dim=1), t_emb)        # (B, 64,  28, 28)
        out = self.dec1_tail(h2)                                     # (B, 1,   28, 28)
        return out


if __name__ == "__main__":

    import torch
    model = SimpleUNet()
    batch_size = 10
    inputs = torch.rand(batch_size, 1, 28, 28)
    timestamp = torch.randint(0, 10000, (batch_size,))
    outputs = model(inputs, timestamp)
    print(outputs.shape)