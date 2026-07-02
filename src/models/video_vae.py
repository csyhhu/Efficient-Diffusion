"""
3D Causal VAE for video compression (Moving MNIST scale).

Design (inspired by Wan-VAE):
  - 3D causal convolutions: future frames never influence past frames.
  - RMSNorm instead of GroupNorm to preserve temporal causality.
  - Encoder: 3× 2× spatial downsampling (64 → 32 → 16 → 8), temporal unchanged.
  - Decoder: symmetric 3× 2× spatial upsampling.
  - Latent: (T, latent_dim, 8, 8), where latent_dim = 8 (grayscale MNIST).
  - ~3M parameters — compact enough for consumer GPUs.

Reference:
    Wan Team, "Wan: Open and Advanced Large-Scale Video Generative Models"
    https://arxiv.org/abs/2503.20314
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal 3D convolution wrapper
# ---------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """3D convolution with causal padding on the time dimension.

    Standard Conv3d pads symmetrically on both sides of the time dim,
    which leaks future information. This wrapper pads only on the past side
    so that output at frame `t` only depends on frames ≤ t.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        assert isinstance(kernel_size, int), "only isotropic kernel supported for simplicity"
        self.kernel_size = kernel_size
        self.stride = stride
        # Causal padding: full padding on H/W, (kernel-1) on past side of T, 0 on future side
        self.time_pad = kernel_size - 1
        self.spatial_pad = padding
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        # Pad: (left_W, right_W, left_H, right_H, left_T, right_T)
        x = F.pad(x, [self.spatial_pad, self.spatial_pad,
                       self.spatial_pad, self.spatial_pad,
                       self.time_pad, 0])
        return self.conv(x)


# ---------------------------------------------------------------------------
# Helper: RMSNorm (from Wan-VAE, preserves causality vs GroupNorm)
# ---------------------------------------------------------------------------

class RMSNorm3d(nn.Module):
    """RMSNorm over the channel dimension for 5D tensors (B, C, T, H, W)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        rms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * rms * self.weight.view(1, -1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Residual block with optional spatial down/up-sampling
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """3D residual block with two causal convolutions and RMSNorm.

    Supports 2× spatial downsampling (stride=2) or upsampling (nearest + conv).
    Temporal dimension is never changed by this block.
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False,
                 upsample: bool = False):
        super().__init__()
        assert not (downsample and upsample), "cannot downsample and upsample simultaneously"

        stride = (1, 2, 2) if downsample else (1, 1, 1)
        self.downsample = downsample
        self.upsample = upsample

        self.norm1 = RMSNorm3d(in_ch)
        self.conv1 = CausalConv3d(in_ch, out_ch, kernel_size=3, stride=stride)
        self.norm2 = RMSNorm3d(out_ch)
        self.conv2 = CausalConv3d(out_ch, out_ch, kernel_size=3)

        # Shortcut
        if in_ch != out_ch or downsample or upsample:
            shortcut_stride = stride if downsample else (1, 1, 1)
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=shortcut_stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut: compute before upsampling, then upsample to match output
        if self.upsample:
            shortcut = self.shortcut(x)
            shortcut = F.interpolate(shortcut, scale_factor=(1, 2, 2), mode='nearest')
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        else:
            shortcut = self.shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + shortcut


# ---------------------------------------------------------------------------
# Video VAE
# ---------------------------------------------------------------------------

class VideoVAE(nn.Module):
    """3D causal VAE for video compression.

    Encoder:   64 → 32 → 16 → 8  (3× 2× spatial downsample)
    Decoder:    8 → 16 → 32 → 64  (3× 2× spatial upsample)

    Args:
        in_channels:      input channels (1 for grayscale MNIST).
        latent_dim:       latent channel dimension.
        base_channels:    base channel count for encoder/decoder.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 8,
                 base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- Encoder ----
        self.enc_conv_in = CausalConv3d(in_channels, base_channels, kernel_size=3)

        # 64 → 32
        self.enc_down1 = ResBlock3D(base_channels, base_channels * 2, downsample=True)
        # 32 → 16
        self.enc_down2 = ResBlock3D(base_channels * 2, base_channels * 4, downsample=True)
        # 16 → 8
        self.enc_down3 = ResBlock3D(base_channels * 4, base_channels * 4, downsample=True)

        self.enc_norm_out = RMSNorm3d(base_channels * 4)
        self.enc_conv_out = CausalConv3d(base_channels * 4, latent_dim * 2, kernel_size=3)

        # ---- Decoder ----
        self.dec_conv_in = nn.Conv3d(latent_dim, base_channels * 4, kernel_size=3, padding=1)

        # 8 → 16
        self.dec_up1 = ResBlock3D(base_channels * 4, base_channels * 4, upsample=True)
        # 16 → 32
        self.dec_up2 = ResBlock3D(base_channels * 4, base_channels * 2, upsample=True)
        # 32 → 64
        self.dec_up3 = ResBlock3D(base_channels * 2, base_channels, upsample=True)

        self.dec_norm_out = RMSNorm3d(base_channels)
        self.dec_conv_out = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)

    # ------------------------------------------------------------------
    # TODO: Implement the encoder forward pass.
    #
    # 1. Pass x through enc_conv_in.
    # 2. Pass through enc_down1, enc_down2, enc_down3.
    # 3. Apply enc_norm_out → SiLU → enc_conv_out.
    # 4. Split output into mu and logvar (each latent_dim channels).
    # 5. Sample z = mu + exp(0.5 * logvar) * eps, where eps ~ N(0, I).
    # 6. Return (z, mu, logvar).
    #
    # HINT:
    #   h = self.enc_conv_in(x)
    #   h = self.enc_down1(h)
    #   h = self.enc_down2(h)
    #   h = self.enc_down3(h)
    #   h = F.silu(self.enc_norm_out(h))
    #   h = self.enc_conv_out(h)
    #   mu, logvar = h.chunk(2, dim=1)
    #   z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    #   return z, mu, logvar
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple:
        """Encode video to latent.

        Args:
            x: (B, C_in, T, H, W) input video frames.

        Returns:
            z:      (B, latent_dim, T, H//8, W//8) latent.
            mu:     (B, latent_dim, T, H//8, W//8) mean.
            logvar: (B, latent_dim, T, H//8, W//8) log variance.
        """
        # --- YOUR CODE BELOW ---
        h = self.enc_conv_in(x)
        h = self.enc_down1(h)
        h = self.enc_down2(h)
        h = self.enc_down3(h)
        h = F.silu(self.enc_norm_out(h))
        h = self.enc_conv_out(h)
        mu, logvar = h.chunk(2, dim=1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar
        # --- END YOUR CODE ---

    # ------------------------------------------------------------------
    # TODO: Implement the decoder forward pass.
    #
    # 1. Pass z through dec_conv_in.
    # 2. Pass through dec_up1, dec_up2, dec_up3.
    # 3. Apply dec_norm_out → SiLU → dec_conv_out.
    # 4. Return reconstructed video.
    #
    # HINT:
    #   h = self.dec_conv_in(z)
    #   h = self.dec_up1(h)
    #   h = self.dec_up2(h)
    #   h = self.dec_up3(h)
    #   h = F.silu(self.dec_norm_out(h))
    #   return self.dec_conv_out(h)
    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to video.

        Args:
            z: (B, latent_dim, T, H_lat, W_lat) latent.

        Returns:
            (B, C_in, T, H, W) reconstructed video.
        """
        # --- YOUR CODE BELOW ---
        h = self.dec_conv_in(z)
        h = self.dec_up1(h)
        h = self.dec_up2(h)
        h = self.dec_up3(h)
        h = F.silu(self.dec_norm_out(h))
        return self.dec_conv_out(h)
        # --- END YOUR CODE ---

    def forward(self, x: torch.Tensor) -> tuple:
        """Full VAE forward: encode → decode → reconstruction.

        Returns:
            (recon, z, mu, logvar)
        """
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, z, mu, logvar

    @torch.no_grad()
    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to latent without sampling (use mu directly)."""
        h = self.enc_conv_in(x)
        h = self.enc_down1(h)
        h = self.enc_down2(h)
        h = self.enc_down3(h)
        h = F.silu(self.enc_norm_out(h))
        h = self.enc_conv_out(h)
        mu, _ = h.chunk(2, dim=1)
        return mu

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video."""
        return self.decode(z)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoVAE(in_channels=1, latent_dim=8, base_channels=32).to(device)
    x = torch.randn(2, 1, 16, 64, 64).to(device)
    recon, z, mu, logvar = model(x)
    print(f"Input:   {tuple(x.shape)}")
    print(f"Latent:  {tuple(z.shape)}")
    print(f"Recon:   {tuple(recon.shape)}")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")
