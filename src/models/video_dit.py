"""
Video DiT (Diffusion Transformer) for latent-space video generation.

Design (inspired by Wan-DiT, simplified for Moving MNIST):
  - Input: latent video (B, C, T, H, W) from VideoVAE.
  - 2D patchify each frame independently: reshapes to (B*T, N_spatial, patch_dim).
  - Spatial position: learnable embedding (shared across frames).
  - Temporal position: learnable embedding (shared across spatial patches).
  - Full joint self-attention over all T * N_spatial tokens.
  - adaLN-Zero time conditioning (reuses DiTBlock from dit.py).
  - Output: velocity field in latent space, same shape as input.

Token flow (Moving MNIST example):
    latent (2, 8, 16, 8, 8)
    → 2D patchify per frame (patch_size=2)
    → (2, 16*16, 32) = (2, 256, 32)   [16 frames × 4×4 patches = 256 tokens]
    → Linear(32, 256)
    → + spatial_pos (1, 16, 256) [tiled per frame]
    → + temporal_pos (1, 16, 256) [tiled per spatial position]
    → 6× DiTBlock (full attention, 256² matrix)
    → LayerNorm → Linear(256, 32)
    → unpatchify → (2, 8, 16, 8, 8)

Reference:
    Wan Team, "Wan: Open and Advanced Large-Scale Video Generative Models"
    https://arxiv.org/abs/2503.20314
"""

import math
import torch
import torch.nn as nn

from .dit import DiTBlock, SinusoidalTimeEmbedding


# ---------------------------------------------------------------------------
# Video DiT
# ---------------------------------------------------------------------------

class VideoDiT(nn.Module):
    """Diffusion Transformer for video latent generation.

    Args:
        in_channels:   number of latent channels (e.g. 8 from VideoVAE).
        num_frames:    number of latent frames T.
        spatial_size:  latent spatial size (e.g. 8 for 8×8 latents).
        patch_size:    size of each 2D spatial patch.
        hidden_dim:    transformer hidden dimension.
        depth:         number of transformer blocks.
        num_heads:     attention heads.
        time_dim:      dimension of the sinusoidal time embedding.
        mlp_ratio:     MLP hidden / transformer hidden ratio.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_frames: int = 16,
        spatial_size: int = 8,
        patch_size: int = 2,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert spatial_size % patch_size == 0, \
            f"spatial_size ({spatial_size}) must be divisible by patch_size ({patch_size})"

        self.in_channels = in_channels
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.num_patches_spatial = (spatial_size // patch_size) ** 2
        self.num_tokens = num_frames * self.num_patches_spatial

        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: (B*T, N_spatial, patch_dim) → (B*T, N_spatial, hidden_dim)
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        # Positional embeddings
        self.spatial_pos = nn.Parameter(
            torch.randn(1, self.num_patches_spatial, hidden_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, num_frames, hidden_dim) * 0.02
        )

        # Time embedding (continuous t ∈ [0, 1])
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Transformer blocks (reuse DiTBlock with full attention)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_dim, mlp_ratio)
            for _ in range(depth)
        ])

        # Output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.mlp[0].weight)
            nn.init.zeros_(block.mlp[0].bias)
            nn.init.xavier_uniform_(block.mlp[-1].weight)
            nn.init.zeros_(block.mlp[-1].bias)

    # ------------------------------------------------------------------
    # TODO: Implement 2D patchify for each video frame.
    #
    # Input:  (B, C, T, H, W)
    # Output: (B, T * num_patches_spatial, patch_dim)
    #
    # Steps:
    #   1. Permute: (B, C, T, H, W) → (B, T, C, H, W)
    #   2. Flatten batch & time: (B*T, C, H, W)
    #   3. Apply image-style patchify (reshape + permute, same as dit.py)
    #   4. Reshape back to (B, T * N_spatial, patch_dim)
    #
    # HINT:
    #   B, C, T, H, W = x.shape
    #   x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    #   p = self.patch_size
    #   x = x.reshape(B * T, C, H // p, p, W // p, p)
    #   x = x.permute(0, 2, 4, 1, 3, 5)
    #   x = x.reshape(B * T, (H // p) * (W // p), C * p * p)
    #   x = x.reshape(B, T * self.num_patches_spatial, C * p * p)
    #   return x
    # ------------------------------------------------------------------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert latent video to patches.

        Args:
            x: (B, C, T, H, W) latent video.

        Returns:
            (B, T * N_spatial, patch_dim) token sequence.
        """
        # --- YOUR CODE BELOW ---
        B, C, T, H, W = x.shape
        p = self.patch_size
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = x.reshape(B * T, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B * T, self.num_patches_spatial, C * p * p)
        x = x.reshape(B, T * self.num_patches_spatial, C * p * p)
        return x
        # --- END YOUR CODE ---

    # ------------------------------------------------------------------
    # TODO: Implement reverse unpatchify.
    #
    # Input:  (B, T * N_spatial, patch_dim)
    # Output: (B, C, T, H, W)
    #
    # HINT:
    #   B = x.shape[0]
    #   p = self.patch_size
    #   C = self.in_channels
    #   H = W = self.spatial_size
    #   h = w = H // p
    #   x = x.reshape(B, self.num_frames, self.num_patches_spatial, C * p * p)
    #   x = x.reshape(B * self.num_frames, h, w, C, p, p)
    #   x = x.permute(0, 3, 1, 4, 2, 5)  # (B*T, C, h, p, w, p)
    #   x = x.reshape(B * self.num_frames, C, H, W)
    #   x = x.reshape(B, self.num_frames, C, H, W).permute(0, 2, 1, 3, 4)
    #   return x
    # ------------------------------------------------------------------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert token sequence back to latent video.

        Args:
            x: (B, T * N_spatial, patch_dim) token sequence.

        Returns:
            (B, C, T, H, W) latent video.
        """
        # --- YOUR CODE BELOW ---
        B = x.shape[0]
        p = self.patch_size
        C = self.in_channels
        H = W = self.spatial_size
        h = w = H // p
        x = x.reshape(B, self.num_frames, self.num_patches_spatial, C * p * p)
        x = x.reshape(B * self.num_frames, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B * self.num_frames, C, H, W)
        x = x.reshape(B, self.num_frames, C, H, W).permute(0, 2, 1, 3, 4)
        return x
        # --- END YOUR CODE ---

    # ------------------------------------------------------------------
    # TODO: Implement the forward pass.
    #
    # 1. Patchify latent video → tokens.
    # 2. Linear projection → hidden_dim.
    # 3. Add spatial positional embedding (tiled over T frames).
    # 4. Add temporal positional embedding (tiled over N_spatial patches).
    # 5. Compute time conditioning t_emb = self.time_embed(t).
    # 6. Pass through all DiT blocks.
    # 7. Apply final_norm → proj → unpatchify.
    # 8. Return velocity field in latent space.
    #
    # HINT for step 3-4:
    #   spatial_pos = self.spatial_pos.unsqueeze(1).expand(-1, T, -1, -1)
    #                   .reshape(1, T * N, hidden_dim)   # (1, T*N, D)
    #   temporal_pos = self.temporal_pos.unsqueeze(2)
    #                   .expand(-1, -1, N, -1)
    #                   .reshape(1, T * N, hidden_dim)   # (1, T*N, D)
    #   tokens = tokens + spatial_pos + temporal_pos
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for flow matching.

        Args:
            x: (B, C, T, H, W) noisy latent video.
            t: (B,) continuous time ∈ [0, 1].

        Returns:
            (B, C, T, H, W) predicted velocity field.
        """
        B = x.shape[0]
        T = self.num_frames
        N = self.num_patches_spatial

        # --- YOUR CODE BELOW ---
        # Patchify + embed
        tokens = self.patchify(x)                           # (B, T*N, patch_dim)
        tokens = self.patch_embed(tokens)                   # (B, T*N, hidden_dim)

        # Add positional embeddings
        spatial_pos = self.spatial_pos.unsqueeze(1) \
            .expand(-1, T, -1, -1).reshape(1, T * N, -1)
        temporal_pos = self.temporal_pos.unsqueeze(2) \
            .expand(-1, -1, N, -1).reshape(1, T * N, -1)
        tokens = tokens + spatial_pos + temporal_pos

        # Time conditioning
        t_emb = self.time_embed(t)                          # (B, time_dim)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb)

        # Output
        tokens = self.final_norm(tokens)
        patches = self.proj(tokens)
        out = self.unpatchify(patches)
        return out
        # --- END YOUR CODE ---


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDiT(
        in_channels=8,
        num_frames=16,
        spatial_size=8,
        patch_size=2,
        hidden_dim=256,
        depth=6,
        num_heads=4,
        time_dim=256,
    ).to(device)

    # Simulated latent video from VideoVAE
    x = torch.randn(2, 8, 16, 8, 8).to(device)
    t = torch.rand(2, device=device)  # continuous time ∈ [0, 1]

    out = model(x, t)
    print(f"Input:   {tuple(x.shape)}")
    print(f"Output:  {tuple(out.shape)}")
    print(f"Tokens:  {model.num_tokens}  (spatial_pos={model.num_patches_spatial} × temporal={model.num_frames})")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")
