"""
Diffusion Transformer (DiT) for image generation.

This implementation supports both unconditional and text-conditional generation:
- Unconditional: Only timestep conditioning (self-attention only)
- Conditional: Timestep + text conditioning (self-attention + cross-attention)

Key design choices:
- Patchify input image into tokens
- adaLN-Zero conditioning on timestep and text
- Optional cross-attention blocks for text conditioning
- ~1.5M parameters for base MNIST model

Usage::

    # Unconditional (MNIST)
    model = DiT(in_channels=1, image_size=28, patch_size=4)
    output = model(x, t)

    # Text-conditional (CIFAR-100 with VAE)
    model = DiT(
        in_channels=4, image_size=8, patch_size=2,
        use_cross_attention=True, cross_attention_dim=768,
    )
    output = model(x, t, encoder_hidden_states=text_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding followed by a small MLP."""

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

class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with zero-initialized residual scaling.

    For an input of shape (B, N, C):
        x = norm(x) * (1 + scale) + shift
        x = x + gate * block(x)

    The modulation parameters (shift, scale, gate) are produced by a shared
    SiLU -> Linear head from the combined time and text embedding.
    """

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 3),
        )
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, N, C) input tensor
            c_emb: (B, time_dim) conditioning embedding (time + text)

        Returns:
            (modulated_x, gate) - gate is applied later after the transformer block.
        """
        shift, scale, gate = self.head(c_emb).chunk(3, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift, gate


# ---------------------------------------------------------------------------
# Self-Attention Block
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """Transformer block with self-attention and adaLN-Zero conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, time_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        self.adaln1 = AdaLNZero(time_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.adaln2 = AdaLNZero(time_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        modulated, gate = self.adaln1(x, c_emb)
        attn_out, _ = self.attn(modulated, modulated, modulated)
        x = x + gate * attn_out

        modulated, gate = self.adaln2(x, c_emb)
        x = x + gate * self.mlp(modulated)

        return x


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and adaLN-Zero."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        time_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        self.adaln1 = AdaLNZero(time_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.adaln_cross = AdaLNZero(time_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        if cross_attention_dim != hidden_dim:
            self.k_proj = nn.Linear(cross_attention_dim, hidden_dim)
            self.v_proj = nn.Linear(cross_attention_dim, hidden_dim)
        else:
            self.k_proj = None
            self.v_proj = None

        self.adaln2 = AdaLNZero(time_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        c_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        modulated, gate = self.adaln1(x, c_emb)
        attn_out, _ = self.self_attn(modulated, modulated, modulated)
        x = x + gate * attn_out

        modulated, gate = self.adaln_cross(x, c_emb)

        if self.k_proj is not None:
            k = self.k_proj(encoder_hidden_states)
            v = self.v_proj(encoder_hidden_states)
        else:
            k = encoder_hidden_states
            v = encoder_hidden_states
        cross_out, _ = self.cross_attn(modulated, k, v)
        x = x + gate * cross_out

        modulated, gate = self.adaln2(x, c_emb)
        x = x + gate * self.mlp(modulated)

        return x


# ---------------------------------------------------------------------------
# Text Embedding Processor
# ---------------------------------------------------------------------------

class TextEmbeddingProcessor(nn.Module):
    """Process text embeddings for conditioning the DiT model.

    Projects text embeddings to match the model's time dimension
    and provides both pooled and sequence-level conditioning.
    """

    def __init__(self, text_dim: int, time_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, time_dim)
        self.norm = nn.LayerNorm(time_dim)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: (B, seq_len, text_dim) text embeddings

        Returns:
            (B, time_dim) pooled and projected text embedding
        """
        pooled = encoder_hidden_states.mean(dim=1)
        pooled = self.proj(pooled)
        pooled = self.norm(pooled)
        return pooled


# ---------------------------------------------------------------------------
# DiT model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer for image generation.

    Supports both unconditional and text-conditional generation.

    Args:
        in_channels: Number of input image channels (1 for MNIST, 4 for VAE latents).
        image_size: Spatial size of the input image.
        patch_size: Size of each square patch.
        hidden_dim: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        time_dim: Dimension of the sinusoidal time embedding.
        mlp_ratio: MLP hidden / transformer hidden ratio.
        use_cross_attention: Whether to use cross-attention for text conditioning.
        cross_attention_dim: Dimension of the text encoder hidden states.
        num_classes: Number of classes for class-conditional generation (optional).
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
        use_cross_attention: bool = False,
        cross_attention_dim: int = 768,
        num_classes: int = 0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.use_cross_attention = use_cross_attention
        self.cross_attention_dim = cross_attention_dim

        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        if use_cross_attention:
            self.text_proj = TextEmbeddingProcessor(cross_attention_dim, time_dim)

        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, time_dim)

        if use_cross_attention:
            self.blocks = nn.ModuleList([
                CrossAttentionBlock(
                    hidden_dim, num_heads, time_dim, cross_attention_dim, mlp_ratio
                )
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                SelfAttentionBlock(hidden_dim, num_heads, time_dim, mlp_ratio)
                for _ in range(depth)
            ])

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

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches: (B, C, H, W) -> (B, N, patch_dim)."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert patches back to image: (B, N, patch_dim) -> (B, C, H, W)."""
        p = self.patch_size
        h, w = H // p, W // p
        C = x.shape[-1] // (p * p)
        x = x.reshape(x.shape[0], h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(x.shape[0], C, H, W)
        return x

    def _compute_conditioning_embedding(
        self,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        class_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute combined conditioning embedding from time and optional text/class inputs."""
        c_emb = self.time_embed(t)

        if encoder_hidden_states is not None and self.use_cross_attention:
            text_emb = self.text_proj(encoder_hidden_states)
            c_emb = c_emb + text_emb

        if class_labels is not None and hasattr(self, 'class_embed'):
            class_emb = self.class_embed(class_labels)
            c_emb = c_emb + class_emb

        return c_emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        class_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image or latent.
            t: (B,) integer timesteps or float time values.
            encoder_hidden_states: (B, seq_len, dim) text embeddings (optional).
            class_labels: (B,) class indices (optional, for class-conditional generation).

        Returns:
            (B, C, H, W) predicted noise or velocity.
        """
        B, C, H, W = x.shape

        tokens = self.patchify(x)
        tokens = self.patch_embed(tokens)
        tokens = tokens + self.pos_embed

        c_emb = self._compute_conditioning_embedding(t, encoder_hidden_states, class_labels)

        if self.use_cross_attention and encoder_hidden_states is None:
            raise ValueError(
                "use_cross_attention=True requires encoder_hidden_states to be provided."
            )

        for block in self.blocks:
            if self.use_cross_attention:
                tokens = block(tokens, c_emb, encoder_hidden_states)
            else:
                tokens = block(tokens, c_emb)

        tokens = self.final_norm(tokens)
        patches = self.proj(tokens)
        out = self.unpatchify(patches, H, W)

        return out


    def decode(self, x: torch.Tensor, latent, text_emb) -> torch.Tensor:
        """Decode the latent space to the image."""
        pass

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # """
    print("=" * 50)
    print("Test 1: Unconditional DiT (MNIST)")
    print("=" * 50)

    model = DiT(in_channels=1, image_size=28, patch_size=4)
    batch_size = 10
    inputs = torch.rand(batch_size, 1, 28, 28)
    timestamp = torch.randint(0, 1000, (batch_size,))
    outputs = model(inputs, timestamp)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # """
    print("\n" + "=" * 50)
    print("Test 2: Text-conditional DiT (CIFAR-100 latent space)")
    print("=" * 50)

    model = DiT(
        in_channels=4, image_size=8, patch_size=2,
        hidden_dim=256, depth=6, num_heads=4,
        use_cross_attention=True, cross_attention_dim=768,
    )
    inputs = torch.rand(4, 4, 8, 8)
    timestamp = torch.rand(4)
    encoder_hidden_states = torch.rand(4, 77, 768)
    outputs = model(inputs, timestamp, encoder_hidden_states=encoder_hidden_states)
    print(f"Input shape: {inputs.shape}")
    print(f"Text embedding shape: {encoder_hidden_states.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 50)
    print("Test 3: Class-conditional DiT")
    print("=" * 50)

    model = DiT(
        in_channels=3, image_size=32, patch_size=4,
        num_classes=100,
    )
    inputs = torch.rand(4, 3, 32, 32)
    timestamp = torch.randint(0, 1000, (4,))
    class_labels = torch.randint(0, 100, (4,))
    outputs = model(inputs, timestamp, class_labels=class_labels)
    print(f"Input shape: {inputs.shape}")
    print(f"Class labels shape: {class_labels.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 50)
    print("Test 4: Backward pass")
    print("=" * 50)

    model = DiT(
        in_channels=4, image_size=8, patch_size=2,
        use_cross_attention=True, cross_attention_dim=768,
    )
    model.train()
    inputs = torch.rand(2, 4, 8, 8)
    timestamp = torch.rand(2)
    encoder_hidden_states = torch.rand(2, 77, 768)
    outputs = model(inputs, timestamp, encoder_hidden_states=encoder_hidden_states)
    loss = outputs.sum()
    loss.backward()
    print(f"Backward pass successful!")
    print(f"Gradients computed for {sum(1 for p in model.parameters() if p.grad is not None)} parameters")

    print("\n" + "=" * 50)
    print("Test 5: CUDA compatibility")
    print("=" * 50)

    if torch.cuda.is_available():
        model = DiT(
            in_channels=4, image_size=8, patch_size=2,
            use_cross_attention=True, cross_attention_dim=768,
        ).cuda()
        inputs = torch.rand(2, 4, 8, 8).cuda()
        timestamp = torch.rand(2).cuda()
        encoder_hidden_states = torch.rand(2, 77, 768).cuda()
        outputs = model(inputs, timestamp, encoder_hidden_states=encoder_hidden_states)
        print(f"CUDA forward pass successful!")
        print(f"Output shape on CUDA: {outputs.shape}")
        print(f"Output device: {outputs.device}")
    else:
        print("CUDA not available, skipping CUDA test")

    print("\n" + "=" * 50)
    print("All DiT tests passed!")
    print("=" * 50)