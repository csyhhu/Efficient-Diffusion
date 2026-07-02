"""
End-to-end video generation training on Moving MNIST using VAE + DiT.

Pipeline (following Wan2.1 architecture at a minimal scale):
  1. Train VideoVAE for video reconstruction.
  2. Freeze VAE, train VideoDiT on VAE latents with Flow Matching.
  3. Sample: DiT generates latent → VAE decoder → video.

Architecture reference:
    Wan Team, "Wan: Open and Advanced Large-Scale Video Generative Models"
    https://arxiv.org/abs/2503.20314

Usage:
    python scripts/mnist_moving_train.py

Outputs:
    outputs/moving_mnist/
        vae.pt                   -- trained VideoVAE checkpoint
        dit.pt                   -- trained VideoDiT checkpoint
        loss_vae.png             -- VAE training loss curve
        loss_dit.png             -- DiT training loss curve
        loss_vae_history.csv     -- per-batch VAE loss records
        loss_dit_history.csv     -- per-batch DiT loss records
        samples_epoch_*.png      -- sampled video grids
"""

import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.video_vae import VideoVAE
from src.models.video_dit import VideoDiT


# ============================================================================
# 1. Data loading — Moving MNIST
# ============================================================================

class MovingMNISTDataset(torch.utils.data.Dataset):
    """Online-generated Moving MNIST videos.

    Each video: two MNIST digits moving on a 64×64 black canvas for `num_frames`
    frames. Digits bounce off boundaries.

    Args:
        num_samples:   number of videos in the dataset (artificial, can be >N).
        num_frames:    frames per video.
        image_size:    spatial resolution (square).
        num_digits:    number of moving digits (1 or 2).
        step_length:   max pixels moved per frame.
    """

    _mnist = None  # class-level cache to avoid re-downloading

    def __init__(self, num_samples: int = 10000, num_frames: int = 16,
                 image_size: int = 64, num_digits: int = 2, step_length: float = 0.1):
        import torchvision.datasets as dsets

        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_digits = num_digits
        self.step_length = step_length

        # Load MNIST once
        if MovingMNISTDataset._mnist is None:
            MovingMNISTDataset._mnist = dsets.MNIST(
                root=os.path.join(os.path.dirname(__file__), "..", "data"),
                train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]),
            )

    def __len__(self):
        return self.num_samples

    def _get_random_digit(self):
        """Return a random MNIST digit tensor of shape (1, 28, 28)."""
        idx = torch.randint(0, len(self._mnist), (1,)).item()
        return self._mnist[idx][0]  # (1, 28, 28)

    def __getitem__(self, idx):
        frames = []
        canvas = torch.zeros(self.num_frames, self.image_size, self.image_size)

        for d in range(self.num_digits):
            digit = self._get_random_digit()  # (1, 28, 28)
            # Random initial position
            x = torch.randint(0, self.image_size - 28, (1,)).item()
            y = torch.randint(0, self.image_size - 28, (1,)).item()
            # Random velocity (in pixels per frame)
            vx = torch.randn(1).item() * self.step_length * self.image_size
            vy = torch.randn(1).item() * self.step_length * self.image_size

            for t in range(self.num_frames):
                x_int, y_int = int(round(x)), int(round(y))
                x_int = max(0, min(x_int, self.image_size - 28))
                y_int = max(0, min(y_int, self.image_size - 28))
                frames.append((t, y_int, x_int, digit.clone()))

                x += vx
                y += vy
                # Bounce
                if x <= 0 or x >= self.image_size - 28:
                    vx = -vx
                if y <= 0 or y >= self.image_size - 28:
                    vy = -vy

        for t, yy, xx, dgt in frames:
            canvas[t, yy:yy+28, xx:xx+28] = torch.clamp(
                canvas[t, yy:yy+28, xx:xx+28] + dgt.squeeze(0), -1, 1
            )

        return canvas.unsqueeze(1)  # (T, 1, 64, 64)


# ============================================================================
# 2. Flow Matching scheduler (same as mnist_train_fm.py)
# ============================================================================

class FlowMatchingScheduler:
    """Conditional Flow Matching with straight-line paths.

    x_t = (1 - t) * x_0 + t * x_1     (interpolation)
    u_t = x_1 - x_0                    (target velocity)
    """

    def __init__(self):
        pass

    def sample_noise(self, x_0: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x_0)

    def sample_time(self, batch_size: int, device: str) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        t_ = t
        while t_.ndim < x_0.ndim:
            t_ = t_.unsqueeze(-1)
        return (1 - t_) * x_0 + t_ * x_1

    def compute_target(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return x_1 - x_0


# ============================================================================
# 3. Sampling utilities
# ============================================================================

@torch.no_grad()
def sample_dit(
    dit: VideoDiT,
    vae: VideoVAE,
    scheduler: FlowMatchingScheduler,
    batch_size: int = 4,
    num_steps: int = 50,
    num_frames: int = 16,
    latent_channels: int = 8,
    latent_size: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate videos via Flow Matching ODE integration + VAE decoder.

    TODO: Implement the ODE integration loop for DiT latent sampling.

    Steps:
      1. Sample x_1 ~ N(0, I)  in latent space (B, C, T, H, W).
      2. Euler-integrate dx/dt = v_θ(x, t)  from t=1 to t=0.
      3. Decode latent → pixel space with VAE decoder.

    HINT:
        dt = 1.0 / num_steps
        x_t = torch.randn(B, C_lat, T, H_lat, W_lat, device=device)
        for i in range(num_steps):
            t_now = 1.0 - i * dt
            t_batch = torch.full((B,), t_now, device=device, dtype=torch.float32)
            v = dit(x_t, t_batch)
            x_t = x_t - v * dt
        return vae.decode_latents(x_t.clamp(-1, 1))
    """
    B = batch_size
    C = latent_channels
    T = num_frames
    H = W = latent_size

    # --- YOUR CODE BELOW ---
    dt = 1.0 / num_steps
    x_t = torch.randn(B, C, T, H, W, device=device)
    for i in range(num_steps):
        t_now = 1.0 - i * dt
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.float32)
        v = dit(x_t, t_batch)
        x_t = x_t - v * dt
    x_0 = x_t.clamp(-1, 1)
    video = vae.decode_latents(x_0)
    return video.clamp(-1, 1)
    # --- END YOUR CODE ---


def save_video_grid(videos: torch.Tensor, save_path: str, nrow: int = 2,
                    n_frames_show: int = 8):
    """Save a grid of generated video frames as a PNG.

    Args:
        videos: (B, C, T, H, W) in [-1, 1].
    """
    videos = (videos + 1) / 2  # [-1, 1] → [0, 1]
    videos = videos.clamp(0, 1)
    B, C, T, H, W = videos.shape
    T_show = min(T, n_frames_show)

    # Select evenly spaced frames
    indices = torch.linspace(0, T - 1, T_show).long()
    selected = videos[:, :, indices]  # (B, C, T_show, H, W)

    # Build grid: rows = samples, columns = time steps
    rows = []
    for b in range(min(B, nrow)):
        row_frames = [selected[b, :, t] for t in range(T_show)]
        rows.append(torch.cat(row_frames, dim=-1))  # (C, H, T_show*W)
    grid = torch.cat(rows, dim=-2)  # (C, nrow*H, T_show*W)

    plt.imsave(save_path, grid.squeeze().cpu().numpy(), cmap="gray")


def save_loss_curve(losses: list, save_path: str, title: str = "Training Loss"):
    """Plot and save the training loss curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# 4. Main training script
# ============================================================================

if __name__ == "__main__":

    # ---- Config ----
    config = {
        # Data
        "num_frames": 16,
        "image_size": 64,
        "num_digits": 2,
        "num_train_samples": 5000,
        "batch_size": 16,
        "num_workers": 2,

        # VAE
        "latent_dim": 8,
        "vae_base_channels": 32,
        "vae_epochs": 0,
        "vae_lr": 1e-3,
        "kl_weight": 1e-5,

        # DiT
        "dit_patch_size": 2,
        "dit_hidden_dim": 256,
        "dit_depth": 6,
        "dit_heads": 4,
        "dit_epochs": 1,
        "dit_lr": 2e-4,
        "time_dim": 256,

        # Sampling
        "num_sample_steps": 50,
        "sample_every": 1,

        # General
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    output_dir = "./outputs/moving_mnist"
    os.makedirs(output_dir, exist_ok=True)
    device = config["device"]
    print(f"Using device: {device}")
    print(f"Config:\n{config}")
    print(f"Output dir: {output_dir}")

    # ---- Data ----
    dataset = MovingMNISTDataset(
        num_samples=config["num_train_samples"],
        num_frames=config["num_frames"],
        image_size=config["image_size"],
        num_digits=config["num_digits"],
    )
    train_loader = DataLoader(
        dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], drop_last=True,
    )
    print(f"Dataset: {len(dataset)} videos of shape ({config['num_frames']}, 1, "
          f"{config['image_size']}, {config['image_size']})")

    # ======================================================================
    # Stage 1: Train VideoVAE
    # ======================================================================

    print("\n" + "=" * 60)
    print("Stage 1: Training VideoVAE")
    print("=" * 60)

    vae = VideoVAE(
        in_channels=1,
        latent_dim=config["latent_dim"],
        base_channels=config["vae_base_channels"],
    ).to(device)

    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config["vae_lr"])
    vae_loss_records = []
    vae_losses = []
    vae_global_step = 0

    for epoch in range(config["vae_epochs"]):
        vae.train()
        pbar = tqdm(train_loader, desc=f"VAE Epoch {epoch}/{config['vae_epochs']}", leave=False)
        epoch_loss = 0.0

        for x, in pbar:
            x = x.to(device)  # (B, 1, T, H, W) — 5D tensor for 3D VAE

            # ---------------------------------------------------------------
            # TODO: VAE training step.
            #
            # 1. Forward: recon, z, mu, logvar = vae(x)
            # 2. Reconstruction loss: L_recon = F.l1_loss(recon, x)
            # 3. KL loss: L_kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch
            # 4. Total loss = L_recon + kl_weight * L_kl
            # 5. Backward + optimizer step
            # ---------------------------------------------------------------
            # HINT:
            #   recon, z, mu, logvar = vae(x)
            #   recon_loss = F.l1_loss(recon, x)
            #   kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            #   loss = recon_loss + config["kl_weight"] * kl_loss
            # ---------------------------------------------------------------

            # --- YOUR CODE BELOW ---
            recon, z, mu, logvar = vae(x)
            recon_loss = F.l1_loss(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + config["kl_weight"] * kl_loss

            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            # --- END YOUR CODE ---

            loss_val = loss.item()
            vae_losses.append(loss_val)
            epoch_loss += loss_val
            vae_global_step += 1
            vae_loss_records.append((vae_global_step, epoch, loss_val))

            pbar.set_postfix(loss=f"{loss_val:.6f}",
                             recon=f"{recon_loss.item():.6f}",
                             kl=f"{kl_loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"VAE Epoch {epoch:3d} | avg loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save(vae.state_dict(), os.path.join(output_dir, "vae.pt"))

    # Save VAE loss records
    csv_path = os.path.join(output_dir, "loss_vae_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss"])
        writer.writerows(vae_loss_records)
    save_loss_curve(vae_losses, os.path.join(output_dir, "loss_vae.png"),  title="VideoVAE Training Loss")

    # Freeze VAE for DiT training
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print("VideoVAE frozen. Computing latent statistics...")

    # Pre-compute all latents once (optional, saves time during DiT training)
    # For simplicity, we encode on-the-fly per batch

    # ======================================================================
    # Stage 2: Train VideoDiT with Flow Matching
    # ======================================================================

    print("\n" + "=" * 60)
    print("Stage 2: Training VideoDiT (Flow Matching in latent space)")
    print("=" * 60)

    latent_size = config["image_size"] // 8  # 64 → 8
    dit = VideoDiT(
        in_channels=config["latent_dim"],
        num_frames=config["num_frames"],
        spatial_size=latent_size,
        patch_size=config["dit_patch_size"],
        hidden_dim=config["dit_hidden_dim"],
        depth=config["dit_depth"],
        num_heads=config["dit_heads"],
        time_dim=config["time_dim"],
    ).to(device)

    fm_scheduler = FlowMatchingScheduler()
    dit_optimizer = torch.optim.Adam(dit.parameters(), lr=config["dit_lr"])
    dit_loss_fn = nn.MSELoss()

    dit_loss_records = []
    dit_losses = []
    dit_global_step = 0
    t0_total = time.time()

    for epoch in range(config["dit_epochs"]):

        # ---- Sampling ----
        if epoch % config["sample_every"] == 0:
            save_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
            # TODO: call sample_dit here
            print(f"  -> Sampling at epoch {epoch} (save to {save_path})")
            # Uncomment once sample_dit is implemented:
            videos = sample_dit(dit, vae, fm_scheduler, batch_size=4,
                                num_steps=config["num_sample_steps"],
                                num_frames=config["num_frames"],
                                latent_channels=config["latent_dim"],
                                latent_size=latent_size, device=device)
            save_video_grid(videos, save_path)
            break

        dit.train()
        pbar = tqdm(train_loader, desc=f"DiT Epoch {epoch}/{config['dit_epochs']}", leave=False)
        epoch_loss = 0.0

        for x, in pbar:
            x = x.to(device)  # (B, 1, T, H, W)

            # Encode video to latent
            with torch.no_grad():
                z_0 = vae.encode_latents(x)  # (B, C_lat, T, H_lat, W_lat)

            # ---------------------------------------------------------------
            # TODO: Flow Matching training step in latent space.
            #
            # 1. Sample noise z_1 ~ N(0, I), same shape as z_0.
            # 2. Sample time t ~ Uniform(0, 1), shape (B,).
            # 3. Interpolate: z_t = (1 - t) * z_0 + t * z_1.
            # 4. Target velocity: u_target = z_1 - z_0.
            # 5. Predict: u_pred = dit(z_t, t).
            # 6. loss = MSELoss(u_pred, u_target).
            # 7. Backward + optimizer step.
            # ---------------------------------------------------------------
            # HINT:
            #   z_1 = fm_scheduler.sample_noise(z_0)
            #   t   = fm_scheduler.sample_time(z_0.shape[0], device)
            #   z_t = fm_scheduler.interpolate(z_0, z_1, t)
            #   u_target = fm_scheduler.compute_target(z_0, z_1)
            #   u_pred   = dit(z_t, t)
            #   loss     = dit_loss_fn(u_pred, u_target)
            # ---------------------------------------------------------------

            # --- YOUR CODE BELOW ---
            z_1 = fm_scheduler.sample_noise(z_0)
            t = fm_scheduler.sample_time(z_0.shape[0], device)
            z_t = fm_scheduler.interpolate(z_0, z_1, t)
            u_target = fm_scheduler.compute_target(z_0, z_1)
            u_pred = dit(z_t, t)
            loss = dit_loss_fn(u_pred, u_target)

            dit_optimizer.zero_grad()
            loss.backward()
            dit_optimizer.step()
            # --- END YOUR CODE ---

            loss_val = loss.item()
            dit_losses.append(loss_val)
            epoch_loss += loss_val
            dit_global_step += 1
            dit_loss_records.append((dit_global_step, epoch, loss_val))

            pbar.set_postfix(loss=f"{loss_val:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        dt_total = time.time() - t0_total
        print(f"DiT Epoch {epoch:3d} | avg loss: {avg_loss:.6f} | elapsed: {dt_total:.0f}s")

        # Save DiT checkpoint
        torch.save(dit.state_dict(), os.path.join(output_dir, "dit.pt"))

        # Save loss records after each epoch
        csv_path = os.path.join(output_dir, "loss_dit_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss"])
            writer.writerows(dit_loss_records)

    # ---- Post-training ----
    dt_total = time.time() - t0_total
    save_loss_curve(dit_losses, os.path.join(output_dir, "loss_dit.png"),
                    title="VideoDiT Flow Matching Loss")
    print(f"\nDone! Total time: {dt_total:.1f}s ({dt_total / 60:.1f} min)")
    print(f"Model & outputs saved to: {output_dir}")
