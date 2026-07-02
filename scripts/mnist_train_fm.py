"""
End-to-end Flow Matching training & sampling on MNIST.

Flow Matching learns a velocity field v_θ(x_t, t) that transports samples
from a simple base distribution (Gaussian noise) to the data distribution (MNIST
digits).  Compared to DDPM, Flow Matching:

  - Uses continuous time t ∈ [0, 1] instead of discrete timesteps.
  - Targets the straight-line velocity x_1 − x_0 instead of Gaussian noise.
  - Samples by solving an ODE (dx/dt = v_θ) with an Euler / RK45 solver.

Usage:
    python scripts/mnist_train_fm.py                # default: unet
    # or edit model_type = "dit" on line ~180

Outputs:
    outputs/mnist_fm_unet/    (or mnist_fm_dit/)
        loss_curve.png          -- training loss curve
        loss_history.csv        -- per-batch loss records
        samples_epoch_*.png     -- sampled images at various epochs
        fm_mnist_<type>.pt      -- model checkpoint
"""

import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import get_mnist_dataloader
from src.models.unet import SimpleUNet
from src.models.dit import DiT


# ============================================================================
# 1. Flow Matching scheduler
# ============================================================================

class FlowMatchingScheduler:
    """Scheduler for conditional Flow Matching with straight-line paths.

    Given a data sample x_0 and a noise sample x_1 ~ N(0, I), the
    conditional probability path is:

        x_t = (1 - t) * x_0 + t * x_1          (interpolation)
        u_t = x_1 - x_0                         (target velocity)

    The model v_θ(x_t, t) is trained to regress the velocity field u_t.
    """

    def __init__(self):
        pass

    def sample_noise(self, x_0: torch.Tensor) -> torch.Tensor:
        """Sample noise x_1 ~ N(0, I) with the same shape as x_0.

        Args:
            x_0: clean image, shape (B, C, H, W)

        Returns:
            x_1: Gaussian noise, same shape as x_0
        """
        return torch.randn_like(x_0)

    def sample_time(self, batch_size: int, device: str) -> torch.Tensor:
        """Sample continuous time t ~ Uniform(0, 1) for each sample.

        Args:
            batch_size: number of samples (B)
            device: torch device

        Returns:
            t: shape (B,), dtype float32, values in [0, 1]
        """
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute interpolation x_t = (1 - t) * x_0 + t * x_1.

        Args:
            x_0: clean image, shape (B, C, H, W)
            x_1: noise,     shape (B, C, H, W)
            t:   time,      shape (B,) in [0, 1]

        Returns:
            x_t: interpolated sample, shape (B, C, H, W)
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_ = t.view(-1, 1, 1, 1)
        return (1 - t_) * x_0 + t_ * x_1

    def compute_target(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Compute the conditional velocity field target u_t = x_1 - x_0.

        Args:
            x_0: clean image, shape (B, C, H, W)
            x_1: noise,      shape (B, C, H, W)

        Returns:
            u_t: target velocity, shape (B, C, H, W)
        """
        return x_1 - x_0


# ============================================================================
# 2. Sampling (ODE integration)
# ============================================================================

def sample_flow(
    model: nn.Module,
    num_steps: int = 100,
    shape: tuple = (16, 1, 28, 28),
    device: str = "cpu",
) -> torch.Tensor:
    """Generate images by integrating the learned velocity field ODE.

    Flow Matching sampling:
      - Start from x_1 ~ N(0, I)  (t = 1)
      - Discretize [0, 1] into `num_steps` equal intervals.
      - For each step i from 0 to num_steps-1:
          t = 1 - i * dt                          # current time
          v = model(x_t, t)                       # predicted velocity
          x_t = x_t - v * dt                      # Euler step (backward in time)
      - Return x_0 (t = 0) clamped to [-1, 1].

    TODO: Implement the ODE integration loop below.

    HINT:
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_now = 1.0 - i * dt
            t_tensor = torch.full((B,), t_now, device=device, dtype=torch.float32)
            v = model(x_t, t_tensor)
            x_t = x_t - v * dt
        return x_t.clamp(-1, 1)
    """
    model.eval()
    B = shape[0]

    # Start from pure noise at t = 1
    x_t = torch.randn(shape, device=device)

    with torch.no_grad():
        # ---------------------------------------------------------------
        # TODO: Implement ODE integration with Euler method.
        #
        # dt = 1.0 / num_steps
        # for each step i = 0, ..., num_steps-1:
        #     t_now = 1.0 - i * dt       # move from t=1 → t=0
        #     t_batch = tensor of shape (B,) filled with t_now
        #     velocity = model(x_t, t_batch)
        #     x_t = x_t - velocity * dt  # Euler step
        #
        # Note: the model takes float t ∈ [0, 1], not integer timesteps.
        # ---------------------------------------------------------------
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_now = 1.0 - i * dt
            t_tensor = torch.full((B,), t_now, device=device, dtype=torch.float32)
            v = model(x_t, t_tensor)
            x_t = x_t - v * dt

    return x_t.clamp(-1, 1)


# ============================================================================
# 3. Utility functions (shared with DDPM script)
# ============================================================================

def save_sample_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """Save a grid of generated images as a PNG."""
    images = (images + 1) / 2  # denormalize [-1, 1] -> [0, 1]
    images = images.clamp(0, 1)
    grid = torch.cat(
        [torch.cat([img for img in row], dim=-1) for row in images.split(nrow)],
        dim=-2,
    )
    plt.imsave(save_path, grid.squeeze().cpu().numpy(), cmap="gray")


def save_loss_curve(losses: list, save_path: str):
    """Plot and save the training loss curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Flow Matching Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# 4. Main training script
# ============================================================================

if __name__ == "__main__":

    # ---- Config ----
    # model_type: "unet" | "dit"
    # model_type = "unet"
    model_type = "dit"

    config = {
        "batch_size": 128,
        "lr": 2e-4,
        "epochs": 5,
        "num_sample_steps": 100,       # ODE discretization steps for sampling
        "time_dim": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sample_every": 1,             # sample every N epochs
    }

    # ---- Model-specific config ----
    if model_type == "unet":
        model_config = {"base_channels": 64}
    else:  # dit
        model_config = {
            "patch_size": 4,
            "hidden_dim": 256,
            "depth": 6,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        }

    # Derive output directory from model type
    output_dir = f"./outputs/mnist_fm_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    device = config["device"]
    print(f"Using device: {device}")
    print(f"Model: {model_type} | Config: {config} | Model config: {model_config}")
    print(f"Output dir: {output_dir}")

    # ---- Data ----
    train_loader = get_mnist_dataloader(batch_size=config["batch_size"], train=True)

    # ---- Model & Scheduler ----
    if model_type == "unet":
        model = SimpleUNet(
            in_channels=1,
            base_channels=model_config["base_channels"],
            time_dim=config["time_dim"],
        ).to(device)
    else:  # dit
        model = DiT(
            in_channels=1,
            image_size=28,
            patch_size=model_config["patch_size"],
            hidden_dim=model_config["hidden_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            time_dim=config["time_dim"],
            mlp_ratio=model_config["mlp_ratio"],
        ).to(device)

    scheduler = FlowMatchingScheduler()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    # ---- Training loop ----
    loss_records = []  # (step, epoch, train_loss) per batch
    losses = []
    global_step = 0
    t0_total = time.time()

    for epoch in range(config["epochs"]):

        # ---- Sampling ----
        if epoch % config["sample_every"] == 0:
            save_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
            # TODO: call sample_flow and save_sample_grid here (after implementing them)
            print(f"  -> Sampling at epoch {epoch} (save to {save_path})")
            # Uncomment once sample_flow is implemented:
            save_sample_grid(
                sample_flow(model, config["num_sample_steps"], device=device),
                save_path,
            )

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}", leave=False)
        epoch_loss = 0.0
        t0_epoch = time.time()
        batch_times = []  # rolling window for batch speed estimation

        for x, _ in pbar:
            x = x.to(device)  # (B, 1, 28, 28)
            t0_batch = time.time()

            # ---------------------------------------------------------------
            # TODO: Flow Matching training step — fill in the code below.
            #
            # Flow Matching with conditional optimal transport:
            #
            #   1. Sample noise  x_1 ~ N(0, I)  , same shape as x (clean image).
            #
            #   2. Sample time   t ~ Uniform(0, 1), shape (B,).
            #
            #   3. Interpolate: x_t = (1 - t) * x + t * x_1
            #      (use scheduler.interpolate)
            #
            #   4. Compute target velocity: u_target = x_1 - x
            #      (use scheduler.compute_target, or simply x_1 - x)
            #
            #   5. Predict velocity: u_pred = model(x_t, t)
            #      NOTE: t is float in [0, 1]; the model's time embedding
            #            handles both int and float timesteps.
            #
            #   6. loss = MSELoss(u_pred, u_target)
            #
            #   7. optimizer.zero_grad() → loss.backward() → optimizer.step()
            #
            #   8. Track loss_val for logging.
            # ---------------------------------------------------------------
            # HINT helpers:
            #   x_1 = scheduler.sample_noise(x)
            #   t   = scheduler.sample_time(x.shape[0], device)
            #   x_t = scheduler.interpolate(x, x_1, t)
            #   u_target = scheduler.compute_target(x, x_1)
            #   u_pred   = model(x_t, t)
            #   loss     = loss_fn(u_pred, u_target)
            # ---------------------------------------------------------------

            x_1 = scheduler.sample_noise(x)
            t = scheduler.sample_time(x.shape[0], device)
            x_t = scheduler.interpolate(x, x_1, t)
            u_target = scheduler.compute_target(x, x_1)
            u_pred = model(x_t, t)
            loss = loss_fn(u_pred, u_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Timing
            dt_batch = time.time() - t0_batch
            batch_times.append(dt_batch)
            if len(batch_times) > 50:
                batch_times.pop(0)
            avg_batch_time = sum(batch_times) / len(batch_times)

            # Logging
            loss_val = loss.item() if loss is not None else 0.0
            losses.append(loss_val)
            epoch_loss += loss_val
            global_step += 1
            loss_records.append((global_step, epoch, loss_val))
            batches_done = global_step - epoch * len(train_loader)

            # Update progress bar every batch
            pbar.set_postfix(
                loss=f"{loss_val:.6f}",
                avg=f"{epoch_loss / batches_done:.6f}",
                ms=f"{avg_batch_time * 1000:.0f}",
            )

        # ---- End of epoch ----
        dt_epoch = time.time() - t0_epoch
        avg_loss = epoch_loss / len(train_loader)
        dt_total = time.time() - t0_total
        print(
            f"Epoch {epoch:3d}/{config['epochs']}"
            f" | avg loss: {avg_loss:.6f}"
            f" | epoch time: {dt_epoch:.1f}s"
            f" | total elapsed: {dt_total:.1f}s"
        )

        # Save loss records after each epoch
        csv_path = os.path.join(output_dir, "loss_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss"])
            writer.writerows(loss_records)

    # ---- Post-training ----
    dt_total = time.time() - t0_total

    save_loss_curve(losses, os.path.join(output_dir, "loss_curve.png"))
    torch.save(model.state_dict(), os.path.join(output_dir, f"fm_mnist_{model_type}.pt"))
    print(f"\nDone! Total time: {dt_total:.1f}s ({dt_total / 60:.1f} min)")
    print(f"Model & outputs saved to: {output_dir}")
