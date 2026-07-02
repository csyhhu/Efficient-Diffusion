"""
End-to-end DDPM training & sampling on MNIST.

Usage:
    python scripts/mnist_train_ddpm.py                # default: unet
    # or edit model_type = "dit" on line ~168

Outputs:
    outputs/mnist_ddpm_unet/    (or mnist_ddpm_dit/)
        loss_curve.png          -- training loss curve
        samples_epoch_*.png     -- sampled images at various epochs
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
from src.models.quantized_dit import QuantizedDiT


# ============================================================================
# 1. DDPM noise schedule
# ============================================================================

class DDPMScheduler:
    """Linear beta schedule DDPM noise scheduler.

    Precomputes all coefficients needed for the forward (noising) and
    reverse (denoising) processes.
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t

        # Precompute constants for forward / reverse processes
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", (1.0 - alpha_cumprod).sqrt())

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Store a tensor as a non-parameter buffer on CPU."""
        setattr(self, name, tensor)

    # ------------------------------------------------------------------
    # TODO: Implement the DDPM forward diffusion step (adding noise).
    #
    # Given a clean image x_0 and a timestep t, the forward process is:
    #   x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise
    #
    # Hint:
    #   - Use self.sqrt_alpha_cumprod[t] and self.sqrt_one_minus_alpha_cumprod[t]
    #   - noise ~ N(0, I), same shape as x_0
    #   - Return (x_t, noise) so that noise can be used as the training target
    # ------------------------------------------------------------------
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.

        Args:
            x_0:   clean image, shape (B, C, H, W)
            t:     timestep index per sample, shape (B,)
            noise: optional pre-sampled noise; generated if None.

        Returns:
            (x_t, noise)  —  noisy image and the noise used (for training target)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Index schedule tensors by t and reshape for broadcasting
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise


# ============================================================================
# 2. Training utilities
# ============================================================================

def sample_ddpm(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple = (16, 1, 28, 28),
    device: str = "cpu",
) -> torch.Tensor:
    """DDPM reverse process: sample from pure noise by iterative denoising.

    TODO (optional, for later):
      - Start from x_T ~ N(0, I)
      - For t = T-1, ..., 0:
          - Compute predicted noise eps = model(x_t, t)
          - Compute x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
            + sqrt(beta_t) * z  (where z ~ N(0, I) if t > 0 else 0)
      - Return x_0
    """
    model.eval()
    T = scheduler.num_timesteps
    B = shape[0]

    # Start from pure Gaussian noise x_T ~ N(0, I)
    x_t = torch.randn(shape, device=device)

    with torch.no_grad():
        # Reverse iteration: t = T-1, T-2, ..., 0
        for t in reversed(range(T)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise: eps_theta(x_t, t)
            eps_theta = model(x_t, t_tensor)

            # DDPM posterior: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps) + sigma_t * z
            alpha_t = scheduler.alphas[t]
            alpha_bar_t = scheduler.alpha_cumprod[t]
            beta_t = scheduler.betas[t]

            coef = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
            mean = (1.0 / alpha_t.sqrt()) * (x_t - coef * eps_theta)

            if t > 0:
                z = torch.randn(shape, device=device)
                mean = mean + beta_t.sqrt() * z

            x_t = mean

    return x_t.clamp(-1, 1)


def save_sample_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """Save a grid of generated images as a PNG."""
    images = (images + 1) / 2  # denormalize [-1, 1] → [0, 1]
    images = images.clamp(0, 1)
    grid = torch.cat([torch.cat([img for img in row], dim=-1) for row in images.split(nrow)], dim=-2)
    plt.imsave(save_path, grid.squeeze().cpu().numpy(), cmap="gray")


def save_loss_curve(losses: list, save_path: str):
    """Plot and save the training loss curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    
    # ---- Config ----
    # model_type: "unet" | "dit" | "quantized_dit"
    # model_type = "unet"
    model_type = "quantized_dit"

    config = {
        "batch_size": 128,
        "lr": 2e-4,
        "epochs": 1,
        "num_timesteps": 1000,
        "time_dim": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sample_every": 2,   # sample every N epochs
    }

    # ---- Model-specific config ----
    if model_type == "unet":
        model_config = {"base_channels": 64}
    elif model_type == "quantized_dit":
        model_config = {"patch_size": 4, "hidden_dim": 256, "depth": 6, "num_heads": 4, "mlp_ratio": 4.0, "bitW": 8, "bitA": 8, "bitG": 32}
    else:  # dit
        model_config = {"patch_size": 4, "hidden_dim": 256, "depth": 6, "num_heads": 4, "mlp_ratio": 4.0}

    # Derive output directory from model type
    output_dir = f"./results/mnist_ddpm_{model_type}"
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
    elif model_type == "quantized_dit":
        model = QuantizedDiT(
            in_channels=1,
            image_size=28,
            patch_size=model_config["patch_size"],
            hidden_dim=model_config["hidden_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            time_dim=config["time_dim"],
            mlp_ratio=model_config["mlp_ratio"],
            bitW=model_config["bitW"],
            bitA=model_config["bitA"],
            bitG=model_config["bitG"],
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

    scheduler = DDPMScheduler(num_timesteps=config["num_timesteps"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    # ---- Training loop ----
    loss_records = []  # (step, epoch, train_loss) per batch
    losses = []
    global_step = 0
    t0_total = time.time()

    # Per-layer quantization error tracking (only for quantized models)
    quant_error_records = {}  # {layer_key: [(step, error_sum), ...]}
    quant_error_dir = os.path.join(output_dir, "quant_error")
    if model_type == "quantized_dit":
        os.makedirs(quant_error_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        
        # ---- Sampling ----
        if epoch % config["sample_every"] == 0:
            save_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
            # TODO: call sample_ddpm and save_sample_grid here (after implementing them)
            save_sample_grid(sample_ddpm(model, scheduler, device=device), save_path)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}", leave=False)
        epoch_loss = 0.0
        t0_epoch = time.time()
        batch_times = []  # rolling window for batch speed estimation

        for idx, (x, _) in enumerate(pbar):

            if idx > 100:
                break
            
            x = x.to(device)                           # (B, 1, 28, 28)
            t0_batch = time.time()

            # ---------------------------------------------------------------
            # TODO: DDPM training step — fill in the missing lines below.
            #
            # 1. Sample random timesteps t ~ Uniform(0, T-1) for each image.
            #    shape: (B,), dtype: torch.long
            #
            # 2. Sample Gaussian noise eps ~ N(0, I), same shape as x.
            #
            # 3. Compute x_t = scheduler.add_noise(x, t, eps).
            #    This applies the forward diffusion: x_t = sqrt(a_bar_t)*x_0 + sqrt(1-a_bar_t)*eps
            #
            # 4. Predict noise: eps_pred = model(x_t, t)
            #
            # 5. Compute loss = MSELoss(eps_pred, eps)
            #
            # 6. Backward + optimizer step + zero_grad
            #
            # 7. Track loss value for logging
            # ---------------------------------------------------------------
            # HINT:
            #   t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
            #   x_t, eps = scheduler.add_noise(x, t)
            #   eps_pred = model(x_t, t)
            #   loss = loss_fn(eps_pred, eps)

            # --- YOUR CODE BELOW ---

            # step 1 — sample timesteps
            t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)

            # step 2 — forward diffusion (returns x_t and the noise used as target)
            x_t, eps = scheduler.add_noise(x, t)

            # step 3 — model predicts noise
            eps_pred = model(x_t, t)

            # step 4 — loss
            loss = loss_fn(eps_pred, eps)

            # step 5 — backward + optimizer
            optimizer.zero_grad()
            loss.backward()

            # Record per-layer quantization error (quantized models only)
            if hasattr(model, 'quantization_error_info') and model.quantization_error_info:
                layer_totals = {}  # {layer_key: sum_of_in_domain+left+right}
                for key, val in model.quantization_error_info.items():
                    if key.endswith('.in_domain_error'):
                        layer = key[:-len('.in_domain_error')]
                    elif key.endswith('.left_domain_error'):
                        layer = key[:-len('.left_domain_error')]
                    elif key.endswith('.right_domain_error'):
                        layer = key[:-len('.right_domain_error')]
                    else:
                        continue
                    layer_totals[layer] = layer_totals.get(layer, 0.0) + val.detach().sum().item()
                for layer, total in layer_totals.items():
                    quant_error_records.setdefault(layer, []).append((global_step + 1, total))
                model.quantization_error_info.clear()

            optimizer.step()

            # --- END YOUR CODE ---

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
            batches_done = global_step - (epoch - 1) * len(train_loader)  # 当前 epoch 已处理 batch 数

            # Update progress bar every batch: show current loss, epoch avg, and batch time
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

        # Save per-layer quantization error CSVs after each epoch
        if quant_error_records:
            for layer, records in quant_error_records.items():
                safe_name = layer.replace('.', '_')
                layer_csv = os.path.join(quant_error_dir, f"{safe_name}.csv")
                with open(layer_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "quant_error_sum"])
                    writer.writerows(records)


    # ---- Post-training ----
    dt_total = time.time() - t0_total

    # save_loss_curve(losses, os.path.join(output_dir, "loss_curve.png"))
    # torch.save(model.state_dict(), os.path.join(output_dir, f"ddpm_mnist_{model_type}.pt"))
    print(f"\nDone! Total time: {dt_total:.1f}s ({dt_total / 60:.1f} min)")
    print(f"Model & outputs saved to: {output_dir}")
