r"""
Usage

    # Flow Matching
    python main.py `
        --model_name=quantized_dit `
        --model_config_path=config/mnist_dit_fm/model.yaml `
        --dataset_name=MNIST `
        --dataset_config_path=config/mnist_dit_fm/dataset.yaml `
        --running_config_path=config/mnist_dit_fm/running.yaml `
        --output_dir=Results/mnist_quantized_dit_fm

    # Consistency Model
    python main.py `
        --model_name=dit `
        --model_config_path=config/mnist_dit_fm/model.yaml `
        --dataset_name=MNIST `
        --dataset_config_path=config/mnist_dit_fm/dataset.yaml `
        --running_config_path=config/mnist_dit_cm/running.yaml `
        --output_dir=Results/mnist_dit_cm
"""
import os, sys
import argparse
import csv
import time

import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure project root is on sys.path so that ``from src.xxx`` works
# regardless of how this file is invoked.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.model_loader import load_model
from src.data_loader import get_dataloader
from src.utils import load_config, save_sample_grid, EMAModel
from src.schedulers import DDPMScheduler, FlowMatchingScheduler, ConsistencyModelScheduler
from src.samplers import fm_sample, ddpm_sample, fm_t2i_sample, cm_sample

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Efficient Diffusion Training")
    parser.add_argument("--model_name", type=str, default="quantized_dit")
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--dataset_config_path", type=str, default=None)
    parser.add_argument("--running_config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Load Model Configuration
    model_config = load_config(args.model_config_path)
    # Model Initailization
    model, vae, tokenizer = load_model(args.model_name, model_config, dry_run=args.dry_run)

    # Dataset Loading
    dataset_config = load_config(args.dataset_config_path)
    train_loader, val_loader = get_dataloader(args.dataset_name, dataset_config, vae, tokenizer)

    # Running Configuration, including training config and sampling config
    running_config = load_config(args.running_config_path)

    # Device & output dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ── Scheduler ──
    sample_type = running_config.get("sample_type", "ddpm")
    scheduler_name = running_config.get("scheduler", sample_type)

    if scheduler_name == "ddpm":
        scheduler = DDPMScheduler(num_timesteps=running_config.get("num_timesteps", 1000))
    elif scheduler_name == "cm":
        scheduler = ConsistencyModelScheduler(
            num_discretization=running_config.get("cm_num_discretization", 40),
            sigma_min=running_config.get("cm_sigma_min", 0.002),
            sigma_max=running_config.get("cm_sigma_max", 80.0),
        )
    else:
        scheduler = FlowMatchingScheduler()  # FM / default

    # ── EMA (Consistency Model only) ──
    ema = None
    if sample_type == "cm":
        ema = EMAModel(model, decay=running_config.get("cm_ema_decay", 0.9999))

    # ── Optimizer & loss ──
    optimizer = torch.optim.Adam(model.parameters(), lr=running_config.get("lr", 1e-4))
    loss_fn = nn.MSELoss()

    # Move model to device
    model = model.to(device)

    # ── Training state ──
    loss_records = []       # (step, epoch, train_loss)
    global_step = 0
    t0_total = time.time()
    n_epochs = running_config.get("epochs", 100)

    # Sample shape deduced from model config
    img_size = model_config.get("image_size", 28)
    in_ch = model_config.get("in_channels", 1)
    sample_shape = (16, in_ch, img_size, img_size)

    quant_error_records = {}  # {module_key: [(step, quant_error), ...]}
    quant_error_dir = os.path.join(output_dir, "quant_error")

    # ==================================================================
    # Training loop
    # ==================================================================
    for epoch in range(n_epochs):

        # ── Sampling ──
        if epoch % running_config.get("sample_interval", 10) == 0:
            save_path = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png")
            if sample_type == "ddpm":
                samples = ddpm_sample(
                    model, scheduler,
                    shape=sample_shape, device=device,
                )
            elif sample_type == "cm":
                samples = cm_sample(
                    model, scheduler,
                    shape=sample_shape, device=device,
                    ema=ema,
                    num_steps=running_config.get("cm_sampling_steps", None),
                )
            else:
                if vae is not None:
                    samples = fm_t2i_sample(model, vae, tokenizer, device=device)
                else:
                    samples = fm_sample(
                        model, num_steps=running_config.get("num_steps", 50),
                        shape=sample_shape, device=device,
                    )
            save_sample_grid(samples.to("cpu"), save_path)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{n_epochs}", leave=False)
        epoch_loss = 0.0
        t0_epoch = time.time()
        batch_times = []

        for x, _ in pbar:
            x = x.to(device)
            B = x.shape[0]
            t0_batch = time.time()

            # ── Forward (DDPM / FM / CM) ──
            if sample_type == "ddpm":
                t = scheduler.sample_timesteps(B, device)          # int, [0, T-1]
                x_t, target = scheduler.add_noise(x, t)            # target = noise ε
                pred = model(x_t, t)
                loss = loss_fn(pred, target)

            elif sample_type == "cm":
                # ── Consistency Model ──
                sigma_n, sigma_np1 = scheduler.sample_timestep_pair(B, device)
                x_sn, x_snp1, _ = scheduler.add_noise_pair(x, sigma_n, sigma_np1)

                # Target: EMA model on less-noisy sample (stop-grad)
                ema.swap_to_ema()
                with torch.no_grad():
                    target = model(x_sn, sigma_n)
                ema.swap_back()

                # Prediction: current model on more-noisy sample
                pred = model(x_snp1, sigma_np1)

                loss = scheduler.pseudo_huber_loss(pred, target)

            else:  # Flow Matching
                x_1 = scheduler.sample_noise(x)                    # noise ~ N(0,I)
                t = scheduler.sample_time(B, device)               # float, [0,1]
                x_t = scheduler.interpolate(x, x_1, t)             # (1-t)*x + t*x_1
                target = scheduler.compute_target(x, x_1)          # v = x_1 - x
                pred = model(x_t, t)
                loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()

            # Record per-layer quantization error (quantized models only)
            quantization_error_info = model.get_quantization_error() if hasattr(model, "get_quantization_error") else {}
            if quantization_error_info:
                for layer, error in quantization_error_info.items():
                    quant_error_records.setdefault(layer, []).append((global_step, error))

            optimizer.step()

            # Update EMA after optimizer step (CM only)
            if sample_type == "cm" and ema is not None:
                ema.update()

            # ── Timing ──
            dt_batch = time.time() - t0_batch
            batch_times.append(dt_batch)
            if len(batch_times) > 50:
                batch_times.pop(0)
            avg_batch_time = sum(batch_times) / len(batch_times)

            # ── Logging ──
            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1
            loss_records.append((global_step, epoch, loss_val))
            batches_done = global_step - epoch * len(train_loader)

            pbar.set_postfix(
                loss=f"{loss_val:.6f}",
                avg=f"{epoch_loss / batches_done:.6f}",
                ms=f"{avg_batch_time * 1000:.0f}",
            )

        # ── End of epoch ──
        dt_epoch = time.time() - t0_epoch
        avg_loss = epoch_loss / len(train_loader)
        dt_total = time.time() - t0_total
        print(f"Epoch {epoch:3d}/{n_epochs}"
              f" | avg loss: {avg_loss:.6f}"
              f" | epoch time: {dt_epoch:.1f}s"
              f" | total elapsed: {dt_total / 60:.1f}m")

        # Save loss CSV
        csv_path = os.path.join(output_dir, "loss_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss"])
            writer.writerows(loss_records)

        # Save per-layer quantization error CSVs
        if quant_error_records:
            os.makedirs(quant_error_dir, exist_ok=True)
            for layer, records in quant_error_records.items():
                safe_name = layer.replace('.', '_')
                with open(os.path.join(quant_error_dir, f"{safe_name}.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "quant_error_sum"])
                    writer.writerows(records)

    # ── Post-training ──
    dt_total = time.time() - t0_total
    print(f"\nDone! Total time: {dt_total:.1f}s ({dt_total / 60:.1f} min)")
    print(f"Output saved to: {output_dir}")
