import sys, os
import argparse

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_loader import load_model

parser = argparse.ArgumentParser(description="UNet architecture inspector")
parser.add_argument("--model", type=str, default="sd", choices=["sd", "sdxl"])
parser.add_argument("--selected-index", type=int, default=0, choices=[0, 1])
parser.add_argument("--load_weights", action="store_true", default=False, help="Load actual weights for histogram analysis (slow, ~3.4GB RAM)")
args = parser.parse_args()

# -------------------------------------------------------------------
# Load UNet
# -------------------------------------------------------------------
if args.load_weights:
    print("[0] Loading pipeline with weights (may take a while) ...")
    pipe, device = load_model(model_name=args.model, mirror="https://hf-mirror.com")
    unet = pipe.unet
    print(f"    Model loaded (device={device})\n")
else:
    print("[0] Building UNet from config (no weights, fast) ...")
    from diffusers import UNet2DConditionModel

    model_id = "runwayml/stable-diffusion-v1-5" if args.model == "sd" else "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config(model_id, subfolder="unet", local_files_only=True)
    )
    print(f"    UNet built from config (type={type(unet).__name__})\n")


stats = []
for name, param in unet.named_parameters():
    if len(param.size()) < 2:
        continue
    n_channels = param.size()[0]
    reshape_param = param.reshape(param.size()[0], -1)
    """
    stat = {
        "name": name,
        "shape": param.size(),
        "numel": param.numel(),
        "reshape_param": reshape_param.detach().cpu().numpy(),
        "per-channel-std": torch.std(reshape_param, dim=-1)
    }
    stats.append(stat)
    """
    for channel_idx in range(n_channels):
        channel_param = reshape_param[channel_idx]
        stat = {
            "name": f"{name}-{channel_idx}",
            "numel": channel_param.numel(),
            "param": channel_param.detach().cpu().numpy(),
            "std": torch.std(channel_param)
        }
        stats.append(stat)
    # print(f"{name:<78s}: {stat['shape']} | {stat['reshape_param'].size()} | {stat['per-channel-std'].size()}")

"""    
stats.sort(key=lambda x:x["numel"], reverse=True)
n_selected_params = 3
fig, axes = plt.subplots(n_selected_params, 3)
fontsize = 5
labelsize = 5
for idx, stat in enumerate(stats):
    if idx >= n_selected_params:
        break
    reshape_param = stat["reshape_param"]
    std_list = stat["0-std"]
    min_std_idx = torch.argmin(std_list)
    max_std_idx = torch.argmax(std_list)
    while True:
        random_index = torch.randint(0, len(std_list), (1,)).item()
        if random_index != max_std_idx and random_index != min_std_idx:
            break
    
    # Plot histograms
    axes[idx, 0].hist(reshape_param[min_std_idx], bins=50, color='coral', alpha=0.8, edgecolor='none')
    axes[idx, 1].hist(reshape_param[max_std_idx], bins=50, color='steelblue', alpha=0.8, edgecolor='none')
    axes[idx, 2].hist(reshape_param[random_index], bins=50, color='seagreen', alpha=0.8, edgecolor='none')
    
    # Set titles (reduce font size)
    axes[idx, 0].set_title(f"Min std (idx={min_std_idx}, std={std_list[min_std_idx].item():.4f})", fontsize=fontsize)
    axes[idx, 1].set_title(f"Max std (idx={max_std_idx}, std={std_list[max_std_idx].item():.4f})", fontsize=fontsize)
    axes[idx, 2].set_title(f"Random (idx={random_index}, std={std_list[random_index].item():.4f})", fontsize=fontsize)

    
    # Set tick label font size
    axes[idx, 0].tick_params(axis='both', which='major', labelsize=labelsize)
    axes[idx, 1].tick_params(axis='both', which='major', labelsize=labelsize)
    axes[idx, 2].tick_params(axis='both', which='major', labelsize=labelsize)
    
    # Add caption (reduce font size)
    caption = f"[Stat {idx}] {stat['name']}  |  shape={list(stat['shape'])}  |  numel={stat['numel']:,}"
    axes[idx, 0].text(-0.1, 1.05, caption, transform=axes[idx, 0].transAxes, fontsize=fontsize, fontweight='bold', verticalalignment='top', wrap=True)


# plt.tight_layout()
plt.subplots_adjust(hspace=0.5) 
# fig.show()
plt.savefig("./outputs/channel-wise-std-histogram.png", dpi=150, bbox_inches="tight")
"""

# stats.sort(key=lambda x:x["std"], reverse=True)
std_list = [stat["std"] for stat in stats]
min_std_idx = torch.argmin(std_list)
max_std_idx = torch.argmax(std_list)
while True:
    random_index = torch.randint(0, len(std_list), (1,)).item()
    if random_index != max_std_idx and random_index != min_std_idx:
        break
