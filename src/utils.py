"""
Visualization, logging, and checkpoint utilities.

Usage::

    from src.utils import (
        save_sample_grid, save_loss_curve,
        LossTracker, CheckpointManager,
    )
"""

import csv
import copy
import math
import os
from typing import List, Optional, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PIL import Image


def load_config(config_path: str) -> dict:
    """Load YAML config file and return a plain dict.

    Args:
        config_path: Absolute or relative path to a ``.yaml`` file.

    Returns:
        Plain ``dict`` with all config keys.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the path is ``None`` or empty.
    """
    import yaml  # lazy import

    if not config_path:
        raise ValueError("config_path must be a non-empty path string, got None or empty")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg if cfg is not None else {}


def save_sample_grid(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 4,
):
    """Arrange a batch of images into a grid and save as PNG.

    Supports both grayscale (C=1) and RGB (C=3) images.  Input tensor is
    assumed to be in [-1, 1] and will be denormalised to [0, 1] before saving.

    Args:
        samples:    image tensor, shape ``(B, C, H, W)`` in [-1, 1].
        save_path:  output PNG file path.
        nrow:       number of images per row in the grid.
    """
    # Denormalize [-1, 1] → [0, 1]
    if torch.min(samples) < 0:
        images = (samples + 1) / 2
    else:
        images = samples
    images = images.clamp(0, 1)

    B, C, H, W = images.shape

    # Pad to a multiple of nrow if necessary
    if B % nrow != 0:
        pad = nrow - B % nrow
        images = torch.cat([images, torch.zeros(pad, C, H, W, dtype=images.dtype, device=images.device)], dim=0)
        B = images.shape[0]

    # Build grid: stack rows of (nrow × C × H × W), then concat
    rows = images.split(nrow, dim=0)            # list of (nrow, C, H, W)
    rows = [torch.cat(list(row), dim=-1)        # (C, H, nrow * W)
            for row in rows]
    grid = torch.cat(rows, dim=-2)              # (C, n_rows * H, nrow * W)

    # PIL expects (H, W) for grayscale, (H, W, C) for RGB
    grid_np = grid.float().cpu().numpy()
    if C == 1:
        grid_np = grid_np.squeeze(0)            # (H, W)
        cmap = "gray"
    else:
        grid_np = grid_np.transpose(1, 2, 0)    # (H, W, C)
        cmap = None

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.imsave(save_path, grid_np, cmap=cmap)


def save_pil_grid(
    images: list,
    save_path: str,
    nrow: int = 4,
    padding: int = 2,
    background_color: tuple = (255, 255, 255)
):
    """Arrange a list of PIL Images into a grid and save as PNG.

    Args:
        images: list of PIL Image objects
        save_path: output PNG file path
        nrow: number of images per row in the grid
        padding: padding between images in pixels
        background_color: RGB tuple for background color
    """
    if not images:
        raise ValueError("No images to save")

    img_width, img_height = images[0].size

    #
    n_total = len(images)
    nrow = min(nrow, n_total)
    ncol = math.ceil(n_total / nrow)

    #
    grid_width = nrow * img_width + (nrow - 1) * padding
    grid_height = ncol * img_height + (ncol - 1) * padding

    #
    grid_image = Image.new('RGB', (grid_width, grid_height), background_color)

    #
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow

        # 计算位置
        x = col * (img_width + padding)
        y = row * (img_height + padding)

        # 粘贴图像
        grid_image.paste(img, (x, y))

    #
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    grid_image.save(save_path, quality=95)


# ============================================================================
# Exponential Moving Average (EMA) — used by Consistency Models as target
# ============================================================================

class EMAModel:
    """Lightweight EMA wrapper that shares the underlying model's parameters.

    Instead of keeping a full model copy, we shadow only the trainable
    parameters.  Use ``swap_to_ema()`` / ``swap_back()`` to temporarily
    replace the model's weights for inference, keeping memory low.

    Typical CM training usage::

        ema = EMAModel(model, decay=0.9999)

        for x, _ in loader:
            # ... forward ...
            optimizer.step()
            ema.update()                    # update EMA *after* optimizer

        # Sampling with EMA weights:
        ema.swap_to_ema()
        samples = scheduler.multistep_sample(model, ...)
        ema.swap_back()

    Args:
        model:  the model whose parameters are being tracked.
        decay:  EMA decay rate (closer to 1 → slower update, more stable).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.model = model

        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self):
        """Single-pole low-pass filter: EMA ← decay·EMA + (1-decay)·θ."""
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                self._shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_to_ema(self):
        """Save current model weights and load EMA weights.

        Must be followed by ``swap_back()`` before resuming training.
        """
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    @torch.no_grad()
    def swap_back(self):
        """Restore the original weights saved by ``swap_to_ema()``."""
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict:
        """Serializable state for checkpointing."""
        return {"decay": self.decay, "shadow": self._shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self._shadow = state_dict["shadow"]


def _find_or_download_component(repo_id, cache_dir, required_files):
    """Find existing component or download it from ModelScope."""
    paths_to_check = [
        os.path.join(cache_dir, repo_id),
        os.path.join(cache_dir, repo_id.replace("/", "_")),
        os.path.join(cache_dir, "._____temp", repo_id),
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            existing_files = [f for f in required_files if os.path.exists(os.path.join(path, f))]
            if len(existing_files) >= len(required_files) // 2:
                print(f"Found component at: {path}")
                return path
    
    print(f"Downloading {repo_id} from ModelScope...")
    from modelscope import snapshot_download
    
    local_path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        allow_patterns=required_files,
    )
    print(f"Downloaded to: {local_path}")
    return local_path