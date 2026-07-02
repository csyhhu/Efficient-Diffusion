"""
Sampling / reverse-process functions for different diffusion families.

Usage::

    from src.samplers import ddim_sample, fm_sample, fm_t2i_sample
"""

import math
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.schedulers import DDPMScheduler, FlowMatchingScheduler, ConsistencyModelScheduler
from src.utils import EMAModel


# ============================================================================
# DDPM reverse sampling (MNIST)
# ============================================================================

@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple = (16, 1, 28, 28),
    device: str = "cpu",
) -> torch.Tensor:
    """DDPM reverse process: sample from pure noise by iterative denoising.

    Args:
        model:     noise-prediction model ε_θ(x_t, t).
        scheduler: DDPM scheduler with precomputed coefficients.
        shape:     (B, C, H, W) latent shape.
        device:    torch device.

    Returns:
        x_0 tensor in [-1, 1].
    """
    model.eval()
    T = scheduler.T
    B = shape[0]

    x_t = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_tensor)

        coeffs = scheduler.get_posterior_coeffs(t, device)
        mean = coeffs["mean_coef_x_t"] * x_t - coeffs["mean_coef_eps"] * eps_theta

        if t > 0:
            z = torch.randn(shape, device=device)
            x_t = mean + coeffs["sigma"] * z
        else:
            x_t = mean

    return x_t.clamp(-1, 1)


# ============================================================================
# Flow Matching ODE sampling (MNIST — no VAE)
# ============================================================================

@torch.no_grad()
def fm_sample(
    model: nn.Module,
    num_steps: int = 100,
    shape: tuple = (16, 1, 28, 28),
    device: str = "cpu",
) -> torch.Tensor:
    """Flow Matching: integrate velocity ODE backward (t=1 → t=0).

    Args:
        model:     velocity-prediction model v_θ(x_t, t).
        num_steps: ODE discretization steps.
        shape:     (B, C, H, W).
        device:    torch device.

    Returns:
        x_0 tensor in [-1, 1].
    """
    model.eval()
    B = shape[0]

    x_t = torch.randn(shape, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_now = 1.0 - i * dt
        t_tensor = torch.full((B,), t_now, device=device, dtype=torch.float32)
        v = model(x_t, t_tensor)
        x_t = x_t - v * dt

    return x_t.clamp(-1, 1)


# ============================================================================
# Flow Matching + VAE sampling (T2I — SD3 / PixArt)
# ============================================================================

@torch.no_grad()
def fm_t2i_sample(
    model: nn.Module,
    vae,
    encoder_hidden_states: torch.Tensor,
    num_steps: int = 50,
    shape: tuple = (8, 16, 32, 32),
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    model_type: str = "sd3",
    pooled_projections: Optional[torch.Tensor] = None,
    image_size: int = 256,
) -> torch.Tensor:
    """T2I Flow Matching: ODE integration + VAE decode.

    Args:
        model: backbone (QuantizedSD3 or QuantizedPixArt).
        vae: frozen VAE decoder (or DummyVAE in dry-run mode).
        encoder_hidden_states: T5 embeddings, shape (B, seq, dim).
        num_steps: ODE discretization steps.
        shape: (B, C, H, W) latent shape.
        device, dtype: compute device / dtype.
        model_type: "sd3" or "pixart".
        pooled_projections: pooled text embeddings (SD3 only).
        image_size: output image resolution (PixArt only).

    Returns:
        images tensor (B, 3, H_img, W_img) in [0, 1].
    """
    model.eval()
    B, C, H, W = shape

    z_t = torch.randn(shape, device=device, dtype=dtype)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_now = 1.0 - i * dt
        t_tensor = torch.full((B,), t_now * 1000, device=device, dtype=torch.float32)

        if model_type == "pixart":
            v = model(
                hidden_states=z_t,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t_tensor.long(),
                encoder_attention_mask=None,
                height=torch.full((B,), image_size, device=device),
                width=torch.full((B,), image_size, device=device),
                return_dict=False,
            )
        else:  # sd3
            v = model(
                hidden_states=z_t,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=t_tensor.long(),
                return_dict=False,
            )

        z_t = z_t - v * dt

    # Decode latents → images
    vae_scale = vae.config.scaling_factor
    z_t = z_t / vae_scale
    images = vae.decode(z_t.to(dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


# ============================================================================
# Consistency Model sampling — multistep / one-step
# ============================================================================

@torch.no_grad()
def cm_sample(
    model: nn.Module,
    scheduler: ConsistencyModelScheduler,
    shape: tuple = (16, 1, 28, 28),
    device: str = "cpu",
    ema: Optional[EMAModel] = None,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """Consistency Model multistep sampling (uses EMA weights if available).

    Args:
        model:     consistency function f_θ (predicts x₀ directly).
        scheduler: ``ConsistencyModelScheduler``.
        shape:     (B, C, H, W).
        device:    torch device.
        ema:       optional ``EMAModel``; if provided, samples with EMA weights.
        num_steps: override the number of sigma steps (rebuilds schedule).

    Returns:
        x₀ tensor in [-1, 1].
    """
    if num_steps is not None:
        scheduler.reset_sigmas(num_steps)

    if ema is not None:
        ema.swap_to_ema()

    samples = scheduler.multistep_sample(
        model, shape, device, use_preconditioning=False,
    )

    if ema is not None:
        ema.swap_back()

    return samples
