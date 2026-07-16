"""
Model-agnostic activation-collection and per-layer Cayley-rotation calibration.

Usage (with any model containing ``NVFP4Linear`` layers)::

    from rotation import CayleyRotation
    from activation_calibrator import calibrate_activation_rotations

    # 1. Build model with a SHARED placeholder CayleyRotation
    placeholder_rot = CayleyRotation(block_size=16)
    model = MyModel(..., rotation=placeholder_rot)

    # 2. Calibrate — this creates per-layer independent CayleyRotation
    #    instances, fits them on real activations, and replaces the model's
    #    shared rotation in place.
    calib_loader = ...   # yields model inputs
    calibrate_activation_rotations(
        model,
        calib_loader,
        forward_fn=lambda model, batch: model(**batch),
        module_class=NVFP4Linear,          # or any module with .rotation attr
        rotation_attr="rotation",
        block_size=16,
        iters=200, lr=1e-2,
    )
    # Model is now ready for inference with per-layer learned rotations.
"""

import copy
import time
import torch
import torch.nn as nn
from typing import Callable, Optional, List, Dict, Type, Any, Union, Iterator
from .rotation import CayleyRotation


def collect_activations(
    model: nn.Module,
    calibration_loader,
    forward_fn: Callable[[nn.Module, Any], Any],
    module_class: Type[nn.Module],
    max_batches: Optional[int] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Run calibration data through *model* and collect the raw inputs to
    every layer of type *module_class*.

    Args:
        model:              the model to hook into (eval mode recommended).
        calibration_loader: iterable yielding model inputs.
        forward_fn:         fn(model, batch) that runs one forward pass.
                            For most models: ``lambda m, b: m(**b)``.
        module_class:       which ``nn.Module`` subclass to hook
                            (e.g. ``NVFP4Linear``).
        max_batches:        optional cap on number of batches (None = use all).

    Returns:
        ``{layer_name: [tensor_1, tensor_2, ...]}`` where each tensor is the
        **raw input** to that layer's ``forward(x, ...)``, i.e. the activation
        **before** rotation/permutation is applied.
    """
    buffers: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def _hook_fn(name: str):
        def _hook(module, input, output):
            # input[0] is the first positional arg (the activation x),
            # before any rotation/permutation in the layer's forward.
            buffers[name].append(input[0].detach().cpu())
        return _hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, module_class):
            buffers[name] = []
            handles.append(module.register_forward_hook(_hook_fn(name)))

    # Run calibration
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    try:
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if max_batches is not None and i >= max_batches:
                    break
                # Move batch to model device if it's a dict of tensors
                if isinstance(batch, dict):
                    batch = {k: v.to(model_device).type(model_dtype) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = tuple(v.to(model_device).type(model_dtype) if isinstance(v, torch.Tensor) else v
                                  for v in batch)
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(model_device).type(model_dtype)
                forward_fn(model, batch)
    finally:
        for h in handles:
            h.remove()

    return buffers


def calibrate_activation_rotations(
    model: nn.Module,
    calibration_loader,
    forward_fn: Callable[[nn.Module, Any], Any],
    module_class: Type[nn.Module] = None,
    rotation_attr: str = "rotation",
    block_size: int = 16,
    iters: int = 200,
    lr: float = 1e-2,
    max_calib_batches: Optional[int] = None,
    init_from_hadamard: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """End-to-end calibration pipeline.

    1. Collect raw activations for every *module_class* layer.
    2. Create a **per-layer independent** ``CayleyRotation`` instance.
    3. Call ``rot.fit_activation(activations)`` for each layer.
    4. Replace the shared rotation on each layer with the fitted one.

    After this call, the model is ready for inference.

    Args:
        model:               the model (modified in place).
        calibration_loader:  iterable of calibration batches.
        forward_fn:          fn(model, batch) -> output.
        module_class:        which module type to calibrate. Auto-detected
                             if the model has any ``NVFP4Linear`` modules.
        rotation_attr:       attribute name on each module_class holding the
                             rotation (default: "rotation").
        block_size:          NVFP4 block size.
        iters:               optimization iterations per layer.
        lr:                  Adam learning rate.
        max_calib_batches:   cap on calibration batches.
        verbose:             print progress.

    Returns:
        ``{layer_name: {"loss_before": ..., "loss_after": ..., "loss_ratio": ...}}``
        for diagnosis.
    """
    from .rotation import CayleyRotation
    from .quantization import NVFP4ActivationQuantization

    # Auto-detect module_class if not provided
    if module_class is None:
        # Try to import NVFP4Linear
        try:
            from ..modules.quantized_linear import NVFP4Linear as _NVL
        except ImportError:
            import sys, os
            _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, _src)
            from modules.quantized_linear import NVFP4Linear as _NVL
        module_class = _NVL

    model_device = next(model.parameters()).device
    stats: Dict[str, Dict[str, float]] = {}

    if verbose:
        print(f"[Calibrator] Collecting activations for {module_class.__name__} layers...")

    # Step 1: Collect activations
    buffers = collect_activations(
        model, calibration_loader, forward_fn,
        module_class=module_class, max_batches=max_calib_batches,
    )

    if not buffers:
        print("[Calibrator] WARNING: No layers found. Check module_class and forward_fn.")
        return stats

    if verbose:
        total_toks = sum(len(v) for v in buffers.values())
        print(f"[Calibrator] Collected activations from {len(buffers)} layers, "
              f"{total_toks} total batches")

    # Step 2: Per-layer fit
    for name, module in model.named_modules():
        if not isinstance(module, module_class):
            continue
        if name not in buffers or not buffers[name]:
            if verbose:
                print(f"[Calibrator]   {name}: no activations collected, skipping")
            continue

        # Create per-layer independent CayleyRotation
        new_rot = CayleyRotation(block_size=block_size)

        activations = buffers[name]
        act_tensors = [a.to(device=model_device, dtype=torch.float32) for a in activations]

        # ---- Compute quantization error BEFORE fitting (baseline, R = I) ----
        in_features = act_tensors[0].shape[-1]
        n = new_rot._padded_dim(in_features)
        all_act = torch.cat([a.reshape(-1, in_features) for a in act_tensors], dim=0)
        all_pad = torch.nn.functional.pad(all_act, (0, n - in_features))
        all_3d = all_pad.unsqueeze(1)
        with torch.no_grad():
            all_q_before = NVFP4ActivationQuantization.apply(all_3d, block_size).squeeze(1)
        loss_before = torch.nn.functional.mse_loss(all_q_before, all_pad).item()

        # ---- Build Hadamard init matrix if requested ----
        init_mat = None
        if init_from_hadamard:
            # Build a Hadamard matrix of size n (needs n to be power of 2)
            from .rotation import HadamardRotation
            hrot = HadamardRotation(block_size=block_size)
            if hrot._padded_dim(in_features) == n:
                init_mat = hrot._ensure_matrix(in_features, model_device, torch.float32)
            else:
                # Hadamard requires power-of-2 padded dim; if it doesn't match
                # Cayley's block-aligned padded dim, skip Hadamard init.
                if verbose:
                    print(f"[Calibrator]   {name}: Hadamard init skipped "
                          f"(Hadamard dim={hrot._padded_dim(in_features)} != Cayley dim={n})")

        # ---- Fit Cayley K matrix ----
        t_layer_start = time.time()
        new_rot.fit_activation(act_tensors, iters=iters, lr=lr, init_from_matrix=init_mat)
        t_layer = time.time() - t_layer_start

        # ---- Compute quantization error AFTER fitting ----
        with torch.no_grad():
            all_rot = new_rot.rotate_activation(all_act)
            all_rot_3d = all_rot.unsqueeze(1)
            all_q_after = NVFP4ActivationQuantization.apply(all_rot_3d, block_size).squeeze(1)
        loss_after = torch.nn.functional.mse_loss(all_q_after, all_rot).item()

        loss_ratio = loss_after / (loss_before + 1e-12)

        # ---- Apply learned rotation to the layer ----
        # Replace the shared rotation with this layer's fitted instance
        setattr(module, rotation_attr, new_rot)
        t_apply_start = time.time()
        # Also rotate the weight: W_new = W @ R (same R as activation side)
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                w_rot = new_rot.rotate_weight(module.weight.data)
                # Pad weight's in_features if needed
                w_n = new_rot._padded_dim(module.weight.data.shape[-1])
                w_pad = w_n - module.weight.data.shape[-1]
                if w_pad > 0:
                    module.weight.data = nn.Parameter(
                        torch.cat([module.weight.data,
                                   torch.zeros(module.weight.data.shape[0], w_pad,
                                               device=module.weight.device,
                                               dtype=module.weight.dtype)],
                                  dim=-1))
                module.weight.data = w_rot
        t_apply = time.time() - t_apply_start

        stats[name] = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_ratio": loss_ratio,
            "n": n,
            "time_fit_s": round(t_layer, 3),
            "time_apply_s": round(t_apply, 3),
        }

        if verbose:
            print(f"[Calibrator]   {name}: MSE {loss_before:.6e} -> {loss_after:.6e} "
                  f"(ratio={loss_ratio:.4f}, n={n}, fit={t_layer:.2f}s, apply={t_apply:.3f}s)")

    # Summary
    if verbose and stats:
        avg_ratio = sum(s["loss_ratio"] for s in stats.values()) / len(stats)
        total_fit = sum(s["time_fit_s"] for s in stats.values())
        total_apply = sum(s["time_apply_s"] for s in stats.values())
        print(f"[Calibrator] Done. Average loss ratio: {avg_ratio:.4f} "
              f"(<1.0 = improvement over no rotation), "
              f"total fit={total_fit:.1f}s, total apply={total_apply:.2f}s")

    return stats


def calibrate_activation_rotations_mha(
    model: nn.Module,
    calibration_loader,
    forward_fn: Callable[[nn.Module, Any], Any],
    attention_class: Optional[Type[nn.Module]] = None,
    rotation_attr: str = "rotation",
    block_size: int = 16,
    iters: int = 200,
    lr: float = 1e-2,
    max_calib_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Like ``calibrate_activation_rotations`` but for attention modules
    where one rotation is shared across Q/K/V projections.

    In a ``NVFP4MultiHeadAttention``, the input activation ``x`` feeds
    into Q, K, V projections via a single shared rotation. This function
    collects the input to the MHA module, learns one CayleyRotation for
    the entire attention block, and updates the weights of all three
    projections.

    Args:
        model:              the model (modified in place).
        calibration_loader: iterable of calibration batches.
        forward_fn:         fn(model, batch) -> output.
        attention_class:    MHA module type (e.g. ``NVFP4MultiHeadAttention``).
        rotation_attr:      attribute name holding the rotation (default: "rotation").
        block_size:         NVFP4 block size.
        iters:              optimization iterations per layer.
        lr:                 Adam learning rate.
        max_calib_batches:  cap on calibration batches.
        verbose:            print progress.

    Returns:
        ``{layer_name: {"loss_before": ..., "loss_after": ..., "loss_ratio": ...}}``.
    """
    from .rotation import CayleyRotation
    from .quantization import NVFP4ActivationQuantization

    # Auto-detect attention_class
    if attention_class is None:
        try:
            from ..modules.quantized_mha import NVFP4MultiHeadAttention as _MHA
        except ImportError:
            import sys, os
            _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, _src)
            from modules.quantized_mha import NVFP4MultiHeadAttention as _MHA
        attention_class = _MHA

    model_device = next(model.parameters()).device
    stats: Dict[str, Dict[str, float]] = {}

    if verbose:
        print(f"[Calibrator-MHA] Collecting MHA inputs for {attention_class.__name__} layers...")

    # Hooks on the attention module itself (not its sub-layers)
    buffers: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def _hook_fn(name: str):
        def _hook(module, input, output):
            # In MHA forward(query, key, value, ...), query is the primary activation.
            buffers[name].append(input[0].detach().cpu())
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, attention_class):
            buffers[name] = []
            handles.append(module.register_forward_hook(_hook_fn(name)))

    try:
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if max_calib_batches is not None and i >= max_calib_batches:
                    break
                if isinstance(batch, dict):
                    batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = tuple(v.to(model_device) if isinstance(v, torch.Tensor) else v
                                  for v in batch)
                forward_fn(model, batch)
    finally:
        for h in handles:
            h.remove()

    if not buffers:
        print("[Calibrator-MHA] WARNING: No MHA layers found.")
        return stats

    # Per-MHA fit
    for name, module in model.named_modules():
        if not isinstance(module, attention_class):
            continue
        if name not in buffers or not buffers[name]:
            continue

        new_rot = CayleyRotation(block_size=block_size)
        activations = buffers[name]
        act_tensors = [a.to(device=model_device, dtype=torch.float32) for a in activations]

        in_features = act_tensors[0].shape[-1]
        n = new_rot._padded_dim(in_features)
        all_act = torch.cat([a.reshape(-1, in_features) for a in act_tensors], dim=0)
        all_pad = torch.nn.functional.pad(all_act, (0, n - in_features))
        with torch.no_grad():
            all_q_before = NVFP4ActivationQuantization.apply(
                all_pad.unsqueeze(1), block_size).squeeze(1)
        loss_before = torch.nn.functional.mse_loss(all_q_before, all_pad).item()

        new_rot.fit_activation(act_tensors, iters=iters, lr=lr)

        with torch.no_grad():
            all_rot = new_rot.rotate_activation(all_act)
            all_q_after = NVFP4ActivationQuantization.apply(
                all_rot.unsqueeze(1), block_size).squeeze(1)
        loss_after = torch.nn.functional.mse_loss(all_q_after, all_rot).item()
        loss_ratio = loss_after / (loss_before + 1e-12)

        # Apply to the attention module
        setattr(module, rotation_attr, new_rot)

        # Rotate in_proj_weight (all Q/K/V concatenated) and out_proj.weight
        if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
            with torch.no_grad():
                w_rot = new_rot.rotate_weight(module.in_proj_weight.data)
                w_n = new_rot._padded_dim(module.in_proj_weight.data.shape[-1])
                w_pad = w_n - module.in_proj_weight.data.shape[-1]
                if w_pad > 0:
                    module.in_proj_weight.data = nn.Parameter(
                        torch.cat([module.in_proj_weight.data,
                                   torch.zeros(module.in_proj_weight.data.shape[0], w_pad,
                                               device=module.in_proj_weight.device,
                                               dtype=module.in_proj_weight.dtype)],
                                  dim=-1))
                module.in_proj_weight.data = w_rot

        if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'weight'):
            with torch.no_grad():
                w_rot = new_rot.rotate_weight(module.out_proj.weight.data)
                w_n = new_rot._padded_dim(module.out_proj.weight.data.shape[-1])
                w_pad = w_n - module.out_proj.weight.data.shape[-1]
                if w_pad > 0:
                    module.out_proj.weight.data = nn.Parameter(
                        torch.cat([module.out_proj.weight.data,
                                   torch.zeros(module.out_proj.weight.data.shape[0], w_pad,
                                               device=module.out_proj.weight.device,
                                               dtype=module.out_proj.weight.dtype)],
                                  dim=-1))
                module.out_proj.weight.data = w_rot

        stats[name] = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_ratio": loss_ratio,
            "n": n,
        }

        if verbose:
            print(f"[Calibrator-MHA]   {name}: MSE {loss_before:.6e} -> {loss_after:.6e} "
                  f"(ratio={loss_ratio:.4f}, n={n})")

    if verbose and stats:
        avg_ratio = sum(s["loss_ratio"] for s in stats.values()) / len(stats)
        print(f"[Calibrator-MHA] Done. Average loss ratio: {avg_ratio:.4f}")

    return stats


# ============================================================================
# Cayley activation calibration loader
# ============================================================================

class CayleyCalibrationLoader:
    """Yields forward-pass batches for Cayley activation calibration.

    Supports two modes:
    1. Synthetic mode: Uses real text embeddings with synthetic noise latents
       to simulate the transformer's forward pass input.
    2. Dataset mode: Uses real prompts from a dataset to generate calibration batches.

    In synthetic mode, this is much faster than running the full decode loop and
    provides sufficient coverage for activation statistics collection.
    """

    def __init__(self, pipe, txt_embeds, in_channels, resolution, device, n_batches=8):
        """Initialize the calibration loader.

        Args:
            pipe: SanaSprintPipeline instance
            txt_embeds: List of text embeddings (each is [1, seq_len, caption_channels])
            in_channels: Number of input channels for the latent
            resolution: Image resolution
            device: Torch device
            n_batches: Number of calibration batches to generate
        """
        self.pipe = pipe
        self.txt_embeds = txt_embeds
        self.in_channels = in_channels
        self.resolution = resolution
        self.device = device
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        config = self.pipe.scheduler.config
        if isinstance(config, dict):
            sigma_data = float(config.get("sigma_data", 1.0))
        else:
            sigma_data = float(config.sigma_data)
        guidance_embeds_scale = getattr(self.pipe.transformer.config, 'guidance_embeds_scale', 0.1)

        for i in range(self.n_batches):
            gen = torch.Generator(device=self.device).manual_seed(1000 + i)
            latent0 = self.pipe.prepare_latents(
                1, self.in_channels, self.resolution, self.resolution,
                torch.float32, self.device, gen, None) * sigma_data
            scm_t = torch.rand(1, device=self.device) * 0.5 + 0.25
            latent_model_input = latent0 / sigma_data
            latent_model_input = latent_model_input * torch.sqrt(
                scm_t.view(-1, 1, 1, 1) ** 2 + (1 - scm_t.view(-1, 1, 1, 1)) ** 2)
            guidance = torch.full([1], 4.5, device=self.device,
                                  dtype=torch.float32) * guidance_embeds_scale

            txt_idx = i % len(self.txt_embeds)
            txt_embed = self.txt_embeds[txt_idx].to(self.device)

            yield {
                "hidden_states": latent_model_input,
                "encoder_hidden_states": txt_embed,
                "timestep": scm_t,
                "guidance": guidance,
                "return_dict": False,
            }


def make_calibration_loader(
    pipe,
    prompts: Optional[List[str]] = None,
    txt_embeds: Optional[List[torch.Tensor]] = None,
    in_channels: int = 32,
    resolution: int = 1024,
    device = None,
    n_batches: int = 8,
) -> CayleyCalibrationLoader:
    """Create a CayleyCalibrationLoader from prompts or pre-computed text embeddings.

    Args:
        pipe: SanaSprintPipeline instance
        prompts: List of prompts to encode (optional, if txt_embeds provided)
        txt_embeds: List of pre-computed text embeddings
        in_channels: Number of input channels for the latent
        resolution: Image resolution
        device: Torch device (auto-detected if None)
        n_batches: Number of calibration batches

    Returns:
        CayleyCalibrationLoader instance

    Example:
        # From prompts:
        loader = make_calibration_loader(
            pipe, prompts=["a cat", "a dog"], in_channels=32, resolution=1024,
            device="cuda", n_batches=8)

        # From pre-computed embeddings:
        loader = make_calibration_loader(
            pipe, txt_embeds=[embed1, embed2], in_channels=32, resolution=1024,
            device="cuda", n_batches=8)
    """
    if device is None:
        device = next(pipe.transformer.parameters()).device

    if txt_embeds is None:
        if prompts is None:
            raise ValueError("Either prompts or txt_embeds must be provided")

        txt_embeds = []
        for prompt in prompts:
            with torch.no_grad():
                out = pipe.encode_prompt(
                    prompt=prompt, device=device,
                    num_images_per_prompt=1,
                )
                if hasattr(out, "prompt_embeds"):
                    embeds = out.prompt_embeds
                elif isinstance(out, tuple):
                    embeds = out[0]
                else:
                    embeds = out
                txt_embeds.append(embeds.detach().cpu())

    return CayleyCalibrationLoader(
        pipe=pipe,
        txt_embeds=txt_embeds,
        in_channels=in_channels,
        resolution=resolution,
        device=device,
        n_batches=n_batches,
    )
