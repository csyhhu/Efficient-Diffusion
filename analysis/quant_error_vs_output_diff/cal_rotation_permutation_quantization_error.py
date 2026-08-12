"""
Sweep: per-sample transformer output error vs a reference model, and per-sample
per-layer quantization error info, across (rotation x permutation x (un)quantized)
configurations.

For every sample we start from a SEEDED random noise latent and define the decode
trajectory as follows:

  - The REFERENCE model (an unquantized `NVFP4QuantizedSana`, mathematically
    identical to the diffusers original) is decoded ONCE per sample, recording
    its intermediate states: the per-step raw transformer outputs (`oq`, a
    NOISE / VELOCITY prediction -- NOT a VAE latent) and the final latent.
  - Each TARGET model (a (rotation, permutation, quantized) combo) is decoded
    from the SAME initial latent (`run_decode_loop` takes a `latent` input and a
    constant per-sample `txt`), recording its own intermediate states.
  - The recorded intermediate states are finally compared: per-step `oq` diff
    (MSE / L1 / cosine) vs the reference, plus the final-latent diff.

The transformer output at each step cannot be decoded directly -- it must go
through the scheduler. Each model advances its latent with ITS OWN prediction,
so every model's final latent reflects that model and can be VAE-decoded to an
image (controlled by `--generate-images`), verifying the whole parameter-loading
chain for that combo.

Per-sample inputs (constant across models within a sample):
  - encoder_hidden_states (txt): the sample's real caption encoded by the
    pipeline's text encoder (encoding is unchanged across models).
  - initial latent: the seeded random noise latent, shared by ref and all targets
    so that every model's decode starts from the SAME state.

Recorded per sample x combo x step:
  - output diff (MSE / L1 / cosine) vs the reference  -> output_diff.csv / .json
      * overall: the transformer's final raw output (`oq`)
      * per-layer: the output of EACH transformer block (key "layers")
Recorded per sample x combo (steps-independent, end-to-end):
  - final diff: the FINAL latent, the VAE-decoded feature (`dec`), and the final
    RGB image, each vs the reference (MSE / L1 / cosine) -> output_diff.json
    under the reserved top-level key "final_diff" + rows in output_diff.csv
    (step="final", layer in {final_latents, dec, image}). These capture the whole
    pipeline error (last-step prediction + scheduler + VAE decode) beyond the
    per-step transformer-output error above.
Recorded per combo (sample-independent, since weights don't depend on the input):
  - parameter quantization error per layer             -> param_quant_errors.json

NOTE: the decode loop starts from a SEEDED random noise latent (the standard
diffusion starting point) -- no real images or VAE encoding are required for the
sweep. The only per-sample input that varies is the prompt embedding `txt`. The
final latent (after `--decode_steps` scheduler steps) is returned by
`run_decode_loop` and can be VAE-decoded (`pipe.vae.decode(final_latents /
scaling_factor)`) to produce an image -- useful for verifying the whole
parameter-loading chain.

Usage (Windows PowerShell — copy & paste directly; each line ends with a
backtick `` ` `` which is PowerShell's line-continuation character):

--rots none,hadamard,random,

    python analysis\\cal_rotation_permutation_quantization_error.py `
        --dataset_path G:\\datasets\\MJHQ-30K `
        --n_samples 1 `
        --output_dir G:\\Outputs\\Efficient-Diffusion\\rot_perm_compare_module_steps_quantized `
        --rots cayley `
        --cayley_calib `
        --perms identity,random,mag `
        --quantized_modes quantized `
        --cosine `
        --decode_steps 2 `
        --generate_images

    # `--generate_images` is OPTIONAL: also VAE-decode each config's final
    # latent into images (verifies the whole parameter-loading chain). Images
    # for the SAME prompt are grouped: `generated_images/{idx:04d}/` holds
    # `ref.png`, one `{config}.png` per configuration, and a combined
    # `grid_{idx:04d}.png` montage (all configs side-by-side). Drop the flag to
    # run the sweep only (final diffs are still recorded regardless).

    # Single-line form (handy for variables / no continuation):
    # python analysis\\cal_rotation_permutation_quantization_error.py --dataset_path G:\\datasets\\MJHQ-30K --n_samples 1 --output_dir G:\\Outputs\\Efficient-Diffusion\\rot_perm_compare_steps --rots none,hadamard,random --perms none,identity,random,mag --quantized_modes both --decode_steps 2 --generate_images

    # DRY MODE: randomly-initialized transformer + VAE (NO checkpoint download).
    # Tiny config so the whole sweep runs in seconds — use it to verify the
    # plumbing, transforms, quantization and saving paths. Outputs are NOT
    # meaningful (random weights); only the code path / shapes matter.
    python analysis\\cal_rotation_permutation_quantization_error.py --dry `
        --n_samples 10 --output_dir G:\\Outputs\\Efficient-Diffusion\\rot_perm_compare_module_steps_dry `
        --rots identity,random,hadamard,cayley --perms identity,random,mag `
        --cayley_calib `
        --quantized_modes quantized --decode_steps 4 --dry_dim 256 --dry_layers 4 --dry_resolution 64

Usage in IPython (use the ``run`` magic; pass args as a single string):

    In [1]: run analysis/cal_rotation_permutation_quantization_error.py --dataset_path G:\\datasets\\MJHQ-30K --n_samples 1 --output_dir G:\\Outputs\\Efficient-Diffusion\\rot_perm_compare_steps --rots none,hadamard,random --perms none,identity,random,mag --quantized_modes both --decode_steps 2 --generate_images

    # DRY MODE in IPython:
    In [2]: run analysis/cal_rotation_permutation_quantization_error.py --dry --n_samples 1 --output_dir G:\\Outputs\\Efficient-Diffusion\\rot_perm_compare_steps_dry --rots none,hadamard,random --perms none,random --quantized_modes both --decode_steps 2 --dry_dim 256 --dry_layers 2 --dry_resolution 64

    # Tip: ``run`` also exposes the script's globals to your IPython session, so
    # after a run you can inspect e.g. ``results`` / ``pipe`` / ``samples``.
"""

import argparse
import csv
import json
import os
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Make the project (and its `src`/`scripts` packages) importable when run as a
# plain script:  python analysis\cal_rotation_permutation_quantization_error.py ...
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.image_generation import load_pipeline  # reuse the working pipeline loader
from src.quant_utils.rotation import (
    RotationBase, IdentityRotation, HadamardRotation, RandomRotation, CayleyRotation,
)
from src.quant_utils.permutation import (
    PermutationBase, IdentityPermutation, RandomPermutation, MagnitudeSortPermutation,
)
from src.modules.quantized_linear import NVFP4Linear
from src.quant_utils.quantization import NVFP4Quantization
from src.models.nvfp4_quantized_Sana import NVFP4QuantizedSana
from src.quant_utils.activation_calibrator import (
    calibrate_activation_rotations, make_calibration_loader)


# ===========================================================================
# Transform factories
# ===========================================================================

def make_rotation(name, block_size=16, seed=None):
    """Build a (shared) RotationBase instance from a name (or None)."""
    if name == "identity":
        return IdentityRotation(block_size=block_size)
    if name == "hadamard":
        return HadamardRotation(block_size=block_size)
    if name == "random":
        return RandomRotation(block_size=block_size, seed=seed)
    if name == "cayley":
        return CayleyRotation(block_size=block_size, seed=seed)
    raise ValueError(f"unknown rotation: {name}")


def make_perm_factory(name, block_size=16, seed=None, metric="norm", order="asc"):
    """Build a PER-LAYER permutation factory (name) -> PermutationBase | None.

    Returns a callable ``factory(in_features) -> PermutationBase | None`` so each
    layer gets its own (layer-specific) permutation instance.
    """
    if name == "identity":
        return lambda in_features: IdentityPermutation(block_size=block_size)
    if name == "random":
        return lambda in_features: RandomPermutation(block_size=block_size, seed=seed)
    if name == "mag":
        return lambda in_features: MagnitudeSortPermutation(
            block_size=block_size, metric=metric, order=order)
    raise ValueError(f"unknown permutation: {name}")


# ===========================================================================
# Sample loading (prompt + image path) for MJHQ-30K
# ===========================================================================

def load_samples(dataset_name, dataset_path, n_samples=None,
                 sampling="sequential", sample_start=0, seed=42):
    """Load MJHQ-30K samples as list of {image_id, prompt, category} dicts."""
    if dataset_name.upper().replace("_", "-") != "MJHQ-30K":
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not supported yet (only MJHQ-30K).")
    meta_file = os.path.join(dataset_path, "meta_data.json")
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(
            f"meta_data.json not found in {dataset_path} (expected MJHQ-30K layout).")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    keys = sorted(meta.keys())
    items = [{
        "image_id": k,
        "prompt": meta[k]["prompt"],
        "category": meta[k].get("category", ""),
    } for k in keys]
    total = len(items)

    if n_samples is None or n_samples >= total:
        return items
    if sampling == "random":
        import random as _random
        _random.seed(seed)
        idx = sorted(_random.sample(range(total), n_samples))
        return [items[i] for i in idx]
    start = max(0, sample_start)
    return items[start:start + n_samples]


def encode_prompt_embeds(pipe, prompt, device):
    """Return the caption embedding (B, N, caption_channels) the transformer eats."""
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
    return embeds.to(device)


def build_transformer(ref_model, device, use_nvfp4, block_size, rotation, permutation):
    """Build an NVFP4QuantizedSana by copying weights from a cached ref_model.

    Reuses ``ref_model`` across all combos (avoids reloading the diffusers
    transformer for every configuration).
    """
    cfg = ref_model.config
    inner_dim = int(cfg["num_attention_heads"]) * int(cfg["attention_head_dim"])
    model = NVFP4QuantizedSana(
        sample_size=cfg.get("sample_size", 32),
        patch_size=cfg.get("patch_size", 1),
        in_channels=cfg.get("in_channels", 32),
        out_channels=cfg.get("out_channels") or cfg.get("in_channels", 32),
        num_layers=cfg["num_layers"],
        attention_head_dim=cfg["attention_head_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        num_cross_attention_heads=cfg.get("num_cross_attention_heads") or cfg["num_attention_heads"],
        cross_attention_head_dim=cfg.get("cross_attention_head_dim") or cfg["attention_head_dim"],
        cross_attention_dim=cfg.get("cross_attention_dim") or inner_dim,
        caption_channels=cfg.get("caption_channels", 2304),
        mlp_ratio=cfg.get("mlp_ratio", 2.5),
        attention_bias=cfg.get("attention_bias", True),
        norm_elementwise_affine=cfg.get("norm_elementwise_affine", False),
        norm_eps=cfg.get("norm_eps", 1e-6),
        interpolation_scale=cfg.get("interpolation_scale", None),
        guidance_embeds=cfg.get("guidance_embeds", True),
        guidance_embeds_scale=cfg.get("guidance_embeds_scale", 0.1),
        qk_norm=cfg.get("qk_norm", None),
        block_size=block_size,
        use_nvfp4=use_nvfp4,
        rotation=rotation,
        permutation=permutation,
    )
    model._copy_weights(ref_model)
    model.fit_all_permutations()
    if model.rotation is not None:
        model.rotation.invalidate()
    model.to(device).eval()
    return model


def build_dry_transformer(dry_ref, dry_cfg, device, use_nvfp4, block_size,
                          rotation, permutation):
    """Build a TARGET transformer for `--dry` mode.

    Same as ``build_transformer`` but the random reference weights come from the
    in-process ``dry_ref`` (no checkpoint download). The new model copies the
    random weights from ``dry_ref`` via ``_copy_weights``, so in unquantized mode
    a target is mathematically identical to the reference (used to sanity-check
    the sweep plumbing in dry mode).
    """
    model = NVFP4QuantizedSana(
        use_nvfp4=use_nvfp4, block_size=block_size,
        rotation=rotation, permutation=permutation, **dry_cfg)
    model._copy_weights(dry_ref)
    model.fit_all_permutations()
    if model.rotation is not None:
        model.rotation.invalidate()
    model.to(device).eval()
    return model


def build_dry_pipeline(device, block_size, dry_dim=256, dry_layers=2, dry_resolution=64):
    """Build a ``SanaSprintPipeline`` with RANDOMLY-INITIALIZED components (no downloads).

    Used by ``--dry`` to exercise the full sweep / code path quickly without
    loading the 0.6B checkpoint. The transformer and VAE are real diffusers
    modules with RANDOM weights (and a small config); the text encoder / tokenizer
    are dummies and ``encode_prompt`` is overridden to return random caption
    embeddings of the correct shape.

    Returns ``(pipe, dry_ref, dry_cfg)`` where ``dry_ref`` is the (random)
    transformer also installed as ``pipe.transformer`` and ``dry_cfg`` is the
    small config dict used to build every target.
    """
    import types
    from diffusers import SanaSprintPipeline, SCMScheduler, AutoencoderDC

    caption_channels = 2304
    dry_cfg = dict(
        sample_size=32, patch_size=1, in_channels=32, out_channels=32,
        num_layers=dry_layers, attention_head_dim=32,
        num_attention_heads=max(1, dry_dim // 32),
        num_cross_attention_heads=max(1, dry_dim // 32),
        cross_attention_head_dim=32, cross_attention_dim=dry_dim,
        caption_channels=caption_channels, mlp_ratio=2.0,
        attention_bias=True, norm_elementwise_affine=False, norm_eps=1e-6,
        interpolation_scale=None, guidance_embeds=True, guidance_embeds_scale=0.1,
        qk_norm=None,
    )

    dry_ref = NVFP4QuantizedSana(
        use_nvfp4=False, block_size=block_size,
        rotation=IdentityRotation(block_size=block_size),
        permutation=lambda in_features: IdentityPermutation(block_size=block_size),
        **dry_cfg).to(device).eval()
    vae = AutoencoderDC().to(device).eval()   # random VAE, latent_channels=32
    scheduler = SCMScheduler()                # sigma_data=0.5

    class _DummyTokenizer:
        pass

    class _DummyTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

    pipe = SanaSprintPipeline(
        tokenizer=_DummyTokenizer(), text_encoder=_DummyTextEncoder().to(device),
        vae=vae, transformer=dry_ref, scheduler=scheduler)
    pipe.set_progress_bar_config(disable=True)

    # encode_prompt override: return random caption embeddings (no real text encoder).
    seq_len = 300

    def _dry_encode_prompt(self, prompt, num_images_per_prompt=1, device=None, **kwargs):
        if device is None:
            device = getattr(self, "_execution_device", "cpu")
        bs = 1 if isinstance(prompt, str) else len(prompt)
        B = bs * num_images_per_prompt
        embeds = torch.randn(B, seq_len, caption_channels, device=device, dtype=torch.float32)
        mask = torch.ones(B, seq_len, device=device, dtype=torch.float32)
        return embeds, mask

    pipe.encode_prompt = types.MethodType(_dry_encode_prompt, pipe)
    return pipe, dry_ref, dry_cfg


# ===========================================================================
# Multi-step decode loop (mirrors SanaSprintPipeline.__call__ denoising loop)
# ===========================================================================

def run_decode_loop(pipe, model, latent, txt, device, guidance_scale,
                    decode_steps, record_activation=False,
                    record_layer_outputs=False, record_linear_outputs=False,
                    include_cosine=False):
    """Decode ONE model from an initial `latent`, recording its intermediate states.

    Faithfully replicates ``SanaSprintPipeline.__call__``'s denoising loop for a
    single model:

        latents = latent                      # caller-supplied, shared across models
        for t in timesteps[:-1]:
            scm_t   = sin(t) / (cos(t) + sin(t))           # timestep fed to transformer
            x_in    = (latents/sigma_data) * sqrt(scm_t^2 + (1-scm_t)^2)
            oq      = model(x_in, txt, scm_t, guidance)[0]  # raw noise/velocity
            latents = scheduler.step(scm_postprocess(oq), t, latents)

    The transformer output at each step is a NOISE / VELOCITY prediction -- NOT a
    VAE latent -- so it cannot be decoded directly; it must go through the
    scheduler. We advance ``latents`` with THIS model's own prediction, so the
    final latent reflects this model and can be VAE-decoded to an image.

    Args:
        latent: initial latent (ALREADY scaled by sigma_data). It is supplied by
            the caller and shared across all models so that every model's decode
            starts from the SAME state -- the only varying factor is the model's
            weights/transforms. (We clone it internally; the caller's tensor is
            never mutated.)
        txt: per-sample caption embedding, CONSTANT across models for a sample
            (the encoding is unchanged). It is passed unchanged into every model.

    Returns ``(step_oqs, final_latents, step_acts, step_layer_oqs, step_linear_oqs)``:
      step_oqs[i]    = raw transformer output `oq` (tensor) at step i
      final_latents  = denoised / sigma_data  (the exact VAE input, pre-scaling)
      step_acts[i]   = {layer: {act_mse, act_mae}} from ``model`` (if requested)
      step_layer_oqs[i] = [(layer_name, layer_out), ...] (one (name, tensor) pair
          per transformer block) at step i, if ``record_layer_outputs`` is True
          (else []). ``layer_name`` is the block's qualified module name (e.g.
          "transformer_blocks.0"). Used to compute per-layer output diff vs the
          reference.
      step_linear_oqs[i] = [(module_name, out), ...] (one (name, tensor) pair per
          NVFP4Linear submodule), if ``record_linear_outputs`` is True (else []).
          ``module_name`` is the full qualified name (e.g.
          "transformer_blocks.0.attn1.to_q"). Used to compute per-NVFP4Linear
          output diff vs the reference.
    """
    scheduler = pipe.scheduler
    sigma_data = float(scheduler.config.sigma_data)
    guidance_embeds_scale = model.config.guidance_embeds_scale

    latents = latent.clone()

    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    guidance = guidance.expand(latents.shape[0]).to(txt.dtype) * guidance_embeds_scale

    # Timestep schedule (mirror retrieve_timesteps + set_begin_index in pipeline).
    scheduler.set_timesteps(
        decode_steps, device=device,
        max_timesteps=1.5708,
        intermediate_timesteps=(1.3 if decode_steps == 2 else None),
    )
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    timesteps = scheduler.timesteps[:-1]

    # IMPORTANT: the SCM scheduler injects random noise `z ~ N(0, I) * sigma_data`
    # between steps during multi-step inference. To make the per-step `oq`
    # comparison meaningful (reference vs target), EVERY model (reference AND
    # each target) must follow the SAME scheduler-noise trajectory, otherwise the
    # latents diverge after step 0 and the comparison measures RNG, not the
    # model. We therefore use a DETERMINISTIC, per-call reseeded generator so the
    # noise sequence is identical across all decode calls.
    _noise_gen = torch.Generator(device=device).manual_seed(1234)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(_noise_gen, 0.0)


    step_oqs = []
    step_acts = []
    step_layer_oqs = []
    step_linear_oqs = []

    # Optional: capture the output of EACH transformer block (per-layer output)
    # via forward hooks, so we can diff reference vs target layer-by-layer.
    #
    # The per-layer NAME used in the outputs is the block's qualified module name
    # (e.g. "transformer_blocks.0"). We take it from ``model.named_modules()`` so
    # it matches the real module hierarchy; if a block isn't found there we fall
    # back to "transformer_blocks.<i>".
    hooks = []
    _layer_buf = []   # list of (layer_name, out_tensor) in forward order
    if record_layer_outputs:
        blocks = getattr(model, "transformer_blocks", None)
        if blocks is not None:
            _mod2name = {id(m): n for n, m in model.named_modules()}
            for i, block in enumerate(blocks):
                bname = _mod2name.get(id(block), f"transformer_blocks.{i}")

                def _make_hook(name):
                    def _hook(m, inp, out):
                        _layer_buf.append((name, out))
                    return _hook

                hooks.append(block.register_forward_hook(_make_hook(bname)))

    # Optional: capture the output of EVERY NVFP4Linear module (per-linear output)
    # via forward hooks, so we can diff reference vs target at the finest granularity.
    # module_name is the full qualified name (e.g. "transformer_blocks.0.attn1.to_q").
    linear_hooks = []
    _linear_buf = []  # list of (module_name, out_tensor) in forward order
    if record_linear_outputs:
        for name, module in model.named_modules():
            if isinstance(module, NVFP4Linear):

                def _make_linear_hook(n):
                    def _hook(m, inp, out):
                        _linear_buf.append((n, out))
                    return _hook

                linear_hooks.append(module.register_forward_hook(_make_linear_hook(name)))

    with torch.no_grad():
        for t in timesteps:
            timestep = t.expand(latents.shape[0])
            latents_model_input = latents / sigma_data
            scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
            scm_te = scm_t.view(-1, 1, 1, 1)
            latent_model_input = latents_model_input * torch.sqrt(
                scm_te ** 2 + (1 - scm_te) ** 2)

            # Raw output (noise/velocity) of THIS model at this state.
            qinfo = {"_cosine": True} if (record_activation and include_cosine) else ({} if record_activation else None)
            _layer_buf.clear()
            _linear_buf.clear()
            oq = model(
                latent_model_input, txt, scm_t,
                guidance=guidance, quantization_error_info=qinfo, return_dict=False,
            )[0].float()
            step_oqs.append(oq)
            if record_activation:
                act, _ = parse_error_info(qinfo)
                step_acts.append(act)
            if record_layer_outputs:
                step_layer_oqs.append(
                    [(name, out.float()) for name, out in _layer_buf])
            if record_linear_outputs:
                step_linear_oqs.append(
                    [(name, out.float()) for name, out in _linear_buf])

            # SCM post-processing -> scheduler step (advance with THIS model).
            noise_pred = (
                (1 - 2 * scm_te) * latent_model_input
                + (1 - 2 * scm_te + 2 * scm_te ** 2) * oq
            ) / torch.sqrt(scm_te ** 2 + (1 - scm_te) ** 2)
            noise_pred = noise_pred * sigma_data
            latents, denoised = scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)

    for h in hooks:
        h.remove()
    for h in linear_hooks:
        h.remove()

    final_latents = denoised / sigma_data
    return step_oqs, final_latents, step_acts, step_layer_oqs, step_linear_oqs


def decode_latents(pipe, final_latents):
    """VAE-decode `final_latents` (as returned by ``run_decode_loop``) to the raw
    decoded tensor (PRE-postprocess, float32).

    Mirrors the ``pipe.vae.encode`` symmetry: the scheduler output is divided by
    the VAE ``scaling_factor`` before ``pipe.vae.decode``. The returned tensor is
    what ``image_processor.postprocess`` later turns into an RGB image, so it is
    the right space for the ``dec`` (decoded-feature) diff.
    """
    scaling_factor = pipe.vae.config.scaling_factor
    # Wrap in no_grad: the VAE parameters require grad, so decoding outside a
    # no_grad context would build an autograd graph and make the output require
    # grad, which then breaks ``image_processor.postprocess`` (it calls .numpy()).
    with torch.no_grad():
        dec = pipe.vae.decode(
            final_latents.to(pipe.vae.dtype) / scaling_factor, return_dict=False)[0]
    return dec.float()


def decode_latents_to_image(pipe, final_latents):
    """VAE-decode `final_latents` (as returned by ``run_decode_loop``) to a PIL image.

    Thin wrapper around ``decode_latents`` + ``image_processor.postprocess``; kept
    for IPython / external use. For the per-sample final diff the sweep computes the
    decoded tensor once (via ``decode_latents``) and reuses it for both the ``dec``
    and final-image diffs.
    """
    dec = decode_latents(pipe, final_latents)
    return pipe.image_processor.postprocess(dec, output_type="pil")[0]


def image_to_tensor(img):
    """Convert a PIL image to a float32 tensor in [0, 1] for pixel-level diff."""
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def make_labeled_grid(images_with_labels, thumb=256, bg=(255, 255, 255)):
    """Compose a single horizontal grid PIL image from a list of
    ``(label, PIL.Image)`` pairs, each thumbnail-ed to ``thumb`` px on the long
    side with its label drawn above. Used to put the SAME prompt's multiple
    configurations side-by-side for easy comparison.

    Returns a PIL.Image (RGB).
    """
    if not images_with_labels:
        return None
    # Normalize: accept either (label, img) tuples or bare images.
    pairs = []
    for item in images_with_labels:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            pairs.append((str(item[0]), item[1]))
        else:
            pairs.append(("", item))

    resized = []
    for label, img in pairs:
        img = img.convert("RGB")
        w, h = img.size
        scale = thumb / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        rimg = img.resize((nw, nh), Image.BILINEAR)
        resized.append((label, rimg))

    cell_w = max(r.size[0] for _, r in resized)
    cell_h = max(r.size[1] for _, r in resized)
    label_h = 22  # space reserved for the label strip above each cell

    grid_w = cell_w * len(resized)
    grid_h = label_h + cell_h
    grid = Image.new("RGB", (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x = 0
    for label, rimg in resized:
        # center the cell horizontally within its column
        off_x = (cell_w - rimg.size[0]) // 2
        grid.paste(rimg, (x + off_x, label_h))
        if label and font is not None:
            draw.text((x + 2, 2), label, fill=(0, 0, 0), font=font)
        x += cell_w
    return grid


# ===========================================================================
# Quantization error + metric helpers
# ===========================================================================

def compute_param_quant_errors(model, block_size, include_cosine=False):
    """Per-layer WEIGHT quantization error (sample-independent)."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Linear):
            prefix = module.layer_prefix or name
            wprefix = prefix + ".weight"
            err = {"_cosine": True} if include_cosine else {}
            NVFP4Quantization.apply(module._effective_weight(), block_size, err, wprefix)
            if err:
                result[prefix] = {
                    "weight_mse": err.get(wprefix + ".nvfp4_error_mse", 0.0),
                    "weight_mae": err.get(wprefix + ".nvfp4_error_mae", 0.0),
                    "weight_cosine": err.get(wprefix + ".nvfp4_error_cosine", 0.0),
                }
    return result


def parse_error_info(error_info):
    """Split a forward's quantization_error_info into (activation, parameter) maps."""
    act, param = {}, {}
    for k, v in error_info.items():
        if k.endswith("nvfp4_act_error_mse"):
            base = k[:-len("nvfp4_act_error_mse")].rstrip(".")
            act[base] = {
                "act_mse": v,
                "act_mae": error_info.get(base + ".nvfp4_act_error_mae", 0.0),
                "act_cosine": error_info.get(base + ".nvfp4_act_error_cosine", 0.0),
            }
        elif k.endswith("nvfp4_error_mse"):
            base = k[:-len("nvfp4_error_mse")].rstrip(".")
            param[base] = {
                "weight_mse": v,
                "weight_mae": error_info.get(base + ".nvfp4_error_mae", 0.0),
                "weight_cosine": error_info.get(base + ".nvfp4_error_cosine", 0.0),
            }
    return act, param


def mse(a, b):
    return F.mse_loss(a.float().reshape(-1), b.float().reshape(-1)).item()


def mae(a, b):
    return F.l1_loss(a.float().reshape(-1), b.float().reshape(-1)).item()


def cosine(a, b):
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()


# ===========================================================================
# Incremental results loading (skip already-computed configs)
# ===========================================================================

def load_existing_results(output_dir):
    """Load previously-computed sweep results from *output_dir* if they exist.

    Returns ``(results_dict, existing_configs_set)`` where *results_dict*
    contains the in-memory data structures ready to be extended by the
    current run.  Keys in *results_dict*:
        "out_target_error", "activation_errors", "param_quant_errors",
        "final_diff", "meta".
    """
    existing = {}
    existing_configs = set()

    # --- output_diff.json ---
    diff_path = os.path.join(output_dir, "output_diff.json")
    if os.path.isfile(diff_path):
        with open(diff_path, "r", encoding="utf-8") as f:
            diff_json = json.load(f)
        out_target = {}
        for cname, per_sample in diff_json.items():
            if cname == "final_diff":
                continue
            out_target[cname] = {}
            for idx_s, step_data in per_sample.items():
                out_target[cname][int(idx_s)] = step_data
            existing_configs.add(cname)
        existing["out_target_error"] = out_target

        fd = diff_json.get("final_diff", {})
        final_d = {}
        for cname, per_sample in fd.items():
            final_d[cname] = {}
            for idx_s, m in per_sample.items():
                final_d[cname][int(idx_s)] = m
        existing["final_diff"] = final_d

    # --- activation_errors.json ---
    act_path = os.path.join(output_dir, "activation_errors.json")
    if os.path.isfile(act_path):
        with open(act_path, "r", encoding="utf-8") as f:
            act_json = json.load(f)
        act = {}
        for idx_s, configs in act_json.items():
            act[int(idx_s)] = configs
        existing["activation_errors"] = act

    # --- param_quant_errors.json ---
    param_path = os.path.join(output_dir, "param_quant_errors.json")
    if os.path.isfile(param_path):
        with open(param_path, "r", encoding="utf-8") as f:
            existing["param_quant_errors"] = json.load(f)

    # --- sweep_metadata.json ---
    meta_path = os.path.join(output_dir, "sweep_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            existing["meta"] = json.load(f)
        for c in existing["meta"].get("configs", []):
            existing_configs.add(c)

    return existing, existing_configs


# ===========================================================================
# Cayley K caching (avoids re-calibrating for different permutations)
# ===========================================================================

def extract_cayley_K(model):
    """Extract per-layer Cayley K matrices from a calibrated model.

    Returns ``{layer_name: K_tensor}`` for every NVFP4Linear that has a
    CayleyRotation with a non-None K.  The K tensors are detached and moved
    to CPU for reuse across different model instances.
    """
    from src.modules.quantized_linear import NVFP4Linear as _NVL
    from src.quant_utils.rotation import CayleyRotation as _CR
    K_cache = {}
    for name, module in model.named_modules():
        if isinstance(module, _NVL):
            rot = getattr(module, 'rotation', None)
            if isinstance(rot, _CR) and rot.K is not None:
                K_cache[name] = rot.K.detach().cpu().clone()
    return K_cache


def apply_cayley_from_cache(model, K_cache):
    """Inject cached Cayley K matrices into a model in-place.

    For each NVFP4Linear layer whose name appears in *K_cache*, this function:
    1. Creates a NEW ``CayleyRotation`` (per-layer, independent of the shared
       placeholder rotation created by the model constructor).
    2. Sets its K from the cache.
    3. Replaces the module's ``.rotation`` attribute with the new instance.
    4. Rotates the module's weight: ``W_rot = W @ R(K)``, optionally padding
       the in_features dimension.
    """
    from src.modules.quantized_linear import NVFP4Linear as _NVL
    from src.quant_utils.rotation import CayleyRotation as _CR

    for name, module in model.named_modules():
        if not isinstance(module, _NVL) or name not in K_cache:
            continue
        new_rot = _CR(block_size=module.block_size)
        new_rot.K = K_cache[name].to(device=module.weight.device,
                                     dtype=module.weight.dtype)
        new_rot.invalidate()
        module.rotation = new_rot
        with torch.no_grad():
            w_rot = new_rot.rotate_weight(module.weight.data)
            w_n = new_rot._padded_dim(module.weight.data.shape[-1])
            w_pad = w_n - module.weight.data.shape[-1]
            if w_pad > 0:
                module.weight.data = torch.cat([
                    module.weight.data,
                    torch.zeros(module.weight.data.shape[0], w_pad,
                                device=module.weight.device,
                                dtype=module.weight.dtype),
                ], dim=-1)
            module.weight.data = w_rot


# ===========================================================================
# Saving
# ===========================================================================

def save_results(args, samples, out_target_error, activation_errors,
                 param_quant_errors, final_diff, missing_images, config_meta):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # output_diff.csv + .json  (per sample x combo x step x layer)
    # The CSV has one row per (sample, config, step, layer). The overall output
    # diff uses layer == "all"; per-transformer-block diffs use the layer index.
    # The end-to-end final diffs are appended as rows with step == "final" and
    # layer in {final_latents, dec, image}.
    rows = []
    od_json = {}
    for config_name, per_sample in out_target_error.items():
        od_json[config_name] = {}
        for idx, per_step in per_sample.items():
            od_json[config_name][str(idx)] = per_step
            for step, metrics in per_step.items():
                # Overall output (the transformer's final `oq`) -> layer "all".
                rows.append([idx, config_name, step, "all",
                             metrics["mse"], metrics["mae"], metrics["cosine"]])
                # Per-layer (per-transformer-block) output diff.
                for li, lmetrics in metrics.get("layers", {}).items():
                    rows.append([idx, config_name, step, li,
                                 lmetrics["mse"], lmetrics["mae"], lmetrics["cosine"]])
                # Per-NVFP4Linear output diff, keyed by full qualified module name.
                for li, lmetrics in metrics.get("linear_layers", {}).items():
                    rows.append([idx, config_name, step, li,
                                 lmetrics["mse"], lmetrics["mae"], lmetrics["cosine"]])
    # End-to-end final diffs (steps-independent): {final_latents, dec, image}.
    fd_json = {}
    for config_name, per_sample in final_diff.items():
        fd_json[config_name] = {}
        for idx, m in per_sample.items():
            fd_json[config_name][str(idx)] = m
            for kind in ("final_latents", "dec", "image"):
                km = m.get(kind)
                if km is None:
                    continue
                rows.append([idx, config_name, "final", kind,
                             km["mse"], km["mae"], km["cosine"]])
    # Store final diffs under the reserved top-level key "final_diff" so the
    # visualization script can enumerate configs from the other top-level keys
    # without treating this aggregate as a config.
    od_json["final_diff"] = fd_json
    with open(os.path.join(out_dir, "output_diff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "config", "step", "layer", "mse", "mae", "cosine"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "output_diff.json"), "w", encoding="utf-8") as f:
        json.dump(od_json, f, indent=2)

    # activation_errors.json  (per sample x combo x step x layer)
    with open(os.path.join(out_dir, "activation_errors.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in activation_errors.items()}, f, indent=2)

    # param_quant_errors.json  (per combo x layer, sample-independent)
    with open(os.path.join(out_dir, "param_quant_errors.json"), "w", encoding="utf-8") as f:
        json.dump(param_quant_errors, f, indent=2)

    meta = {
        "model_id": args.model_id,
        "n_samples": len(samples),
        "rots": args.rots, "perms": args.perms, "quantized_modes": args.quantized_modes,
        "block_size": args.block_size,
        "decode_steps": args.decode_steps,
        "guidance_scale": args.guidance_scale,
        "configs": list(out_target_error.keys()),
        "missing_images": missing_images,
    }
    meta.update(config_meta)
    with open(os.path.join(out_dir, "sweep_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-sample transformer output error + per-layer quantization "
                    "error across (rotation x permutation x quant) configurations.")
    # Dataset / prompts
    p.add_argument("--dataset_name", type=str, default="MJHQ-30K")
    p.add_argument("--dataset_path", type=str, default="G://datasets/MJHQ-30K")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--sampling", type=str, default="sequential",
                   choices=["sequential", "random"])
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--seed_start", type=int, default=42)
    # Model
    p.add_argument("--model_id", type=str,
                   default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    p.add_argument("--cache_dir", type=str, default="G://models")
    p.add_argument("--download_source", type=str, default="modelscope",
                   choices=["modelscope", "huggingface"])
    p.add_argument("--block_size", type=int, default=16)
    # Sweep axes
    p.add_argument("--rots", type=str, default="none,hadamard,random",
                   help="Comma-separated rotations: none,hadamard,random,cayley")
    p.add_argument("--perms", type=str, default="none,identity,random,mag",
                   help="Comma-separated permutations: none,identity,random,mag")
    p.add_argument("--quantized_modes", type=str, default="quantized",
                   help="One of: quantized, unquantized (run separate invocations "
                        "for each, possibly with different --output_dir).")
    # -- Cayley calibration --
    p.add_argument("--cayley_calib", action="store_true",
                   help="For Cayley rotations, learn K from activations (calls "
                        "fit_activation on each layer). Without this flag, Cayley "
                        "uses K=0 (equivalent to identity).")
    p.add_argument("--cayley_calib_iters", type=int, default=100,
                   help="Adam iterations per layer during Cayley calibration (halved "
                        "from the original 200; still provides good convergence for "
                        "most layers while being 2x faster).")
    p.add_argument("--cayley_calib_lr", type=float, default=1e-2,
                   help="Learning rate for Cayley calibration.")
    p.add_argument("--cayley_calib_batches", type=int, default=8,
                   help="Number of synthetic batches for activation collection.")
    p.add_argument("--cayley_calib_dataset_path", type=str, default=None,
                   help="Path to calibration dataset for Cayley activation calibration. "
                        "If provided, uses real prompts from this dataset instead of "
                        "synthetic inputs.")
    p.add_argument("--cayley_calib_n_samples", type=int, default=30,
                   help="Number of prompts to use from the calibration dataset.")
    p.add_argument("--no_layer_diff", action="store_true",
                   help="Disable per-transformer-block output diff (saves memory "
                        "by not keeping layer outputs in output_diff.json). The "
                        "overall output diff is still recorded.")
    p.add_argument("--no_linear_diff", action="store_true",
                   help="Disable per-NVFP4Linear output diff (saves memory by "
                        "not keeping every NVFP4Linear's output). The overall "
                        "output diff and block-level diff are still recorded.")
    p.add_argument("--cosine", action="store_true",
                   help="Also compute per-layer cosine similarity errors "
                        "(adds overhead; off by default for speed).")
    # Transform seeds / permutation metric
    p.add_argument("--rot_seed", type=int, default=0)
    p.add_argument("--perm_seed", type=int, default=0)
    p.add_argument("--perm_metric", type=str, default="norm", choices=["norm", "max"])
    p.add_argument("--perm_order", type=str, default="asc", choices=["asc", "desc"])
    # Inputs
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--decode_steps", type=int, default=2,
                   help="Number of denoising (scheduler) steps in the decode loop. "
                        "Sana Sprint uses 2. The sweep runs this many steps per "
                        "sample and records the output/activation error at EACH step.")
    p.add_argument("--guidance_scale", type=float, default=4.5)
    # Misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dry", action="store_true",
                   help="DRY MODE: build a tiny RANDOMLY-INITIALIZED pipeline "
                        "(transformer + VAE random, no checkpoint download) so the "
                        "whole sweep / code path runs in seconds to test integrity. "
                        "Results are NOT meaningful (random weights); use it to "
                        "verify the plumbing, transforms, quantization and saving.")
    p.add_argument("--dry_resolution", type=int, default=64,
                   help="Image/latent resolution used in --dry mode (small = fast).")
    p.add_argument("--dry_dim", type=int, default=256,
                   help="Transformer hidden dim in --dry mode (small = fast).")
    p.add_argument("--dry_layers", type=int, default=2,
                   help="Number of transformer layers in --dry mode (small = fast).")
    p.add_argument("--generate_images", action="store_true",
                   help="After the sweep, VAE-decode each config's final latent "
                        "(including the reference) into images, written to "
                        "<output_dir>/generated_images/<config_name>/. This verifies "
                        "the whole parameter-loading chain for every combo.")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("MODEL_CACHE_DIR", cache_dir)

    # --- Load samples (prompt + image path) ---
    if args.dry and not os.path.isfile(os.path.join(args.dataset_path, "meta_data.json")):
        # DRY MODE without the dataset present: synthesize dummy samples so the
        # full sweep still runs to verify code/plumbing integrity.
        samples = [{
            "image_id": f"dry_{i:05d}",
            "prompt": f"a dry-mode synthetic test prompt number {i}",
            "category": "dry",
        } for i in range(max(1, args.n_samples))]
        print(f"[DRY MODE] Dataset not found at {args.dataset_path}; "
              f"using {len(samples)} synthetic samples.")
    else:
        samples = load_samples(
            args.dataset_name, args.dataset_path,
            n_samples=args.n_samples, sampling=args.sampling,
            sample_start=args.sample_start, seed=args.seed_start,
        )
    num_images = len(samples)
    print(f"Loaded {num_images} samples from {args.dataset_name}")

    # Save run arguments
    with open(os.path.join(args.output_dir, "arguments.txt"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    block_size = args.block_size
    dry_mode = args.dry
    if dry_mode:
        args.resolution = args.dry_resolution

    # --- Build base pipeline (origin) to get VAE + text encoder for inputs ---
    if dry_mode:
        print("\n[DRY MODE] Building randomly-initialized pipeline (NO downloads) ...")
        pipe, dry_ref, dry_cfg = build_dry_pipeline(
            device, block_size, dry_dim=args.dry_dim,
            dry_layers=args.dry_layers, dry_resolution=args.dry_resolution)
        base_model_ref = pipe.transformer  # the random dry_ref
        ref_model = None                   # not used in dry mode
        in_channels = dry_cfg["in_channels"]
        sample_size = dry_cfg["sample_size"]
        print(f"  DRY transformer: layers={args.dry_layers}, dim={args.dry_dim}, "
              f"resolution={args.dry_resolution}, "
              f"caption_channels={dry_cfg['caption_channels']}")
    else:
        print(f"\nLoading base pipeline: {args.model_id}")
        pipe = load_pipeline(
            args.model_id, cache_dir, False, block_size, device,
            download_source=args.download_source,
            scheduler_type=None, scheduler_config=None,
            rotation=IdentityRotation(block_size=block_size),
            permutation=lambda in_features: IdentityPermutation(block_size=block_size),
        )
        base_model_ref = pipe.transformer  # NVFP4QuantizedSana(use_nvfp4=False)

        # --- Resolve + load the reference diffusers transformer once (reused) ---
        local_path = NVFP4QuantizedSana._resolve_checkpoint_path(
            args.model_id, download_source=args.download_source, cache_dir=cache_dir)
        from diffusers import SanaTransformer2DModel
        ref_model = SanaTransformer2DModel.from_pretrained(
            local_path, subfolder="transformer", local_files_only=True)

        # Latent shape from the model config (metadata / sanity only).
        in_channels = int(ref_model.config.get("in_channels", 32))
        sample_size = int(ref_model.config.get("sample_size", 32))

    # --- Precompute per-sample prompt embeddings (the only per-sample input; the
    #     decode loop starts from a seeded random noise latent, no VAE/images). ---
    print("\nEncoding per-sample prompt embeddings ...")
    txts = [encode_prompt_embeds(pipe, s["prompt"], device) for s in tqdm(
        samples, desc="Encoding", unit="sample", dynamic_ncols=True)]
    missing_images = []

    # The UNQUANTIZED transformer (pipe.transformer) is the reference every combo
    # is compared against. Keep it alive for the whole sweep; do NOT delete it.
    base_model_ref = base_model_ref.to(device).eval()
    print(f"  decode_steps={args.decode_steps}, guidance={args.guidance_scale}, "
          f"resolution={args.resolution}, n_samples={num_images}")

    # --- Reference decode (ONCE per sample) ---
    # Every target model below starts from the SAME seeded latent, so the only
    # varying factor across combos is the (rotation, permutation, quantize) combo.
    # We decode the reference model once per sample and keep its intermediate
    # states (per-step raw outputs `oq` + final latent) for later comparison.
    sigma_data = float(pipe.scheduler.config.sigma_data)

    rotations = [s.strip() for s in args.rots.split(",") if s.strip()]
    perms = [s.strip() for s in args.perms.split(",") if s.strip()]
    use_nvfp4 = (args.quantized_modes == "quantized")

    print("\nDecoding reference model (once per sample) ...")
    latent0_list = {}   # idx -> shared initial latent (already * sigma_data)
    ref_states = {}     # idx -> {"oqs", "final", "layer_oqs", "linear_oqs", "dec", "image_t"}
    record_layer_outputs = not getattr(args, "no_layer_diff", False)
    record_linear_outputs = not getattr(args, "no_linear_diff", False)
    for idx, txt in enumerate(tqdm(
            txts, desc="Ref", unit="sample", dynamic_ncols=True)):
        gen = torch.Generator(device=device).manual_seed(args.seed_start + idx)
        latent0 = pipe.prepare_latents(
            1, in_channels, args.resolution, args.resolution,
            torch.float32, device, gen, None) * sigma_data
        latent0_list[idx] = latent0
        step_oqs, ref_final, _, step_layer_oqs, step_linear_oqs = run_decode_loop(
            pipe, base_model_ref, latent0, txt, device, args.guidance_scale,
            args.decode_steps, record_activation=False,
            record_layer_outputs=record_layer_outputs,
            record_linear_outputs=record_linear_outputs)
        # Decode the reference final latent once -> decoded feature (`dec`) and
        # a float image tensor (`image_t`) for the end-to-end final diffs. Both
        # are reused by every target combo (ref is shared across combos).
        ref_dec = decode_latents(pipe, ref_final)
        ref_img = pipe.image_processor.postprocess(ref_dec, output_type="pil")[0]
        ref_states[idx] = {"oqs": step_oqs, "final": ref_final,
                           "layer_oqs": step_layer_oqs,
                           "linear_oqs": step_linear_oqs,
                           "dec": ref_dec, "image_t": image_to_tensor(ref_img)}

        # Optional: save the reference image to disk. Images for the SAME prompt
        # (sample idx) are grouped into one folder `generated_images/{idx:04d}/`
        # so that the reference and every configuration live together; a combined
        # montage `grid_{idx:04d}.png` is built at the end of the sweep.
        if args.generate_images:
            sample_img_dir = os.path.join(args.output_dir, "generated_images", f"{idx:04d}")
            sample_img_dir = os.path.join(args.output_dir, "generated_images")
            os.makedirs(sample_img_dir, exist_ok=True)
            ref_img.save(os.path.join(sample_img_dir, "ref.png"))

    # --- Load existing results to skip already-computed configs ---
    existing_results, existing_configs = load_existing_results(args.output_dir)
    if existing_configs:
        print(f"\nFound {len(existing_configs)} existing config(s) in output_dir, will skip:")
        for c in sorted(existing_configs):
            print(f"  [skip] {c}")

    # Count new configs to compute (always include q/uq suffix for clarity).
    _new_config_names = []
    # suffix = "q" if use_nvfp4 else "uq"
    for pn in perms:
        for rn in rotations:
            # _cn = f"{rn}_{pn}_{suffix}"
            _cn = f"{rn}_{pn}"
            if _cn not in existing_configs:
                _new_config_names.append(_cn)

    if not _new_config_names and existing_configs:
        print(f"\nAll {len(existing_configs)} configs already computed in "
              f"{args.output_dir}. Nothing to do.")
        sys.exit(0)

    if _new_config_names:
        print(f"  {len(_new_config_names)} new config(s) to compute: {_new_config_names}")

    # --- Sweep over (permutation x rotation) ---
    # Rotation is the INNERMOST loop so that Cayley K matrices (which are
    # permutation-independent) can be calibrated ONCE and reused across all
    # permutations, avoiding redundant expensive optimization.
    #
    # Loop order:  for each permutation:
    #                for each rotation:
    #                  build model, calibrate (or reuse cached K), eval
    #
    # Initialize accumulators from existing data (extended by the sweep below).
    out_target_error = existing_results.get("out_target_error", {})
    activation_errors = existing_results.get("activation_errors", {})
    param_quant_errors = existing_results.get("param_quant_errors", {})
    final_diff = existing_results.get("final_diff", {})
    all_config_names = existing_results.get("meta", {}).get("configs", [])
    if not all_config_names:
        all_config_names = list(existing_configs)

    # Per-layer Cayley K cache (shared across permutations, reset per run).
    cayley_K_cache = {}

    for pn in perms:
        perm_factory = make_perm_factory(
            pn, block_size, args.perm_seed, args.perm_metric, args.perm_order)
        for rn in rotations:
            rot = make_rotation(rn, block_size, args.rot_seed)
            config_name = f"{rn}_{pn}_{suffix}"
            # Skip already-computed configs (loaded from output_dir)
            if config_name in existing_configs:
                print(f"\n=== Config: {config_name} (SKIP: already computed) ===")
                continue
            if config_name not in all_config_names:
                all_config_names.append(config_name)
            print(f"\n=== Config: {config_name} (use_nvfp4={use_nvfp4}) ===")

            # Build a fresh transformer with this (rotation, permutation,
            # quantize) setting, reusing the cached ref_model (real mode) or
            # the in-process random dry_ref (--dry mode).
            if dry_mode:
                model = build_dry_transformer(
                    dry_ref, dry_cfg, device, use_nvfp4, block_size, rot, perm_factory)
            else:
                model = build_transformer(
                    ref_model, device, use_nvfp4, block_size, rot, perm_factory)

            # ---- Cayley activation calibration (with K-caching) ----
            if rn == "cayley" and args.cayley_calib:
                if cayley_K_cache:
                    # Reuse previously calibrated K matrices (same for any
                    # permutation — Cayley only depends on the layer dimension
                    # and activation statistics, not on the permutation).
                    print(f"  [Cayley] Reusing cached K from earlier calibration "
                          f"({len(cayley_K_cache)} layers)")
                    t_cached = time.time()
                    apply_cayley_from_cache(model, cayley_K_cache)
                    model.fit_all_permutations()
                    print(f"  [Cayley] Cache apply + re-fit done "
                          f"({time.time() - t_cached:.2f}s)")
                else:
                    # Prepare calibration text embeddings
                    if args.cayley_calib_dataset_path:
                        # Load calibration dataset prompts
                        calib_samples = load_samples(
                            args.dataset_name, args.cayley_calib_dataset_path,
                            n_samples=args.cayley_calib_n_samples,
                            sampling="random", seed=42,
                        )
                        calib_txts = [encode_prompt_embeds(pipe, s["prompt"], device)
                                      for s in tqdm(calib_samples, desc="Encoding calib prompts",
                                                    unit="sample", dynamic_ncols=True)]
                        print(f"  [Cayley] Using {len(calib_txts)} prompts from calibration dataset")
                        calib_n_batches = max(len(calib_txts), args.cayley_calib_batches)
                    else:
                        # Use synthetic calibration with first sample's embedding
                        calib_txts = [txts[0]]
                        calib_n_batches = args.cayley_calib_batches
                        print(f"  [Cayley] Using synthetic calibration with 1 prompt")

                    print(f"  [Cayley] Running activation calibration "
                          f"(iters={args.cayley_calib_iters}, batches={calib_n_batches})...")
                    calib_loader = make_calibration_loader(
                        pipe=pipe,
                        txt_embeds=calib_txts,
                        in_channels=in_channels,
                        resolution=args.resolution,
                        device=device,
                        n_batches=calib_n_batches,
                    )
                    t_calib_start = time.time()
                    calibrate_activation_rotations(
                        model, calib_loader,
                        forward_fn=lambda m, batch: m(**batch),
                        iters=args.cayley_calib_iters,
                        lr=args.cayley_calib_lr,
                        max_calib_batches=calib_n_batches,
                        init_from_hadamard=True,
                        verbose=True,
                    )
                    t_calib = time.time() - t_calib_start
                    # Re-fit permutations after rotation changed the weights
                    model.fit_all_permutations()
                    # Extract K for reuse across other permutations
                    cayley_K_cache = extract_cayley_K(model)
                    print(f"  [Cayley] Calibration done in {t_calib:.1f}s "
                          f"({len(cayley_K_cache)} layers cached for reuse)")

            if use_nvfp4:
                param_quant_errors[config_name] = compute_param_quant_errors(
                    model, block_size, include_cosine=args.cosine)

            out_target_error[config_name] = {}
            with torch.no_grad():
                for idx, txt in enumerate(tqdm(
                        txts, desc=config_name, unit="sample", dynamic_ncols=True)):
                    # Same initial latent as the reference -> consistent decode
                    # inputs across models (only the model itself varies).
                    latent0 = latent0_list[idx]
                    step_oqs, final_latents, step_acts, step_layer_oqs, step_linear_oqs = run_decode_loop(
                        pipe, model, latent0, txt, device,
                        args.guidance_scale, args.decode_steps,
                        record_activation=use_nvfp4,
                        record_layer_outputs=record_layer_outputs,
                        record_linear_outputs=record_linear_outputs,
                        include_cosine=args.cosine)
                    ref_oqs = ref_states[idx]["oqs"]
                    ref_layer_oqs = ref_states[idx]["layer_oqs"]
                    ref_linear_oqs = ref_states[idx]["linear_oqs"]
                    # Compare each step's raw output to the reference.
                    per_step = {}
                    for s in range(len(step_oqs)):
                        per_step[str(s)] = {
                            "mse": mse(ref_oqs[s], step_oqs[s]),
                            "mae": mae(ref_oqs[s], step_oqs[s]),
                            "cosine": cosine(ref_oqs[s], step_oqs[s]),
                        }
                        # Per-layer (per-transformer-block) output diff, keyed
                        # by the block's qualified module name.
                        if record_layer_outputs and step_layer_oqs and ref_layer_oqs:
                            layers = {}
                            for (rname, rL), (_tname, tL) in zip(
                                    ref_layer_oqs[s], step_layer_oqs[s]):
                                layers[rname] = {
                                    "mse": mse(rL, tL),
                                    "mae": mae(rL, tL),
                                    "cosine": cosine(rL, tL),
                                }
                            per_step[str(s)]["layers"] = layers
                        # Per-NVFP4Linear output diff, keyed by the module's
                        # full qualified name (e.g. "transformer_blocks.0.attn1.to_q").
                        if record_linear_outputs and step_linear_oqs and ref_linear_oqs:
                            linear_layers = {}
                            for (rname, rL), (_tname, tL) in zip(
                                    ref_linear_oqs[s], step_linear_oqs[s]):
                                linear_layers[rname] = {
                                    "mse": mse(rL, tL),
                                    "mae": mae(rL, tL),
                                    "cosine": cosine(rL, tL),
                                }
                            per_step[str(s)]["linear_layers"] = linear_layers
                    out_target_error[config_name][idx] = per_step
                    if use_nvfp4:
                        activation_errors.setdefault(idx, {})[config_name] = {
                            str(s): a for s, a in enumerate(step_acts)}

                    # --- End-to-end FINAL diff vs the reference ---
                    # Decode this combo's final latent once -> the VAE-decoded
                    # feature (`dec`) and the final RGB image, then diff each
                    # of {final_latents, dec, image} against the reference.
                    # These capture the WHOLE pipeline error (last-step
                    # prediction + scheduler + VAE decode), complementary to
                    # the per-step transformer-output error above.
                    ref_final_latents = ref_states[idx]["final"]
                    ref_dec = ref_states[idx]["dec"]
                    ref_img_t = ref_states[idx]["image_t"]
                    tgt_dec = decode_latents(pipe, final_latents)
                    tgt_img = pipe.image_processor.postprocess(
                        tgt_dec, output_type="pil")[0]
                    tgt_img_t = image_to_tensor(tgt_img)
                    final_diff.setdefault(config_name, {})[idx] = {
                        "final_latents": {
                            "mse": mse(ref_final_latents, final_latents),
                            "mae": mae(ref_final_latents, final_latents),
                            "cosine": cosine(ref_final_latents, final_latents),
                        },
                        "dec": {
                            "mse": mse(ref_dec, tgt_dec),
                            "mae": mae(ref_dec, tgt_dec),
                            "cosine": cosine(ref_dec, tgt_dec),
                        },
                        "image": {
                            "mse": mse(ref_img_t, tgt_img_t),
                            "mae": mae(ref_img_t, tgt_img_t),
                            "cosine": cosine(ref_img_t, tgt_img_t),
                        },
                    }

                    # Optional: save the generated image to disk (verifies the
                    # whole parameter-loading chain for this combo). Save it
                    # into the SAME per-sample folder as the reference so the
                    # same prompt's configurations live together; a combined
                    # montage for this sample is assembled after the sweep.
                    if args.generate_images:
                        sample_img_dir = os.path.join(args.output_dir, "generated_images")
                        os.makedirs(sample_img_dir, exist_ok=True)
                        tgt_img.save(os.path.join(sample_img_dir, f"{config_name}.png"))

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # --- Assemble per-prompt montage grids (same prompt, all configs together) ---
    # For each sample idx, compose `ref` + every config into one horizontal grid
    # `grid_{idx:04d}.png` inside `generated_images/`, so the multiple configs of
    # the same prompt are visually placed side-by-side instead of in separate
    # folders. The individual `ref.png` / `{config}.png` files are kept alongside.
    if args.generate_images:
        print("\nComposing per-prompt montage grids ...")
        for idx in range(num_images):
            sample_img_dir = os.path.join(args.output_dir, "generated_images")
            ref_path = os.path.join(sample_img_dir, "ref.png")
            if not os.path.isfile(ref_path):
                continue
            cells = [("ref", Image.open(ref_path))]
            for cn in all_config_names:
                cp = os.path.join(sample_img_dir, f"{cn}.png")
                if os.path.isfile(cp):
                    cells.append((cn, Image.open(cp)))
            grid = make_labeled_grid(cells)
            if grid is not None:
                grid.save(os.path.join(sample_img_dir, f"grid_{idx:04d}.png"))

    # --- Save everything ---
    save_results(
        args, samples, out_target_error, activation_errors,
        param_quant_errors, final_diff, missing_images,
        config_meta={"in_channels": in_channels, "sample_size": sample_size},
    )
