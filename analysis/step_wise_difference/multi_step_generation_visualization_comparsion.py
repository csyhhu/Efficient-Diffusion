"""One-step vs Multi-step generation visualization comparison.

Pipeline:
  1. Always run a full multi-step baseline (num_steps, no skip).
  2. Run one or more ``dit_step_patterns``, each specifying which steps run DiT.
     Steps not in the list reuse the cached noise_pred of the most recent DiT step.
  3. Combine all runs into one figure: one subplot per run, with caption
     labeling the pattern (e.g. "full", "step 0", "steps 0,2").

``gen.generate`` returns ``(images, intermediates_recorder)`` where
``intermediates_recorder["final_output"]`` is the final image tensor.
The relative L2 vs the full baseline is computed inline (no helper function).

Usage:
  python -m analysis.step_wise_difference.multi_step_generation_visualization_comparsion `
    --model_id Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers `
    --prompt "A cute cat" `
    --output_dir G://Outputs//Efficient-Diffusion//multi_step_compare//Sana//2-step

  python -m analysis.step_wise_difference.multi_step_generation_visualization_comparsion `
    --model_id stabilityai/stable-diffusion-3.5-medium `
    --use_origin_model `
    --prompt "A cute cat" `
    --num_steps 28 `
    --dit_step_patterns "0" "0,1,27" "0,1,26,27" `
    --output_dir G://Outputs//Efficient-Diffusion//multi_step_compare//SD3_origin//28-step

  python -m analysis.step_wise_difference.multi_step_generation_visualization_comparsion `
    --model_id stabilityai/stable-diffusion-3.5-medium `
    --use_origin_model `
    --plans_json G://Outputs//Efficient-Diffusion//step_wise_output//SD3-origin-MJHQ30K//analysis//recommended_skip_plans.json `
    --prompt "A cute cat" `
    --num_steps 28 `
    --output_dir G://Outputs//Efficient-Diffusion//multi_step_compare//SD3_origin//28-step

This produces a single comparison image containing 4 subplots:
  - full   (baseline, all 4 steps run DiT)
  - step 0 (only step 0 runs DiT)
  - steps 0,2
  - steps 0,1,3
"""

import os
import json
import argparse
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.image_generator import SanaImageGenerator, SD3ImageGenerator


def parse_pattern(s):
    """Parse '0,2' into [0, 2] (sorted unique)."""
    steps = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    return sorted(set(steps))


def pattern_label(steps):
    """Human-readable label: 'full' if None, else 'step 0' / 'steps 0,2'."""
    if steps is None:
        return "full"
    if len(steps) == 1:
        return f"step {steps[0]}"
    return "steps " + ",".join(str(s) for s in steps)


def pattern_caption(steps, rel_l2=None):
    """Multi-line caption for a subplot."""
    if steps is None:
        cap = "full (all steps run DiT)"
    else:
        cap = f"dit_inference_steps = {steps}"
    if rel_l2 is not None:
        cap += f"\nrel_l2 vs full = {rel_l2:.4f}"
    return cap


def save_comparison_grid(images_captions, save_path):
    """Arrange images and captions into a single matplotlib figure."""
    n = len(images_captions)
    nrow = min(4, n)
    ncol = (n + nrow - 1) // nrow

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 5 * nrow))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, (img, caption) in zip(axes, images_captions):
        img_np = img.detach().cpu().float().squeeze(0).numpy()  # [C, H, W]
        if img_np.shape[0] == 1:
            img_np = img_np[0]
            cmap = "gray"
        else:
            img_np = img_np.transpose(1, 2, 0)
            img_np = (img_np + 1) / 2  # [-1,1] -> [0,1]
            img_np = img_np.clip(0, 1)
            cmap = None
        ax.imshow(img_np, cmap=cmap)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="One-step vs Multi-step generation comparison"
    )
    parser.add_argument("--model_id", type=str,
                        default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    parser.add_argument("--use_origin_model", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--prompt", type=str,
                        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--use_nvfp4", action="store_true")
    parser.add_argument("--rotation", type=str, default="identity")
    parser.add_argument("--permutation", type=str, default="identity")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dit_step_patterns", nargs="+", default=None,
                        help="One or more DiT step patterns, e.g. '0' '0,2' '0,1,3'. "
                             "Each becomes one subplot. The full baseline is always included. "
                             "Ignored when --plans_json is provided.")
    parser.add_argument("--plans_json", type=str, default=None,
                        help="Path to recommended_skip_plans.json from analyze_step_wise_difference.py. "
                             "When provided, all plans in the JSON are loaded as patterns. "
                             "Overrides --dit_step_patterns.")
    parser.add_argument("--output_dir", type=str,
                        default="G://Outputs//Efficient-Diffusion//step_wise_output//compare_grid")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build generator ----
    if args.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
        gen = SanaImageGenerator(
            model_id=args.model_id,
            use_origin_model=args.use_origin_model,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=torch.bfloat16,
            use_nvfp4=args.use_nvfp4, block_size=args.block_size,
            rotation=args.rotation, permutation=args.permutation,
        )
    elif args.model_id == "stabilityai/stable-diffusion-3.5-medium":
        gen = SD3ImageGenerator(
            model_id=args.model_id,
            use_origin_model=args.use_origin_model,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=torch.bfloat16,
            use_nvfp4=args.use_nvfp4, block_size=args.block_size,
            rotation=args.rotation, permutation=args.permutation,
        )
    else:
        raise ValueError(f"Unknown model ID: {args.model_id}")

    # ---- Build run list: full baseline + patterns from JSON or CLI ----
    patterns = [(None, "full")]

    if args.plans_json is not None:
        # Load all plans from recommended_skip_plans.json
        with open(args.plans_json, "r", encoding="utf-8") as f:
            plans_data = json.load(f)
        plans_dict = plans_data.get("plans", {})
        for key, val in plans_dict.items():
            steps = val["dit_inference_steps"]
            n_dit = val.get("n_dit_steps", len(steps))
            dp_cost = val.get("dp_cost", None)
            label = f"{key} (n={n_dit}, cost={dp_cost:.2f})" if dp_cost is not None else f"{key} (n={n_dit})"
            patterns.append((steps, label))
        print(f"[plans_json] loaded {len(plans_dict)} plans from {args.plans_json}")
    elif args.dit_step_patterns is not None:
        for ps in args.dit_step_patterns:
            steps = parse_pattern(ps)
            patterns.append((steps, pattern_label(steps)))

    # Filter out duplicates (e.g. user passes '0,1,2,3' for num_steps=4 == full)
    seen = set()
    unique_patterns = []
    for steps, label in patterns:
        key = tuple(steps) if steps is not None else None
        if key in seen:
            continue
        seen.add(key)
        unique_patterns.append((steps, label))
    patterns = unique_patterns

    print(f"[config] prompt:  {args.prompt}")
    print(f"[config] num_steps={args.num_steps}, patterns ({len(patterns)}):")
    for steps, label in patterns:
        steps_str = "ALL" if steps is None else str(steps)
        print(f"  - {label:25s}: dit_inference_steps = {steps_str}")

    # Common kwargs forwarded to gen.generate for every run
    gen_kwargs = dict(
        prompt=args.prompt,
        num_samples=1,
        seed=args.seed,
        num_steps=args.num_steps,
        # return_intermediates=True,
        # guidance_scale=args.guidance,
    )

    # ---- Step 1: full baseline (no skip) ----
    print("\n[Step 1] full baseline (no skip)")
    full_image = gen.generate(**gen_kwargs)
    full_final_output = full_image.float()
    # print(f"  [gen] full baseline -> image shape {tuple(full_image.shape)}")

    images_captions = [(full_image, pattern_caption(None, None))]
    metrics = {"full": {"dit_inference_steps": None, "rel_l2_vs_full": 0.0}}

    # ---- Step 2: each dit_step_pattern ----
    print("\n[Step 2] each dit_step_pattern")
    for steps, label in patterns:
        if steps is None:
            continue  # already done as baseline

        img = gen.generate(dit_inference_steps=steps, **gen_kwargs)
        final_latent = img.float()
        # Inline relative L2 vs full baseline (no helper function)
        diff = (final_latent - full_final_output).norm().item()
        rel_l2 = diff / (full_final_output.norm().item() + 1e-8)

        print(f"  [gen] {label:25s} -> image shape {tuple(img.shape)}, rel_l2={rel_l2:.4f}")
        images_captions.append((img, pattern_caption(steps, rel_l2)))
        metrics[label] = {"dit_inference_steps": steps, "rel_l2_vs_full": rel_l2}

    # ---- Step 3: combined comparison image ----
    print("\n[Step 3] saving combined comparison image")
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    save_comparison_grid(images_captions, grid_path)
    print(f"[save] image grid -> {grid_path}")

    # ---- Save metrics.json ----
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "prompt": args.prompt,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "patterns": [{"label": lbl,
                          "dit_inference_steps": (None if st is None else list(st)),
                          "rel_l2_vs_full": metrics[lbl]["rel_l2_vs_full"]}
                         for st, lbl in patterns],
        }, f, indent=2, ensure_ascii=False)
    print(f"[save] metrics    -> {metrics_path}")
    print(f"\n[done] comparison complete -> {args.output_dir}")
