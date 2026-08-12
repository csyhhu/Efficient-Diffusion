"""Save step-wise intermediate outputs (dit_outputs / noise_preds /
scheduler_outputs) for step-wise difference analysis.

Workflow:
  1. ``generate(dataset_name=..., return_intermediates=True)`` runs the
     dataset-mode loop: it loads prompts, generates one image per prompt
     (saved as ``00000.png`` ... under ``save_root``), and returns a list
     of per-prompt intermediates.
  2. This script saves each entry as ``sample_XXXX.pt`` containing
     ``dit_outputs`` / ``noise_preds`` / ``scheduler_outputs`` (per-step
     tensor lists), plus ``num_steps`` / ``prompt`` / ``seed`` metadata,
     matching the format expected by ``analyze_step_wise_difference.py``.

Run from the repository root using ``python -m``:

    python -m analysis.step_wise_difference.save_step_wise_output \\
        --model_id "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" \\
        --dataset_name mjhq30k --dataset_path "G://datasets/MJHQ-30K" \\
        --n_samples 10 --num_steps 2 --seed 42 \\
        --output_dir "G://Outputs//Efficient-Diffusion//step_wise_output//Sana-origin-MJHQ30K"

    python -m analysis.step_wise_difference.save_step_wise_output `
        --model_id "stabilityai/stable-diffusion-3.5-medium" `
        --use_origin_model `
        --dataset_name mjhq30k --dataset_path "G://datasets/MJHQ-30K" `
        --n_samples 10 --num_steps 28 --seed 42 `
        --output_dir "G://Outputs//Efficient-Diffusion//step_wise_output//SD3-origin-MJHQ30K"

"""

import argparse
import json
import os

import torch

from src.image_generator import SanaImageGenerator, SD3ImageGenerator


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Save step-wise dit_outputs / noise_preds for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_id", type=str,
        default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    )
    parser.add_argument("--use_origin_model", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--dataset_name", type=str, default="mjhq30k")
    parser.add_argument("--dataset_path", type=str, default="G://datasets/MJHQ-30K")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--use_nvfp4", action="store_true")
    parser.add_argument("--rotation", type=str, default="identity",
                        choices=["identity", "hadamard", "random", "cayley"])
    parser.add_argument("--permutation", type=str, default="identity",
                        choices=["identity", "random", "mag"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str,
                        default="G://Outputs//Efficient-Diffusion//step_wise_output//Sana-MJHQ30K")

    args = parser.parse_args()

    # ---- Build generator ----
    ImageGenerator = SanaImageGenerator if args.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" else SD3ImageGenerator
    gen = ImageGenerator(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=torch.bfloat16,
        use_nvfp4=args.use_nvfp4,
        block_size=args.block_size,
        rotation=args.rotation,
        permutation=args.permutation,
        use_origin_model=args.use_origin_model
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate: dataset mode handles the prompt loop and image saving.
    # It returns a list of per-prompt intermediates for .pt saving here.
    _, all_intermediates = gen.generate(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        num_steps=args.num_steps,
        save_root=args.output_dir,
        return_intermediates=True,
        n_generated_sample=args.n_samples,
        guidance_scale=args.guidance,
    )

    # Save per-sample intermediates as sample_XXXX.pt for
    # analyze_step_wise_difference.py. Each file contains:
    # dit_outputs / noise_preds / scheduler_outputs (per-step lists),
    # num_steps, prompt, seed.
    for idx, inter in enumerate(all_intermediates):
        torch.save(inter, os.path.join(args.output_dir, f"sample_{idx:04d}.pt"))
    print(f"[intermediates] saved {len(all_intermediates)} samples -> {args.output_dir}")

    # Save metadata
    metadata = {
        "model_id": args.model_id,
        "use_nvfp4": bool(args.use_nvfp4),
        "rotation": args.rotation,
        "permutation": args.permutation,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "n_samples": len(all_intermediates),
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "device": args.device,
        "dtype": "bfloat16",
        "scheduler_type": type(gen.scheduler).__name__,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[metadata] saved -> {os.path.join(args.output_dir, 'metadata.json')}")
    print(f"[config] scheduler={metadata['scheduler_type']} num_steps={args.num_steps} n_samples={len(all_intermediates)}")
