"""
python -m analysis.computation_diff.test_token_skip `
    --skip_plan_path G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K//token_skip_plan_global_10.json
    
python -m analysis.computation_diff.test_token_skip `
    --dit_inference_steps "0,1,2,4,8,14,22,27" `
    --skip_plan_path G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K//token_skip_plan_M8_l2_global_10.json
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
        default="stabilityai/stable-diffusion-3.5-medium",
    )
    parser.add_argument("--use_origin_model", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--dataset_name", type=str, default="mjhq30k")
    parser.add_argument("--dataset_path", type=str, default="G://datasets/MJHQ-30K")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--dit_inference_steps", type=str, default=None)
    parser.add_argument("--skip_plan_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K")

    args = parser.parse_args()
    
    if args.skip_plan_path is not None:
        with open(args.skip_plan_path, "r") as f:
            skip_plan = json.load(f)
    else:
        skip_plan = None
    
    ImageGenerator = SD3ImageGenerator if args.model_id == "stabilityai/stable-diffusion-3.5-medium" else SD3ImageGenerator
    gen = ImageGenerator(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=torch.bfloat16
    )
    os.makedirs(args.output_dir, exist_ok=True)
    gen.generate(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        save_root=args.output_dir,
        seed=args.seed,
        dit_inference_steps=args.dit_inference_steps,
        skip_plan=skip_plan,
    )