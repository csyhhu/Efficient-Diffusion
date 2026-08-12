"""
Generate images for evaluation using ImageGenerator.

python -m eval.generate_for_eval --quantized --n_generated_sample 10
"""

import os
import sys
import argparse
import torch

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_generator import SanaImageGenerator, SD3ImageGenerator

parser = argparse.ArgumentParser(description="Step-wise Cayley calibration")
parser.add_argument("--model_id", type=str, default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
parser.add_argument("--cache_dir", type=str, default="G://models")
parser.add_argument("--rotation_ckpt_path", type=str, default=None)
parser.add_argument("--quantized", action="store_true", default=False)
parser.add_argument("--permutation", type=str, default="identity")
parser.add_argument("--rotation", type=str, default="identity")
parser.add_argument("--block_size", type=int, default=16)
parser.add_argument("--dataset_name", type=str, default="MJHQ-30K")
parser.add_argument("--dataset_path", type=str, default="G://datasets//MJHQ-30K")
parser.add_argument("--n_generated_sample", type=int, default=-1)
parser.add_argument("--save_root", type=str, default="G://Outputs//Efficient-Diffusion//eval_gen//test_quantized")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_steps", type=int, default=4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initalize ImageGenerator
IG = SanaImageGenerator if args.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" else SD3ImageGenerator
gen = IG(
    model_id=args.model_id,
    cache_dir=args.cache_dir,
    device=device,
    dtype=torch.bfloat16,
    use_nvfp4=args.quantized, block_size=args.block_size,
    rotation=args.rotation, permutation=args.permutation
)
if args.rotation_ckpt_path is not None:
    gen.load_rotation(args.rotation_ckpt_path)

# Generate images for given dataset's prompt
gen.generate(
    dataset_name=args.dataset_name,
    seed=args.seed,
    save_root=args.save_root,
    dataset_path=args.dataset_path,
    n_generated_sample=args.n_generated_sample,
    num_steps=args.num_steps
)
