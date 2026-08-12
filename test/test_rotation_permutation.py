import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

from src.image_generation import ImageGeneration


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test ImageGeneration class")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//rot_perm_compare")
    parser.add_argument("--cayley_batches", type=int, default=8)
    parser.add_argument("--cayley_iters", type=int, default=20)
    parser.add_argument("--cayley_lr", type=float, default=0.01)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    rotations = ["identity", "random", "hadamard"]
    permutations = ["identity", "random"]
    quantizations = ["unquantized", "quantized"]
    
    cayley_K_cache = None
    
    for quantization in quantizations:
        for permutation in permutations:
            for rotation in rotations:
                use_nvfp4 = (quantization == "quantized")
                print(f"\nTesting: rotation={rotation}, permutation={permutation}, quantization={quantization}")
                
                gen = ImageGeneration(
                    model_id=args.model_id,
                    cache_dir=args.cache_dir,
                    device=args.device,
                    block_size=args.block_size,
                    dtype=torch.bfloat16,
                    rotation=rotation,
                    permutation=permutation,
                    use_nvfp4=use_nvfp4,
                )
                
                # image = gen.generate(args.prompt, seed=args.seed, used_origin_pipe=True)
                # filename = f"{rotation}-{permutation}-{quantization}-image-origin-pipe.png"
                image = gen.generate(args.prompt, seed=args.seed)
                filename = f"{rotation}-{permutation}-{quantization}-image.png"
                # save_path = f"{args.output_dir}/generation/{filename}"
                save_path = f"{args.output_dir}/generation_sd3/{filename}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
                print(f"    Image saved: {save_path}")
