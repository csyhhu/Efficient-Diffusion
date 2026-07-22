import os
import sys
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_generation import ImageGeneration


def test_dry_mode(args):
    """Test dry mode with randomly initialized model.
    
    In dry mode, a small dummy model is used for quick testing without loading
    the actual Sana model weights. This is useful for testing code logic and
    calibration pipelines.
    """
    gen = ImageGeneration(
        dry_mode=True,
        dry_config={
            "dim": 256,
            "layers": 4,
            "resolution": 64,
        },
        device=args.device,
        dtype=torch.bfloat16,
    )
    """
    gen.build_transformer(rotation="cayley", permutation="identity", use_nvfp4=False)
    
    print("  Calibrating Cayley rotation in dry mode...")
    gen.calibrate_cayley(
        prompt=args.prompt,
        n_batches=args.cayley_batches,
        iters=args.cayley_iters,
        lr=args.cayley_lr,
    )
    filename = f"random-random-image.png"
    """
    gen.build_transformer(rotation="random", permutation="random", use_nvfp4=True)
    filename = f"random-random-image.png"
    
    image = gen.generate(args.prompt, seed=args.seed)
    os.makedirs(f"{args.output_dir}/generation_dry", exist_ok=True)
    image.save(f"{args.output_dir}/generation_dry/{filename}")
    print(f"  Dry mode image saved to: {args.output_dir}/generation_dry/{filename}")


def test_real_model(args):
    """Test rotation + permutation + quantization combinations with real model.
    
    Tests various combinations of rotation strategies, permutation strategies,
    and quantization modes using the actual Sana model. Supports K-cache
    optimization for Cayley rotations to avoid repeated calibration.
    """
    # rotations = ["cayley"]
    rotations = ["random"]
    permutations = ["identity"]
    quantizations = ["quantized"]
    
    gen = ImageGeneration(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        block_size=args.block_size,
        dtype=torch.bfloat16,
    )
    
    cayley_K_cache = None
    
    for rotation in rotations:
        for permutation in permutations:
            for quantization in quantizations:
                use_nvfp4 = (quantization == "quantized")
                print(f"\nTesting: rotation={rotation}, permutation={permutation}, quantization={quantization}")
                
                gen.build_transformer(
                    rotation=rotation,
                    permutation=permutation,
                    use_nvfp4=use_nvfp4,
                )
                if rotation == "cayley":
                    if cayley_K_cache is None:
                        print("    Calibrating Cayley rotation (first time)...")
                        gen.calibrate_cayley(
                            prompt=args.prompt,
                            n_batches=args.cayley_batches,
                            iters=args.cayley_iters,
                            lr=args.cayley_lr,
                        )
                        cayley_K_cache = gen.extract_cayley_K()
                        print(f"    Extracted K matrices from {len(cayley_K_cache)} layers")
                    else:
                        print("    Reusing cached Cayley K matrices...")
                        gen.apply_cayley_from_cache(cayley_K_cache)
                
                # image = gen.generate(args.prompt, seed=args.seed, used_origin_pipe=True)
                # filename = f"{rotation}-{permutation}-{quantization}-image-origin-pipe.png"
                image = gen.generate(args.prompt, seed=args.seed)
                filename = f"{rotation}-{permutation}-{quantization}-image.png"
                os.makedirs(f"{args.output_dir}/generation", exist_ok=True)
                image.save(f"{args.output_dir}/generation/{filename}")
                print(f"    Image saved: {filename}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test ImageGeneration class")
    parser.add_argument("--model_id", type=str, 
                        default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--prompt", type=str, 
                        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//rot_perm_compare")
    parser.add_argument("--cayley_batches", type=int, default=8)
    parser.add_argument("--cayley_iters", type=int, default=20)
    parser.add_argument("--cayley_lr", type=float, default=0.01)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # test_dry_mode(args)
    test_real_model(args)
