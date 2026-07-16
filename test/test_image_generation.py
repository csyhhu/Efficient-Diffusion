import os
import sys
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_generation import ImageGeneration


def test_dry_mode(args):
    """Test dry mode with randomly initialized model."""
    gen = ImageGeneration(
        dry_mode=True,
        dry_config={
            "dim": 256,
            "layers": 2,
            "resolution": 64,
        },
        device=args.device,
    )
    """
    gen.build_transformer(rotation="identity", permutation="identity", use_nvfp4=False)
    image = gen.generate(args.prompt, seed=args.seed)
    """
    gen.build_transformer(rotation="cayley", permutation="identity", use_nvfp4=False)
    gen.calibrate_cayley(
        prompts=[args.prompt],
        n_batches=args.cayley_batches,
        iters=args.cayley_iters,
        lr=args.cayley_lr,
    )
    image = gen.generate(args.prompt, seed=args.seed, used_origin_pipe=True)
    filename = f"cayley-identity-image.png"
    os.makedirs(f"{args.output_dir}/generation_dry", exist_ok=True)
    image.save(f"{args.output_dir}/generation_dry/{filename}")
    print(f"  Dry mode image saved to: {args.output_dir}/generation_dry/{filename}")    


def test_rotation_permutation_quantization(args, config_only_mode: bool):
    """Test rotation + permutation + quantization combinations with K-cache optimization."""
    # rotations = ["identity", "hadamard", "random", "cayley"]
    """
    rotations = ["cayley"]
    permutations = ["identity", "identity", "random", "mag"]
    quantizations = ["unquantized", "quantized"]
    """
    rotations = ["identity"]
    permutations = ["identity"]
    quantizations = ["quantized"]
    
    gen = ImageGeneration(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        block_size=args.block_size,
        config_only_mode=config_only_mode,
        # dtype=torch.float32
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
                            prompts=[args.prompt],
                            n_batches=args.cayley_batches,
                            iters=args.cayley_iters,
                            lr=args.cayley_lr,
                        )
                        cayley_K_cache = gen.extract_cayley_K()
                        print(f"    Extracted K matrices from {len(cayley_K_cache)} layers")
                    else:
                        print("    Reusing cached Cayley K matrices...")
                        gen.apply_cayley_from_cache(cayley_K_cache)
                
                image = gen.generate(args.prompt, seed=args.seed, used_origin_pipe=False)
                filename = f"{rotation}-{permutation}-{quantization}-image.png"
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
    parser.add_argument("--output_dir", type=str, default="test_output")
    parser.add_argument("--cayley_batches", type=int, default=8)
    parser.add_argument("--cayley_iters", type=int, default=20)
    parser.add_argument("--cayley_lr", type=float, default=0.01)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # test_dry_mode(args)
    test_rotation_permutation_quantization(args, config_only_mode=True)
