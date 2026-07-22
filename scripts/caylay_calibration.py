"""
A script to perform Cayley calibration on different ranges.
"""

import os
import sys
import argparse
import torch
from PIL import Image

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_generation import ImageGeneration
from src.data_loader import get_dataloader_prompt
from src.utils import save_sample_grid, save_pil_grid

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Step-wise Cayley calibration")
    parser.add_argument("--model_id", type=str, default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    parser.add_argument("--config_path", type=str, default="config/cifar100_dit_fm")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--calib_dataset_name", type=str, default="MJHQ-30K", help="Name of the dataset to use for calibration")
    parser.add_argument("--calib_dataset_path", type=str, default="G://datasets/MJHQ-30K", help="Path to dataset directory with .txt caption files")
    parser.add_argument("--calib_n_sample", type=int, default=8, help="Number of prompts to sample from dataset")
    parser.add_argument("--prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", help="Single prompt for calibration (used if calib_dataset_path is not provided)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//MJHQ-Sana")
    # parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//cifar100_dit_fm")
    parser.add_argument("--cayley_batches", type=int, default=8)
    parser.add_argument("--cayley_iters", type=int, default=20)
    parser.add_argument("--cayley_lr", type=float, default=0.01)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # """
    # Existing model
    gen = ImageGeneration(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=device,
        dtype=torch.bfloat16,
        # dtype=torch.float32,
        use_nvfp4=True, block_size=args.block_size, 
        rotation="cayley", permutation="identity"
        # rotation=None, permutation=None
    )
    # """
    # """
    # gen.build_transformer(rotation="hadamard", permutation="identity", use_nvfp4=True)
    # generated_tensor = gen.generate(args.prompt, num_steps=2, num_samples=4, seed=args.seed)
    # save_sample_grid(generated_tensor, f"{args.output_dir}/generation/identity-identity-unquantized-image.jpg", 2)
    # save_sample_grid(generated_tensor, f"{args.output_dir}/generation/none-none-unquantized-image.jpg", 2)
    # save_sample_grid(generated_tensor, f"{args.output_dir}/generation/origin-model-num-step-2.jpg", 2)
    # generated_images = gen.generate(args.prompt, num_samples=4, seed=args.seed, used_origin_pipe=True)
    # save_pil_grid(generated_images, f"{args.output_dir}/generation/origin-pipe.jpg", 2)
    # """
    # """
    # """
    error_info = gen.calibrate_cayley(
        calibrate_dataset_name=args.calib_dataset_name,
        # iters=args.cayley_iters,
        # lr=args.cayley_lr,
        iters=1,
        lr=0,
        criterion="module-wise",
        save_path=args.output_dir,
        test_mode=True
    )
    gen.plot_cayley_loss(error_info, save_root=args.output_dir)
    # """
    # """
    # generated_tensor = gen.generate(args.prompt, num_samples=4, seed=args.seed)
    # save_sample_grid(generated_tensor, f"{args.output_dir}/generation/step-wise-cayley-identity-quantized-image.jpg", 2)
    # ---
    # Local Mode
    # ---
    """
    gen = ImageGeneration(
        local_mode=True,
        local_config_path=args.config_path,
        device=device,
        dtype=torch.float32,
        use_nvfp4=True,
    )
    error_info = gen.calibrate_cayley(
        calibrate_dataset_name="cifar100",
        iters=args.cayley_iters,
        lr=args.cayley_lr,
        criterion="module-wise",
        save_path=args.output_dir,
    )
    """
    # error_info = f"{args.output_dir}/cayley_error_info.json"
    # gen.plot_cayley_loss(error_info, save_root=args.output_dir)
    # """
    # Dry mode
    """
    gen = ImageGeneration(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=device,
        dtype=torch.bfloat16,
        dry_mode=True,
        dry_config={
            "dim": 256,
            "layers": 4,
            "resolution": 64,
        },
        use_nvfp4=True, block_size=args.block_size,
        rotation="cayley", permutation="identity"
    )
    gen.calibrate_cayley(
        calibrate_dataset_name=args.calib_dataset_name,
        iters=args.cayley_iters,
        lr=args.cayley_lr,
        criterion="step-wise",
    )
    """