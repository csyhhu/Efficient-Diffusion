"""
A script to perform Cayley calibration on different ranges.

python -m scripts.caylay_calibration `
    --model_id stabilityai/stable-diffusion-3.5-medium `
    --calib_dataset_name MJHQ-30K --calib_dataset_path G://datasets/MJHQ-30K --calib_n_sample 16 --cayley_iters 20`
    --cayley_method module-wise`
    --output_dir G://Outputs//Efficient-Diffusion//SD3-MJHQ//module-wise`

python -m scripts.caylay_calibration `
    --model_id Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers `
    --calib_dataset_name cifar100 --calib_n_sample 2 --cayley_iters 20 `
    --cayley_method module-wise --num_steps 2 `
    --output_dir G://Outputs//Efficient-Diffusion//Caylay-Sana-cifar100//module-wise

python -m scripts.caylay_calibration `
    --model_id dit_cifar100_fm `
    --calib_dataset_name cifar100 --calib_n_sample 128 --cayley_iters 100 `
    --cayley_method module-wise --num_steps 4 `
    --output_dir G://Outputs//Efficient-Diffusion//Caylay-DiT-cifar100//module-wise
"""

import os
import sys
import argparse

# Reduce CUDA memory fragmentation (helps with Cayley rotation backward graph)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image

from src.image_generator import BaseImageGenerator, SanaImageGenerator, SD3ImageGenerator
from src.data_loader import get_dataloader_prompt
from src.utils import save_sample_grid, save_pil_grid

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Step-wise Cayley calibration")
    parser.add_argument("--model_id", type=str, default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    parser.add_argument("--config_path", type=str, default="config/cifar100_dit_fm")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--calib_dataset_name", type=str, default="MJHQ-30K", help="Name of the dataset to use for calibration")
    parser.add_argument("--calib_dataset_path", type=str, default=None, help="Path to dataset directory with .txt caption files")
    parser.add_argument("--calib_n_sample", type=int, default=8, help="Number of prompts to sample from dataset")
    parser.add_argument("--prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", help="Single prompt for calibration (used if calib_dataset_path is not provided)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//MJHQ-Sana")
    # parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//cifar100_dit_fm")
    parser.add_argument("--cayley_method", type=str, default="module-wise")
    parser.add_argument("--cayley_batches", type=int, default=8)
    parser.add_argument("--cayley_iters", type=int, default=20)
    parser.add_argument("--cayley_lr", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--single_step_mode", action="store_true", default=False)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
        Gen = SanaImageGenerator
    elif args.model_id == "stabilityai/stable-diffusion-3.5-medium":
        Gen = SD3ImageGenerator
    else:
        Gen = BaseImageGenerator
        local_mode = True
    gen = Gen(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=device,
        dtype=torch.bfloat16,
        use_nvfp4=True, block_size=args.block_size, 
        rotation="cayley", permutation=None,
        # use_origin_model=True,
        local_mode=local_mode,
        local_config_path=args.config_path
    )
    
    error_info = gen.calibrate_cayley(
        calibrate_dataset_name=args.calib_dataset_name, calib_dataset_path=args.calib_dataset_path, calib_n_sample=args.calib_n_sample,
        criterion=args.cayley_method, iters=args.cayley_iters, lr=args.cayley_lr, 
        num_steps=args.num_steps, single_step_mode=args.single_step_mode,
        save_path=args.output_dir,
        # test_mode=True
    )
    gen.plot_cayley_loss(error_info, save_root=args.output_dir)
    # gen.save_rotation(os.path.join(args.output_dir, "rotation_ckpt.pt"))
    gen.generate(args.prompt, num_samples=4, visual_n_row=2, save_root=args.output_dir, save_name="cayley-test.png", seed=args.seed)
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