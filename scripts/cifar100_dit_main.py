"""
python -m scripts.cifar100_dit_main
"""

import argparse
import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.image_generator.base import BaseImageGenerator
from src.utils import save_sample_grid

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train CIFAR100+DiT model")
    parser.add_argument("--config_path", type=str, default="config/cifar100_dit_fm")
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//cifar100_dit_fm")
    parser.add_argument("--prompt", type=str, default="a cut cat")
    parser.add_argument("--dtype", type=str, default=torch.float32)
    args = parser.parse_args()

    dtype = args.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gen = BaseImageGenerator(
        local_mode=True,
        local_config_path=args.config_path,
        device=device,
        dtype=dtype,
    )
    # """
    gen.prepare_local_training()
    gen.train(output_dir=args.output_dir)
    # """
    # """
    # ckpt load test
    gen.load_checkpoint(ckpt_path=f"{args.output_dir}/best_model.pth")
    gen.generate(prompt=args.prompt, num_samples=4, visual_n_row=2, num_steps=50, save_root=args.output_dir, save_name="best.png")
    # """