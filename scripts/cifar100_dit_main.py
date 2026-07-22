import argparse
import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.image_generation import ImageGeneration
from src.utils import save_sample_grid

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train CIFAR100+DiT model")
    parser.add_argument("--config_path", type=str, default="config/cifar100_dit_fm")
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//cifar100_dit_fm")
    parser.add_argument("--prompt", type=str, default="a photo of a dog")
    parser.add_argument("--dtype", type=str, default=torch.float32)
    args = parser.parse_args()

    dtype = args.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gen = ImageGeneration(
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
    gen.load_ckpt(ckpt_path=f"{args.output_dir}/best_model.pth")
    img = gen.generate(prompt=args.prompt, num_samples=4, num_steps=50)
    save_sample_grid(img, f"{args.output_dir}/cifar100-dit-fm.png", nrow=2)
    # """