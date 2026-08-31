"""
python -m scripts.cifar100_dit_main
"""

import argparse
import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.image_generator.base import BaseImageGenerator
from src.utils import save_sample_grid, _shutdown_loaders

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train CIFAR100+DiT model")
    parser.add_argument("--config_path", type=str, default="config/cifar100_dit_fm")
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
    try:
        gen.prepare_local_training()
        gen.train()
    except KeyboardInterrupt:
        print("\n[Interrupt] Ctrl+C received — shutting down DataLoader workers...")
        _shutdown_loaders(gen)
        print("[Interrupt] Cleanup done. Exiting.")
        # Hard exit: Python's normal interpreter shutdown can still hang on
        # lingering worker threads under Windows spawn; os._exit avoids that.
        os._exit(0)
    # """
    """
    # ckpt load test
    gen.load_checkpoint("last_model.pth")
    gen.generate(prompt=args.prompt, num_samples=4, visual_n_row=2, num_steps=50, save_name="last.png")
    """
