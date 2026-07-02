"""
MNIST data loading and visualisation utilities.

Usage::

    from src.data.mnist import get_mnist_dataloader, visualize_mnist_samples

    train_loader = get_mnist_dataloader(batch_size=128, train=True)
    fig = visualize_mnist_samples(train_loader, num_samples=16)
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def get_mnist_dataloader(
    batch_size: int = 128,
    train: bool = True,
    data_dir: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create an MNIST DataLoader with pixel values normalized to [-1, 1].

    Args:
        batch_size:  Number of samples per batch.
        train:       True → training set (shuffled), False → test set.
        data_dir:    Directory to store / download MNIST data.
        num_workers: Number of subprocesses for data loading.
        pin_memory:  Enable pin_memory (recommended for GPU training).

    Returns:
        torch.utils.data.DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                # [0, 1]
        transforms.Normalize((0.5,), (0.5,)), # [-1, 1]
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return loader


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_mnist_samples(
    dataloader: DataLoader = None,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    show_labels: bool = True,
) -> plt.Figure:
    """Visualize a grid of MNIST samples.

    Args:
        dataloader:   DataLoader; auto-creates a training loader if None.
        num_samples:  Number of samples to display.
        save_path:    If provided, save the figure to this path.
        show_labels:  Whether to display label text on each subplot.

    Returns:
        matplotlib.figure.Figure
    """
    if dataloader is None:
        dataloader = get_mnist_dataloader(batch_size=num_samples, train=True)

    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Denormalize: [-1, 1] → [0, 1]
    images = (images + 1) / 2

    # Compute grid layout
    ncols = int(num_samples ** 0.5)
    nrows = (num_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 1.8, nrows * 1.8),
    )
    # Flatten to 1D index for uniform handling
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        if show_labels:
            axes[i].set_title(f"Label: {labels[i].item()}", fontsize=10)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Image saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 50)
    print("Testing MNIST DataLoader")
    print("=" * 50)

    train_loader = get_mnist_dataloader(batch_size=64, train=True)
    test_loader  = get_mnist_dataloader(batch_size=64, train=False)

    x, y = next(iter(train_loader))
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Batch shape:   x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"x range:       [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"y labels:      {y[:8].tolist()}")

    print("\n" + "=" * 50)
    print("Testing visualization")
    print("=" * 50)

    fig = visualize_mnist_samples(
        train_loader,
        num_samples=16,
        save_path="./outputs/mnist_samples.png",
        show_labels=True,
    )
    print("Visualization complete! Check ./outputs/mnist_samples.png")
    print("Use plt.show(fig) to display interactively.")
