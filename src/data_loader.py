"""
Unified dataloader entry point: dispatches between MNIST and T2I datasets.

Usage::

    from src.data_loader import get_dataloader

    train_loader, val_loader = get_dataloader(
        "MNIST", dataset_config, vae=None, tokenizer=None,
    )
    train_loader, val_loader = get_dataloader(
        "pokemon", dataset_config, vae=vae, tokenizer=tokenizer,
    )
"""
import os, sys
# Ensure project root is on sys.path so that ``from src.xxx`` works
# regardless of how this file is invoked.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

from src.data.mnist import get_mnist_dataloader, visualize_mnist_samples

from src.data.t2i import (  # noqa: F401 – re-export everything
    LatentDataset,
    RawImageDataset,
    load_t2i_data,
    load_t2i_data_raw,
    load_t2i_data_tokenized,
    T2ILatentDataset,
    process_t2i_data,
    visualize_raw_batch,
    visualize_t2i_dataloader,
    visualize_t2i_samples,
)

# ---------------------------------------------------------------------------
# Dataset name classification
# ---------------------------------------------------------------------------

_T2I_DATASETS = {"pokemon", "coco", "flickr30k", "cc12m"}
_DEFAULT_TOKEN_LENGTHS = {"quantized_sd3": 77, "quantized_pixart": 300}


def get_dataloader(
    dataset_name: str,
    dataset_config: dict,
    vae=None,
    tokenizer=None,
) -> tuple:
    """Create train and validation DataLoaders.

    Args:
        dataset_name:
            | ``"mnist"`` — MNIST (grayscale, no VAE/tokenizer needed).
            | ``"pokemon"`` | ``"coco"`` | ``"flickr30k"`` | ``"cc12m"`` —
              T2I datasets requiring VAE and tokenizer.
        dataset_config:
            Flat configuration dict.  Supported keys:

            **Common**
            - ``dtype`` (str): compute dtype, e.g. ``"float16"`` (default ``"float16"``).
            - ``batch_size`` (int): batch size (default ``8``).
            - ``num_workers`` (int): DataLoader workers (default ``0``).
            - ``pin_memory`` (bool): pin_memory for GPU (default ``False``).

            **T2I only**
            - ``max_samples`` (int): cap on training samples (default ``20000``).
            - ``image_size`` (int): resize target (default ``256``).
            - ``val_samples`` (int): number of validation samples (default ``1000``).
            - ``cc12m_path`` (str): glob pattern for CC12M shards (required for cc12m).
            - ``drop_last`` (bool): drop incomplete last batch (default ``True``).
            - ``max_token_length`` (int): tokenizer max length (default ``77``).

            **MNIST only**
            - ``data_dir`` (str): download / storage directory (default ``"./data"``).

        vae:
            Frozen VAE (AutoencoderKL) for image→latent encoding.  Required
            for T2I datasets; ignored for MNIST.
        tokenizer:
            Tokenizer (CLIPTokenizer / T5TokenizerFast) for caption
            tokenization.  Required for T2I datasets; ignored for MNIST.

    Returns:
        ``(train_loader, val_loader)``.

        **MNIST**: Each batch is ``(image, label)`` — ``image`` shape ``(B, 1, 28, 28)``
        normalised to [-1, 1].

        **T2I**: Each batch is ``(latent, tokens_dict)`` —
        ``latent`` shape ``(B, C, H, W)``,
        ``tokens_dict`` = ``{"input_ids": (B, seq_len), "attention_mask": (B, seq_len)}``.
    """

    dataset_name_lower = dataset_name.lower()

    # ── MNIST ──────────────────────────────────────────────────────────
    if dataset_name_lower == "mnist":
        mnist_cfg = {
            "batch_size": dataset_config.get("batch_size", 128),
            "data_dir": dataset_config.get("data_dir", "./data"),
            "num_workers": dataset_config.get("num_workers", 0),
            "pin_memory": dataset_config.get("pin_memory", False),
        }
        train_loader = get_mnist_dataloader(**mnist_cfg, train=True)
        val_loader = get_mnist_dataloader(**mnist_cfg, train=False)
        return train_loader, val_loader

    # ── T2I ────────────────────────────────────────────────────────────
    if dataset_name_lower in _T2I_DATASETS:
        if vae is None or tokenizer is None:
            raise ValueError(
                f"T2I dataset '{dataset_name}' requires both vae and tokenizer. "
                f"Got vae={vae}, tokenizer={tokenizer}"
            )

        # Auto-detect device & dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_dtype(dataset_config.get("dtype", None), vae)

        max_token_length = dataset_config.get("max_token_length", 77)
        return load_t2i_data_tokenized(
            dataset_name=dataset_name_lower,
            vae=vae,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            batch_size=dataset_config.get("batch_size", 8),
            max_samples=dataset_config.get("max_samples", 20000),
            image_size=dataset_config.get("image_size", 256),
            cc12m_path=dataset_config.get("cc12m_path", None),
            val_samples=dataset_config.get("val_samples", 1000),
            num_workers=dataset_config.get("num_workers", 0),
            pin_memory=dataset_config.get("pin_memory", False),
            drop_last=dataset_config.get("drop_last", True),
            max_token_length=max_token_length,
        )

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Supported: mnist, {', '.join(sorted(_T2I_DATASETS))}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str | None, vae) -> torch.dtype:
    """Resolve compute dtype: explicit string → VAE param dtype → float16."""
    if dtype_str is not None:
        dtype = getattr(torch, dtype_str, None)
        if dtype is not None:
            return dtype
    # Fall back to VAE's dtype
    try:
        return next(vae.parameters()).dtype
    except (StopIteration, AttributeError):
        return torch.float16


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    os.makedirs("./outputs", exist_ok=True)

    # ── Test 1: MNIST ──
    print("=" * 60)
    print("Test 1: MNIST via get_dataloader")
    train_loader, val_loader = get_dataloader(
        "MNIST",
        {"batch_size": 64, "data_dir": "./data", "num_workers": 0},
    )
    x, y = next(iter(train_loader))
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"  Batch shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"  x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print()

    # ── Test 2: T2I with RandomVAE + RandomTokenizer ──
    print("=" * 60)
    print("Test 2: T2I (pokemon) via get_dataloader (dry-run VAE/tokenizer)")

    # Minimal stubs for testing without real models
    class _DummyVAE:
        config = type("obj", (), {"scaling_factor": 0.13025})
        def encode(self, x):
            B, C, H, W = x.shape
            latent = torch.randn(B, 4, H // 8, W // 8, device=x.device, dtype=x.dtype)
            return type("LD", (), {"latent_dist": type("D", (), {"sample": lambda self: latent})()})()

    class _DummyTokenizer:
        def __call__(self, text, max_length=77, padding="max_length",
                     truncation=True, return_tensors="pt", **kwargs):
            if isinstance(text, (list, tuple)):
                bs = len(text)
            else:
                bs = 1
            return {
                "input_ids": torch.randint(0, 49408, (bs, max_length)),
                "attention_mask": torch.ones(bs, max_length, dtype=torch.long),
            }

    vae_dummy = _DummyVAE()
    tok_dummy = _DummyTokenizer()

    train_loader, val_loader = get_dataloader(
        "pokemon",
        {
            "batch_size": 4,
            "max_samples": 10,
            "val_samples": 4,
            "image_size": 128,
            "dtype": "float32",
            "max_token_length": 77,
        },
        vae=vae_dummy,
        tokenizer=tok_dummy,
    )
    latent, tokens = next(iter(train_loader))
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"  Latent shape: {tuple(latent.shape)}")
    print(f"  input_ids shape: {tuple(tokens['input_ids'].shape)}")
    print(f"  attention_mask shape: {tuple(tokens['attention_mask'].shape)}")
    print()

    print("All data_loader tests passed!")
