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
import os, sys, inspect
# Ensure project root is on sys.path so that ``from src.xxx`` works
# regardless of how this file is invoked.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

from src.data.mnist import get_mnist_dataloader
from src.data.cifar import get_cifar100_dataloader
from src.data.coco2017 import get_coco2017_dataloader
from src.data.mqjh import get_mqjh30k_dataloader

from src.data.t2i import (  # noqa: F401 – re-export everything
    LatentDataset,
    RawImageDataset,
    _build_paths_and_captions,
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

_T2I_DATASETS = {"pokemon", "coco", "flickr30k", "cc12m", "mjhq-30k"}
_DEFAULT_TOKEN_LENGTHS = {"quantized_sd3": 77, "quantized_pixart": 300}


def get_dataloader(
    dataset_name: str,
    vae=None,
    tokenizer=None,
    text_encoder=None,
    dataset_config: dict = None,
    **kwargs
   ) -> tuple:
    """Create train and validation DataLoaders.

    Args:
        dataset_name:
            | ``"mnist"`` — MNIST (grayscale, no VAE/tokenizer needed).
            | ``"cifar100"`` — CIFAR-100 (RGB, prompt from class labels).
            | ``"coco2017"`` — COCO 2017 **validation** split only, via ModelScope.
              Uses instance-segmentation categories to generate prompts
              (same template strategy as CIFAR-100).
            | ``"pokemon"`` | ``"coco"`` | ``"flickr30k"`` | ``"cc12m"`` —
              T2I datasets requiring VAE and tokenizer.
        dataset_config:
            Flat configuration dict. Any key that matches a parameter of the
            target ``get_xxx_dataloader`` is forwarded; keys the caller omits
            fall back to that function's own defaults — this dispatcher holds
            no duplicated defaults (edit the function to change a default).

            - ``data_dir`` is accepted for every dataset and is aliased to
              ``root`` for CIFAR-100 / MJHQ-30K (whose functions name the path
              argument ``root``).
            - ``dtype`` is given as a string (e.g. ``"bfloat16"``) and resolved
              to a ``torch.dtype`` automatically in latent mode (inheriting the
              VAE dtype when omitted).

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

        **CIFAR-100 / COCO2017 (raw mode)**: Each batch is
        ``(image, category_idx, prompt)`` — ``image`` shape ``(B, 3, H, W)``
        normalised to [-1, 1].

        **COCO2017 (latent mode, when vae+tokenizer+text_encoder given)**:
        Each batch is ``(latent, encoder_hidden_states)``.

        **T2I**: Each batch is ``(latent, tokens_dict)`` —
        ``latent`` shape ``(B, C, H, W)``,
        ``tokens_dict`` = ``{"input_ids": (B, seq_len), "attention_mask": (B, seq_len)}``.
    """

    # Normalise the config: tolerate ``None`` and fold in extra **kwargs so
    # callers may pass options either as a dict or as keyword arguments.
    dataset_config = dict(dataset_config or {})
    dataset_config.update(kwargs)

    if dataset_name == "mnist":
        cfg = _filter_config(dataset_config, get_mnist_dataloader)
        train_loader = get_mnist_dataloader(**cfg, train=True)
        val_loader = get_mnist_dataloader(**cfg, train=False)
        return train_loader, val_loader

    elif dataset_name == "cifar100":
        # CIFAR names its path argument ``root``; alias ``data_dir`` → ``root``.
        cfg = _filter_config(dataset_config, get_cifar100_dataloader, aliases={"data_dir": "root"})
        # NOTE: get_dataloader keeps CIFAR in RAW mode here (matches prior
        # behaviour). The function itself supports latent mode — call it
        # directly with vae/tokenizer/text_encoder to enable VAE latents.
        if vae is not None and tokenizer is not None and text_encoder is not None:
            _inject_latent_kwargs(cfg, dataset_config, vae, tokenizer, text_encoder)
        train_loader = get_cifar100_dataloader(**cfg, train=True)
        val_loader = get_cifar100_dataloader(**cfg, train=False)
        return train_loader, val_loader

    # ── COCO 2017 (validation split only, ModelScope) ─────────────────
    elif dataset_name in ("coco2017", "coco2017val"):
        cfg = _filter_config(dataset_config, get_coco2017_dataloader)
        # Latent mode only when all three model components are provided.
        if vae is not None and tokenizer is not None and text_encoder is not None:
            _inject_latent_kwargs(cfg, dataset_config, vae, tokenizer, text_encoder)
        train_flag = (dataset_name != "coco2017val")
        train_loader = get_coco2017_dataloader(**cfg, train=train_flag)
        val_loader = get_coco2017_dataloader(**cfg, train=False)
        return train_loader, val_loader

    # ── MJHQ-30K (latent + text embedding) ────────────────────────────
    elif dataset_name in ("mjhq-30k", "mjhq30k", "MJHQ-30K", "MJHQ30K"):
        # MJHQ names its path argument ``root``; alias ``data_dir`` → ``root``.
        cfg = _filter_config(dataset_config, get_mqjh30k_dataloader, aliases={"data_dir": "root"})
        if vae is not None and tokenizer is not None and text_encoder is not None:
            _inject_latent_kwargs(cfg, dataset_config, vae, tokenizer, text_encoder)
        train_loader = get_mqjh30k_dataloader(**cfg, train=True)
        val_loader = get_mqjh30k_dataloader(**cfg, train=False)
        return train_loader, val_loader

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: mnist, cifar100, coco2017, mjhq-30k"
        )


# ---------------------------------------------------------------------------
# Prompt-based dataloader
# ---------------------------------------------------------------------------
def get_dataset_prompts(
    dataset_name: str,
    dataset_path: str,
    n_sample: int = -1,
) -> list:
    """Return a list of text prompts from a T2I dataset.

    Uses ``_build_paths_and_captions`` to locate images + captions, then
    returns only the caption strings.  When ``n_sample`` is -1, all available
    prompts are returned.

    Args:
        dataset_name: e.g. ``"mjhq-30k"``, ``"coco"``, ``"pokemon"`` …
        dataset_path: Local path to the dataset directory.
        n_sample: Maximum number of prompts to return (-1 = all).

    Returns:
        List[str]: Prompts from the dataset.
    """
    # Normalise: "MJHQ-30K" -> "mjhq30k" to match _build_paths_and_captions
    dataset_key = dataset_name.lower().replace("-", "").replace("_", "")
    max_total = n_sample if n_sample > 0 else 10 ** 9
    _, captions, _ = _build_paths_and_captions(
        dataset_key, max_total=max_total, dataset_path=dataset_path,
    )
    return captions


def get_dataloader_prompt(
    dataset_name: str,
    dataset_path: str,
    n_sample: int,
) -> torch.utils.data.DataLoader:

    image_paths, captions, tmpdir = _build_paths_and_captions(
        dataset_name, max_total=n_sample, dataset_path=dataset_path,
    )
        
    class PromptDataset(torch.utils.data.Dataset):
        def __init__(self, prompts_list):
            self.prompts = prompts_list
        
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            return self.prompts
    
    return torch.utils.data.DataLoader(
        PromptDataset(captions),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_config(dataset_config, fn, aliases=None, exclude=("dtype",)):
    """Build keyword args for ``fn`` from ``dataset_config``.

    Only keys that are **both** present in ``dataset_config`` and accepted by
    ``fn`` (per its signature) are forwarded. Keys the caller omits are
    deliberately NOT inserted, so ``fn``'s own parameter defaults take effect
    — this dispatcher therefore holds no duplicated defaults; edit the target
    ``get_xxx_dataloader`` to change a default.

    Args:
        dataset_config: Source config dict (read-only; not mutated).
        fn: Target ``get_xxx_dataloader`` callable. Its signature decides
            which keys are accepted (functions here have fixed signatures,
            no ``**kwargs``).
        aliases: Optional ``{config_key: fn_param_name}`` map for naming
            mismatches, e.g. ``{"data_dir": "root"}`` when the dataset
            function names the path argument ``root`` but the config uses
            ``data_dir``.
        exclude: Keys to skip even when present. ``"dtype"`` is excluded by
            default because the config stores it as a string (e.g.
            ``"bfloat16"``) whereas the functions expect a ``torch.dtype``;
            it is resolved separately in :func:`_inject_latent_kwargs`.
    """
    aliases = aliases or {}
    exclude = set(exclude)
    accepted = {
        name for name, p in inspect.signature(fn).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    cfg = {}
    for k, v in dataset_config.items():
        if k in exclude or v is None:
            continue
        target = aliases.get(k, k)
        if target in accepted:
            cfg[target] = v
    return cfg


def _inject_latent_kwargs(cfg, dataset_config, vae, tokenizer, text_encoder):
    """Add vae / tokenizer / text_encoder / device / dtype for latent mode.

    Called only when all three model components are supplied. ``dtype`` is
    resolved from a config string (e.g. ``"bfloat16"``) — or, when absent,
    inherited from the VAE — into a real ``torch.dtype`` before injection,
    so it never reaches the dataset function as a raw string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(dataset_config.get("dtype", None), vae)
    cfg.update({
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "device": device,
        "dtype": dtype,
    })
    return cfg


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
# 
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    """
    python -m src.data_loader
    """
    
    import os
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

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

    class _DummyTextEncoder:
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
    text_encoder_dummy = _DummyTextEncoder()

    os.makedirs("./outputs", exist_ok=True)

    # ── Test 1: MNIST ──
    """
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
    """

    # ── Test 2: T2I with RandomVAE + RandomTokenizer ──
    """
    print("=" * 60)
    print("Test 2:  pokemon via get_dataloader (dry-run VAE/tokenizer)")

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

    # ── Test 3: T2I (coco) via get_dataloader ──
    print("=" * 60)
    print("Test 3: T2I (coco) via get_dataloader (dry-run VAE/tokenizer)")

    train_loader, val_loader = get_dataloader(
        "coco",
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
    """
    # --- coco2017val ---
    train_loader, val_loader = get_dataloader(
        "coco2017val",
        vae=vae_dummy,
        tokenizer=tok_dummy,
        text_encoder=text_encoder_dummy
    )
    data_batch = next(iter(val_loader))

    # Test prompt loading
    """
    calibrate_dataloader = get_dataloader_prompt(
        "mjhq30k",
        r"G://datasets//MJHQ-30K",
        8
    )
    for prompt in calibrate_dataloader:
        print(prompt)
    """
