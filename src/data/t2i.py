"""
Text-to-Image data pipelines: Pokemon / COCO / Flickr30k / CC12M.

Three-stage API:

- ``load_t2i_data_raw``     – download → raw DataLoader (no VAE/T5)
- ``process_t2i_data``      – raw DataLoader → latent DataLoader (VAE+T5)
- ``load_t2i_data``         – convenience one-shot

Usage::

    from src.data.t2i import load_t2i_data_raw, process_t2i_data, load_t2i_data

    # Two-stage (inspect data before VAE)
    train_raw, val_raw, tmpdir = load_t2i_data_raw("pokemon", max_samples=600)
    train_latent, attn_mask = process_t2i_data(
        train_raw, vae, tokenizer, text_encoder, device, dtype,
        cleanup_tmpdir=tmpdir,
    )

    # One-shot
    train_loader, val_loader, attn_mask = load_t2i_data(
        "pokemon", vae, tokenizer, text_encoder,
        device, dtype, batch_size=8, max_samples=600,
    )
"""

import math
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# RawImageDataset  (on-the-fly loading, no VAE/T5)
# ---------------------------------------------------------------------------

class RawImageDataset(Dataset):
    """Lazy-load images from disk — **no VAE or T5 needed**.

    Yields ``{"image": (3,H,W) in [-1,1], "caption": str}``.

    Useful for data-inspection / visualisation before VAE pre-computation.
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        image_size: int = 256,
    ):
        self.image_paths = image_paths
        self.captions = captions
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5  # [-1, 1]
        return {"image": img_tensor, "caption": self.captions[idx]}


# ---------------------------------------------------------------------------
# Raw-data visualisation  (no VAE/T5)
# ---------------------------------------------------------------------------

def visualize_t2i_samples(
    image_paths: List[str],
    captions: List[str],
    num_samples: int = 8,
    seed: int = 42,
    save_path: Optional[str] = None,
    display_size: int = 256,
) -> plt.Figure:
    """Display a grid of image-caption pairs (from file paths, no VAE/T5)."""

    total = min(len(image_paths), len(captions))
    rng = np.random.default_rng(seed)

    n_show = min(num_samples, total)
    if total > n_show:
        indices = sorted(rng.choice(total, size=n_show, replace=False))
    else:
        indices = list(range(total))

    ncols = int(math.ceil(n_show ** 0.5))
    nrows = int(math.ceil(n_show / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.8),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
            img = img.resize((display_size, display_size), Image.BICUBIC)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "(load error)", ha="center", va="center", fontsize=8)

        cap = captions[idx]
        cap_display = (cap[:55] + "...") if len(cap) > 55 else cap
        ax.set_title(cap_display, fontsize=7, pad=4)
        ax.axis("off")

    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"T2I Raw Samples  ({n_show} / {total})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved to {save_path}")

    return fig


def visualize_raw_batch(
    dataloader: DataLoader,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    display_size: int = 256,
) -> plt.Figure:
    """Inspect a **raw** T2I DataLoader — display actual images and captions.

    Args:
        dataloader:  DataLoader yielding ``{"image": (3,H,W), "caption": str}``.
        num_samples: max images to show.
        save_path:   optional output path.
    """
    batch = next(iter(dataloader))
    images = batch["image"]
    captions = batch["caption"]

    n_show = min(num_samples, images.shape[0])

    ncols = int(math.ceil(n_show ** 0.5))
    nrows = int(math.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False)
    axes = axes.flatten()

    for i in range(n_show):
        ax = axes[i]
        # Denormalise [-1, 1] → [0, 1]
        img = images[i].permute(1, 2, 0).clamp(-1, 1)
        img = (img + 1) / 2
        img = img.cpu().float().numpy()
        img = img.clip(0, 1)
        ax.imshow(img)

        cap = captions[i] if isinstance(captions, (list, tuple)) else captions
        if isinstance(cap, torch.Tensor):
            cap = str(cap.item()) if cap.numel() == 1 else "..."
        cap_display = (str(cap)[:55] + "...") if len(str(cap)) > 55 else str(cap)
        ax.set_title(cap_display, fontsize=7, pad=4)
        ax.axis("off")

    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Raw T2I DataLoader  ({n_show} / {images.shape[0]})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# LatentDataset  (pre-computed VAE + T5 embeddings)
# ---------------------------------------------------------------------------

class LatentDataset(Dataset):
    """Pre-compute VAE latents and T5 embeddings for a set of (image, caption) pairs.

    This avoids running VAE/T5 at every training step, saving ~50% training time.

    Args:
        image_paths:  list of image file paths.
        captions:     list of text captions (same length).
        vae:          frozen VAE (``AutoencoderKL``).
        tokenizer:    T5 tokenizer.
        text_encoder: frozen T5 encoder.
        device:       torch device.
        dtype:        compute dtype.
        image_size:   resize target (square).
        cache_dir:    (unused, reserved for future disk-cache support).
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        vae,
        tokenizer,
        text_encoder,
        device: torch.device,
        dtype: torch.dtype,
        image_size: int = 256,
        cache_dir: Optional[str] = None,
    ):
        self.latents: List[torch.Tensor] = []
        self.encoder_hidden_states: List[torch.Tensor] = []

        vae_scale = vae.config.scaling_factor

        print(f"[data] Pre-computing VAE latents + T5 embeddings for {len(image_paths)} images ...")
        for i, (img_path, caption) in enumerate(tqdm(zip(image_paths, captions), total=len(image_paths))):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size), Image.BICUBIC)
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor - 0.5) / 0.5  # [-1, 1]

                with torch.no_grad():
                    latent = vae.encode(img_tensor.unsqueeze(0).to(device, dtype)).latent_dist.sample()
                    latent = latent * vae_scale

                    text_inputs = tokenizer(
                        caption,
                        max_length=300,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_emb = text_encoder(
                        text_inputs.input_ids.to(device),
                        attention_mask=text_inputs.attention_mask.to(device),
                    ).last_hidden_state

                self.latents.append(latent.cpu().squeeze(0))
                self.encoder_hidden_states.append(text_emb.cpu().squeeze(0))

            except Exception as e:
                if i < 3:
                    print(f"[data] WARNING: skipping {img_path}: {e}")
                continue

        if not self.latents:
            raise RuntimeError("[data] No valid samples after pre-computation!")

        self.attention_mask = torch.ones(
            self.encoder_hidden_states[0].shape[0], device="cpu"
        ).bool().unsqueeze(0)  # (1, seq_len)

        print(f"[data] Pre-computed {len(self.latents)} valid samples.")

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "latent": self.latents[idx],
            "encoder_hidden_states": self.encoder_hidden_states[idx],
        }


# ---------------------------------------------------------------------------
# Dataset download / locate  (COCO / Flickr30k / CC12M)
# ---------------------------------------------------------------------------

def _build_paths_and_captions(
    dataset_name: str,
    max_total: int,
    cc12m_path: Optional[str] = None,
) -> Tuple[List[str], List[str], str]:
    """Download / locate images + captions, return ``(file_paths, captions, tmpdir)``.

    ``tmpdir`` should be cleaned by the caller after use.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"{dataset_name}_cache_")

    if dataset_name == "coco":
        from datasets import load_dataset
        print("[data] Loading COCO 2017 train split ...")
        ds = load_dataset("HuggingFaceM4/COCO", split="train", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(max_total, len(ds))))

        image_paths: List[str] = []
        captions: List[str] = []
        for i, item in enumerate(ds):
            pil_img = item["image"]
            fpath = os.path.join(tmpdir, f"{i:06d}.jpg")
            pil_img.save(fpath)
            image_paths.append(fpath)

            caps = item.get("sentences", item.get("captions", []))
            cap = caps[0] if isinstance(caps, list) and caps else ""
            captions.append(cap)

    elif dataset_name == "flickr30k":
        from datasets import load_dataset
        print("[data] Loading Flickr30k ...")
        ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(max_total, len(ds))))

        image_paths = []
        captions = []
        for i, item in enumerate(ds):
            pil_img = item["image"]
            fpath = os.path.join(tmpdir, f"{i:06d}.jpg")
            pil_img.save(fpath)
            image_paths.append(fpath)

            cap = item.get("caption", item.get(
                "sentences", [""]
            )[0] if isinstance(item.get("sentences"), list) else "")
            captions.append(cap)

    elif dataset_name == "pokemon":
        from datasets import load_dataset
        print("[data] Loading Pokemon BLIP captions (parquet) ...")
        ds = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
        ds = ds.shuffle(seed=42).select(range(min(max_total, len(ds))))

        image_paths = []
        captions = []
        for i, item in enumerate(ds):
            pil_img = item["image"]
            fpath = os.path.join(tmpdir, f"{i:06d}.png")
            pil_img.save(fpath)
            image_paths.append(fpath)

            cap = item.get("en_text", item.get("text", ""))
            captions.append(cap)

    elif dataset_name == "cc12m":
        if cc12m_path is None:
            raise ValueError("--cc12m_path is required for CC12M dataset.")
        import webdataset as wds
        print(f"[data] Loading CC12M shards from {cc12m_path} ...")
        ds = wds.WebDataset(cc12m_path).shuffle(1000).decode("pil")
        image_paths = []
        captions = []
        for i, sample in enumerate(ds):
            if i >= max_total:
                break
            pil_img = sample.get("jpg") or sample.get("png") or sample.get("image")
            cap = sample.get("txt") or sample.get("caption") or ""
            if pil_img is None:
                continue
            fpath = os.path.join(tmpdir, f"{i:06d}.jpg")
            pil_img.save(fpath)
            image_paths.append(fpath)
            captions.append(cap)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return image_paths, captions, tmpdir


# ---------------------------------------------------------------------------
# Stage 1 — Data reading  (download + raw DataLoader, NO VAE/T5)
# ---------------------------------------------------------------------------

def load_t2i_data_raw(
    dataset_name: str,
    max_samples: int = 20000,
    image_size: int = 256,
    cc12m_path: Optional[str] = None,
    val_samples: int = 1000,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, str]:
    """Download / locate images + captions, return **raw** DataLoaders.

    Raw DataLoader yields ``{"image": (3,H,W) in [-1,1], "caption": str}``.

    **No VAE or T5 is needed** — use this for data inspection / visualisation.

    Args:
        dataset_name:  ``"pokemon"`` | ``"coco"`` | ``"flickr30k"`` | ``"cc12m"``.
        max_samples:   cap on training samples.
        image_size:    resize target (square).
        cc12m_path:    glob pattern for webdataset shards (required for cc12m).
        val_samples:   number of samples reserved for validation.
        batch_size:    batch size for DataLoaders.
        num_workers:   DataLoader worker processes.
        pin_memory:    pin_memory for GPU training.

    Returns:
        ``(train_loader, val_loader, tmpdir)``
        - **train_loader** / **val_loader**: raw DataLoader (image + caption).
        - **tmpdir**: path to temp image files — pass to ``process_t2i_data``
          for cleanup after pre-computation.
    """
    max_total = max_samples + val_samples
    image_paths, captions, tmpdir = _build_paths_and_captions(
        dataset_name, max_total, cc12m_path,
    )

    n_train = min(max_samples, len(image_paths) - val_samples)
    train_paths = image_paths[:n_train]
    train_captions = captions[:n_train]
    val_paths = image_paths[n_train:n_train + val_samples]
    val_captions = captions[n_train:n_train + val_samples]

    print(f"[data] Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_ds = RawImageDataset(train_paths, train_captions, image_size)
    val_ds = RawImageDataset(val_paths, val_captions, image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, tmpdir


# ---------------------------------------------------------------------------
# Stage 2 — Data processing  (VAE + T5 pre-computation)
# ---------------------------------------------------------------------------

class _PrecomputedLatentDataset(Dataset):
    """Thin wrapper around pre-computed lists of latents + encoder-hidden-states."""

    def __init__(self, latents: List[torch.Tensor], encoder_hidden_states: List[torch.Tensor]):
        self.latents = latents
        self.encoder_hidden_states = encoder_hidden_states

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "latent": self.latents[idx],
            "encoder_hidden_states": self.encoder_hidden_states[idx],
        }


def process_t2i_data(
    raw_loader: DataLoader,
    vae,
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    cleanup_tmpdir: Optional[str] = None,
) -> Tuple[DataLoader, torch.Tensor]:
    """Pre-compute VAE latents + T5 embeddings from a **raw** DataLoader.

    Consumes ``raw_loader`` (yielding ``{"image", "caption"}``) and produces
    a latent DataLoader (yielding ``{"latent", "encoder_hidden_states"}``).

    Args:
        raw_loader:       DataLoader from ``load_t2i_data_raw``.
        vae:              frozen VAE (``AutoencoderKL``).
        tokenizer:        T5 tokenizer.
        text_encoder:     frozen T5 encoder.
        device, dtype:    compute device / dtype.
        batch_size:       batch size for output DataLoader.
        num_workers:      DataLoader worker processes.
        pin_memory:       pin_memory for GPU training.
        drop_last:        drop incomplete last batch.
        cleanup_tmpdir:   if provided, delete this temp directory after
                          pre-computation (pass the ``tmpdir`` from
                          ``load_t2i_data_raw``).

    Returns:
        ``(latent_loader, attention_mask)``
        - **latent_loader**: ``DataLoader`` yielding
          ``{"latent": (C,H,W), "encoder_hidden_states": (seq, dim)}``.
        - **attention_mask**: ``(1, seq_len)`` bool tensor on CPU.
    """
    all_latents: List[torch.Tensor] = []
    all_enc_hidden: List[torch.Tensor] = []

    vae_scale = vae.config.scaling_factor
    total = len(raw_loader.dataset)

    print(f"[process] Pre-computing VAE latents + T5 embeddings for {total} samples ...")
    for batch in tqdm(raw_loader, total=len(raw_loader)):
        images = batch["image"].to(device, dtype)
        captions = batch["caption"]

        # Ensure captions is a list of strings
        if isinstance(captions, torch.Tensor):
            captions = [str(c.item()) for c in captions]
        else:
            captions = list(captions)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae_scale

            text_inputs = tokenizer(
                captions,
                max_length=300,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_emb = text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            ).last_hidden_state

        all_latents.extend([lat.cpu() for lat in latents])
        all_enc_hidden.extend([emb.cpu() for emb in text_emb])

    if not all_latents:
        raise RuntimeError("[process] No valid samples after pre-computation!")

    # Cleanup temp image files (latents are now in memory)
    if cleanup_tmpdir:
        shutil.rmtree(cleanup_tmpdir, ignore_errors=True)

    attention_mask = torch.ones(
        all_enc_hidden[0].shape[0], device="cpu"
    ).bool().unsqueeze(0)  # (1, seq_len)

    latent_ds = _PrecomputedLatentDataset(all_latents, all_enc_hidden)

    latent_loader = DataLoader(
        latent_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    print(f"[process] Pre-computed {len(all_latents)} latent samples.")
    return latent_loader, attention_mask


# ---------------------------------------------------------------------------
# Convenience — one-shot  (download + pre-compute, returns latent DataLoaders)
# ---------------------------------------------------------------------------

def load_t2i_data(
    dataset_name: str,
    vae,
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
    max_samples: int = 20000,
    image_size: int = 256,
    cc12m_path: Optional[str] = None,
    val_samples: int = 1000,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """One-shot: download + pre-compute → latent DataLoaders.

    Equivalent to calling ``load_t2i_data_raw`` + ``process_t2i_data``.

    Returns:
        ``(train_loader, val_loader, attention_mask)``
        - **train_loader** / **val_loader**: ``DataLoader`` yielding
          ``{"latent": (C,H,W), "encoder_hidden_states": (seq, dim)}``.
        - **attention_mask**: ``(1, seq_len)`` bool tensor on CPU.
    """
    train_raw, val_raw, tmpdir = load_t2i_data_raw(
        dataset_name,
        max_samples=max_samples,
        image_size=image_size,
        cc12m_path=cc12m_path,
        val_samples=val_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_latent, attn_mask = process_t2i_data(
        train_raw, vae, tokenizer, text_encoder, device, dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        cleanup_tmpdir=tmpdir,  # train set consumes the temp dir
    )

    val_latent, _ = process_t2i_data(
        val_raw, vae, tokenizer, text_encoder, device, dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return train_latent, val_latent, attn_mask


# ---------------------------------------------------------------------------
# Latent DataLoader inspection
# ---------------------------------------------------------------------------

def visualize_t2i_dataloader(
    dataloader: DataLoader,
    num_samples: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Inspect a T2I DataLoader — show latent statistics and text-embedding norms.

    Does **not** require VAE (works directly on pre-computed latents).

    Args:
        dataloader:   DataLoader yielding ``{"latent": ..., "encoder_hidden_states": ...}``.
        num_samples:  max number of samples to plot.
        save_path:    optional output path.

    Returns:
        matplotlib.figure.Figure
    """
    batch = next(iter(dataloader))
    latents = batch["latent"]
    enc_hidden = batch["encoder_hidden_states"]

    n_show = min(num_samples, latents.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    # Top: latent per-channel means
    for i in range(n_show):
        axes[0].plot(
            latents[i].mean(dim=(1, 2)).cpu().float(),
            alpha=0.5, lw=0.8,
            label=f"sample {i}" if i < 5 else "",
        )
    axes[0].set_title(f"Latent per-channel means  (shape = {tuple(latents.shape)})", fontsize=10)
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Mean")
    if n_show <= 5:
        axes[0].legend(fontsize=7)

    # Bottom: text-embedding per-token L2 norms
    for i in range(n_show):
        axes[1].plot(
            enc_hidden[i].norm(dim=-1).cpu().float(),
            alpha=0.5, lw=0.8,
            label=f"sample {i}" if i < 5 else "",
        )
    axes[1].set_title(
        f"Text-embedding per-token L2 norm  (shape = {tuple(enc_hidden.shape)})",
        fontsize=10,
    )
    axes[1].set_xlabel("Token index")
    axes[1].set_ylabel("L2 norm")
    if n_show <= 5:
        axes[1].legend(fontsize=7)

    fig.suptitle("T2I DataLoader Inspection", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# T2ILatentDataset — VAE + tokenizer only (no separate text_encoder)
# ---------------------------------------------------------------------------

class T2ILatentDataset(Dataset):
    """Pre-compute VAE latents + **tokenized** captions (no text encoder).

    Yields ``(latent, token_info)`` tuples where:
    - ``latent``: ``(C, H, W)`` VAE latent tensor
    - ``token_info``: dict with ``"input_ids"`` and ``"attention_mask"``

    This avoids the need for a separate T5/CLIP text encoder — the model's
    forward should handle embedding ``input_ids`` internally.
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        vae,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        image_size: int = 256,
        max_token_length: int = 77,
    ):
        self.latents: List[torch.Tensor] = []
        self.input_ids_list: List[torch.Tensor] = []
        self.attention_mask_list: List[torch.Tensor] = []

        vae_scale = vae.config.scaling_factor

        print(f"[data] Pre-computing VAE latents + tokenized captions "
              f"for {len(image_paths)} images ...")
        for img_path, caption in tqdm(
            zip(image_paths, captions), total=len(image_paths),
            desc="[T2I] VAE+tokenizer",
        ):
            try:
                # --- Load & preprocess image ---
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size), Image.BICUBIC)
                img_tensor = (
                    torch.from_numpy(np.array(img))
                    .permute(2, 0, 1).float() / 255.0
                )
                img_tensor = (img_tensor - 0.5) / 0.5  # [-1, 1]

                # --- VAE encode ---
                with torch.no_grad():
                    latent = vae.encode(
                        img_tensor.unsqueeze(0).to(device, dtype)
                    ).latent_dist.sample()
                    latent = latent * vae_scale

                # --- Tokenize caption (no text-encoder) ---
                tokens = tokenizer(
                    caption,
                    max_length=max_token_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self.latents.append(latent.cpu().squeeze(0))
                self.input_ids_list.append(tokens["input_ids"].squeeze(0))
                self.attention_mask_list.append(tokens["attention_mask"].squeeze(0))

            except Exception as e:
                if len(self.latents) < 3:
                    print(f"[data] WARNING: skipping {img_path}: {e}")
                continue

        if not self.latents:
            raise RuntimeError("[data] No valid samples after pre-computation!")

        print(f"[data] Pre-computed {len(self.latents)} valid samples.")

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int):
        """Return ``(latent, tokens_dict)``."""
        return (
            self.latents[idx],
            {
                "input_ids": self.input_ids_list[idx],
                "attention_mask": self.attention_mask_list[idx],
            },
        )


def load_t2i_data_tokenized(
    dataset_name: str,
    vae,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
    max_samples: int = 20000,
    image_size: int = 256,
    cc12m_path: Optional[str] = None,
    val_samples: int = 1000,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    max_token_length: int = 77,
) -> Tuple[DataLoader, DataLoader]:
    """One-shot T2I pipeline using **only VAE + tokenizer** (no text encoder).

    Downloads / locates images, pre-computes VAE latents and tokenised
    captions, then returns DataLoaders.

    Each batch yields ``(latent, tokens_dict)`` where:
    - ``latent``: ``(B, C, H, W)``
    - ``tokens_dict``: ``{"input_ids": (B, seq_len), "attention_mask": (B, seq_len)}``

    Returns:
        ``(train_loader, val_loader)``
    """
    max_total = max_samples + val_samples
    image_paths, captions, tmpdir = _build_paths_and_captions(
        dataset_name, max_total, cc12m_path,
    )

    n_train = min(max_samples, len(image_paths) - val_samples)
    train_paths = image_paths[:n_train]
    train_captions = captions[:n_train]
    val_paths = image_paths[n_train:n_train + val_samples]
    val_captions = captions[n_train:n_train + val_samples]

    print(f"[data] Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Pre-compute for train set (reuse tmpdir for train raw images)
    train_ds = T2ILatentDataset(
        train_paths, train_captions,
        vae=vae, tokenizer=tokenizer,
        device=device, dtype=dtype,
        image_size=image_size,
        max_token_length=max_token_length,
    )

    # Cleanup temp image files after pre-computation
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Pre-compute for val set (val raw images are re-downloaded inside
    # _build_paths_and_captions for val if needed — but here we just
    # process the val split we already have)
    val_ds = T2ILatentDataset(
        val_paths, val_captions,
        vae=vae, tokenizer=tokenizer,
        device=device, dtype=dtype,
        image_size=image_size,
        max_token_length=max_token_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last,
    )

    print(f"[data] T2I loaders ready — train: {len(train_loader)} batches, "
          f"val: {len(val_loader)} batches")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    os.makedirs("./outputs", exist_ok=True)

    # ---- Test 1: Raw DataLoader ----
    print("=" * 50)
    print("Stage 1 — Raw DataLoader  (load_t2i_data_raw)")
    print("=" * 50)

    train_raw, val_raw, tmpdir = load_t2i_data_raw(
        "pokemon", max_samples=50, val_samples=10, batch_size=9,
    )
    print(f"[data] train batches: {len(train_raw)}, val batches: {len(val_raw)}")

    batch = next(iter(train_raw))
    print(f"[data] batch keys: {list(batch.keys())}")
    print(f"[data] image shape: {tuple(batch['image'].shape)}, caption[0]: {batch['caption'][0][:60]}...")

    visualize_raw_batch(
        train_raw, num_samples=9,
        save_path="./outputs/t2i_raw_batch.png",
    )

    shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- Test 2: Two-stage workflow ----
    """
    print("\n" + "=" * 50)
    print("Stage 1+2 — Two-stage workflow  (raw → latent)")
    print("=" * 50)

    train_raw, val_raw, tmpdir = load_t2i_data_raw(
        "pokemon", max_samples=20, val_samples=5, batch_size=4,
    )
    visualize_raw_batch(train_raw, num_samples=4, save_path="./outputs/t2i_two_stage_raw.png")

    
    from diffusers import AutoencoderKL
    from transformers import T5EncoderModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_id = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"

    print(f"[pipeline] Loading {model_id} ...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae.eval()
    text_encoder.eval()

    train_latent, attn_mask = process_t2i_data(
        train_raw, vae, tokenizer, text_encoder, device, dtype,
        batch_size=4,
        cleanup_tmpdir=tmpdir,
    )

    print(f"[data] latent train batches: {len(train_latent)}")
    print(f"[data] attention_mask shape: {tuple(attn_mask.shape)}")

    visualize_t2i_dataloader(
        train_latent, num_samples=4,
        save_path="./outputs/t2i_two_stage_latent.png",
    )
    shutil.rmtree(tmpdir, ignore_errors=True)
    """