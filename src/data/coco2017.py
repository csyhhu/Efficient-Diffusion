"""
COCO 2017 validation data loading with prompt generation for text-conditional
diffusion training.

Uses ModelScope to download COCO2017 validation dataset (5000 images with
instance segmentation annotations). Since the ModelScope dataset provides
instance-level categories rather than captions, we generate text prompts from
category names using template-based augmentation (same approach as CIFAR-100).

Only the **validation** split is downloaded / used, per project requirements.

Usage::

    from src.data.coco2017 import get_coco2017_dataloader, COCO_CATEGORIES

    # Raw mode (image, category_id, prompt)
    val_loader = get_coco2017_dataloader(batch_size=32, train=False)
    images, labels, prompts = next(iter(val_loader))

    # Latent mode (with VAE + tokenizer + text_encoder)
    train_loader, val_loader = get_coco2017_dataloader(
        batch_size=64, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.float32,
    )
"""

import os
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# COCO 2017 Category Names (80 classes)
# ---------------------------------------------------------------------------

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# COCO category ids are not contiguous (e.g. 0=N/A, 1=person, 2=bicycle, ...)
# Map: original COCO category id -> 0..79 index
COCO_ID_TO_INDEX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
    33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
    42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
    52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
    61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
    74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
    84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


# ---------------------------------------------------------------------------
# Prompt Generation Templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "a {} in natural setting",
    "a close-up of a {}",
    "a {} on white background",
    "a beautiful {}",
    "a realistic {}",
    "an image of a {}",
    "a {} isolated on white",
    "a {} in the wild",
    "a {} in its natural habitat",
    "a detailed photo of a {}",
    "a high-quality image of a {}",
    "a {} looking at camera",
    "a {} from above",
    "a {} side view",
    "a {} front view",
    "a small {}",
    "a large {}",
    "a colorful {}",
    "a black and white photo of a {}",
    "a {} with green background",
    "a {} with blue sky",
    "a {} in the forest",
    "a {} in the city",
    "a {} at night",
    "a {} during the day",
    "multiple {}s together",
    "a {} on the street",
    "a {} indoors",
]


def generate_prompt(category_name: str, template_idx: Optional[int] = None) -> str:
    """Generate a text prompt from a COCO category name using templates.

    Args:
        category_name: COCO category name (e.g., "person", "car")
        template_idx: Optional template index for deterministic generation.
                      If None, randomly selects a template.

    Returns:
        str: Generated prompt string
    """
    import random
    if template_idx is not None:
        template = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    else:
        template = random.choice(PROMPT_TEMPLATES)
    # Handle vowels for a/an (simple heuristic)
    if template.startswith("a ") and category_name[0].lower() in "aeiou":
        template = "An " + template[2:]
    return template.format(category_name.replace("_", " "))


def generate_prompts_for_labels(category_indices: List[int]) -> List[str]:
    """Generate prompts for a list of COCO category indices (0..79).

    Args:
        category_indices: List of class indices in [0, 80)

    Returns:
        List[str]: Generated prompts
    """
    prompts = []
    for idx in category_indices:
        if 0 <= idx < len(COCO_CATEGORIES):
            class_name = COCO_CATEGORIES[idx]
        else:
            class_name = "object"
        prompts.append(generate_prompt(class_name))
    return prompts


# ---------------------------------------------------------------------------
# ModelScope dataset download / local cache
# ---------------------------------------------------------------------------

def _download_coco2017_val_via_modelscope(
    cache_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Download COCO2017 validation split from ModelScope.

    Uses ``modelscope.msdatasets.MsDataset`` to download
    ``COCO2017_Instance_Segmentation`` validation split and resolve the
    local cache paths.

    Returns:
        ``(img_dir, ann_file)`` — paths to ``val2017/`` image folder and
        ``instances_val2017.json`` annotation file.
    """
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.constant import DownloadMode

    cache_dir = cache_dir or os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope"
    )
    os.makedirs(cache_dir, exist_ok=True)

    print("[coco2017] Loading COCO2017 validation split from ModelScope ...")
    ds = MsDataset.load(
        "COCO2017_Instance_Segmentation",
        split="validation",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        cache_dir=cache_dir,
    )

    # MsDataset.config_kwargs contains the resolved data paths
    config_kwargs = getattr(ds, "config_kwargs", {}) or {}
    data_dir = config_kwargs.get("data_dir") or config_kwargs.get("split_data_dir")

    # Fallback: try to find the dataset directory from the MsDataset internals
    if not data_dir or not os.path.isdir(data_dir):
        # Walk the ModelScope cache to find val2017 + instances_val2017.json
        base = cache_dir
        for root, dirs, files in os.walk(base):
            if "val2017" in dirs and "annotations" in os.listdir(root):
                ann_dir = os.path.join(root, "annotations")
                if any(f.startswith("instances_val2017") for f in os.listdir(ann_dir)):
                    data_dir = root
                    break

    if not data_dir or not os.path.isdir(data_dir):
        raise RuntimeError(
            "[coco2017] Could not locate downloaded COCO2017 validation data. "
            f"cache_dir={cache_dir}"
        )

    img_dir = os.path.join(data_dir, "val2017")
    ann_dir = os.path.join(data_dir, "annotations")

    # Locate the instances annotation file (name may vary slightly)
    ann_file = None
    for fname in os.listdir(ann_dir):
        if "instances" in fname and "val2017" in fname and fname.endswith(".json"):
            ann_file = os.path.join(ann_dir, fname)
            break

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"[coco2017] val2017 image folder not found at {img_dir}"
        )
    if ann_file is None or not os.path.isfile(ann_file):
        raise FileNotFoundError(
            f"[coco2017] instances_val2017.json not found under {ann_dir}"
        )

    print(f"[coco2017] Images: {img_dir}")
    print(f"[coco2017] Annotations: {ann_file}")
    return img_dir, ann_file


def _build_index_from_annotations(
    ann_file: str,
    img_dir: str,
    max_samples: int = -1,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """Parse COCO instances JSON → list of (image_path, primary_category_index).

    For each image, the most frequent category is used as "primary" category
    for prompt generation.

    Returns:
        ``(image_paths, category_indices)`` — parallel lists, category
        indices are in [0, 80).
    """
    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Count categories per image
    img_to_cats: dict = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        idx = COCO_ID_TO_INDEX.get(cat_id, -1)
        if idx < 0:
            continue
        if img_id not in img_to_cats:
            img_to_cats[img_id] = {}
        img_to_cats[img_id][idx] = img_to_cats[img_id].get(idx, 0) + 1

    # Build id -> file_name map
    id_to_fname = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

    image_paths: List[str] = []
    category_indices: List[int] = []

    all_img_ids = list(id_to_fname.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_img_ids)

    for img_id in all_img_ids:
        fname = id_to_fname[img_id]
        fpath = os.path.join(img_dir, fname)
        if not os.path.isfile(fpath):
            continue
        counts = img_to_cats.get(img_id, {})
        if counts:
            primary_idx = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            # No annotations → skip (or use 0 fallback)
            continue
        image_paths.append(fpath)
        category_indices.append(primary_idx)
        if max_samples > 0 and len(image_paths) >= max_samples:
            break

    print(f"[coco2017] Built index: {len(image_paths)} images.")
    return image_paths, category_indices


# ---------------------------------------------------------------------------
# COCO2017 Dataset Classes
# ---------------------------------------------------------------------------

class COCO2017RawDataset(Dataset):
    """Raw COCO 2017 validation dataset returning images and generated prompts.

    Yields ``(image, category_idx, prompt)`` tuples where:
    - ``image``: (3, H, W) tensor in [-1, 1]
    - ``category_idx``: int in [0, 80)
    - ``prompt``: str generated from category name via templates
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        train: bool = False,  # kept for API compatibility; only val is used
        image_size: int = 512,
        max_samples: int = -1,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True,
    ):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        data_dir = f"{data_dir}/modelscope/COCO2017_Instance_Segmentation/master/data_files//extracted//d376d18ebab3e013b155c78acc3f3b8fb038e6e910724cc2f0482441efa5a74a/COCO2017val"

        # Resolve image dir + annotation file
        if local_files_only or data_dir:
            if not os.path.isdir(data_dir):
                raise ValueError(f"local_files_only=True, but data_dir={data_dir} is not a directory")
            img_dir = os.path.join(data_dir, "val2017")
            ann_dir = os.path.join(data_dir, "annotations")
            ann_file = None
            if os.path.isdir(ann_dir):
                for fname in os.listdir(ann_dir):
                    if "instances" in fname and "val2017" in fname and fname.endswith(".json"):
                        ann_file = os.path.join(ann_dir, fname)
                        break
            if not os.path.isdir(img_dir) or ann_file is None:
                raise FileNotFoundError(
                    f"[coco2017] Local data_dir '{data_dir}' is missing "
                    f"val2017/ or annotations/instances_val2017.json"
                )
        else:
            img_dir, ann_file = _download_coco2017_val_via_modelscope(cache_dir)

        self.img_dir = img_dir
        self.ann_file = ann_file

        image_paths, category_indices = _build_index_from_annotations(
            ann_file, img_dir, max_samples=max_samples,
        )

        # 90/10 split for train/val (even though the source is the val split)
        n_total = len(image_paths)
        n_train = int(n_total * 0.9)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_total)

        if train:
            sel = perm[:n_train]
        else:
            sel = perm[n_train:]

        self.image_paths = [image_paths[i] for i in sel]
        self.category_indices = [category_indices[i] for i in sel]

        print(
            f"[coco2017] COCO2017RawDataset({'train' if train else 'val'}): "
            f"{len(self.image_paths)} samples, image_size={image_size}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        cat_idx = self.category_indices[idx]
        class_name = COCO_CATEGORIES[cat_idx] if 0 <= cat_idx < len(COCO_CATEGORIES) else "object"
        prompt = generate_prompt(class_name)

        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, cat_idx, prompt


class COCO2017LatentDataset(Dataset):
    """COCO 2017 latent dataset for text-conditional diffusion training.

    Computes VAE latents and text embeddings on-the-fly during training,
    avoiding memory overhead of pre-computing all data.

    Yields ``(latent, encoder_hidden_states)`` tuples.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        train: bool = False,
        image_size: int = 512,
        max_samples: int = -1,
        cache_dir: Optional[str] = None,
        vae=None,
        tokenizer=None,
        text_encoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_token_length: int = 77,
    ):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_token_length = max_token_length

        if vae is not None:
            self.vae_scale = vae.config.scaling_factor

        # Reuse the raw dataset logic to build index
        raw_ds = COCO2017RawDataset(
            data_dir=data_dir, train=train, image_size=image_size,
            max_samples=max_samples, cache_dir=cache_dir,
        )
        self.image_paths = raw_ds.image_paths
        self.category_indices = raw_ds.category_indices
        self.transform = raw_ds.transform

        print(
            f"[coco2017] COCO2017LatentDataset({'train' if train else 'val'}): "
            f"{len(self.image_paths)} samples"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_idx = self.category_indices[idx]
        class_name = COCO_CATEGORIES[cat_idx] if 0 <= cat_idx < len(COCO_CATEGORIES) else "object"
        prompt = generate_prompt(class_name)

        img = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(img)

        with torch.no_grad():
            encoder_output = self.vae.encode(image.unsqueeze(0).to(self.device, self.dtype))
            if hasattr(encoder_output, "latent_dist"):
                latent = encoder_output.latent_dist.sample()
            elif hasattr(encoder_output, "latent"):
                latent = encoder_output.latent
            elif hasattr(encoder_output, "latents"):
                latent = encoder_output.latents
            elif isinstance(encoder_output, torch.Tensor):
                latent = encoder_output
            else:
                latent = encoder_output
            latent = latent * self.vae_scale

            text_inputs = self.tokenizer(
                prompt,
                max_length=self.max_token_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_emb = self.text_encoder(
                text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
            ).last_hidden_state

        return latent.squeeze(0), text_emb.squeeze(0)


# ---------------------------------------------------------------------------
# DataLoader Functions
# ---------------------------------------------------------------------------

def get_coco2017_dataloader(
    batch_size: int = 128,
    train: bool = False,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    image_size: int = 256,
    max_samples: int = -1,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    vae=None,
    tokenizer=None,
    text_encoder=None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    max_token_length: int = 77,
) -> DataLoader:
    """Create a COCO 2017 (validation split) DataLoader.

    Supports two modes:

    1. **Raw mode** (no VAE/tokenizer/text_encoder) — returns
       ``(image, category_idx, prompt)`` tuples.
    2. **Latent mode** (VAE + tokenizer + text_encoder all provided) —
       returns ``(latent, encoder_hidden_states)`` tuples, computing them
       on-the-fly.

    The dataset is downloaded from ModelScope (``PAI/COCO2017`` →
    ``COCO2017_Instance_Segmentation`` validation split) when ``data_dir``
    is not provided or does not contain the expected files.

    Args:
        batch_size: Samples per batch.
        train: True → 90% split (shuffled), False → 10% split.  Note that
               the source data is always COCO 2017 *validation* split
               (5000 images), per project requirements.
        data_dir: Optional local root containing ``val2017/`` and
                  ``annotations/instances_val2017.json``.  When provided
                  and valid, ModelScope download is skipped.
        cache_dir: ModelScope download cache override.
        image_size: Resize target (square).
        max_samples: Cap on total images loaded from the annotation file
                     (-1 = all 5000 val images).
        num_workers: DataLoader subprocesses.
        pin_memory: pin_memory for GPU.
        persistent_workers: persistent_workers flag.
        vae: Frozen VAE for image→latent encoding (latent mode).
        tokenizer: Tokenizer for prompt processing (latent mode).
        text_encoder: Text encoder for embedding generation (latent mode).
        device: Compute device (latent mode).
        dtype: Compute dtype (latent mode).
        max_token_length: Maximum token sequence length.

    Returns:
        torch.utils.data.DataLoader
    """
    if vae is not None and tokenizer is not None and text_encoder is not None:
        dataset = COCO2017LatentDataset(
            data_dir=data_dir,
            train=train,
            image_size=image_size,
            max_samples=max_samples,
            cache_dir=cache_dir,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            dtype=dtype,
            max_token_length=max_token_length,
        )
    else:
        dataset = COCO2017RawDataset(
            data_dir=data_dir,
            train=train,
            image_size=image_size,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )

    # drop_last only makes sense for training: a partial final batch would
    # skew training statistics.  For validation / test we must keep the last
    # (possibly partial) batch, otherwise datasets smaller than batch_size
    # yield zero batches and ``next(iter(loader))`` raises StopIteration.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=train,
    )

    return loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # import sys
    # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # os.makedirs("./outputs", exist_ok=True)
    """
    python -m src.data.coco2017
    """

    # ---- Test 1: Raw DataLoader (small subset) ----
    print("=" * 60)
    print("Test 1: COCO2017 Raw DataLoader")
    print("=" * 60)

    val_loader = get_coco2017_dataloader(
        batch_size=8,
        train=False,
        image_size=256,
        max_samples=50,
        # cache_dir="G://datasets",
        data_dir=r"G://datasets//",  # uncomment to use local
    )
    images, labels, prompts = next(iter(val_loader))
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch shape: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    print(f"  Image range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"  Sample labels: {labels[:5].tolist()}")
    print(f"  Sample prompts:")
    for i, p in enumerate(prompts[:5]):
        print(f"    {i+1}. {p}")
    # print("[OK] Raw DataLoader test passed.\n")

    # ---- Test 2: Latent DataLoader (requires real models) ----
    # """
    from src.utils import _find_or_download_component
    from diffusers import AutoencoderKL
    from transformers import BertTokenizer, BertModel, BertConfig

    cache_dir = "G://models"
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_path = _find_or_download_component(
        "stabilityai/sd-vae-ft-mse", cache_dir,
        ["config.json", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
    )
    text_encoder_path = _find_or_download_component(
        "iic/multi-modal_clip-vit-base-patch16_zh", cache_dir,
        ["config.json", "pytorch_model.bin", "text_model_config.json", "vocab.txt"],
    )

    vae = AutoencoderKL.from_pretrained(
        vae_path, cache_dir=cache_dir, local_files_only=True,
    ).to(device).to(dtype)
    tokenizer = BertTokenizer.from_pretrained(
        text_encoder_path, cache_dir=cache_dir, local_files_only=True,
    )
    config = BertConfig.from_dict({
        "vocab_size": 21128, "hidden_size": 768, "num_hidden_layers": 12,
        "num_attention_heads": 12, "intermediate_size": 3072, "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512, "type_vocab_size": 2, "initializer_range": 0.02,
    })
    text_encoder = BertModel(config).to(device).to(dtype)

    val_loader = get_coco2017_dataloader(
        batch_size=4, train=False, image_size=256, max_samples=20,
        data_dir=r"G://datasets//",  # use local data (same as Test 1)
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=dtype,
    )
    latents, txt_embs = next(iter(val_loader))
    print(f"Latent shape: {tuple(latents.shape)}")
    print(f"Text emb shape: {tuple(txt_embs.shape)}")
    # print("[OK] Latent DataLoader test passed.")
    # """
