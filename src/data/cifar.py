"""
CIFAR-100 data loading with prompt generation for text-conditional diffusion training.

CIFAR-100 dataset contains 100 classes grouped into 20 superclasses.
Since CIFAR-100 does not contain text prompts, we generate prompts from class labels
using template-based augmentation for text-conditional diffusion training.

Usage::

    from src.data.cifar import get_cifar100_dataloader, CIFAR100_CLASSES, generate_prompt

    train_loader = get_cifar100_dataloader(batch_size=128, train=True)
    images, labels, prompts = next(iter(train_loader))

    # With VAE and tokenizer for latent space training
    train_loader, val_loader = get_cifar100_dataloader(
        batch_size=64, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.float32,
    )
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CIFAR-100 Class Names
# ---------------------------------------------------------------------------

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
]

CIFAR100_SUPERCLASSES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals",
    "trees", "vehicles_1", "vehicles_2"
]


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
    "a cute {}",
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
    "a {} in the ocean",
    "a {} on the beach",
    "a {} in the city",
    "a {} at night",
    "a {} during the day",
]


def generate_prompt(class_name: str, template_idx: Optional[int] = None) -> str:
    """Generate a text prompt from a CIFAR-100 class name using templates.

    Args:
        class_name: CIFAR-100 class name (e.g., "apple", "cat")
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
    return template.format(class_name.replace("_", " "))


def generate_prompts_for_labels(labels: torch.Tensor) -> List[str]:
    """Generate prompts for a batch of class labels.

    Args:
        labels: Tensor of class indices, shape (B,)

    Returns:
        List[str]: List of generated prompts
    """
    prompts = []
    for label in labels:
        class_name = CIFAR100_CLASSES[label.item()]
        prompts.append(generate_prompt(class_name))
    return prompts


# ---------------------------------------------------------------------------
# CIFAR-100 Dataset Classes
# ---------------------------------------------------------------------------

class CIFAR100RawDataset(Dataset):
    """Raw CIFAR-100 dataset returning images and generated prompts.

    Yields (image, label, prompt) tuples where:
    - image: (3, H, W) tensor in [-1, 1]
    - label: int class index
    - prompt: str generated text prompt
    """

    def __init__(self, root: str = "G://datasets//cifar-100-python", train: bool = True, image_size: int = 64):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = datasets.CIFAR100(
            root=root, train=train, download=True, transform=self.transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image, label = self.dataset[idx]
        class_name = CIFAR100_CLASSES[label]
        prompt = generate_prompt(class_name)
        return image, label, prompt


def _resolve_precomputed_path(precomputed_path: str, train: bool, image_size: int) -> str:
    """Resolve the concrete cache file path.

    If ``precomputed_path`` points to a directory (or a path with missing
    extension), a split-specific filename is constructed inside it so train
    and test sets with different image sizes do not collide.
    """
    if precomputed_path is None:
        return None
    ext = os.path.splitext(precomputed_path)[1].lower()
    if ext in (".pt", ".pth", ".bin"):
        return precomputed_path
    os.makedirs(precomputed_path, exist_ok=True)
    split = "train" if train else "test"
    fname = f"{split}_imgsz_{image_size}.pt"
    return os.path.join(precomputed_path, fname)


class CIFAR100LatentDataset(Dataset):
    """CIFAR-100 latent dataset for text-conditional diffusion training.

    Two modes:

    1. **Precomputed-cache mode** (``precomputed_path is not None``):
       VAE latents and text embeddings are loaded from / saved to a single
       ``.pt`` file on disk.  Encoding happens in one batched pass the first
       time, which is much faster than per-sample GPU kernels.  After that,
       ``__getitem__`` returns pure CPU tensors so ``num_workers > 0`` and
       ``pin_memory = True`` become safe and beneficial.

    2. **On-the-fly mode** (``precomputed_path is None``):
       Keeps the original behaviour — ``vae.encode`` and ``text_encoder`` are
       called inside ``__getitem__``.  Requires ``num_workers = 0`` because
       CUDA ops cannot run in DataLoader worker processes.

    Yields tuples:
    - latent: (C, H, W) VAE latent tensor
    - encoder_hidden_states: (seq_len, dim) text encoder output
    """

    def __init__(
        self,
        root: str = "G://datasets//cifar-100-python",
        train: bool = True,
        image_size: int = 64,
        vae=None,
        tokenizer=None,
        text_encoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_token_length: int = 77,
        precomputed_path: str = None,
    ):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_token_length = max_token_length
        self.image_size = image_size
        self.train = train

        if vae is not None:
            self.vae_scale = vae.config.scaling_factor

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )

        # ---- Precomputed cache handling ---------------------------------
        # When precomputed_path is None we fall through and use the legacy
        # on-the-fly encode path in __getitem__ (requires num_workers=0).
        self._cached_latents = None
        self._cached_class_text_embs = None  # (num_classes, seq_len, dim)
        self._cached_labels = None           # (N,) int64 — for label lookup
        self._cache_path = _resolve_precomputed_path(precomputed_path, train, image_size)

        if self._cache_path is not None and os.path.exists(self._cache_path):
            print(f"[data] Loading precomputed CIFAR-100 cache from: {self._cache_path}")
            obj = torch.load(self._cache_path, map_location="cpu")
            self._cached_latents = obj["latents"]
            self._cached_class_text_embs = obj["class_text_embs"]
            self._cached_labels = obj["labels"]
            if len(self._cached_latents) != len(self.dataset):
                raise RuntimeError(
                    f"Cached latents length {len(self._cached_latents)} does not match "
                    f"dataset length {len(self.dataset)} at {self._cache_path}. "
                    f"Please delete the stale cache and retry."
                )
            print(f"[data] Loaded cache: latents {tuple(self._cached_latents.shape)}, "
                  f"class_text_embs {tuple(self._cached_class_text_embs.shape)}, "
                  f"labels {tuple(self._cached_labels.shape)}")

        elif self._cache_path is not None:
            # Need to generate the cache.  Models must be supplied.
            missing = [n for n, m in (("vae", vae), ("tokenizer", tokenizer), ("text_encoder", text_encoder)) if m is None]
            if missing:
                raise RuntimeError(
                    f"precomputed_path={self._cache_path} does not exist, "
                    f"but {missing} were not provided to build the cache. "
                    f"Pass vae/tokenizer/text_encoder or an existing cache file."
                )
            print(f"[data] precomputed_path not yet on disk; building cache once: {self._cache_path}")
            latents, class_text_embs, labels = self._precompute_all(precompute_batch_size=256)
            self._cached_latents = latents
            self._cached_class_text_embs = class_text_embs
            self._cached_labels = labels
            save_dir = os.path.dirname(self._cache_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {"latents": latents, "class_text_embs": class_text_embs, "labels": labels},
                self._cache_path,
            )
            mem_mb = (latents.nelement() * latents.element_size()
                      + class_text_embs.nelement() * class_text_embs.element_size()
                      + labels.nelement() * labels.element_size()) / 1024 / 1024
            print(f"[data] Cache written to disk ({mem_mb:.1f} MB).")

        # Fall-through: if precomputed_path is None we keep the legacy path.
        print(f"[data] CIFAR100LatentDataset initialized: {len(self.dataset)} samples "
              f"(cached={self._cached_latents is not None})")

    # ------------------------------------------------------------------
    # Batched precomputation
    # ------------------------------------------------------------------
    def _precompute_all(self, precompute_batch_size: int = 256):
        """Encode the whole split in large batches and return CPU tensors.

        To avoid the 11.8 GB memory blow-up from caching per-sample text
        embeddings (50000 x 77 x 768 x 4B), we only cache:
          - image latents:  (N, C, H, W)            — ~12.8 MB for CIFAR-100
          - class_text_embs:(num_classes, seq, dim) — ~23.7 MB (100 classes)
          - labels:         (N,) int64              — ~0.4 MB
        ``__getitem__`` looks up the text embedding by label at access time,
        so prompt diversity is reduced to one fixed template per class in
        cached mode (on-the-fly mode still uses random templates).
        """
        N = len(self.dataset)
        num_classes = len(CIFAR100_CLASSES)

        # Probe latent shape with a single sample.
        img0, _ = self.dataset[0]
        with torch.no_grad():
            enc = self.vae.encode(img0.unsqueeze(0).to(self.device, self.dtype))
            if hasattr(enc, "latent_dist"):
                z0 = enc.latent_dist.sample()
            elif hasattr(enc, "latent"):
                z0 = enc.latent
            elif hasattr(enc, "latents"):
                z0 = enc.latents
            else:
                z0 = enc
            z0 = (z0 * self.vae_scale).squeeze(0).cpu()
        latent_shape = tuple(z0.shape)  # e.g. (4, 4, 4)

        all_latents = torch.empty((N,) + latent_shape, dtype=z0.dtype, device="cpu")
        all_labels = torch.empty(N, dtype=torch.long, device="cpu")

        # Build a raw-image DataLoader so we can feed the VAE with big batches.
        raw_loader = DataLoader(
            self.dataset,
            batch_size=precompute_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        offset = 0
        pbar = tqdm(raw_loader, desc=f"Precomputing {'train' if self.train else 'test'} latents")
        with torch.no_grad():
            for images, labels in pbar:
                B = images.shape[0]
                images_dev = images.to(self.device, self.dtype)

                # --- VAE encode ---
                enc = self.vae.encode(images_dev)
                if hasattr(enc, "latent_dist"):
                    lat = enc.latent_dist.sample()
                elif hasattr(enc, "latent"):
                    lat = enc.latent
                elif hasattr(enc, "latents"):
                    lat = enc.latents
                else:
                    lat = enc
                lat = (lat * self.vae_scale).cpu()

                all_latents[offset:offset + B].copy_(lat)
                all_labels[offset:offset + B].copy_(labels)
                offset += B

        # --- Encode one prompt per class (fixed template) ----------------
        # Using PROMPT_TEMPLATES[0] = "a photo of a {}" for deterministic,
        # compact cache.  On-the-fly mode (precomputed_path=None) still
        # uses random templates via generate_prompt().
        class_prompts = [PROMPT_TEMPLATES[0].format(c.replace("_", " ")) for c in CIFAR100_CLASSES]
        with torch.no_grad():
            tok = self.tokenizer(
                class_prompts, max_length=self.max_token_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            class_text_embs = self.text_encoder(
                tok.input_ids.to(self.device),
                attention_mask=tok.attention_mask.to(self.device),
            ).last_hidden_state.cpu()

        return all_latents, class_text_embs, all_labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cached mode: pure CPU tensor lookup — safe with num_workers > 0.
        if self._cached_latents is not None:
            label = self._cached_labels[idx].item()
            return self._cached_latents[idx], self._cached_class_text_embs[label], F.one_hot(torch.tensor(label), num_classes=100)

        # On-the-fly mode (precomputed_path=None): per-sample GPU encode.
        # Requires num_workers=0 because VAE/text_encoder live on CUDA.
        image, label = self.dataset[idx]
        class_name = CIFAR100_CLASSES[label]
        prompt = generate_prompt(class_name)

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

        return latent.squeeze(0), text_emb.squeeze(0), F.one_hot(torch.tensor(label), num_classes=100)


# ---------------------------------------------------------------------------
# DataLoader Functions
# ---------------------------------------------------------------------------

def get_cifar100_dataloader(
    batch_size: int = 128,
    train: bool = True,
    root: str = "G://datasets//cifar-100-python",
    image_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    vae=None,
    tokenizer=None,
    text_encoder=None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    max_token_length: int = 77,
    precomputed_path: str = None,
) -> DataLoader:
    """Create a CIFAR-100 DataLoader.

    Supports two modes:
    1. Raw mode (no VAE/tokenizer): Returns (image, label, prompt) tuples
    2. Latent mode (with VAE/tokenizer/text_encoder): Returns dicts with latents and text embeddings

    Args:
        batch_size: Number of samples per batch
        train: True for training set, False for test set
        root: Directory to store/download CIFAR-100 data
        image_size: Target image size
        num_workers: DataLoader worker processes
        pin_memory: Enable pin_memory for GPU training
        vae: Frozen VAE for image→latent encoding (optional)
        tokenizer: Tokenizer for text processing (optional)
        text_encoder: Text encoder for embedding generation (optional)
        device: Target device for pre-computation
        dtype: Data type for pre-computation
        max_token_length: Maximum token length for text encoder
        precomputed_path: Path to a ``.pt`` file OR a directory.  When set,
            VAE latents and text embeddings are encoded once in large
            batches (256) and persisted on disk; subsequent runs load them
            directly from disk so ``__getitem__`` becomes a pure CPU
            tensor lookup (safe and fast with ``num_workers > 0``).

    Returns:
        DataLoader for CIFAR-100
    """
    print(
        f"[data] Loading cifar100 from [{root}] [{image_size} x {image_size}] batch size: {batch_size}"
        f" using vae={vae is not None}, tokenizer={tokenizer is not None}, text_encoder={text_encoder is not None}"
        f" precomputed_path={precomputed_path}"
    )
    # If an existing precomputed cache is specified, we can enter latent mode
    # without the models — they are only needed the very first time to build
    # the cache file.
    has_latent_parts = (
        (vae is not None and tokenizer is not None and text_encoder is not None)
        or precomputed_path is not None
    )
    if has_latent_parts:
        dataset = CIFAR100LatentDataset(
            root=root,
            train=train,
            image_size=image_size,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            dtype=dtype,
            max_token_length=max_token_length,
            precomputed_path=precomputed_path,
        )
    else:
        dataset = CIFAR100RawDataset(
            root=root,
            train=train,
            image_size=image_size,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    return loader


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import torch
    from diffusers import AutoencoderKL
    from transformers import BertTokenizer
    from transformers import BertModel, BertConfig

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils import _find_or_download_component

    # Raw Mode
    """
    print("=" * 50)
    print("Test 1: CIFAR-100 Raw DataLoader")
    print("=" * 50)
    train_loader = get_cifar100_dataloader(batch_size=16, train=True, image_size=64, root="G://datasets//cifar-100-python")
    test_loader = get_cifar100_dataloader(batch_size=16, train=False, image_size=64, root="G://datasets//cifar-100-python")

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels, prompts = next(iter(train_loader))
    print(f"Batch shape: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    print(f"Image range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"Sample labels: {labels[:8].tolist()}")
    print(f"Sample prompts: {prompts[:3]}")
    """

    # Latent Mode
    cache_dir = "G://models"
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
     
    vae_path = _find_or_download_component("stabilityai/sd-vae-ft-mse", cache_dir, ["config.json", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"])
    text_encoder_path = _find_or_download_component("iic/multi-modal_clip-vit-base-patch16_zh", cache_dir, ["config.json", "pytorch_model.bin", "text_model_config.json", "vocab.txt"])
        
    vae = AutoencoderKL.from_pretrained(
        vae_path, 
        cache_dir=cache_dir, 
        local_files_only=True
    ).to(device).to(dtype)
    
    tokenizer = BertTokenizer.from_pretrained(
        text_encoder_path,
        cache_dir=cache_dir, 
        local_files_only=True
    )

    config = BertConfig.from_dict({
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
    })
    text_encoder = BertModel(config).to(device).to(dtype)
    train_loader = get_cifar100_dataloader(
        # batch_size=16, train=True, image_size=64, 
        # root="G://datasets//cifar-100-python", 
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, device=device, dtype=dtype,
        precomputed_path="G://datasets//cifar-100-python//latent_cache"
    )
    latents, txt_embs = next(iter(train_loader))
    print(latents.shape, txt_embs.shape)

    """
    print("\n" + "=" * 50)
    print("Test 2: Prompt Generation")
    print("=" * 50)

    for i in range(5):
        prompt = generate_prompt(CIFAR100_CLASSES[10])
        print(f"  {i+1}: {prompt}")

    print("\n" + "=" * 50)
    print("Test 3: CIFAR-100 Class Names")
    print("=" * 50)
    print(f"Total classes: {len(CIFAR100_CLASSES)}")
    print(f"First 10 classes: {CIFAR100_CLASSES[:10]}")
    print(f"Last 10 classes: {CIFAR100_CLASSES[-10:]}")

    print("\nAll CIFAR-100 dataloader tests passed!")
    """