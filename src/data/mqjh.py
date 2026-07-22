"""
MJHQ-30K data loading for text-conditional diffusion training.

MJHQ-30K is a dataset containing 30K high-quality text-image pairs organized by category.
The dataset structure:
- Root directory contains category subdirectories (animals, art, fashion, etc.)
- Each category contains image files named by hash (e.g., abc123.jpg)
- meta_data.json maps hash names to prompts and categories

Since this dataset contains actual text prompts, we don't need to generate them.

Usage::

    from src.data.mqjh import get_mqjh30k_dataloader

    train_loader = get_mqjh30k_dataloader(batch_size=128, train=True)
    images, prompts = next(iter(train_loader))

    # With VAE and tokenizer for latent space training
    train_loader, val_loader = get_mqjh30k_dataloader(
        batch_size=64, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.float32,
    )
"""

import os
import json
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MJHQ-30K Dataset Classes
# ---------------------------------------------------------------------------

class MJHQ30KRawDataset(Dataset):
    """Raw MJHQ-30K dataset returning images and prompts.

    Yields (image, prompt) tuples where:
    - image: (3, H, W) tensor in [-1, 1]
    - prompt: str text prompt
    """

    def __init__(self, root: str = "./data", train: bool = True, image_size: int = 512):
        self.root = root
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.data = []
        self._load_data(train)

    def _load_data(self, train: bool):
        meta_path = os.path.join(self.root, "meta_data.json")
        if not os.path.exists(meta_path):
            raise ValueError(f"meta_data.json not found in {self.root}")
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        
        categories = [d for d in os.listdir(self.root) 
                     if os.path.isdir(os.path.join(self.root, d)) and d != "_gitee_clone"]
        
        all_data = []
        for category in categories:
            cat_dir = os.path.join(self.root, category)
            for filename in os.listdir(cat_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(cat_dir, filename)
                    hash_name = os.path.splitext(filename)[0]
                    if hash_name in meta_data:
                        prompt = meta_data[hash_name]["prompt"]
                    else:
                        prompt = ""
                    all_data.append((image_path, prompt))
        
        if train:
            self.data = all_data[:int(len(all_data) * 0.9)]
        else:
            self.data = all_data[int(len(all_data) * 0.9):]
        
        print(f"[data] MJHQ30KRawDataset initialized: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path, prompt = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, prompt


class MJHQ30KLatentDataset(Dataset):
    """MJHQ-30K latent dataset for text-conditional diffusion training.

    Computes VAE latents and text embeddings on-the-fly during training,
    avoiding memory overhead of pre-computing all data.

    Yields tuples:
    - latent: (C, H, W) VAE latent tensor
    - encoder_hidden_states: (seq_len, dim) text encoder output
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 512,
        vae=None,
        tokenizer=None,
        text_encoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_token_length: int = 77,
    ):
        self.root = root
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_token_length = max_token_length

        if vae is not None:
            self.vae_scale = vae.config.scaling_factor

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.transform = transform
        self.data = []
        self._load_data(train)

    def _load_data(self, train: bool):
        meta_path = os.path.join(self.root, "meta_data.json")
        if not os.path.exists(meta_path):
            raise ValueError(f"meta_data.json not found in {self.root}")
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        
        categories = [d for d in os.listdir(self.root) 
                     if os.path.isdir(os.path.join(self.root, d)) and d != "_gitee_clone"]
        
        all_data = []
        for category in categories:
            cat_dir = os.path.join(self.root, category)
            for filename in os.listdir(cat_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(cat_dir, filename)
                    hash_name = os.path.splitext(filename)[0]
                    if hash_name in meta_data:
                        prompt = meta_data[hash_name]["prompt"]
                    else:
                        prompt = ""
                    all_data.append((image_path, prompt))
        
        if train:
            self.data = all_data[:int(len(all_data) * 0.9)]
        else:
            self.data = all_data[int(len(all_data) * 0.9):]
        
        print(f"[data] MJHQ30KLatentDataset initialized: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, prompt = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        with torch.no_grad():
            encoder_output = self.vae.encode(image.unsqueeze(0).to(self.device, self.dtype))
            if hasattr(encoder_output, 'latent_dist'):
                latent = encoder_output.latent_dist.sample()
            elif hasattr(encoder_output, 'latent'):
                latent = encoder_output.latent
            elif hasattr(encoder_output, 'latents'):
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

def get_mqjh30k_dataloader(
    batch_size: int = 128,
    train: bool = True,
    root: str = "./data",
    image_size: int = 512,
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
    """Create a MJHQ-30K DataLoader.

    Supports two modes:
    1. Raw mode (no VAE/tokenizer): Returns (image, prompt) tuples
    2. Latent mode (with VAE/tokenizer/text_encoder): Returns dicts with latents and text embeddings

    Args:
        batch_size: Number of samples per batch
        train: True for training set, False for test/val set
        root: Directory containing MJHQ-30K data (should have train/val subdirs)
        image_size: Target image size
        num_workers: DataLoader worker processes
        pin_memory: Enable pin_memory for GPU training
        persistent_workers: Enable persistent_workers for faster data loading
        vae: Frozen VAE for image→latent encoding (optional)
        tokenizer: Tokenizer for text processing (optional)
        text_encoder: Text encoder for embedding generation (optional)
        device: Target device for pre-computation
        dtype: Data type for pre-computation
        max_token_length: Maximum token length for text encoder

    Returns:
        DataLoader for MJHQ-30K
    """
    if vae is not None and tokenizer is not None and text_encoder is not None:
        dataset = MJHQ30KLatentDataset(
            root=root,
            train=train,
            image_size=image_size,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            dtype=dtype,
            max_token_length=max_token_length,
        )
    else:
        dataset = MJHQ30KRawDataset(
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


if __name__ == "__main__":
    
    import torch
    from diffusers import AutoencoderKL
    from transformers import BertTokenizer
    from transformers import BertModel, BertConfig

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils import _find_or_download_component

    # Raw Mode
    # """
    print("=" * 50)
    print("Test 1: Raw Dataloader")
    print("=" * 50)
    train_loader = get_mqjh30k_dataloader(batch_size=16, train=True, image_size=512, root="G://datasets//MJHQ-30K")
    test_loader = get_mqjh30k_dataloader(batch_size=16, train=False, image_size=512, root="G://datasets//MJHQ-30K")

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, prompts = next(iter(train_loader))
    print(f"Batch shape: images={tuple(images.shape)}")
    # print(f"Image range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"Sample prompts: {prompts[:3]}")
    # """

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
    train_loader = get_mqjh30k_dataloader(
        batch_size=16, train=True, image_size=512, 
        root="G://datasets//MJHQ-30K", 
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, device=device, dtype=dtype
    )
    latents, txt_embs = next(iter(train_loader))
    print(latents.shape, txt_embs.shape)
