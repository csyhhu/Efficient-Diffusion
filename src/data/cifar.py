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


class CIFAR100LatentDataset(Dataset):
    """CIFAR-100 latent dataset for text-conditional diffusion training.

    Computes VAE latents and text embeddings on-the-fly during training,
    avoiding memory overhead of pre-computing all data.

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
    ):
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
        self.dataset = datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )

        print(f"[data] CIFAR100LatentDataset initialized: {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return latent.squeeze(0), text_emb.squeeze(0)


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

    Returns:
        DataLoader for CIFAR-100
    """
    print(
        f">> [data] Loading cifar100 from [{root}] [{image_size} x {image_size}] batch size: {batch_size}" 
        f" using vae={vae is not None}, tokenizer={tokenizer is not None}, text_encoder={text_encoder is not None}"
    )
    if vae is not None and tokenizer is not None and text_encoder is not None:
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
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, device=device, dtype=dtype
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