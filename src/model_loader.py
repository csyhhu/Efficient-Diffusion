"""
Model loader: unified entry point for DiT / QuantizedDiT / QuantizedSD3 / QuantizedPixArt.

Returns (model, vae, tokenizer) — for DiT-family models, vae/tokenizer are None.
Supports dry-run mode: all components (DiT, VAE, tokenizer) are randomly initialized.

Usage::

    from src.model_loader import load_model

    model, vae, tokenizer = load_model("quantized_sd3", {
        "patch_size": 2, "in_channels": 16, "num_layers": 24,
        "attention_head_dim": 64, "num_attention_heads": 24,
        ...
        "pretrained_path": "stabilityai/stable-diffusion-3.5-medium",
        "subfolder": "transformer",
        "dtype": "float16",
        "huggingface_mirror": "https://hf-mirror.com",
    }, dry_run=False)
"""

import os
import sys
import copy
import logging

# Ensure project root is on sys.path so that ``from src.xxx`` works
# regardless of how this file is invoked.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn

from src.models.dit import DiT
from src.models.quantized_dit import QuantizedDiT
from src.models.quantized_SD3 import QuantizedSD3
from src.models.quantized_PixArt import QuantizedPixArt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name → class mapping
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "dit":              DiT,
    "simple":           DiT,          # alias
    "quantized_dit":    QuantizedDiT,
    "quantized_sd3":    QuantizedSD3,
    "quantized_pixart": QuantizedPixArt,
}

# Models that require VAE + tokenizer (T2I pipeline)
_T2I_MODELS = {"quantized_sd3", "quantized_pixart"}


# ---------------------------------------------------------------------------
# Random-initialised VAE (AutoencoderKL stub) for dry-run
# ---------------------------------------------------------------------------


class _RandomVAE(nn.Module):
    """A randomly-initialised VAE stub usable for dry-run T2I experiments.

    This is NOT a real AutoencoderKL — it only provides the minimal
    ``encode`` / ``decode`` signatures so that the training loop does not
    crash.  Internal weights are randomly initialised Conv2d layers.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 4,
                 scaling_factor: float = 0.13025):
        super().__init__()
        scaling_factor: float = 0.13025
        self.scaling_factor = scaling_factor

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 2× downsample
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor):
        """Return a LatentDistribution-like object with ``.latent_dist``."""
        class _LatentDist:
            def __init__(self, tensor, scale):
                self._tensor = tensor
                self.scale = scale

            def sample(self):
                return self._tensor * self.scale

        latent = self.encoder(x)
        return _LatentDist(latent, self.scaling_factor)

    def decode(self, z: torch.Tensor):
        """Decode latent back to pixel space."""
        return self.decoder(z / self.scaling_factor)


# ---------------------------------------------------------------------------
# Random-initialised tokenizer for dry-run
# ---------------------------------------------------------------------------

class _RandomTokenizer:
    """A dummy tokenizer that returns fixed-length random token IDs.

    Mimics the minimal interface required by T2I data loaders:
    ``tokenizer(text, padding, truncation, return_tensors, max_length)``.
    """

    def __init__(self, vocab_size: int = 49408, max_length: int = 77):
        self.vocab_size = vocab_size
        self.model_max_length = max_length

    def __call__(self, text, padding="max_length", truncation=True,
                 return_tensors="pt", max_length=None, **kwargs):
        max_len = max_length or self.model_max_length
        if isinstance(text, (list, tuple)):
            batch_size = len(text)
        else:
            batch_size = 1

        input_ids = torch.randint(0, self.vocab_size, (batch_size, max_len))
        attention_mask = torch.ones(batch_size, max_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __repr__(self):
        return f"RandomTokenizer(vocab_size={self.vocab_size}, max_length={self.model_max_length})"


# ---------------------------------------------------------------------------
# Real VAE loading (diffusers AutoencoderKL)
# ---------------------------------------------------------------------------

def _load_real_vae(pretrained_path: str, subfolder: str = "vae",
                   dtype: torch.dtype = torch.float16,
                   device: str = "cpu",
                   mirror: str | None = None) -> nn.Module:
    """Load diffusers AutoencoderKL from a pretrained HF repo."""
    from diffusers import AutoencoderKL

    kwargs = {}
    if mirror:
        os.environ.setdefault("HF_ENDPOINT", mirror)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    vae = AutoencoderKL.from_pretrained(
        pretrained_path,
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


# ---------------------------------------------------------------------------
# Real tokenizer loading
# ---------------------------------------------------------------------------

def _load_real_clip_tokenizer(pretrained_path: str, subfolder: str = "tokenizer",
                              mirror: str | None = None):
    """Load CLIPTokenizer (used by SD3)."""
    from transformers import CLIPTokenizer

    if mirror:
        os.environ.setdefault("HF_ENDPOINT", mirror)

    return CLIPTokenizer.from_pretrained(
        pretrained_path,
        subfolder=subfolder,
    )


def _load_real_t5_tokenizer(pretrained_path: str = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
                             subfolder: str = "tokenizer",
                             mirror: str | None = None):
    """Load T5TokenizerFast (used by PixArt)."""
    from transformers import AutoTokenizer

    if mirror:
        os.environ.setdefault("HF_ENDPOINT", mirror)

    return AutoTokenizer.from_pretrained(
        pretrained_path,
        subfolder=subfolder,
    )


# ---------------------------------------------------------------------------
# from_pretrained helpers
# ---------------------------------------------------------------------------

def _load_sd3_pretrained(model: QuantizedSD3, cfg: dict):
    """Load official SD3 weights into QuantizedSD3."""
    pretrained_path = cfg.get("pretrained_path",
                              "stabilityai/stable-diffusion-3.5-medium")
    subfolder = cfg.get("subfolder", "transformer")
    mirror = cfg.get("huggingface_mirror", None)
    if mirror:
        os.environ.setdefault("HF_ENDPOINT", mirror)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    logger.info("Loading pretrained SD3 weights from %s (subfolder=%s) ...",
                pretrained_path, subfolder)
    model.from_pretrained(pretrained_path, subfolder=subfolder)
    logger.info("SD3 weights loaded successfully.")


def _load_pixart_pretrained(model: QuantizedPixArt, cfg: dict):
    """Load official PixArt-Σ weights into QuantizedPixArt."""
    pretrained_path = cfg.get("pretrained_path",
                              "PixArt-alpha/PixArt-Sigma-XL-2-512-MS")
    subfolder = cfg.get("subfolder", "transformer")
    mirror = cfg.get("huggingface_mirror", None)
    if mirror:
        os.environ.setdefault("HF_ENDPOINT", mirror)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    logger.info("Loading pretrained PixArt weights from %s (subfolder=%s) ...",
                pretrained_path, subfolder)
    model.from_pretrained(pretrained_path, subfolder=subfolder)
    logger.info("PixArt weights loaded successfully.")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    model_config: dict | None = None,
    dry_run: bool = False,
) -> tuple[nn.Module, nn.Module | None, object | None]:
    """Load a model (and optionally VAE + tokenizer for T2I backbones).

    Args:
        model_name:
            One of ``"dit"``, ``"simple"``, ``"quantized_dit"``,
            ``"quantized_sd3"``, ``"quantized_pixart"``.
        model_config:
            Flat configuration dict.  All keys are forwarded directly to the
            model constructor.  Additional optional keys used by the loader:

            - ``dtype`` (str): ``"float16"``, ``"float32"``, etc. (default ``"float16"``).
            - ``pretrained_path`` (str): HF repo id for from_pretrained.
            - ``subfolder`` (str): subfolder inside the HF repo (default ``"transformer"``).
            - ``huggingface_mirror`` (str): optional HF mirror endpoint.
            - ``vae_subfolder`` (str): VAE subfolder (default ``"vae"``).
            - ``tokenizer_subfolder`` (str): tokenizer subfolder (default ``"tokenizer"``).

        dry_run:
            If True, all components (model, VAE, tokenizer) use random
            initialisation.  No weights are downloaded from HuggingFace.

    Returns:
        ``(model, vae, tokenizer)``.
        For DiT-family models, ``vae = None``, ``tokenizer = None``.
    """
    cfg: dict = copy.deepcopy(model_config) if model_config else {}

    # --- dtype from model_config, device auto-detected ---
    # dtype_str = cfg.pop("dtype", "float16")
    # dtype = getattr(torch, dtype_str, torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model class lookup ---
    model_cls = _MODEL_REGISTRY.get(model_name.lower())
    if model_cls is None:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    # --- Instantiate model (remaining cfg passed to constructor) ---
    logger.info("Instantiating %s with cfg keys: %s", model_name, list(cfg.keys()))
    model = model_cls(**cfg)

    # --- Load pretrained weights (non-dry-run, T2I only) ---
    if not dry_run and model_name.lower() in _T2I_MODELS:
        if model_name.lower() == "quantized_sd3":
            _load_sd3_pretrained(model, cfg)
        elif model_name.lower() == "quantized_pixart":
            _load_pixart_pretrained(model, cfg)

    # model = model.to(device=device, dtype=dtype)
    # model.train()

    # --- VAE + tokenizer ---
    vae = None
    tokenizer = None

    if model_name.lower() in _T2I_MODELS:
        pretrained_path = cfg.get(
            "pretrained_path",
            "stabilityai/stable-diffusion-3.5-medium"
            if model_name.lower() == "quantized_sd3"
            else "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        )
        mirror = cfg.get("huggingface_mirror", None)

        if dry_run:
            vae_in_channels = cfg.get("vae_in_channels", 3)
            vae_latent_channels = cfg.get("in_channels", 4)
            scaling_factor = cfg.get("vae_scaling_factor",
                                     0.13025 if "sd3" in model_name.lower() else 0.13025)
            vae = _RandomVAE(in_channels=vae_in_channels,
                             latent_channels=vae_latent_channels,
                             scaling_factor=scaling_factor)
            vae = vae.to(device=device, dtype=dtype)

            tokenizer = _RandomTokenizer(
                vocab_size=cfg.get("tokenizer_vocab_size", 49408),
                max_length=cfg.get("tokenizer_max_length", 77),
            )
            logger.info("Dry-run mode: using random VAE and tokenizer.")
        else:
            vae_subfolder = cfg.get("vae_subfolder", "vae")
            vae = _load_real_vae(pretrained_path, subfolder=vae_subfolder,
                                 dtype=dtype, device=device, mirror=mirror)
            vae.eval()

            if model_name.lower() == "quantized_sd3":
                tokenizer = _load_real_clip_tokenizer(
                    pretrained_path,
                    subfolder=cfg.get("tokenizer_subfolder", "tokenizer"),
                    mirror=mirror,
                )
            else:
                tokenizer = _load_real_t5_tokenizer(
                    pretrained_path=pretrained_path,
                    subfolder=cfg.get("tokenizer_subfolder", "tokenizer"),
                    mirror=mirror,
                )

    return model, vae, tokenizer


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("=" * 60)
    print("Test 1: QuantizedDiT")
    model, vae, tok = load_model("quantized_dit", {
        "in_channels": 1, "image_size": 28, "patch_size": 4,
        "hidden_dim": 128, "depth": 4, "num_heads": 4,
        "bitW": 8, "bitA": 8, "bitG": 8,
        "dtype": "float32",
    }, dry_run=False)
    print(f"  model type: {type(model).__name__}, vae={vae}, tokenizer={tok}")
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    print(f"  output shape: {out.shape}")
    print()

    print("=" * 60)
    print("Test 2: QuantizedSD3 (dry-run)")
    model, vae, tok = load_model("quantized_sd3", {
        "sample_size": 128, "patch_size": 2, "in_channels": 16,
        "num_layers": 4, "attention_head_dim": 16, "num_attention_heads": 4,
        "joint_attention_dim": 256, "caption_projection_dim": 64,
        "pooled_projection_dim": 128, "out_channels": 16,
        "pos_embed_max_size": 64,
        "dtype": "float32",
    }, dry_run=True)
    print(f"  model type: {type(model).__name__}")
    print(f"  vae type: {type(vae).__name__}, tokenizer type: {type(tok).__name__}")

    # Test VAE
    fake_img = torch.randn(2, 3, 128, 128)
    latent = vae.encode(fake_img).sample()
    print(f"  VAE encode: {fake_img.shape} → {latent.shape}")
    recon = vae.decode(latent)
    print(f"  VAE decode: {latent.shape} → {recon.shape}")

    # Test tokenizer
    tokens = tok(["a cat", "a dog"])
    print(f"  tokenizer output keys: {list(tokens.keys())}, input_ids shape: {tokens['input_ids'].shape}")

    # Test SD3 forward
    hidden = torch.randn(2, 16, 64, 64)
    enc_hidden = torch.randn(2, 77, 256)
    pooled = torch.randn(2, 128)
    timestep = torch.randint(0, 1000, (2,))
    out = model(hidden_states=hidden, encoder_hidden_states=enc_hidden,
                pooled_projections=pooled, timestep=timestep)
    print(f"  SD3 output shape: {out.shape}")
    print()
    """
    print("=" * 60)
    print("Test 3: QuantizedPixArt (dry-run)")
    model, vae, tok = load_model("quantized_pixart", {
        "sample_size": 64, "patch_size": 2, "in_channels": 4,
        "num_layers": 4, "attention_head_dim": 16, "num_attention_heads": 4,
        "cross_attention_dim": 256, "caption_channels": 256,
        "out_channels": 4, "pos_embed_max_size": 64,
        "dtype": "float32",
    }, dry_run=True)
    print(f"  model type: {type(model).__name__}")
    print(f"  vae type: {type(vae).__name__}, tokenizer type: {type(tok).__name__}")

    # Test VAE
    fake_img = torch.randn(2, 3, 64, 64)
    latent = vae.encode(fake_img).sample()
    print(f"  VAE encode: {fake_img.shape} → {latent.shape}")

    # Test PixArt forward
    hidden = torch.randn(2, 4, 32, 32)
    enc_hidden = torch.randn(2, 77, 256)
    timestep = torch.randint(0, 1000, (2,))
    h_tensor = torch.full((2,), 32, dtype=torch.long)
    w_tensor = torch.full((2,), 32, dtype=torch.long)
    out = model(hidden_states=hidden, encoder_hidden_states=enc_hidden,
                timestep=timestep, height=h_tensor, width=w_tensor)
    print(f"  PixArt output shape: {out.shape}")
    print()

    print("All tests passed!")
    """
