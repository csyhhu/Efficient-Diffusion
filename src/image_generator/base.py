"""Base image generator with local training mode support.

Provides common functionality shared by Sana and SD3 image generators:
  - Local DiT model initialization, training, and generation
  - DDPM and FlowMatching schedulers for local mode
  - Dataset loading (CIFAR100, MJHQ-30K)
  - Cayley rotation calibration utilities
  - Prompt encoding utilities (Sana CHI and SD3 dual-CLIP)
  - save_rotation / load_rotation / plot_cayley_loss
  - Dataset-mode generation in ``generate``
"""

import os
import json
import time
import copy
import contextlib

os.environ["HF_ENDPOINT"] = "https://www.modelscope.cn/api/v1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift

from PIL import Image
from tqdm import tqdm

from src.quant_utils.permutation import IdentityPermutation, MagnitudeSortPermutation, PermutationBase, RandomPermutation, make_permutation
from src.quant_utils.rotation import CayleyRotation, HadamardRotation, IdentityRotation, RandomRotation, RotationBase, make_rotation
from src.modules.quantized_linear import NVFP4Linear
from src.models.nvfp4_quantized_dit import NVFP4DiT
from src.utils import save_sample_grid, EMAModel
from src.schedulers import DDPMScheduler, FlowMatchingScheduler, ConsistencyModelScheduler
from src.data.cifar import get_cifar100_dataloader, generate_prompt, CIFAR100_CLASSES
from src.data.mqjh import get_mqjh30k_dataloader
from src.data_loader import get_dataset_prompts


class BaseImageGenerator:
    """Base class for image generators.

    Supports two modes:
      1. **Pretrained mode** (default): load Sana / SD3 pipelines, generate
         images with custom or original transformers.
      2. **Local mode** (``local_mode=True``): train a small NVFP4DiT on
         CIFAR100 or MJHQ-30K using DDPM or FlowMatching scheduler.

    Subclasses (SanaImageGenerator, SD3ImageGenerator) override:
      - ``load_pipe``: load the pretrained pipeline and transformer
      - ``_custom_generate``: model-specific generation logic
      - ``compute_distillation_loss``: model-specific distillation loss
    """

    # Sana CHI prompt prefix (same as SanaSprintPipeline.__call__ default)
    _SANA_CHI = [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]

    def __init__(
        self,
        model_id=None,
        use_origin_model=False,
        download_source="modelscope",
        cache_dir="G://models",
        use_nvfp4=False,
        block_size=16,
        rotation="identity",
        permutation="identity",
        local_mode=False,
        local_config_path=None,
        load_origin_model=False,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        """Initialize the image generator.

        Args:
            model_id: HuggingFace / ModelScope model identifier.
            use_origin_model: Whether to use the original model.
            download_source: Download source ("modelscope" or "huggingface").
            cache_dir: Local cache directory for model weights.
            use_nvfp4: Whether to load NVFP4-quantized transformer.
            block_size: Block size for NVFP4 quantization.
            rotation: Rotation type ("identity", "cayley", "hadamard", "random").
            permutation: Permutation type ("identity", "magnitude", "random").
            local_mode: Whether to use local DiT training mode.
            local_config_path: Path to local config directory (required for local_mode).
            device: Target device.
            dtype: Data type for all components.
        """
        self.model_id = model_id
        self.block_size = block_size
        self.download_source = download_source
        self.cache_dir = cache_dir
        self.local_mode = local_mode
        self.local_config_path = local_config_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        self.use_nvfp4 = use_nvfp4
        self.rotation = rotation
        self.permutation = permutation

        # Core components
        self.pipe = None
        self.transformer = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.use_origin_model = use_origin_model

        self._sigma_data = 1.0
        self._max_token_length = 77
        self._transformer_config = None

        # Local mode configs
        self._local_model_config = None
        self._local_dataset_config = None
        self._local_running_config = None

        # Training components
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.ema = None

        # Initialize components based on mode
        if self.local_mode:
            self.build_local_pipeline()
        else:
            self.load_pipe()

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def in_channels(self):
        """Get the number of input channels from the transformer config."""
        if self.transformer is None:
            return 32
        if hasattr(self.transformer, 'config') and hasattr(self.transformer.config, 'in_channels'):
            return self.transformer.config.in_channels
        elif hasattr(self.transformer, 'in_channels'):
            return self.transformer.in_channels
        return 32

    @property
    def resolution(self):
        """Get the default resolution from the transformer config."""
        if self.transformer is None:
            return 1024
        if hasattr(self.transformer, 'config') and hasattr(self.transformer.config, 'sample_size'):
            return self.transformer.config.sample_size * 8
        elif hasattr(self.transformer, 'image_size'):
            return self.transformer.image_size * 8
        return 1024

    @property
    def use_local_dit(self):
        """Check if using local DiT model (no config attribute)."""
        return self.transformer is not None and hasattr(self.transformer, 'image_size') and not hasattr(self.transformer, 'config')

    @property
    def latent_resolution(self):
        """Get the latent space resolution from the transformer config."""
        if self.transformer is None:
            return 32
        if hasattr(self.transformer, 'config') and hasattr(self.transformer.config, 'sample_size'):
            return self.transformer.config.sample_size
        elif hasattr(self.transformer, 'image_size'):
            return self.transformer.image_size
        return 32

    # ==================================================================
    # Local mode: pipeline / scheduler / transformer building
    # ==================================================================

    def _resolve_local_model_path(self, model_id, cache_dir):
        """Resolve the local model path from model_id and cache_dir."""
        if cache_dir:
            local_dir = os.path.join(cache_dir, model_id.replace("/", os.sep))
            if os.path.exists(local_dir):
                return local_dir
            raise FileNotFoundError(f"Model config not found in local cache directory: {local_dir}")
        if model_id and os.path.exists(model_id):
            return model_id
        raise FileNotFoundError(f"Model path not found: {model_id}")

    def build_local_pipeline(self):
        """Build the local DiT pipeline: scheduler, transformer, configs."""
        # Load local configs
        config_dir = self.local_config_path or os.path.dirname(self.model_id) if self.model_id else "."
        model_config_path = os.path.join(config_dir, "model_config.json")
        dataset_config_path = os.path.join(config_dir, "dataset_config.json")
        running_config_path = os.path.join(config_dir, "running_config.json")

        with open(model_config_path, "r", encoding="utf-8") as f:
            self._local_model_config = json.load(f)
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            self._local_dataset_config = json.load(f)
        with open(running_config_path, "r", encoding="utf-8") as f:
            self._local_running_config = json.load(f)

        self.build_local_scheduler()
        self.transformer = self.build_local_transformer()

        # Build a simple VAE + tokenizer for local mode
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self._max_token_length = self._local_model_config.get("max_token_length", 77)

    def build_local_scheduler(self):
        """Build scheduler based on running config."""
        scheduler_type = self._local_running_config.get("scheduler", "fm")

        if scheduler_type == "fm":
            self.scheduler = FlowMatchingScheduler()
        elif scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler()
        elif scheduler_type == "cm":
            self.scheduler = ConsistencyModelScheduler()
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def build_local_transformer(self, is_ref=False):
        """Build the local transformer model from config."""
        cfg = self._local_model_config
        rotation = cfg.get("rotation", self.rotation)
        permutation = cfg.get("permutation", self.permutation)

        if is_ref:
            transformer = NVFP4DiT(
                in_channels=cfg.get("in_channels", 4),
                image_size=cfg.get("image_size", 8),
                patch_size=cfg.get("patch_size", 2),
                hidden_dim=cfg.get("hidden_dim", 256),
                depth=cfg.get("depth", 6),
                num_heads=cfg.get("num_heads", 4),
                time_dim=cfg.get("time_dim", 256),
                mlp_ratio=cfg.get("mlp_ratio", 4.0),
                use_cross_attention=cfg.get("use_cross_attention", True),
                cross_attention_dim=cfg.get("cross_attention_dim", 768),
                use_nvfp4=False, rotation=None, permutation=None,
            )
        else:
            transformer = NVFP4DiT(
                in_channels=cfg.get("in_channels", 4),
                image_size=cfg.get("image_size", 8),
                patch_size=cfg.get("patch_size", 2),
                hidden_dim=cfg.get("hidden_dim", 256),
                depth=cfg.get("depth", 6),
                num_heads=cfg.get("num_heads", 4),
                time_dim=cfg.get("time_dim", 256),
                mlp_ratio=cfg.get("mlp_ratio", 4.0),
                use_cross_attention=cfg.get("use_cross_attention", True),
                cross_attention_dim=cfg.get("cross_attention_dim", 768),
                use_nvfp4=self.use_nvfp4, rotation=rotation, permutation=permutation,
            )

        return transformer.to(self.device, dtype=self.dtype)

    # ==================================================================
    # Local mode: dataset / training
    # ==================================================================

    def get_dataloader(self, dataset_name=None):
        """Create train and validation data loaders."""
        dataset_config = self._local_dataset_config if self._local_dataset_config else {}
        if dataset_name is None:
            dataset_name = dataset_config.get("dataset_name", "cifar100")

        if dataset_name == "cifar100":
            self.train_loader = get_cifar100_dataloader(
                train=True,
                root=dataset_config.get("data_dir", "./data"),
                batch_size=dataset_config.get("batch_size", 128),
                image_size=dataset_config.get("image_size", 32),
                vae=self.vae,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                dtype=self.dtype,
                max_token_length=dataset_config.get("max_token_length", self._max_token_length),
                num_workers=dataset_config.get("num_workers", 0),
                pin_memory=dataset_config.get("pin_memory", False),
                persistent_workers=dataset_config.get("persistent_workers", False),
            )
            self.val_loader = get_cifar100_dataloader(
                train=False,
                root=dataset_config.get("data_dir", "./data"),
                batch_size=dataset_config.get("batch_size", 128),
                image_size=dataset_config.get("image_size", 32),
                vae=self.vae,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                dtype=self.dtype,
                max_token_length=dataset_config.get("max_token_length", self._max_token_length),
                num_workers=dataset_config.get("num_workers", 0),
                pin_memory=dataset_config.get("pin_memory", False),
                persistent_workers=dataset_config.get("persistent_workers", False),
            )
        elif dataset_name == "MJHQ-30K":
            self.train_loader = get_mqjh30k_dataloader(
                train=True,
                root=dataset_config.get("data_dir", "G://datasets//MJHQ-30K"),
                batch_size=dataset_config.get("batch_size", 1),
                image_size=dataset_config.get("image_size", 512),
                vae=self.vae,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                dtype=self.dtype,
                max_token_length=dataset_config.get("max_token_length", self._max_token_length),
                num_workers=dataset_config.get("num_workers", 0),
                pin_memory=dataset_config.get("pin_memory", False),
                persistent_workers=dataset_config.get("persistent_workers", False),
            )
            self.val_loader = get_mqjh30k_dataloader(
                train=False,
                root=dataset_config.get("data_dir", "G://datasets//MJHQ-30K"),
                batch_size=dataset_config.get("batch_size", 1),
                image_size=dataset_config.get("image_size", 512),
                vae=self.vae,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                dtype=self.dtype,
                max_token_length=dataset_config.get("max_token_length", self._max_token_length),
                num_workers=dataset_config.get("num_workers", 0),
                pin_memory=dataset_config.get("pin_memory", False),
                persistent_workers=dataset_config.get("persistent_workers", False),
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        return self.train_loader, self.val_loader

    def prepare_local_training(self):
        """Prepare data loaders and optimizer for local training."""
        dataset_config = self._local_dataset_config
        self.get_dataloader(dataset_name=dataset_config.get("dataset_name", "cifar100"))
        print(f"[train] Train loader: {len(self.train_loader)} batches, {len(self.train_loader.dataset)} samples")
        print(f"[train] Val loader: {len(self.val_loader)} batches, {len(self.val_loader.dataset)} samples")

        running_config = self._local_running_config
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=running_config.get("lr", 0.0001),
            betas=(0.9, 0.999),
        )
        self.ema = EMAModel(self.transformer, decay=0.999)

        mixed_precision = running_config.get("mixed_precision", None)
        if mixed_precision and torch.cuda.is_available():
            print(f"[train] Enabling {mixed_precision} mixed precision training...")
            if mixed_precision == "bf16":
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)
                self.autocast_dtype = torch.bfloat16
            elif mixed_precision == "fp16":
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
                self.autocast_dtype = torch.float16
        else:
            self.scaler = None
            self.autocast_dtype = None

    def train(self, output_dir=None):
        """Train local DiT model."""
        if output_dir is None:
            output_dir = self._local_running_config.get("output_dir", "./outputs/cifar100_dit_fm")
        os.makedirs(output_dir, exist_ok=True)

        running_config = self._local_running_config
        epochs = running_config.get("epochs", 200)
        sample_interval = running_config.get("sample_interval", 10)
        test_prompt = running_config.get("test_prompt", "A cat")
        num_steps = running_config.get("num_steps", 50)
        record_interval = running_config.get("record_interval", 10)

        loss_history = []
        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(epochs):
            if (epoch + 1) % sample_interval == 0 or epoch == 0:
                samples = self.generate(prompt=test_prompt, num_samples=4, num_steps=num_steps)
                save_sample_grid(samples, os.path.join(output_dir, f"samples_epoch_{epoch+1}.png"), nrow=2)

            start_time = time.time()
            self.transformer.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for latents, encoder_hidden_states in pbar:
                latents = latents.to(self.device, dtype=self.dtype, non_blocking=True)
                encoder_hidden_states = encoder_hidden_states.to(self.device, dtype=self.dtype, non_blocking=True)

                self.optimizer.zero_grad()
                batch_size = latents.shape[0]

                autocast_context = torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.autocast_dtype else contextlib.nullcontext()

                with autocast_context:
                    if isinstance(self.scheduler, FlowMatchingScheduler):
                        t = torch.rand(batch_size, device=self.device)
                        noise = torch.randn_like(latents)
                        x_t = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
                        target = noise - latents
                        model_output = self.transformer(x_t, t, encoder_hidden_states=encoder_hidden_states)
                        loss = F.mse_loss(model_output, target)
                    else:
                        t = self.scheduler.sample_timesteps(batch_size, str(self.device))
                        noise = torch.randn_like(latents)
                        x_t, _ = self.scheduler.add_noise(latents, t, noise)
                        model_output = self.transformer(x_t, t.float(), encoder_hidden_states=encoder_hidden_states)
                        loss = F.mse_loss(model_output, noise)

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.ema.update()
                total_loss += loss.detach()
                num_batches += 1
                global_step += 1

                if (global_step + 1) % record_interval == 0:
                    current_loss = loss.detach().item()
                    avg_loss = (total_loss / num_batches).item()
                    loss_history.append({"epoch": epoch, "step": global_step, "loss": current_loss, "avg_loss": avg_loss})
                    pbar.set_postfix({"loss": f"{current_loss:.3e}", "avg_loss": f"{avg_loss:.3e}"})

            avg_train_loss = total_loss / num_batches
            avg_val_loss = self.eval()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.transformer.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "ema_state_dict": self.ema.state_dict(),
                    "loss": best_val_loss,
                }, os.path.join(output_dir, "best_model.pth"))

            torch.save({
                "epoch": epoch,
                "model_state_dict": self.transformer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "loss": avg_val_loss,
            }, os.path.join(output_dir, "last_model.pth"))

            with open(os.path.join(output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
                json.dump(loss_history, f, ensure_ascii=False, indent=2)

    def eval(self):
        """Evaluate the model on validation dataset."""
        self.transformer.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for latents, encoder_hidden_states in self.val_loader:
                latents = latents.to(self.device, dtype=self.dtype)
                encoder_hidden_states = encoder_hidden_states.to(self.device, dtype=self.dtype)
                batch_size = latents.shape[0]

                if isinstance(self.scheduler, FlowMatchingScheduler):
                    t = torch.rand(batch_size, device=self.device)
                    noise = torch.randn_like(latents)
                    x_t = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
                    target = noise - latents
                    model_output = self.transformer(x_t, t, encoder_hidden_states=encoder_hidden_states)
                    loss = F.mse_loss(model_output, target)
                else:
                    t = self.scheduler.sample_timesteps(batch_size, str(self.device))
                    noise = torch.randn_like(latents)
                    x_t, _ = self.scheduler.add_noise(latents, t, noise)
                    model_output = self.transformer(x_t, t.float(), encoder_hidden_states=encoder_hidden_states)
                    loss = F.mse_loss(model_output, noise)

                val_loss += loss.item()
                val_batches += 1

        return val_loss / val_batches

    def load_checkpoint(self, ckpt_path=None):
        """Load a checkpoint for local DiT model."""
        if ckpt_path is None:
            output_dir = self._local_running_config.get("output_dir", "./outputs")
            ckpt_path = os.path.join(output_dir, "best_model.pth")

        print(f"Loading checkpoint from {ckpt_path}...")
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint file not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.transformer = self.build_local_transformer()
        self.transformer.load_state_dict(checkpoint["model_state_dict"])

        if self.ema is None:
            self.ema = EMAModel(self.transformer, decay=0.999)
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        print(f"Checkpoint loaded (epoch={checkpoint.get('epoch', '?')}, loss={checkpoint.get('loss', '?')})")

    # ==================================================================
    # Prompt encoding
    # ==================================================================

    def encode_prompt(self, prompt, max_sequence_length=300, num_images_per_prompt=1,
                      do_classifier_free_guidance=True, negative_prompt=None):
        """Encode prompt, dispatching to model-specific encoder.

        Returns:
            (prompt_embeds, prompt_attention_mask, pooled_prompt_embeds)
            For Sana: pooled_prompt_embeds is None.
            For SD3: prompt_attention_mask is None.
        """
        if self.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
            prompt_embeds, prompt_attention_mask = self._encode_prompt_sana(
                prompt, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            return prompt_embeds, prompt_attention_mask, None
        elif self.model_id == "stabilityai/stable-diffusion-3.5-medium":
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = \
                self._encode_prompt_sd3(
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                )
            return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds
        else:
            raise ValueError(f"Unknown model_id: {self.model_id}")

    def _encode_prompt_sana(self, prompt, num_images_per_prompt=1,
                            max_sequence_length=300):
        """Sana prompt encoding: Gemma + CHI prefix.

        Mirrors ``SanaSprintPipeline._get_gemma_prompt_embeds`` exactly.
        """
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        if isinstance(prompt, str):
            prompt = [prompt]
        prompt = [p.strip() for p in prompt]

        chi_prompt = "\n".join(self._SANA_CHI)
        prompt = [chi_prompt + p for p in prompt]
        num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
        max_length_all = num_chi_prompt_tokens + max_sequence_length - 2

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask.to(self.device)

        text_enc_out = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=prompt_attention_mask,
        )
        prompt_embeds = text_enc_out[0].to(dtype=self.dtype, device=self.device)

        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))
        prompt_embeds = prompt_embeds[:, select_index]
        prompt_attention_mask = prompt_attention_mask[:, select_index]

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def _encode_prompt_sd3(self, prompt, num_images_per_prompt=1,
                           max_sequence_length=256,
                           do_classifier_free_guidance=True,
                           negative_prompt=None):
        """SD3 prompt encoding: CLIP-L + CLIP-G concat, then T5 zero-padding.

        Mirrors ``StableDiffusion3Pipeline.encode_prompt`` without T5.
        """
        tokenizers = self.tokenizer
        text_encoders = self.text_encoder

        if isinstance(prompt, str):
            prompt = [prompt]
        prompt = [p.strip() for p in prompt]

        if negative_prompt is None:
            negative_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        negative_prompt = [p.strip() for p in negative_prompt]

        clip_max_length = max(
            t.model_max_length for t in tokenizers if hasattr(t, "model_max_length")
        ) if isinstance(tokenizers, (list, tuple)) else tokenizers.model_max_length

        joint_dim = self._transformer_config.get("joint_attention_dim", 4096)

        def _encode_one(prompt_list):
            prompts_embeds_list = []
            pooled_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if hasattr(tokenizer, "padding_side"):
                    tokenizer.padding_side = "right"
                text_inputs = tokenizer(
                    prompt_list,
                    padding="max_length",
                    max_length=clip_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(self.device)
                out = text_encoder(input_ids, output_hidden_states=True)
                emb = out.hidden_states[-2].to(self.dtype)
                pooled = out[0].to(self.dtype)
                prompts_embeds_list.append(emb)
                pooled_prompt_embeds_list.append(pooled)
            prompt_embeds = torch.cat(prompts_embeds_list, dim=-1)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=-1)
            pad = joint_dim - prompt_embeds.shape[-1]
            if pad > 0:
                prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, pad))
            batch_size = prompt_embeds.shape[0]
            t5_zeros = torch.zeros(
                (batch_size, max_sequence_length, joint_dim),
                device=self.device, dtype=self.dtype,
            )
            prompt_embeds = torch.cat([prompt_embeds, t5_zeros], dim=1)
            return prompt_embeds, pooled_prompt_embeds

        prompt_embeds, pooled_prompt_embeds = _encode_one(prompt)
        if do_classifier_free_guidance:
            neg_embeds, neg_pooled = _encode_one(negative_prompt)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled, pooled_prompt_embeds], dim=0)

        def _repeat(t):
            b, *rest = t.shape
            t = t.unsqueeze(1).repeat(1, num_images_per_prompt, *([1] * len(rest)))
            return t.view(b * num_images_per_prompt, *rest)

        prompt_embeds = _repeat(prompt_embeds)
        pooled_prompt_embeds = _repeat(pooled_prompt_embeds)
        attention_mask = None

        return prompt_embeds, attention_mask, pooled_prompt_embeds

    # ==================================================================
    # Generation (common entry point)
    # ==================================================================

    def generate(
        self,
        prompt=None,
        num_samples=8,
        visual_n_row=4,
        dataset_name=None,
        dataset_path=None,
        seed=None,
        num_steps=None,
        used_origin_pipe=False,
        save_root=None,
        save_name=None,
        return_intermediates=False,
        **kwargs,
    ):
        """Generate images and optionally save to disk.

        Three generation modes:
          1. ``used_origin_pipe=True``: call the original diffusers pipeline.
          2. ``used_origin_pipe=False, use_origin_model=True``: custom generate
             with the pipeline's original (unquantized) transformer.
          3. ``used_origin_pipe=False, use_origin_model=False``: custom generate
             with the NVFP4 quantized transformer.

        Args:
            prompt: Text prompt (prompt mode).
            num_samples: Number of images per prompt.
            visual_n_row: Images per row in saved grid.
            dataset_name: Dataset name for dataset mode.
            dataset_path: Local path to dataset.
            seed: Random seed.
            num_steps: Sampling steps.
            used_origin_pipe: Use original pipeline.
            save_root: Output directory.
            save_name: Output filename.
            return_intermediates: Return intermediate tensors.
            **kwargs: Extra args (n_generated_sample, guidance, etc.).
        """
        # ---- Dataset mode ----
        if dataset_name is not None:
            n_generated_sample = kwargs.get("n_generated_sample", -1)
            prompts = get_dataset_prompts(dataset_name, dataset_path, n_sample=n_generated_sample)
            print(f"[Dataset mode] {len(prompts)} prompts from '{dataset_name}' -> {save_root}")
            # Forward all kwargs except the dataset-mode control arg.
            forward_kwargs = {k: v for k, v in kwargs.items() if k != "n_generated_sample"}
            # When return_intermediates=True, collect per-prompt intermediates
            # (dit_outputs / noise_preds / scheduler_outputs / num_steps) and
            # return them to the caller for saving. Image saving still happens
            # inside _custom_generate via save_root / save_name.
            all_intermediates = [] if return_intermediates else None
            for idx, p in enumerate(prompts):
                result = self.generate(
                    prompt=p, num_samples=1, seed=seed,
                    num_steps=num_steps, used_origin_pipe=used_origin_pipe,
                    return_intermediates=return_intermediates,
                    save_root=save_root,
                    # save_name=f"{p.replace('/', '_').replace('.', '_')}.png",
                    save_name=f"{idx:05d}.png",
                    visual_n_row=1,
                    **forward_kwargs,
                )
                if return_intermediates and result is not None:
                    _, inter = result
                    inter["prompt"] = p
                    inter["seed"] = seed
                    all_intermediates.append(inter)
                if (idx + 1) % 10 == 0:
                    print(f"  [{idx + 1}/{len(prompts)}] saved")

            if return_intermediates:
                return None, all_intermediates
            return None

        # ---- Prompt mode ----
        cur_time = time.time()
        if seed is not None:
            torch.manual_seed(seed)

        if prompt is None:
            prompts = [generate_prompt(CIFAR100_CLASSES[torch.randint(0, 100, ()).item()]) for _ in range(num_samples)]
        else:
            prompts = [prompt] * num_samples

        # ---- used_origin_pipe: call the original pipeline directly ----
        if used_origin_pipe:
            if self.pipe is None:
                raise ValueError("Origin pipe is not loaded")
            pipe_kwargs = dict(kwargs)
            if seed is not None:
                pipe_kwargs["generator"] = torch.Generator(
                    device=self.device
                ).manual_seed(seed)
            image = self.pipe(prompt, **pipe_kwargs).images[0]
            if save_root is not None and save_name is not None:
                os.makedirs(save_root, exist_ok=True)
                image.save(os.path.join(save_root, save_name))
        else:
            result = self._custom_generate(
                prompt=prompts[0],
                num_samples=num_samples,
                seed=seed,
                num_steps=num_steps,
                return_intermediates=return_intermediates,
                **kwargs,
            )
            if return_intermediates:
                image, intermediates_recorder = result
            else:
                image = result
            if save_root is not None and save_name is not None:
                os.makedirs(save_root, exist_ok=True)
                save_sample_grid(image, os.path.join(save_root, save_name), nrow=visual_n_row)

        print(f">> [{time.time() - cur_time:.2f}] Finish Generation")
        return result

    def _custom_generate(self, *args, **kwargs):
        """Subclass-specific custom generation logic. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _custom_generate")

    # ==================================================================
    # Calibration utilities
    # ==================================================================

    def build_reference_model(self):
        """Build a reference model with the same weights but no quantization."""
        if self.local_mode:
            ref_model = self.build_local_transformer(is_ref=True)
            ref_model.load_state_dict(self.transformer.state_dict(), strict=False)
        else:
            # Subclasses should override to build the appropriate reference model
            raise NotImplementedError("Subclasses must implement build_reference_model")
        return ref_model.to(self.device, dtype=self.dtype)

    def save_rotation(self, path):
        """Save only the Cayley rotation parameters (K and _R_init)."""
        rotation_state = {}
        for name, module in self.transformer.named_modules():
            if isinstance(module, CayleyRotation):
                rotation_state[name] = {
                    "K": module.K.data.cpu(),
                    "_R_init": module._R_init.data.cpu(),
                }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(rotation_state, path)
        print(f"Saved {len(rotation_state)} rotation params to: {path}")

    def load_rotation(self, path):
        """Load Cayley rotation parameters into the transformer."""
        rotation_state = torch.load(path, map_location=self.device)
        loaded = 0
        for name, module in self.transformer.named_modules():
            if isinstance(module, CayleyRotation) and name in rotation_state:
                module.K.data = rotation_state[name]["K"].to(device=self.device, dtype=self.dtype)
                module._R_init.data = rotation_state[name]["_R_init"].to(device=self.device, dtype=self.dtype)
                loaded += 1
        print(f"Loaded {loaded} rotation params from: {path}")

    def plot_cayley_loss(self, stats, save_root=None):
        """Plot Cayley calibration loss history."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if type(stats) != dict:
            print(f"Load error info from {stats}")
            stats = json.load(open(stats))

        step_keys = [k for k in stats.keys() if isinstance(k, int)]
        step_keys.sort()

        if not step_keys:
            print("Warning: No step data found in stats")
            return

        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        for step_idx in step_keys:
            step_data = stats.get(step_idx, {})
            step_loss = step_data.get('step_loss', 0)
            axes.scatter(step_idx, step_loss, label=f'Step {step_idx}', s=50)

        final_loss = stats.get('final', 0)
        axes.axhline(y=final_loss, color='r', linestyle='--', label='Final Loss')
        axes.set_xlabel('Step')
        axes.set_ylabel('Loss')
        axes.set_title('Step-wise Loss')
        axes.legend()
        axes.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = f"{save_root}/cayley_loss"
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, "cayley_step_loss.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Step loss plot saved to: {plot_path}")
        plt.close()

        if step_keys:
            module_loss = stats[step_keys[0]].get('module_loss', {})
            if module_loss:
                fig, axes = plt.subplots(1, 1, figsize=(12, 6))
                modules = list(module_loss.keys())
                act_losses = [module_loss[m].get('act', 0) for m in modules]
                param_losses = [module_loss[m].get('param', 0) for m in modules]

                x = range(len(modules))
                width = 0.35
                axes.bar([i - width/2 for i in x], act_losses, width, label='Activation Loss')
                axes.bar([i + width/2 for i in x], param_losses, width, label='Parameter Loss')
                axes.set_xlabel('Module')
                axes.set_ylabel('Quantization Error')
                axes.set_title('Module-wise Quantization Error')
                axes.legend()
                axes.grid(True, alpha=0.3)
                plt.xticks(x, modules, rotation=90)
                plt.tight_layout()

                if save_path is not None:
                    plot_path = os.path.join(save_path, "module_loss.png")
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    print(f"Module loss plot saved to: {plot_path}")
                plt.close()

    # ==================================================================
    # Common: load_pipe (to be overridden by subclasses)
    # ==================================================================

    def load_pipe(self):
        """Load the pretrained pipeline. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement load_pipe")

    # ==================================================================
    # Common: compute_distillation_loss (to be overridden by subclasses)
    # ==================================================================

    def compute_distillation_loss(self, ref_model, batch_data, criterion, num_steps,
                                  loss_fn=F.mse_loss, single_step_mode=True, **kwargs):
        """Compute distillation loss. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement compute_distillation_loss")
