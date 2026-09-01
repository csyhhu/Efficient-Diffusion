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
import gc
import json, yaml
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
from src.data_loader import get_dataset_prompts, get_dataloader

from src.utils import memory_check
from src.recoder import Recorder


def _plot_per_module_split(x_axis, per_module_data, all_names, save_dir,
                           prefix, ylabel, title, modules_per_fig=10):
    """Plot per-module loss trajectories, split into multiple PNGs.

    Each PNG shows at most ``modules_per_fig`` (default 10) modules.
    Files are named ``{prefix}_0.png``, ``{prefix}_1.png``, ...

    Args:
        x_axis: 1D numpy array of iteration indices.
        per_module_data: dict {module_name: [val_per_iter, ...]}.
        all_names: ordered list of module names to plot.
        save_dir: Directory to save PNGs.  If None, nothing is saved.
        prefix: Filename prefix (e.g. "module_wise").
        ylabel: Y-axis label.
        title: Plot title.
        modules_per_fig: Max modules per figure (default 10).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not all_names:
        print(f"  No modules to plot for {prefix}")
        return

    n_figs = (len(all_names) + modules_per_fig - 1) // modules_per_fig
    for fig_idx in range(n_figs):
        start = fig_idx * modules_per_fig
        end = min(start + modules_per_fig, len(all_names))
        chunk = all_names[start:end]

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        for mname in chunk:
            series = per_module_data.get(mname, [0.0] * len(x_axis))
            # Shorten module name for legend readability.
            parts = mname.split(".")
            short = ".".join(parts[-2:]) if len(parts) >= 2 else mname
            ax.plot(x_axis, series, linewidth=1.0, alpha=0.8, label=short)
        ax.set_xlabel("Calibration iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(f"{title} [{start}:{end}]")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=7)
        plt.tight_layout()
        if save_dir is not None:
            p = os.path.join(save_dir, f"{prefix}_{fig_idx}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"  {prefix} plot {fig_idx} (modules {start}:{end}) saved to: {p}")
        plt.close(fig)


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

    def __init__(
        self,
        model_id=None,
        use_origin_model=False,
        download_source="modelscope",
        cache_dir="G://models",
        output_dir=None,
        use_nvfp4=False,
        block_size=16,
        rotation=None,
        permutation=None,
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
        self.output_dir = output_dir
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
        self.recorder = None
        self.start_epoch = 0
        # Cached unconditional (empty-string) embedding for CFG dropout training.
        # Computed once in prepare_local_training via encode_prompt("").
        self._uncond_embeds = None

        # For Analysis
        self.step_wise_computation_diff = {}

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
        config_dir = self.local_config_path
        model_config_path = os.path.join(config_dir, "model.yaml")
        dataset_config_path = os.path.join(config_dir, "dataset.yaml")
        running_config_path = os.path.join(config_dir, "running.yaml")

        with open(model_config_path, "r", encoding="utf-8") as f:
            self._local_model_config = yaml.safe_load(f)
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            self._local_dataset_config = yaml.safe_load(f)
        with open(running_config_path, "r", encoding="utf-8") as f:
            self._local_running_config = yaml.safe_load(f)

        self.build_local_scheduler()
        self.transformer = self.build_local_transformer()

        # Load VAE, tokenizer, text_encoder for local mode (needed for
        # latent-space datasets like CIFAR-100 latent). Reuses logic from
        # ImageGeneration.build_local_pipeline: resolve local model path,
        # load AutoencoderKL + BertTokenizer + BertModel.
        cache_dir = self._local_model_config.get("cache_dir", self.cache_dir)
        vae_repo = self._local_model_config.get("vae", "stabilityai/sd-vae-ft-mse")
        text_encoder_repo = self._local_model_config.get("text_encoder", "iic/multi-modal_clip-vit-base-patch16_zh")

        vae_path = self._find_or_download_component(
            vae_repo, cache_dir,
            ["config.json", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
        )
        te_path = self._find_or_download_component(
            text_encoder_repo, cache_dir,
            ["config.json", "pytorch_model.bin", "vocab.txt"],
        )

        print(f">> [local] Loading VAE from: {vae_path}")
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=self.dtype
        ).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        print(f">> [local] Loading tokenizer from: {te_path}")
        from transformers import BertTokenizer, BertModel, BertConfig
        self.tokenizer = BertTokenizer.from_pretrained(te_path)

        print(f">> [local] Loading text encoder from: {te_path}")
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
        self.text_encoder = BertModel(config)
        state_dict = torch.load(
            os.path.join(te_path, "pytorch_model.bin"), map_location="cpu"
        )
        state_dict = state_dict.get("state_dict", state_dict)
        bert_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module.bert."):
                bert_state_dict[key.replace("module.bert.", "")] = value
        self.text_encoder.load_state_dict(bert_state_dict, strict=False)
        self.text_encoder = self.text_encoder.to(self.dtype).to(self.device).eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self._max_token_length = self._local_model_config.get("max_token_length", 77)

    def _find_or_download_component(self, repo_id, cache_dir, required_files):
        """Find existing component or download it from ModelScope."""
        paths_to_check = [
            os.path.join(cache_dir, repo_id),
            os.path.join(cache_dir, repo_id.replace("/", "_")),
            os.path.join(cache_dir, "._____temp", repo_id),
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                existing_files = [f for f in required_files if os.path.exists(os.path.join(path, f))]
                if len(existing_files) >= len(required_files) // 2:
                    print(f">> [local] Found component at: {path}")
                    return path
        print(f">> [local] Downloading {repo_id} from ModelScope...")
        from modelscope import snapshot_download
        local_path = snapshot_download(
            repo_id, cache_dir=cache_dir, allow_patterns=required_files,
        )
        print(f">> [local] Downloaded to: {local_path}")
        return local_path

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
        rotation = self.rotation if self.rotation else cfg.get("rotation", None)
        permutation = self.permutation if self.permutation else cfg.get("permutation", None)

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

    def prepare_local_training(self):
        """Prepare data loaders and optimizer for local training."""
        dataset_config = self._local_dataset_config
        self.train_loader, self.val_loader = get_dataloader(
            dataset_name=dataset_config.get("dataset_name", "cifar100"),
            dataset_config=dataset_config,
            vae=self.vae, tokenizer=self.tokenizer, text_encoder=self.text_encoder
        )
        print(f"[train] Train loader: {len(self.train_loader)} batches, {len(self.train_loader.dataset)} samples")
        print(f"[train] Val loader: {len(self.val_loader)} batches, {len(self.val_loader.dataset)} samples")

        # Precompute the unconditional (empty-string) embedding once, so CFG
        # dropout in train() can swap dropped samples with the SAME null token
        # used at inference (encode_prompt("")). Shape: (seq, dim).
        with torch.no_grad():
            uncond, _, _ = self.encode_prompt(
                "", num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
            self._uncond_embeds = uncond[0].detach().to(self.device, dtype=self.dtype)

        running_config = self._local_running_config
        self.output_dir = running_config.get("output_dir", "./outputs") if self.output_dir is None else self.output_dir

        self.start_epoch = 0
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=running_config.get("lr", 1e-1),
            betas=(0.9, 0.999),
        )
        self.ema = EMAModel(self.transformer, decay=0.999)
        self.recorder = Recorder(save_path=os.path.join(self.output_dir, "record.json"))
        self.load_checkpoint("last_model.pth")

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

    def train(self):
        """Train local DiT model."""
        os.makedirs(self.output_dir, exist_ok=True)

        running_config = self._local_running_config
        epochs = running_config.get("epochs", 200)
        test_prompt = running_config.get("test_prompt", "A cat")
        num_steps = running_config.get("num_steps", 50)
        sample_interval = running_config.get("sample_interval", 10)
        save_interval = running_config.get("save_interval", 10)
        record_interval = running_config.get("record_interval", 1)

        # CFG training: drop each sample's conditioning with prob cfg_dropout
        # (replaced by the cached uncond embedding); guidance_scale is the
        # inference CFG scale used for the periodic qualitative samples.
        cfg_dropout = float(running_config.get("cfg_dropout", 0.0))
        guidance_scale = float(running_config.get("guidance_scale", 1.0))

        global_step = self.start_epoch * len(self.train_loader)
        for epoch in range(self.start_epoch, epochs):

            start_time = time.time()
            self.transformer.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for latents, encoder_hidden_states, labels in pbar:
                latents = latents.to(self.device, dtype=self.dtype, non_blocking=True)
                encoder_hidden_states = encoder_hidden_states.to(self.device, dtype=self.dtype, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_size = latents.shape[0]

                self.optimizer.zero_grad()

                # Classifier-free guidance dropout: per-sample, with prob
                # cfg_dropout, swap the conditioning with the cached uncond
                # embedding (empty-string BERT output). This trains the model
                # to handle the unconditional branch used at inference.
                if cfg_dropout > 0:
                    drop = torch.rand(batch_size, device=self.device) < cfg_dropout
                    if drop.any():
                        uncond = self._uncond_embeds.to(encoder_hidden_states).expand_as(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            drop.view(-1, 1, 1), uncond, encoder_hidden_states
                        )

                autocast_context = torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.autocast_dtype else contextlib.nullcontext()

                with autocast_context:
                    if isinstance(self.scheduler, FlowMatchingScheduler):
                        t = torch.rand(batch_size, device=self.device)
                        noise = torch.randn_like(latents)
                        x_t = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
                        target = noise - latents
                        model_output, logits = self.transformer(x_t, t, encoder_hidden_states=encoder_hidden_states)
                        loss = F.mse_loss(model_output, target)
                    else:
                        t = self.scheduler.sample_timesteps(batch_size, str(self.device))
                        noise = torch.randn_like(latents)
                        x_t, _ = self.scheduler.add_noise(latents, t, noise)
                        model_output, logits = self.transformer(x_t, t.float(), encoder_hidden_states=encoder_hidden_states)
                        loss = F.mse_loss(model_output, noise)
                    if logits is not None:
                        classifier_loss = F.cross_entropy(logits, labels)
                    else:
                        classifier_loss = None


                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if classifier_loss is not None:
                        (loss + classifier_loss).backward()
                    else:
                        loss.backward()
                    grad_norm = self.transformer.get_grad_norm()
                    self.optimizer.step()

                self.ema.update()
                total_loss += loss.detach()
                num_batches += 1
                global_step += 1

                # Classifier Acc
                if logits is not None:
                    acc = (torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).float().mean().item()
                else:
                    acc = 0.0

                if (global_step + 1) % record_interval == 0:
                    current_loss = loss.detach().item()
                    self.recorder.update(
                        {
                            "loss": current_loss, 
                            "acc": acc, 
                            "grad_norm": grad_norm
                        }, 
                        global_step
                    )
                    pbar.set_postfix({"loss": f"{current_loss:.3e} | {acc:.3f}"})
            
            eval_loss, eval_acc = self.eval()
            avg_train_loss = total_loss / num_batches
            if (epoch + 1) % sample_interval == 0 or epoch == 0:
                self.recorder.update({"eval_loss": eval_loss, "eval_acc": eval_acc}, global_step)
                samples = self.generate(
                    prompt=test_prompt, num_samples=4, num_steps=num_steps,
                    guidance_scale=guidance_scale, negative_prompt="",
                )
                save_sample_grid(samples, os.path.join(self.output_dir, f"samples_epoch_{epoch + 1}.png"), nrow=2)
                torch.save(
                        {
                        "epoch": epoch,
                        "model_state_dict": self.transformer.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "ema_state_dict": self.ema.state_dict(),
                        "loss": eval_loss,
                        "acc": eval_acc,
                    }, os.path.join(self.output_dir, "last_model.pth")
                )
                self.recorder.save()


    def eval(self):
        """Evaluate the model on validation dataset."""
        self.transformer.eval()
        eval_loss = 0.0
        eval_correct = 0.0
        eval_total = 0.0

        pbar = tqdm(self.val_loader, desc=f"Evaluation")
        for latents, encoder_hidden_states, labels in pbar:
            latents = latents.to(self.device, dtype=self.dtype)
            encoder_hidden_states = encoder_hidden_states.to(self.device, dtype=self.dtype)
            labels = labels.to(self.device)
            batch_size = latents.shape[0]
            with torch.no_grad():
                if isinstance(self.scheduler, FlowMatchingScheduler):
                    t = torch.rand(batch_size, device=self.device)
                    noise = torch.randn_like(latents)
                    x_t = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
                    target = noise - latents
                    model_output, logits = self.transformer(x_t, t, encoder_hidden_states=encoder_hidden_states)
                else:
                    t = self.scheduler.sample_timesteps(batch_size, str(self.device))
                    noise = torch.randn_like(latents)
                    x_t, _ = self.scheduler.add_noise(latents, t, noise)
                    model_output, logits = self.transformer(x_t, t.float(), encoder_hidden_states=encoder_hidden_states)
                
                eval_loss += F.mse_loss(logits, labels).detach().item()
                eval_correct += (torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).float().sum().item()
                eval_total += batch_size
                pbar.set_postfix({"loss": f"{eval_loss / eval_total:.3e}", "acc": f"{eval_correct / eval_total:.3f}"})

        return eval_loss / eval_total, eval_correct / eval_total


    def load_checkpoint(self, post_fix=None):
        """Load a checkpoint for local DiT model."""
        ckpt_path = os.path.join(self.output_dir, post_fix)
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.transformer.load_state_dict(checkpoint["model_state_dict"])
            if "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
            self.ema = EMAModel(self.transformer, decay=0.999)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.recorder.load()
            print(f"Checkpoint loaded (epoch={self.start_epoch}, loss={checkpoint.get('loss', '?')})")
        else:
            print(f"No checkpoint found for [{ckpt_path}]. Starting from scratch.")

    # ==================================================================
    # Prompt encoding (generic local-DiT path)
    # Subclasses (SanaImageGenerator / SD3ImageGenerator) override this
    # with model-specific encoding logic.
    # ==================================================================
    def encode_prompt(self, prompt, max_sequence_length=300, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=None):
        """Encode prompt text into embeddings for local DiT mode (BERT).

        Generic single-tokenizer path used by ``dit_cifar100_fm`` etc.
        Subclasses override this for Sana (Gemma+CHI) and SD3 (dual-CLIP).

        When ``do_classifier_free_guidance=True`` the negative prompt(s) are
        encoded alongside the positive prompt(s) and prepended in batch order
        ``[neg..., pos...]`` so a downstream ``chunk(2)`` yields
        ``(uncond, cond)``.

        Returns:
            (prompt_embeds, prompt_attention_mask, pooled_prompt_embeds=None)
        """
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        max_length = getattr(self, "_max_token_length", max_sequence_length)
        select_index = [0] + list(range(-max_length + 1, 0))

        if isinstance(prompt, str):
            prompt = [prompt]
        prompt = [p.lower().strip() for p in prompt]

        # Classifier-free guidance: build [neg..., pos...] batch in one pass.
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            negative_prompt = [p.lower().strip() for p in negative_prompt]
            if len(negative_prompt) < len(prompt):
                negative_prompt = negative_prompt * len(prompt)
            encode_list = list(negative_prompt) + list(prompt)
        else:
            encode_list = prompt

        text_inputs = self.tokenizer(
            encode_list,
            padding="max_length",
            max_length=max_length,
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
        prompt_embeds = prompt_embeds[:, select_index]
        prompt_attention_mask = prompt_attention_mask[:, select_index]

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask, None

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
        return_computation_diff=False,
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
            prompts = get_dataset_prompts(dataset_name, dataset_path, n_sample=num_samples)
            print(f"[Dataset mode] {len(prompts)} prompts from '{dataset_name}' -> {save_root}")
            # Forward all kwargs except the dataset-mode control arg.
            # forward_kwargs = {k: v for k, v in kwargs.items() if k != "num_samples"}
            # When return_intermediates=True, collect per-prompt intermediates
            # (dit_outputs / noise_preds / scheduler_outputs / num_steps) and
            # return them to the caller for saving. Image saving still happens
            # inside _custom_generate via save_root / save_name.
            all_intermediates = [] if return_intermediates else None
            # all_step_wise_computation_diff = [] if return_computation_diff else None
            for idx, p in enumerate(prompts):
                result = self.generate(
                    prompt=p, num_samples=1, seed=seed,
                    num_steps=num_steps, used_origin_pipe=used_origin_pipe,
                    return_intermediates=return_intermediates,
                    return_computation_diff=return_computation_diff,
                    save_root=save_root,
                    save_name=f"{idx:05d}",
                    visual_n_row=1,
                    **kwargs,
                )
                if return_intermediates and result is not None:
                    _, inter = result
                    inter["prompt"] = p
                    inter["seed"] = seed
                    all_intermediates.append(inter)
            if return_intermediates:
                return all_intermediates
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
                return_computation_diff=return_computation_diff,
                **kwargs,
            )
            if return_intermediates:
                image, intermediates_recorder = result
            elif return_computation_diff:
                image, step_wise_computation_diff = result
                save_postfix = kwargs.get('save_postfix', None)
                if save_postfix is None:
                    computation_diff_save_root = f"{save_root}/diff_dict"
                else:
                    computation_diff_save_root = f"{save_root}/diff_dict/{save_postfix}"
                os.makedirs(computation_diff_save_root, exist_ok=True)
                torch.save(step_wise_computation_diff, f"{computation_diff_save_root}/{save_name}.pt")
            else:
                image = result
            if save_root is not None and save_name is not None:
                save_postfix = kwargs.get('save_postfix', None)
                save_name = f"{save_name}_{save_postfix}" if save_postfix is not None else save_name
                print(f"  Saved -> {os.path.join(save_root, f"{save_name}.png")}")
                os.makedirs(save_root, exist_ok=True)
                save_sample_grid(image, os.path.join(save_root, f"{save_name}.png"), nrow=visual_n_row)

        print(f">> [{time.time() - cur_time:.2f}] Finish Generation")
        return result

    def _custom_generate(self, *args, **kwargs):
        """Custom generation for local DiT models (DDPM / Flow-Matching).

        Subclasses (SanaImageGenerator / SD3ImageGenerator) override with
        pipeline-specific logic. The default implementation supports local
        mode: sample initial noise, run ``num_steps`` scheduler iterations,
        pass prompt embeddings to ``self.transformer``, then decode with
        ``self.vae``.

        Accepted kwargs (all forwarded from :meth:`generate`):

        - prompt (str): Text prompt (used only if ``encoder_hidden_states`` is
          not already cached by the caller). For local DiT models, the generic
          :meth:`encode_prompt` single-tokenizer path is used.
        - num_samples (int): Number of images to generate.
        - seed (int | None): Random seed for initial noise sampling.
        - num_steps (int | None): Number of denoising steps — defaults to
          ``20`` for FM and ``50`` for DDPM.
        - return_intermediates (bool): If ``True`` return
          ``(pil_images, intermediates_dict)`` instead of just images.
        - guidance_scale (float): CFG scale. ``> 1.0`` enables classifier-free
          guidance (encodes the negative prompt alongside the positive one and
          mixes ``uncond + scale * (cond - uncond)`` each step). Default ``1.0``
          disables CFG and reproduces the previous single-prompt behaviour.
        - negative_prompt (str | None): Negative prompt for CFG. Defaults to
          an empty string (true unconditional) when CFG is active.
        """
        # ---- Parse args (align with generate() signature for subclass parity) -
        prompt = kwargs.get("prompt", None)
        num_samples = int(kwargs.get("num_samples", 1))
        seed = kwargs.get("seed", None)
        num_steps = kwargs.get("num_steps", None)
        return_intermediates = bool(kwargs.get("return_intermediates", False))
        guidance_scale = float(kwargs.get("guidance_scale", 1.0))
        negative_prompt = kwargs.get("negative_prompt", None)
        do_cfg = guidance_scale > 1.0

        # ---- Scheduler selection ----------------------------------------------
        is_fm = isinstance(self.scheduler, FlowMatchingScheduler)
        is_ddpm = isinstance(self.scheduler, DDPMScheduler)
        if not (is_fm or is_ddpm):
            raise NotImplementedError(
                f"BaseImageGenerator._custom_generate only supports "
                f"FlowMatchingScheduler / DDPMScheduler, got {type(self.scheduler).__name__}. "
                f"Override _custom_generate in the subclass."
            )
        if num_steps is None:
            num_steps = 20 if is_fm else 50

        # ---- Seed for reproducibility -----------------------------------------
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        # ---- Latent / image shape ---------------------------------------------
        # Local DiT operates in VAE latent space (image_size in config is the
        # latent spatial size, e.g. 4 for CIFAR100 with 8× VAE downsample).
        cfg = self._local_model_config or {}
        latent_height = cfg.get("image_size", 4)
        latent_width = cfg.get("image_size", 4)
        latent_channels = cfg.get("in_channels", 4)
        latent_shape = (num_samples, latent_channels, latent_height, latent_width)

        # ---- Prompt encoding ---------------------------------------------------
        # ``encode_prompt`` is overridden by SanaImageGenerator / SD3ImageGenerator
        # but base version works for single-tokenizer local DiT (BERT / CIFAR).
        # When CFG is on, encode_prompt returns batch [neg, pos]; we expand each
        # to num_samples via repeat_interleave -> [neg*n, pos*n] so that
        # torch.cat([latents]*2) pairs uncond/cond correctly for chunk(2).
        prompt_embeds, prompt_attention_mask, _pooled = self.encode_prompt(
            prompt or "a photo",
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )
        if do_cfg:
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples, dim=0)
            if prompt_attention_mask is not None:
                prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_samples, dim=0)
        elif prompt_embeds.shape[0] != num_samples:
            if prompt_embeds.shape[0] == 1:
                prompt_embeds = prompt_embeds.repeat(num_samples, 1, 1)
                if prompt_attention_mask is not None:
                    prompt_attention_mask = prompt_attention_mask.repeat(num_samples, 1)
            else:
                prompt_embeds = prompt_embeds[:num_samples]
                if prompt_attention_mask is not None:
                    prompt_attention_mask = prompt_attention_mask[:num_samples]

        # ---- Initialise latents (start from pure noise) -----------------------
        with torch.no_grad():
            if is_fm:
                # Flow-Matching samples x ~ N(0,1) at t=1 (pure noise regime).
                latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=self.dtype)
            else:  # DDPM
                # Same noise initialisation — scheduler step handles scale.
                latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=self.dtype)

        # ---- Build timestep sequence (mirrors compute_distillation_loss) ------
        if is_ddpm:
            self.scheduler.set_timesteps(num_steps, device=self.device)
            boundary_timesteps = self.scheduler.timesteps.to(self.device)
            timesteps = boundary_timesteps[:-1] if len(boundary_timesteps) == num_steps + 1 else boundary_timesteps
        else:  # FM
            boundary_timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
            timesteps = boundary_timesteps[:-1]

        # ---- Optional intermediate recorder (for save_intermediates.py) -------
        if return_intermediates:
            intermediate_latents = []
            intermediate_denoised = []
            dit_outputs = []
        else:
            intermediate_latents = intermediate_denoised = dit_outputs = None

        # ---- Denoising loop ----------------------------------------------------
        self.transformer.eval()
        with torch.no_grad():
            for i in range(num_steps):
                t = timesteps[i] if i < len(timesteps) else boundary_timesteps[-1]
                # Per-sample timestep (batch n) for the scheduler step; the DiT
                # forward instead sees a 2n batch when CFG duplicates latents.
                t_sample = t.expand(num_samples).to(self.dtype)
                t_model = t_sample.repeat(2) if do_cfg else t_sample
                latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents

                # Local NVFP4DiT forward signature:
                #   forward(x, t, encoder_hidden_states=None, quantization_error_info=None)
                # Pass ``prompt_embeds`` as the 3rd positional arg, do not pass
                # ``encoder_attention_mask`` because local DiT ignores it.
                output, logit = self.transformer(
                    latent_model_input.to(self.dtype),
                    t_model,
                    prompt_embeds.to(self.dtype),
                )

                # Classifier-free guidance: chunk the 2n batch into (uncond,
                # cond) and mix. Without CFG, noise_pred is just the raw output.
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = output.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    noise_pred = output

                if return_intermediates:
                    dit_outputs.append(noise_pred.detach().cpu())
                    intermediate_latents.append(latents.detach().cpu())

                # Update latents via scheduler (mirrors compute_distillation_loss)
                if is_fm:
                    dt = (boundary_timesteps[i + 1] - boundary_timesteps[i]) if (i + 1) < len(boundary_timesteps) else -1.0 / num_steps
                    latents = latents + noise_pred * dt
                else:  # DDPM
                    latents = self.scheduler.step(
                        noise_pred, t_sample, latents, return_dict=False,
                    )[0]

                if return_intermediates:
                    intermediate_denoised.append(latents.detach().cpu())

        # ---- Decode latents → images ------------------------------------------
        # ``_decode_latents_to_images`` returns clamped [-1, 1] CPU float32 tensor.
        images = self._decode_latents_to_images(latents)

        if return_intermediates:
            intermediates = {
                "dit_outputs": dit_outputs,
                "latents": intermediate_latents,
                "denoised": intermediate_denoised,
                "num_steps": num_steps,
                "scheduler_type": "fm" if is_fm else "ddpm",
                "final_output": images.detach().cpu(),
                "final_latents": latents.detach().cpu(),
            }
            return images, intermediates
        return images

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
        """Plot figures from Cayley calibration ``error_info``.

        Expects the raw structure produced by :meth:`calibrate_cayley`::

            {
              "iterations": {
                iter_idx: {  # raw step_error_info from compute_distillation_loss
                  step_idx: {
                    "module_loss": {name: {"act": float, "param": float}},
                    "layer_loss":  {name: float},
                    "step_loss":   float,
                    "dit_loss":    float,
                  }, ...
                  "dit_loss_sum": float,
                }, ...
              },
              "opt_loss": {iter_idx: float, ...},
              "final": float,
              "criterion": str,
              ...
            }

        This method does all aggregation from the raw per-step dicts.

        Output structure under ``save_root/cayley_loss``::

          cayley_criterion_sums.png
            — Plot 1: x=iteration, y=sum of each criterion's aggregated loss
              (module-wise Σ, layer-wise Σ, step-wise Σ, dit_loss Σ).

          module_wise/   (folder)
            — Plot 2 (split): per-module module-wise loss vs iteration.
              One PNG per 10 modules, named ``module_wise_0.png``, ``module_wise_1.png``, ...

          layer_wise/    (folder)
            — Plot 3 (split): per-module layer-wise loss vs iteration.
              One PNG per 10 modules, named ``layer_wise_0.png``, ``layer_wise_1.png``, ...
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if not isinstance(stats, dict):
            print(f"Load error info from {stats}")
            stats = json.load(open(stats))

        # ---- Normalise per-iteration records into a sorted list ---------------
        iters_raw = stats.get("iterations", {})
        if not iters_raw:
            print("Warning: No 'iterations' key found in stats")
            return

        sorted_iters = sorted(
            ((int(k), v) for k, v in iters_raw.items()),
            key=lambda kv: kv[0],
        )
        iter_indices = [idx for idx, _ in sorted_iters]
        x_axis = np.asarray(iter_indices, dtype=int)
        n_iters = len(sorted_iters)

        opt_loss_dict = stats.get("opt_loss", {})

        save_dir = None
        if save_root is not None:
            save_dir = os.path.join(save_root, "cayley_loss")
            os.makedirs(save_dir, exist_ok=True)

        criterion_label = stats.get("criterion", "unknown")
        fig_title_suffix = f" (criterion={criterion_label})"

        # ---- Aggregate raw per-step dicts into per-iteration sums + per-module ----
        # For each iteration, sum across all steps within that iteration.
        mod_sum_list = []      # per-iteration: Σ all modules' (act+param)
        layer_sum_list = []    # per-iteration: Σ all modules' layer_loss
        step_sum_list = []     # per-iteration: Σ step_loss
        dit_sum_list = []      # per-iteration: Σ dit_loss
        opt_list = []           # per-iteration: opt_loss

        # per-module trajectories: {module_name: [val_per_iter, ...]}
        per_module_mod = {}    # act+param per module per iteration
        per_module_layer = {}  # layer_loss per module per iteration

        for _iter_idx, iter_data in sorted_iters:
            mod_sum = 0.0
            layer_sum = 0.0
            step_sum = 0.0
            dit_sum = 0.0
            iter_mod = {}
            iter_layer = {}

            # iter_data is the raw step_error_info: {step_idx: {...}, "dit_loss_sum": float}
            for s_key, s_rec in iter_data.items():
                if s_key == "dit_loss_sum" or not isinstance(s_rec, dict):
                    continue
                # module_loss: {name: {"act": float, "param": float}}
                for mname, mval in s_rec.get("module_loss", {}).items():
                    combined = float(mval.get("act", 0.0)) + float(mval.get("param", 0.0))
                    iter_mod[mname] = iter_mod.get(mname, 0.0) + combined
                    mod_sum += combined
                # layer_loss: {name: float}
                for lname, lval in s_rec.get("layer_loss", {}).items():
                    lv = float(lval)
                    iter_layer[lname] = iter_layer.get(lname, 0.0) + lv
                    layer_sum += lv
                # step_loss
                step_sum += float(s_rec.get("step_loss", 0.0))
                # dit_loss
                dit_sum += float(s_rec.get("dit_loss", 0.0))

            mod_sum_list.append(mod_sum)
            layer_sum_list.append(layer_sum)
            step_sum_list.append(step_sum)
            dit_sum_list.append(dit_sum)
            opt_list.append(float(opt_loss_dict.get(str(_iter_idx), opt_loss_dict.get(_iter_idx, mod_sum))))

            # Accumulate per-module trajectories
            for mname, val in iter_mod.items():
                if mname not in per_module_mod:
                    per_module_mod[mname] = [0.0] * n_iters
                per_module_mod[mname][_iter_idx] = val
            for lname, val in iter_layer.items():
                if lname not in per_module_layer:
                    per_module_layer[lname] = [0.0] * n_iters
                per_module_layer[lname][_iter_idx] = val

        # ==================================================================
        # Plot 1: criterion sums vs iteration
        # ==================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(x_axis, mod_sum_list, marker='o', linewidth=2, label='module-wise (Σ act+param)')
        ax.plot(x_axis, layer_sum_list, marker='s', linewidth=2, label='layer-wise (Σ per-module output diff)')
        ax.plot(x_axis, step_sum_list, marker='^', linewidth=2, label='step-wise (Σ DiT output diff)')
        ax.plot(x_axis, dit_sum_list, marker='D', linewidth=2, label='dit_loss (Σ diffusion training loss)')
        ax.plot(x_axis, opt_list, marker='*', linewidth=2, linestyle='--', label='opt_loss')
        ax.set_xlabel("Calibration iteration")
        ax.set_ylabel("Aggregated loss (log scale)")
        ax.set_yscale("log")
        ax.set_title("Criterion-summed loss vs iteration" + fig_title_suffix)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        if save_dir is not None:
            p = os.path.join(save_dir, "cayley_criterion_sums.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Plot 1 (criterion sums) saved to: {p}")
        plt.close(fig)

        # ==================================================================
        # Plot 2: per-module module-wise loss (folder, 10 modules per image)
        # ==================================================================
        if save_dir is not None:
            mod_dir = os.path.join(save_dir, "module_wise")
            os.makedirs(mod_dir, exist_ok=True)
        else:
            mod_dir = None

        all_mod_names = list(per_module_mod.keys())
        _plot_per_module_split(
            x_axis, per_module_mod, all_mod_names,
            save_dir=mod_dir, prefix="module_wise",
            ylabel="Module-wise loss (act+param, log scale)",
            title="Per-module module-wise loss vs iteration" + fig_title_suffix,
        )

        # ==================================================================
        # Plot 3: per-module layer-wise loss (folder, 10 modules per image)
        # ==================================================================
        if save_dir is not None:
            layer_dir = os.path.join(save_dir, "layer_wise")
            os.makedirs(layer_dir, exist_ok=True)
        else:
            layer_dir = None

        all_layer_names = list(per_module_layer.keys())
        _plot_per_module_split(
            x_axis, per_module_layer, all_layer_names,
            save_dir=layer_dir, prefix="layer_wise",
            ylabel="Layer-wise loss (output diff, log scale)",
            title="Per-module layer-wise loss vs iteration" + fig_title_suffix,
        )

    def _clear_intermediates(self, _models_to_clear: list = None):
        """Clear stored intermediates in all NVFP4Linear modules.

        After ``loss.backward()`` the computation graph is freed, but the
        tensors stored as instance attributes (``x_eff``, ``W_eff``,
        ``x_quant``, ``W_quant``, ``output``) still hold GPU memory.
        Without clearing them, the next forward pass allocates new tensors
        before the old ones are overwritten, causing an OOM spike.
        """
        # Clear intermediates on both the quantized transformer and the
        # reference model (kept as ``self._calib_ref_model`` during calibration)
        # so stale tensors from the previous iteration are released before the
        # next forward pass allocates fresh ones.
        for model_to_clear in _models_to_clear:
            if model_to_clear is None:
                continue
            for _, module in model_to_clear.named_modules():
                if hasattr(module, 'store_intermediates'):
                    module.x_eff = None
                    module.W_eff = None
                    module.x_quant = None
                    module.W_quant = None
                    module.output = None

    def _decode_latents_to_images(self, latents):
        """Decode latents to images for test_mode visualisation.

        During calibration the VAE is offloaded to CPU, so decoding happens
        on CPU with no_grad. Subclasses override to apply model-specific
        scaling (Sana: /sigma_data/scaling_factor, SD3: *scaling+shift).

        Args:
            latents: (B, C, H, W) latent tensor (on any device).

        Returns:
            (B, 3, H*8, W*8) image tensor in [-1, 1] on CPU.
        """
        with torch.no_grad():
            vae_input = latents.detach().to(self.device, dtype=self.dtype)
            scaling = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0
            vae_input = vae_input / scaling
            images = self.vae.decode(vae_input, return_dict=False)[0]
            return images.clamp(-1, 1).to(torch.float32)

    def _offload_to_cpu(self):
        """Offload text encoder(s) and VAE to CPU to free GPU memory.

        During calibration the text encoder is not needed (embeddings are
        pre-computed by the dataloader) and the VAE is only needed for
        ``test_mode``. Moving them to CPU frees several GB of VRAM for
        the Cayley rotation backward graph.
        """
        offloaded = []
        if self.text_encoder is not None:
            if isinstance(self.text_encoder, (list, tuple)):
                for te in self.text_encoder:
                    if hasattr(te, 'device') and te.device.type == 'cuda':
                        te.to("cpu")
                        offloaded.append(type(te).__name__)
            elif hasattr(self.text_encoder, 'device') and self.text_encoder.device.type == 'cuda':
                self.text_encoder.to("cpu")
                offloaded.append(type(self.text_encoder).__name__)
        if self.vae is not None and hasattr(self.vae, 'device') and self.vae.device.type == 'cuda':
            self.vae.to("cpu")
            offloaded.append("VAE")
        if offloaded and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f">> [Sys Opt] Offloaded to CPU: {offloaded}")

    def _restore_to_gpu(self):
        """Restore text encoder(s) and VAE from CPU back to GPU.

        Called after calibration to re-enable image generation.
        """
        restored = []
        if self.text_encoder is not None:
            if isinstance(self.text_encoder, (list, tuple)):
                for te in self.text_encoder:
                    if hasattr(te, 'device') and te.device.type == 'cpu':
                        te.to(self.device, dtype=self.dtype)
                        restored.append(type(te).__name__)
            elif hasattr(self.text_encoder, 'device') and self.text_encoder.device.type == 'cpu':
                self.text_encoder.to(self.device, dtype=self.dtype)
                restored.append(type(self.text_encoder).__name__)
        if self.vae is not None and hasattr(self.vae, 'device') and self.vae.device.type == 'cpu':
            self.vae.to(self.device, dtype=self.dtype)
            restored.append("VAE")
        if restored:
            print(f">> [Sys Opt] Restored to GPU: {restored}")

    # ==================================================================
    # Common: Cayley rotation calibration
    # ==================================================================
    def calibrate_cayley(
        self,
        calibrate_dataset_name=None, calib_dataset_path=None, calib_n_sample=None, calib_batch_size=None,
        criterion="step-wise", iters=200, lr=0.01, 
        num_steps=4, single_step_mode=False,
        save_path=None, 
        test_mode=False
    ):
        """Calibrate Cayley rotation by running a full decode loop (DiT + Scheduler update).

        For each optimization iteration we:
          1. Sample a calibration batch from the dataset (latents + text embeds).
          2. Run a decode loop: for each step, call DiT on ``x_t``, then call
             ``scheduler.step(...)`` to update the latents to ``x_{t-1}``.
             **Noise is randomly sampled each iteration** (not fixed) to enable
             proper diffusion training loss (``dit_loss``) computation — this
             is equivalent to a normal training step with rotation applied.
          3. Compute a distillation loss between quantized transformer and
             unquantized ``ref_model`` based on ``criterion``:

               - ``module-wise``: Σ per-NVFP4Linear activation and weight
                 quantization errors (``get_differentiable_quantization_error``).
               - ``layer-wise``: Σ per-NVFP4Linear output diff
                 (``loss_fn(ref_module.output.detach(), module.output)``).
               - ``step-wise``: error on the DiT final output
                 (``loss_fn(output_target, output_ref)``).

          4. Also compute ``dit_loss`` (standard diffusion training loss) —
             recorded in error_info but NOT used for optimization (unless
             criterion is set to "dit").
          5. Back-prop through the chosen criterion and take one Adam step on
             every Cayley ``K`` parameter.

        The returned ``error_info`` stores the **raw per-step** dicts from
        ``compute_distillation_loss`` (not aggregated).  Aggregation and
        plotting is done in :meth:`plot_cayley_loss`.

        Args:
            calibrate_dataset_name: Dataset name (e.g. "MJHQ-30K", "coco2017val").
            calib_dataset_path: Local path to the dataset.
            calib_n_sample: Number of calibration samples.
            calib_batch_size: Batch size for calibration.
            criterion: Loss criterion for back-prop / optimizer update
                (``'module-wise'`` | ``'layer-wise'`` | ``'step-wise'``).
            iters: Number of Adam optimization iterations.
            lr: Adam learning rate.
            num_steps: Number of scheduler decode steps per iteration.
            save_path: Directory to dump ``cayley_error_info.json``.
            test_mode: If True, decode the final latents and save a sample grid.
            single_step_mode: If True, sample one random decode-step per
                iteration.  If False, run all ``num_steps`` per iteration.

        Returns:
            dict: ``error_info`` with the structure::

                {
                  "iterations": {
                    iter_idx: {  # raw step_error_info from compute_distillation_loss
                      step_idx: {
                        "module_loss": {name: {"act": float, "param": float}},
                        "layer_loss":  {name: float},
                        "step_loss":   float,
                        "dit_loss":    float,
                      }, ...
                      "dit_loss_sum": float,
                    }, ...
                  },
                  "opt_loss": {iter_idx: float, ...},
                  "final": float,
                  "criterion": str,
                  "iters": int,
                  "num_steps": int,
                }
        """
        cur_time = time.time()
        # Build dataset_config for src.data_loader.get_dataloader.
        # ``image_size`` controls the VAE-encoded latent resolution
        # (image_size / 8). A smaller image_size reduces the token count and
        # thus the GPU memory needed to store per-module intermediates for
        # module/layer-wise loss across all 232 NVFP4Linear modules.
        calib_batch_size = calib_batch_size if calib_batch_size is not None else calib_n_sample
        dataset_config = {
            "batch_size": calib_batch_size,
            "data_dir": calib_dataset_path
        }
        cali_dataloader, _ = get_dataloader(
            dataset_name=calibrate_dataset_name,
            dataset_config=dataset_config,
            vae=self.vae, tokenizer=self.tokenizer, text_encoder=self.text_encoder
        )

        print(f">> [{time.time() - cur_time:.2f}] Finish loading dataset "
              f"{calibrate_dataset_name} with size {len(cali_dataloader)}")

        # Fetch the calibration batch BEFORE offloading VAE/text encoder to
        # CPU, because the dataloader's __getitem__ uses them for on-the-fly
        # encoding (MJHQ-30K encodes images with the VAE at fetch time).
        calib_batch = next(iter(cali_dataloader))

        # Build batch_data dict once from the fetched calib_batch.
        batch_data = {
            "x": calib_batch[0].to(self.device, dtype=self.dtype),
            "encoder_hidden_states": calib_batch[1].to(self.device, dtype=self.dtype),
        }

        # Offload text encoder and VAE to CPU to free GPU memory for the
        # computation graph (Cayley rotation backward graph is large).
        self._offload_to_cpu()

        cur_time = time.time()
        if test_mode:
            ref_model = None
        else:
            ref_model = self.build_reference_model()
        # Keep ref_model as an attribute so ``_clear_intermediates`` can reset
        # its stored tensors each iteration. The reference model also needs
        # ``store_intermediates=True`` so layer-wise loss can read each module's
        # ``.output`` without hooks (compute_distillation_loss must not use hooks).
        print(f">> [{time.time() - cur_time:.2f}] Finish building reference model")

        # Enable intermediate storage in NVFP4Linear on BOTH the quantized
        # transformer and the reference model: the quantized model exposes
        # (x_eff, W_eff, x_quant, W_quant) for module-wise loss, and both
        # models expose ``.output`` for layer-wise loss.
        for model_to_cfg in (self.transformer, ref_model):
            for _, module in model_to_cfg.named_modules():
                if hasattr(module, 'store_intermediates'):
                    module.store_intermediates = True

        memory_check("Memory after ref_model")

        # Collect Cayley rotation K parameters (the only learnable params here).
        rot_instances = []
        k_params = []
        for name, module in self.transformer.named_modules():
            if isinstance(module, CayleyRotation):
                rot_instances.append(module)
                k_params.append(module.K)
        optimizer = torch.optim.Adam(k_params, lr=lr)
        print(f">> [Optimizer] Finish initializing optimizer with [{len(rot_instances)}] rotations")

        pbar = tqdm(total=iters, desc="[Cayley Calibration]", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", ncols=100)

        iterations_metrics = {}
        opt_loss_dict = {}
        final_opt_loss = 0.0

        for iter_idx in range(iters):

            optimizer.zero_grad()

            if iter_idx == 0:
                memory_check("GPU mem iter 0 start")
                print("\n")

            # compute_distillation_loss returns (latents, step_error_info, loss).
            # test_mode image decode is done here via _decode_latents_to_images,
            # NOT inside compute_distillation_loss, to keep loss computation
            # decoupled from visualisation.
            latents, step_error_info, loss = self.compute_distillation_loss(
                ref_model, batch_data, criterion, num_steps,
                single_step_mode=single_step_mode,
            )
            if test_mode and iter_idx == 0:
                images = self._decode_latents_to_images(latents)
                nrow = calib_batch_size or calib_n_sample or 4
                save_path = save_path or "."
                os.makedirs(save_path, exist_ok=True)
                save_sample_grid(images, os.path.join(save_path, "test.png"), nrow=nrow)
                print(f" >> [Monitor] Cayley Update Test images saved to: {os.path.join(save_path, 'test.png')}")

            loss_val = loss.item()
            loss.backward()
            optimizer.step()

            # Store raw step_error_info — aggregation happens in plot_cayley_loss.
            step_error_info["opt_loss"] = loss_val
            iterations_metrics[iter_idx] = step_error_info
            opt_loss_dict[iter_idx] = loss_val
            final_opt_loss = loss_val

            dit_sum = step_error_info.get("dit_loss_sum", 0.0)
            pbar.set_postfix({"Loss": f"{loss_val:.4e}", "dit": f"{dit_sum:.4e}"})
            pbar.update(1)

            del loss
            # Clear stored intermediates to free GPU memory before the next
            # forward pass (avoids OOM from stale tensor references).
            self._clear_intermediates([self.transformer, ref_model])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if iter_idx == 0:
                memory_check("GPU mem iter 0 after backward+clear")

        pbar.close()

        # Synchronize GPU before cleanup to avoid "device not ready" errors.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Free calibration graph and intermediates.
        self._clear_intermediates([self.transformer, ref_model])
        del ref_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Restore VAE / text encoder to GPU for downstream generation.
        self._restore_to_gpu()
        if torch.cuda.is_available():
            # Sync after restore so subsequent generate() doesn't hit
            # "device not ready" from pending async copies.
            torch.cuda.synchronize()

        # Wrap raw per-iteration metrics into the structure expected by
        # plot_cayley_loss: {"iterations": {...}, "opt_loss": {...},
        # "criterion": str, "final": float}.
        error_info = {
            "iterations": iterations_metrics,
            "opt_loss": opt_loss_dict,
            "criterion": criterion,
            "final": final_opt_loss,
        }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            error_path = os.path.join(save_path, "cayley_error_info.json")
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
            print(f"Error info saved to: {error_path}")

        return error_info

    # ==================================================================
    # Common: load_pipe (to be overridden by subclasses)
    # ==================================================================

    def load_pipe(self):
        """Load the pretrained pipeline. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement load_pipe")

    # ==================================================================
    # Common: compute_distillation_loss (to be overridden by subclasses)
    # ==================================================================

    def compute_distillation_loss(self, ref_model, batch_data, criterion, num_steps, loss_fn=F.mse_loss, single_step_mode=False, **kwargs):
        """Compute distillation loss for the local DiT model.

        Supports ``FlowMatchingScheduler`` and ``DDPMScheduler``.  No hooks —
        all losses come from NVFP4Linear's exposed intermediates.
        Noise is **randomly sampled** each call (not fixed) to enable proper
        diffusion training loss (``dit_loss``) computation.

        For each decode step:
          1. Run the quantized ``self.transformer`` on ``x_t`` (with grad).
          2. Run the unquantized ``ref_model`` (no grad).
          3. Per-NVFP4Linear: module-wise (act+param) + layer-wise (output diff).
          4. step-wise: ``loss_fn(output_target, output_ref)``.
          5. dit_loss: standard diffusion training loss.
             - FM: target = noise - x_0 (velocity), MSE(output, target)
             - DDPM: target = noise, MSE(output, noise)
          6. Record raw per-step ``error_info[step_idx]``.
          7. ``scheduler.step`` to update latents.

        Returns ``(latents, step_error_info, total_loss)`` where
        ``step_error_info`` is the raw per-step dict (not aggregated).
        """
        hidden_states = batch_data.get('x')
        encoder_hidden_states = batch_data.get('encoder_hidden_states')

        # Random noise each call — enables proper diffusion training loss.
        noise = torch.randn_like(hidden_states)
        clean_target = hidden_states

        is_fm = isinstance(self.scheduler, FlowMatchingScheduler)
        is_ddpm = isinstance(self.scheduler, DDPMScheduler)
        if not (is_fm or is_ddpm):
            raise ValueError(
                f"Local DiT compute_distillation_loss only supports "
                f"FlowMatchingScheduler / DDPMScheduler, got {type(self.scheduler)}"
            )

        # ---- Scheduler timesteps -----------------------------------------------
        # DDPM exposes set_timesteps(); FlowMatchingScheduler does not (its
        # t are continuous in [0,1]). For FM we build a decreasing sequence
        # 1→0 so the decode loop walks the probability-flow ODE from noise
        # toward data, matching how trainer.sample_time is used at training.
        if is_ddpm:
            self.scheduler.set_timesteps(num_steps, device=self.device)
            boundary_timesteps = self.scheduler.timesteps.to(self.device)
            timesteps = boundary_timesteps[:-1] if len(boundary_timesteps) == num_steps + 1 else boundary_timesteps
        else:  # FlowMatchingScheduler
            # Boundary points t=1 (pure noise) → t=0 (clean data), num_steps+1
            # points → num_steps transitions. Each transition uses dt = t_{n+1}-t_n.
            boundary_timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
            timesteps = boundary_timesteps[:-1]  # num_steps entries

        if single_step_mode:
            sampled_step_idx = torch.randint(0, num_steps, (1,)).item()
            steps_to_run = [sampled_step_idx]
        else:
            steps_to_run = list(range(num_steps))

        # ---- Accumulators (float32) ---------------------------------------------
        loss_acm_act = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_acm_param = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_acm_layer_wise = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_acm_step_wise = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_acm_dit = torch.zeros((), device=self.device, dtype=torch.float32)
        error_info = {}

        # ---- Helper: mix x_0 and noise at timestep t ----------------------------
        def _mix(t_scalar):
            t_batch = t_scalar.expand(hidden_states.shape[0])
            t_e = t_batch.view(-1, *([1] * (hidden_states.dim() - 1)))
            if is_fm:
                return (1 - t_e) * clean_target + t_e * noise
            else:  # DDPM
                # Use scheduler.add_noise for correct DDPM schedule
                return self.scheduler.add_noise(clean_target, noise, t_batch)

        # ---- Initialise decode-loop latents -------------------------------------
        t_init = boundary_timesteps[0]
        x_t_target = _mix(t_init).to(self.dtype)
        x_t_ref = x_t_target.detach().clone()
        # print(steps_to_run)
        for i in steps_to_run:
            # print(f"Step: {i}")
            step_module_loss_dict = {}
            step_layer_loss_dict = {}

            t = timesteps[i] if i < len(timesteps) else boundary_timesteps[-1]
            t_batch = t.expand(hidden_states.shape[0]).to(self.dtype)

            # --- Transformer forward (quantized target, with grad) --------------
            output_target = self.transformer(x_t_target, t_batch, encoder_hidden_states)

            # --- Reference transformer forward (no grad) -------------------------
            with torch.no_grad():
                output_ref = ref_model(x_t_ref, t_batch, encoder_hidden_states)

            # --- Per-NVFP4Linear losses (no hooks — stored intermediates) --------
            for name, module in self.transformer.named_modules():
                if isinstance(module, NVFP4Linear):
                    # module-wise
                    loss_act, loss_param = module.get_differentiable_quantization_error(loss_fn)
                    step_module_loss_dict[name] = {
                        "act": float(loss_act.item()),
                        "param": float(loss_param.item()),
                    }
                    loss_acm_act = loss_acm_act + loss_act.to(torch.float32)
                    loss_acm_param = loss_acm_param + loss_param.to(torch.float32)

                    # layer-wise: per-module output diff (vs ref model)
                    output_ref_mod = ref_model.get_submodule(name).output
                    loss_layer_wise = loss_fn(output_ref_mod.detach(), module.output)
                    step_layer_loss_dict[name] = float(loss_layer_wise.item())
                    loss_acm_layer_wise = loss_acm_layer_wise + loss_layer_wise.to(torch.float32)

            # --- step-wise: DiT output diff ---------------------------------------
            loss_step_wise = loss_fn(output_target, output_ref)
            loss_acm_step_wise = loss_acm_step_wise + loss_step_wise.to(torch.float32)

            # --- dit_loss: standard diffusion training loss ----------------------
            if is_fm:
                # Flow Matching: target = noise - x_0 (velocity)
                target_velocity = noise.float() - clean_target.float()
                dit_loss = loss_fn(output_target.float(), target_velocity.detach())
            else:
                # DDPM: target = noise (epsilon prediction)
                dit_loss = loss_fn(output_target.float(), noise.float().detach())
            loss_acm_dit = loss_acm_dit + dit_loss.to(torch.float32)

            # --- Record raw per-step error_info ----------------------------------
            error_info[0 if single_step_mode else i] = {
                "module_loss": step_module_loss_dict,
                "layer_loss": step_layer_loss_dict,
                "step_loss": float(loss_step_wise.item()),
                "dit_loss": float(dit_loss.item()),
            }

            # --- Scheduler step --------------------------------------------------
            # FM: x_{t+dt} = x_t + v·dt  where dt = t_{n+1} - t_n (< 0).
            # DDPM: use scheduler.step (its schedule-aware formula).
            if is_fm:
                dt = (boundary_timesteps[i + 1] - boundary_timesteps[i]) if (i + 1) < len(boundary_timesteps) else -1.0 / num_steps
                x_t_target = x_t_target + output_target * dt
                with torch.no_grad():
                    x_t_ref = x_t_ref + output_ref * dt
            else:
                x_t_target = self.scheduler.step(
                    output_target, t_batch, x_t_target, return_dict=False,
                )[0]
                with torch.no_grad():
                    x_t_ref = self.scheduler.step(
                        output_ref, t_batch, x_t_ref, return_dict=False,
                    )[0]

        error_info["dit_loss_sum"] = float(loss_acm_dit.item())

        # ---- Criterion selection (strict) ---------------------------------------
        if criterion == "module-wise":
            total_loss = loss_acm_act + loss_acm_param
        elif criterion == "layer-wise":
            total_loss = loss_acm_layer_wise
        elif criterion == "step-wise":
            total_loss = loss_acm_step_wise
        else:
            raise ValueError(f"Criterion type {criterion} is not supported.")

        return x_t_target, error_info, total_loss
