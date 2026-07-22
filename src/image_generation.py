import json
import os
import csv
import time
import copy
import contextlib
from typing import List, Optional, Union, Dict, Any

from setuptools.command.develop import develop
from transformers import Pipeline

os.environ["HF_ENDPOINT"] = "https://www.modelscope.cn/api/v1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DPMSolverMultistepScheduler, SCMScheduler
from diffusers import SanaSprintPipeline

from PIL import Image
from tqdm import tqdm

from src.quant_utils.activation_calibrator import calibrate_activation_rotations, make_calibration_loader
from src.quant_utils.permutation import IdentityPermutation, MagnitudeSortPermutation, PermutationBase, RandomPermutation, make_permutation
from src.quant_utils.rotation import CayleyRotation, HadamardRotation, IdentityRotation, RandomRotation, RotationBase, make_rotation
from src.modules.quantized_linear import NVFP4Linear
from src.models.nvfp4_quantized_Sana import NVFP4QuantizedSana
from src.models.nvfp4_quantized_dit import NVFP4DiT

from src.utils import load_config, save_sample_grid, EMAModel
from src.schedulers import DDPMScheduler, FlowMatchingScheduler, ConsistencyModelScheduler
from src.data.cifar import get_cifar100_dataloader, generate_prompt, CIFAR100_CLASSES
from src.data.mqjh import get_mqjh30k_dataloader


class _DummyScheduler:
    """Simple dummy scheduler for dry mode testing."""
    config = {"sigma_data": 1.0}
    init_noise_sigma = 1.0

    def set_timesteps(self, num_steps, device=None, **kwargs):
        """Set timesteps for the scheduler."""
        self.timesteps = torch.linspace(1.5708, 0.0, num_steps + 1, device=device)

    def set_begin_index(self, index):
        """Set begin index for timesteps."""
        pass

    def step(self, model_output, timestep, latent, return_dict=True):
        """Perform a single denoising step."""
        prev_sample = latent - model_output
        denoised = prev_sample
        if return_dict:
            return type("DummyOutput", (), {"prev_sample": prev_sample, "denoised": denoised})
        return prev_sample, denoised


class ImageGeneration:
    """Image generation class for Sana models with support for quantization learning.
    
    This class supports three modes:
    - Dry mode: Uses a small dummy model for quick testing
    - Real mode: Uses the actual Sana model for production
    - Local mode: Uses local DiT model for training (e.g., CIFAR100)
    
    Key features:
    - Supports learning rotation/permutation matrices via Cayley calibration
    - Efficient decode loop with explicit SCM post-processing
    - Unified dtype management across all components
    - Local training mode for custom DiT models on datasets like CIFAR100
    """
    
    def __init__(
        self, model_id=None, 
        download_source="modelscope", cache_dir="G://models", 
        use_nvfp4 = False, block_size=16,
        rotation = "identity", permutation = "identity",
        dry_mode=False, dry_config=None, 
        local_mode=False, local_config_path=None,
        device="cuda", dtype=torch.bfloat16
    ):
        """Initialize the ImageGeneration class.
        
        Args:
            model_id: Path or identifier for the Sana model
            block_size: Block size for quantization
            download_source: Source for model download (modelscope/huggingface)
            cache_dir: Directory for model caching
            dry_mode: Whether to use dry mode (small dummy model)
            dry_config: Configuration for dry mode model
            local_mode: Whether to use local training mode (for CIFAR100+DiT)
            local_config_path: Path to local config directory (required for local_mode)
            device: Target device (cuda/cpu)
            dtype: Data type for all components
        """
        self.model_id = model_id
        self.block_size = block_size
        self.download_source = download_source
        self.cache_dir = cache_dir
        self.dry_mode = dry_mode
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

        #
        self._sigma_data = 1.0
        self._max_token_length = 77
        
        # Config caches
        self._dry_config = dry_config or {}
        self._transformer_config = None
        self._local_model_config = None
        self._local_dataset_config = None
        self._local_running_config = None
        
        # Training components
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.ema = None
        
        # Initialize components based on mode
        if self.dry_mode:
            self.build_dry_pipeline()
        elif self.local_mode:
            self.build_local_pipeline()
        else:
            self.load_pipe()


    def _resolve_local_model_path(self, model_id, cache_dir):
        """Resolve the local model path from model_id and cache_dir.
        
        Args:
            model_id: Model identifier or path
            cache_dir: Cache directory
            
        Returns:
            str: Local path to the model
            
        Raises:
            FileNotFoundError: If model path is not found
        """
        if cache_dir:
            local_dir = os.path.join(cache_dir, model_id.replace("/", os.sep))
            if os.path.exists(local_dir):
                return local_dir
            raise FileNotFoundError(f"Model config not found in local cache directory: {local_dir}")
        if model_id and os.path.exists(model_id):
            return model_id
        raise FileNotFoundError(f"Model path not found: {model_id}")
    

    def build_dry_pipeline(self):
        """Build dummy components for dry mode testing."""
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = _DummyScheduler()
        self._sigma_data = 1.0
        # self.transformer = self._new_dry_transformer(False, None, None)
    

    def _download_component(self, repo_id, component_type, source="modelscope", cache_dir=None):
        """Download a component from ModelScope (default) or HuggingFace.
        
        Args:
            repo_id: ModelScope or HuggingFace repository ID
            component_type: Type of component (vae/tokenizer/text_encoder)
            source: Download source ("modelscope" or "huggingface")
            cache_dir: Cache directory for downloaded components
            
        Returns:
            str: Local path to the downloaded component
        """
        import os
        
        cache_dir = cache_dir or self._local_model_config.get("cache_dir", "./models")
        os.makedirs(cache_dir, exist_ok=True)
        
        modelscope_path = os.path.join(cache_dir, repo_id)
        hf_path = os.path.join(cache_dir, repo_id.replace("/", "_"))
        
        if os.path.exists(modelscope_path):
            print(f"Component {repo_id} already exists at {modelscope_path}")
            return modelscope_path
        
        if os.path.exists(hf_path):
            print(f"Component {repo_id} already exists at {hf_path}")
            return hf_path
        
        print(f"Downloading {component_type}: {repo_id} from {source}...")
        
        if source == "modelscope":
            try:
                from modelscope import snapshot_download
                local_path = snapshot_download(
                    repo_id,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                print(f"  ModelScope cache: {local_path}")
                return local_path
            except Exception as exc:
                print(f"  Local ModelScope cache not found; attempting download: {exc}")
                try:
                    from modelscope import snapshot_download
                    local_path = snapshot_download(
                        repo_id,
                        cache_dir=cache_dir,
                    )
                    print(f"  ModelScope download: {local_path}")
                    return local_path
                except Exception as exc2:
                    print(f"  ModelScope download failed: {exc2}")
                    print(f"  Falling back to HuggingFace...")
        
        from huggingface_hub import snapshot_download as hf_snapshot_download
        hf_kwargs = {"local_files_only": True}
        if cache_dir:
            hf_kwargs["cache_dir"] = cache_dir
        try:
            local_path = hf_snapshot_download(repo_id, **hf_kwargs)
        except Exception:
            print("  (first download: pulling from HuggingFace ...)")
            hf_kwargs["local_files_only"] = False
            local_path = hf_snapshot_download(repo_id, **hf_kwargs)
        return local_path
    
    def build_local_pipeline(self):
        """Build local mode components for CIFAR100+DiT training.
        
        Loads config files and downloads required components:
        - VAE: from ModelScope via diffusers
        - Tokenizer: from ModelScope via transformers (BERT-style)
        - Text Encoder: from ModelScope via transformers (RobertaModel)
        """
        import yaml
        
        model_config_path = os.path.join(self.local_config_path, "model.yaml")
        dataset_config_path = os.path.join(self.local_config_path, "dataset.yaml")
        running_config_path = os.path.join(self.local_config_path, "running.yaml")
        
        with open(model_config_path, "r", encoding="utf-8") as f:
            self._local_model_config = yaml.safe_load(f)
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            self._local_dataset_config = yaml.safe_load(f)
        with open(running_config_path, "r", encoding="utf-8") as f:
            self._local_running_config = yaml.safe_load(f)

        self.transformer = self.build_local_transformer()
        
        cache_dir = self._local_model_config.get("cache_dir", "./models")
        
        vae_repo = self._local_model_config.get("vae", "stabilityai/sd-vae-ft-mse")
        text_encoder_repo = self._local_model_config.get("text_encoder", "iic/multi-modal_clip-vit-base-patch16_zh")
        
        vae_path = self._find_or_download_component(vae_repo, cache_dir, ["config.json", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"])
        text_encoder_path = self._find_or_download_component(text_encoder_repo, cache_dir, ["config.json", "pytorch_model.bin", "text_model_config.json", "vocab.txt"])
        
        print(f"Loading VAE from local path: {vae_path}")
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=self.dtype
        ).to(self.device)
        
        print(f"Loading tokenizer from local path: {text_encoder_path}")
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder_path)
        
        print(f"Loading text encoder from local path: {text_encoder_path}")
        from transformers import BertModel, BertConfig
        
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
        self.text_encoder = BertModel(config).to(self.device)
        
        state_dict = torch.load(os.path.join(text_encoder_path, "pytorch_model.bin"), map_location="cpu")
        state_dict = state_dict["state_dict"]
        
        bert_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module.bert."):
                bert_key = key.replace("module.bert.", "")
                bert_state_dict[bert_key] = value
        
        self.text_encoder.load_state_dict(bert_state_dict)
        self.text_encoder = self.text_encoder.to(self.dtype).to(self.device)
        
        self._sigma_data = 1.0
        
        self.build_local_scheduler()
    
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
                    print(f"Found component at: {path}")
                    return path
        
        print(f"Downloading {repo_id} from ModelScope...")
        from modelscope import snapshot_download
        
        local_path = snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            allow_patterns=required_files,
        )
        print(f"Downloaded to: {local_path}")
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
        running_cfg = self._local_running_config
        
        rotation = cfg.get("rotation", self.rotation)
        permutation = cfg.get("permutation", self.permutation)
        
        if is_ref:
            print(f"[model] Building reference model with rotation=None, permutation=None")
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
                use_nvfp4=False, rotation=None, permutation=None
            )
        else:
            print(f"[model] Using rotation={rotation}, permutation={permutation} from config")
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
                use_nvfp4=self.use_nvfp4, rotation=rotation, permutation=permutation
            )
        
        """
        if running_cfg.get("torch_compile", False) and torch.cuda.is_available():
            compile_mode = running_cfg.get("compile_mode", "default")
            print(f"[model] Compiling transformer with torch.compile(mode={compile_mode})...")
            self.transformer = torch.compile(self.transformer, mode=compile_mode)
        
        if running_cfg.get("gradient_checkpointing", False):
            print("[model] Enabling gradient checkpointing...")
            self.transformer.gradient_checkpointing_enable()
        """
        return transformer.to(self.device, dtype=self.dtype)


    def get_dataloader(self, dataset_name=None):
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
        self.get_dataloader()
        print(f"[train] Train loader: {len(self.train_loader)} batches, {len(self.train_loader.dataset)} samples")
        print(f"[train] Val loader: {len(self.val_loader)} batches, {len(self.val_loader.dataset)} samples")
        
        running_cfg = self._local_running_config
        
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=running_cfg.get("lr", 0.0001),
            betas=(0.9, 0.999),
        )
        
        self.ema = EMAModel(self.transformer, decay=0.999)
        
        mixed_precision = running_cfg.get("mixed_precision", None)
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
        """Train local DiT model on CIFAR100 dataset.
        
        Args:
            output_dir: Directory to save checkpoints and samples.
                        If None, uses value from running config.
        """
        if output_dir is None:
            output_dir = self._local_running_config.get("output_dir", "./outputs/cifar100_dit_fm")
        os.makedirs(output_dir, exist_ok=True)
        
        running_cfg = self._local_running_config
        epochs = running_cfg.get("epochs", 200)
        sample_interval = running_cfg.get("sample_interval", 10)
        test_prompt = running_cfg.get("test_prompt", "A cat")
        num_steps = running_cfg.get("num_steps", 50)
        record_interval = running_cfg.get("record_interval", 10)
        
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
                    loss_history.append({
                        "epoch": epoch,
                        "step": global_step,
                        "loss": current_loss,
                        "avg_loss": avg_loss,
                    })
                    pbar.set_postfix({
                        "loss": f"{current_loss:.3e}",
                        "avg_loss": f"{avg_loss:.3e}",
                    })
            
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
            
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.transformer.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "ema_state_dict": self.ema.state_dict(),
                    "loss": avg_val_loss,
                }, os.path.join(output_dir, "last_model.pth")
            )
            
            with open(os.path.join(output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
                json.dump(loss_history, f, ensure_ascii=False, indent=2)


    def decode(self, num_steps, num_samples, latents, encoder_hidden_states, encoder_attention_mask=None, **kwargs):
        """
        Multistep decode for ViT.
        Support various schedulers.
        """
        current_step = kwargs.get('current_step', [0])
        
        if isinstance(self.scheduler, FlowMatchingScheduler):
            timesteps = torch.linspace(0, 1, num_steps + 1, device=self.device)
            for i in range(num_steps):
                current_step[0] = i
                t_batch = timesteps[i].expand(num_samples)
                output = self.transformer(latents, t_batch, encoder_hidden_states, quantization_error_info=kwargs.get('quantization_error_info'))
                dt = 1.0 / num_steps
                latents = latents + dt * output

        elif isinstance(self.scheduler, DDPMScheduler):
            timesteps = torch.arange(self.scheduler.T - 1, -1, -1, device=self.device)
            for i in range(num_steps):
                current_step[0] = i
                t_batch = timesteps[i].expand(num_samples)
                output = self.transformer(latents, t_batch, encoder_hidden_states, quantization_error_info=kwargs.get('quantization_error_info'))
                coeffs = self.scheduler.get_posterior_coeffs(int(t)).to(self.device)
                mean = coeffs["mean_coef_x_t"] * latents + coeffs["mean_coef_eps"] * output
                if coeffs["sigma"] > 0:
                    latents = mean + coeffs["sigma"] * torch.randn_like(latents)
                else:
                    latents = mean

        elif isinstance(self.scheduler, SCMScheduler):
            sigma_data = float(self.scheduler.config.sigma_data) if hasattr(self.scheduler, 'config') else self._sigma_data
            guidance_embeds_scale = getattr(getattr(self.transformer, 'config', None), "guidance_embeds_scale", 0.1)

            guidance = torch.full([latents.shape[0]], kwargs.get("guidance", 4.5), device=self.device, dtype=torch.float32)
            guidance = guidance.to(self.dtype) * guidance_embeds_scale

            self.scheduler.set_timesteps(
                num_steps, device=self.device,
                max_timesteps=1.5708,
                intermediate_timesteps=(1.3 if num_steps == 2 else None),
            )
            self.scheduler.set_begin_index(0)
            timesteps = self.scheduler.timesteps[:-1].to(self.device).type(self.dtype)

            latents = latents * sigma_data

            denoised = None
            for step_idx, t in enumerate(timesteps):
                current_step[0] = step_idx
                timestep = t.expand(latents.shape[0])
                latents_model_input = latents / sigma_data

                scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                scm_t = scm_t.to(self.dtype)
                scm_t_expanded = scm_t.view(-1, 1, 1, 1)
                model_input = latents_model_input * torch.sqrt(
                    scm_t_expanded**2 + (1 - scm_t_expanded) ** 2
                )
                model_input = model_input.to(self.dtype)
                encoder_hidden_states = encoder_hidden_states.to(self.dtype)

                if hasattr(self.transformer, 'config'):
                    noise_pred = self.transformer(
                        hidden_states=model_input,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=scm_t, guidance=guidance, return_dict=False,
                        quantization_error_info=kwargs.get('quantization_error_info')
                    )[0]
                else:
                    noise_pred = self.transformer(
                        x=model_input, t=scm_t, encoder_hidden_states=encoder_hidden_states,
                        quantization_error_info=kwargs.get('quantization_error_info')
                    )

                noise_pred = (
                    (1 - 2 * scm_t_expanded) * model_input
                    + (1 - 2 * scm_t_expanded + 2 * scm_t_expanded**2) * noise_pred
                ) / torch.sqrt(scm_t_expanded**2 + (1 - scm_t_expanded) ** 2)
                noise_pred = noise_pred.float() * sigma_data

                latents, denoised = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)

            output = (denoised / sigma_data).to(self.dtype)

        else:
            raise ValueError(f"Unsupported scheduler type: {type(self.scheduler)}")

        return output


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
        
        avg_val_loss = val_loss / val_batches
        return avg_val_loss


    def load_checkpoint(self, ckpt_path=None):
        """Load a checkpoint for local DiT model.
        
        Args:
            ckpt_path: Path to the checkpoint file. If None, loads from default path.
        """
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
        
        print(f"Checkpoint loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")


    def encode_prompt(self, prompt, max_sequence_length=300, num_images_per_prompt=1):
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))

        if isinstance(prompt, str):
            prompt = [prompt]
        
        prompt = [p.lower().strip() for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask.to(self.device)
        
        text_emb = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=prompt_attention_mask,
        )[0]
        
        text_emb = text_emb.to(self.dtype)
        text_emb = text_emb[:, select_index]
        prompt_attention_mask = prompt_attention_mask[:, select_index]
        
        bs_embed, seq_len, _ = text_emb.shape
        text_emb = text_emb.repeat(1, num_images_per_prompt, 1)
        text_emb = text_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        
        return text_emb, prompt_attention_mask

    def generate(
        self, prompt=None, num_samples=8, seed=42, num_steps=50, used_origin_pipe=False, **kwargs
    ):
        """Generate images and return as tensor.
        
        Args:
            prompt: Text prompt for generation (if None, uses random CIFAR100 class prompts)
            num_samples: Number of images to generate
            seed: Random seed for reproducibility
            num_steps: Number of sampling steps
            
        Returns:
            torch.Tensor: Generated images tensor, shape (B, C, H, W) in [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        if prompt is None:
            from .data.cifar import generate_prompt, CIFAR100_CLASSES
            prompts = [generate_prompt(CIFAR100_CLASSES[torch.randint(0, 100, ()).item()]) for _ in range(num_samples)]
        else:
            prompts = [prompt] * num_samples

        if used_origin_pipe:
            if self.pipe is None:
                raise ValueError("Origin pipe is not loaded")
            return self.pipe(prompts).images

        with torch.no_grad():

            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            for p in prompts:
                emb, mask = self.encode_prompt(p, num_images_per_prompt=1)
                encoder_hidden_states_list.append(emb)
                prompt_attention_masks_list.append(mask)
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0)
            encoder_attention_mask = torch.cat(prompt_attention_masks_list, dim=0)

            latents = torch.randn(
                num_samples, 
                self.in_channels, self.latent_resolution, self.latent_resolution, 
                device=self.device, dtype=self.dtype
            )
        
            self.transformer.eval()
            model_output = self.decode(num_steps, num_samples, latents, encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            self.vae.eval()
            images = self.vae.decode(model_output / self.vae.config.scaling_factor, return_dict=False)[0]
        
        return images.clamp(-1, 1)


    def load_pipe(self):
        """Load the full SanaSprintPipeline from local files.
        
        Args:
            model_id: Optional override for model path
            cache_dir: Optional override for cache directory
            
        Returns:
            ImageGeneration: Self for chaining
        """
        load_path = self._resolve_local_model_path(self.model_id, self.cache_dir)
        
        if self.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
            cur_time = time.time()
            pipe = SanaSprintPipeline.from_pretrained(
                load_path,
                torch_dtype=self.dtype,
                local_files_only=True
            )
            print(f">> [{time.time() - cur_time:.2f}] Finish Pipeline Loading")
            cur_time = time.time()
            # """
            self.transformer = NVFP4QuantizedSana.from_pretrained(
                self.model_id, download_source=self.download_source, cache_dir=self.cache_dir,
                block_size=self.block_size, use_nvfp4=self.use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
                torch_dtype=self.dtype
            ).to(self.device, dtype=self.dtype)
            pipe.transformer = self.transformer
            print(f">> [{time.time() - cur_time:.2f}] Finish Custom Transformer Loading and Replacement")
            """
            self.transformer = pipe.transformer.to(self.device, dtype=self.dtype)
            print("Finish Origin Transformer Assignment")
            """
        else:
            raise ValueError(f"Empty {self.model_id} is not supported")
        
        # Extract components
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.vae = pipe.vae.to(self.device)
        self.scheduler = pipe.scheduler
        self._sigma_data = float(pipe.scheduler.config.sigma_data)
        
        # Load transformer config for later use
        with open(os.path.join(load_path, "transformer", "config.json"), "r", encoding="utf-8") as f:
            self._transformer_config = json.load(f)
        
        # Clean up to save memory
        self.pipe = pipe.to(self.device)
    

    def build_dry_transformer(self, is_ref=False):
        """Create a dummy transformer for dry mode.
        
        Args:
            use_nvfp4: Whether to use NVFP4 quantization
            rotation: Rotation instance
            permutation: Permutation factory
            
        Returns:
            NVFP4QuantizedSana: Dummy transformer model
        """
        from .models.nvfp4_quantized_Sana import NVFP4QuantizedSana
        from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
        cfg = self._dry_config or {}
        dim = cfg.get("dim", 256)
        heads = cfg.get("num_heads", 8)
        if is_ref:
            transformer = SanaTransformer2DModel(
                sample_size=cfg.get("resolution", 64) // 8, patch_size=1,
                in_channels=4, out_channels=4, num_layers=cfg.get("layers", 2),
                attention_head_dim=dim // heads, num_attention_heads=heads,
                num_cross_attention_heads=heads, cross_attention_head_dim=dim // heads,
                cross_attention_dim=dim, caption_channels=cfg.get("caption_channels", 768),
                mlp_ratio=2.5
            )
        else:
            transformer = NVFP4QuantizedSana(
                sample_size=cfg.get("resolution", 64) // 8, patch_size=1,
                in_channels=4, out_channels=4, num_layers=cfg.get("layers", 2),
                attention_head_dim=dim // heads, num_attention_heads=heads,
                num_cross_attention_heads=heads, cross_attention_head_dim=dim // heads,
                cross_attention_dim=dim, caption_channels=cfg.get("caption_channels", 768),
                mlp_ratio=2.5, 
                block_size=self.block_size, use_nvfp4=self.use_nvfp4,
                rotation=self.rotation, permutation=self.permutation
            )
        return transformer.to(self.device, dtype=self.dtype)
    
    
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
    

    def calibrate_cayley(
        self, calibrate_dataset_name=None, cali_dataloader=None, 
        prompt=None, n_batches=8, iters=200, lr=0.01, 
        criterion="step-wise", num_steps=4, save_path = None, test_mode=False
    ):
        """
        Calibrate Cayley rotation using different criteria for loss computation.
        
        Supports four criteria for updating rotation matrices:
        1. module-wise: loss_fn(quantized_activation, activation) at each NVFP4Linear module
        2. layer-wise: loss_fn(quantized layer output, layer output) at each transformer block
        3. step-wise: loss_fn(quantized transformer output, transformer output) after single forward
        4. multi-step-wise: loss_fn(quantized decoder output, decoder output) after multi-step decode
        
        Args:
            calibrate_dataset_name: Name of calibration dataset (used with get_dataloader)
            cali_dataloader: Optional calibration dataloader yielding batches
            prompt: Optional single text prompt for quick testing
            n_batches: Number of calibration batches per iteration (used with prompt)
            iters: Number of optimization iterations
            lr: Learning rate for Adam optimizer
            criterion: Calibration criterion ('module-wise', 'layer-wise', 'step-wise', 'multi-step-wise')
            num_steps: Number of decoding steps for multi-step-wise criterion
        
        Returns:
            dict: Calibration statistics including per-iteration loss and all error metrics
        
        Raises:
            ValueError: If neither cali_dataloader, calibrate_dataset_name, nor prompt is provided
        """
        cur_time = time.time()
        cali_dataloader, _ = self.get_dataloader(calibrate_dataset_name)
        print(f">> [{time.time() - cur_time:.2f}] Finish loading dataset {calibrate_dataset_name} with size {len(cali_dataloader)}")
        
        cur_time = time.time()
        ref_model = self.build_reference_model()
        print(f">> [{time.time() - cur_time:.2f}] Finish building reference model")
        
        # k_params, rot_instances = self.init_cayley_rotations()
        rot_instances = []
        k_params = []
        for name, module in self.transformer.named_modules():
            if isinstance(module, CayleyRotation):
                rot_instances.append(module)
                k_params.append(module.K)
        optimizer = torch.optim.Adam(k_params, lr=lr)
        print(f"Finish initializing optimizer with [{len(rot_instances)}] rotations")

        pbar = tqdm(total=iters, desc="[Cayley Calibration]", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", ncols=100)
        
        calib_batch = next(iter(cali_dataloader))
        
        all_error_info = {
            'module_wise_loss': [],
            'layer_wise_loss': [],
            'step_wise_loss': [],
            'final_loss': [],
            'loss_history': [],
        }
        
        for iter_idx in range(iters):
            
            optimizer.zero_grad()

            if isinstance(calib_batch, tuple) and len(calib_batch) == 2:
                latents, txt_embs = calib_batch
                batch_data = {
                    "x": latents.to(self.device, dtype=self.dtype),
                    "t": torch.randint(0, 1000, (latents.shape[0],), device=self.device),
                    "encoder_hidden_states": txt_embs.to(self.device, dtype=self.dtype),
                }
            elif isinstance(calib_batch, dict):
                batch_data = {}
                for k, v in calib_batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(self.device, dtype=self.dtype)
                if 'x' in batch_data and 't' not in batch_data:
                    batch_data['t'] = torch.randint(0, 1000, (batch_data['x'].shape[0],), device=self.device)
                elif 'hidden_states' in batch_data and 'x' not in batch_data:
                    batch_data['x'] = batch_data.pop('hidden_states')
                    batch_data['t'] = torch.randint(0, 1000, (batch_data['x'].shape[0],), device=self.device)
            elif isinstance(calib_batch, list):
                if len(calib_batch) == 2:
                    latents, txt_embs = calib_batch[0], calib_batch[1]
                    batch_data = {
                        "x": latents.to(self.device, dtype=self.dtype),
                        "t": torch.randint(0, 1000, (latents.shape[0],), device=self.device),
                        "encoder_hidden_states": txt_embs.to(self.device, dtype=self.dtype),
                    }
                else:
                    raise ValueError(f"List length {len(calib_batch)} not supported")
            else:
                raise ValueError(f"Data type {type(calib_batch)} is not supported")

            latents, error_info, loss = self.compute_distillation_loss(
                ref_model, batch_data, criterion, num_steps
            )
            
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Loss": f"{loss_val:.6f}"})
            pbar.update(1)
            # Pack loss infomation
            all_error_info['loss_history'].append(loss_val)
            
            del loss
            torch.cuda.empty_cache()
            
            iter_module_wise = {}
            iter_layer_wise = {}
            iter_step_wise = {}
            
            for step_idx in range(num_steps):
                if step_idx in error_info:
                    step_data = error_info[step_idx]
                    iter_module_wise[step_idx] = {}
                    iter_layer_wise[step_idx] = {}
                    iter_step_wise[step_idx] = step_data['step_loss']
                    
                    for module_name, module_data in step_data['module_loss'].items():
                        iter_module_wise[step_idx][module_name] = {
                            'act': module_data['act'],
                            'param': module_data['param']
                        }
                    
                    for layer_name, layer_loss in step_data['layer_loss'].items():
                        iter_layer_wise[step_idx][layer_name] = layer_loss
            
            all_error_info['module_wise_loss'].append(iter_module_wise)
            all_error_info['layer_wise_loss'].append(iter_layer_wise)
            all_error_info['step_wise_loss'].append(iter_step_wise)
            all_error_info['final_loss'].append(error_info['final'])
        
        pbar.close()
        
        if save_path is not None:
            import json
            os.makedirs(save_path, exist_ok=True)
            error_path = os.path.join(save_path, "cayley_error_info.json")
            with open(error_path, 'w') as f:
                json.dump(all_error_info, f, indent=2)
            print(f"Error info saved to: {error_path}")
        
        if test_mode:
            self.vae.eval()
            images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            images = images.clamp(0, 1)
            from src.utils import save_sample_grid
            save_sample_grid(images, os.path.join(save_path, "test_images.png"), nrow=1)

        return all_error_info
    
    def build_reference_model(self):
        """Build a reference model with the same weights but no quantization."""
        if self.dry_mode:
            ref_model = self.build_dry_transformer(is_ref=True)
            ref_model.load_state_dict(self.transformer.state_dict(), strict=False)
        elif self.local_mode:
            ref_model = self.build_local_transformer(is_ref=True)
            ref_model.load_state_dict(self.transformer.state_dict(), strict=False)
        else:
            if self.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
                ref_model = NVFP4QuantizedSana.from_pretrained(
                    self.model_id, download_source=self.download_source, cache_dir=self.cache_dir,
                    block_size=self.block_size,
                    rotation=None, permutation=None, use_nvfp4=False, 
                    torch_dtype=self.dtype
                )
            else:
                raise ValueError(f"{self.model_id} is not supported")
        
        return ref_model.to(self.device, dtype=self.dtype)
    
    def plot_cayley_loss(self, stats, save_root=None):
        """Plot Cayley calibration loss history and all error metrics.
        
        Args:
            stats: Dictionary containing all error metrics from calibrate_cayley
            save_path: Directory to save the plot (if None, displays only)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if type(stats) != dict:
            print(f"Load error info from {stats}")
            stats = json.load(open(stats))

        module_wise_loss = stats.get('module_wise_loss', [])
        layer_wise_loss = stats.get('layer_wise_loss', [])
        step_wise_loss = stats.get('step_wise_loss', [])
        final_loss = stats.get('final_loss', [])

        # print(module_wise_loss)
        # print(layer_wise_loss)
        # print(step_wise_loss)
        # print(final_loss)
        
        if not module_wise_loss:
            print("Warning: module_wise_loss is empty")
            return
        
        num_steps = len(module_wise_loss[0]) if module_wise_loss else 0
        num_iters = len(module_wise_loss)
        
        step_keys = list(module_wise_loss[0].keys()) if module_wise_loss else []
        
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(final_loss, label='Final Loss')
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Loss')
        axes.set_title('Final Loss')
        # axes.set_yscale('log')
        axes.legend()
        axes.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = f"{save_root}/cayley_loss"
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, "cayley_final_loss.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Final loss plot saved to: {plot_path}")
        plt.close()
        
        for step_idx in range(num_steps):
            step_key = str(step_idx) if step_keys and isinstance(step_keys[0], str) else step_idx
            modules_in_step = set()
            for iter_data in module_wise_loss:
                if step_key in iter_data:
                    modules_in_step.update(iter_data[step_key].keys())
            # print(modules_in_step)
            if not modules_in_step:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for module_name in list(modules_in_step)[:10]:
                act_errors = []
                param_errors = []
                for iter_data in module_wise_loss:
                    if step_key in iter_data and module_name in iter_data[step_key]:
                        act_errors.append(iter_data[step_key][module_name]['act'])
                        param_errors.append(iter_data[step_key][module_name]['param'])
                    else:
                        act_errors.append(0.0)
                        param_errors.append(0.0)
                axes[0].plot(act_errors, label=module_name, alpha=0.7)
                axes[0].set_xticks(range(len(act_errors))) 
                axes[1].plot(param_errors, label=module_name, alpha=0.7)
                axes[1].set_xticks(range(len(param_errors))) 
            
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Activation Quantization Loss')
            axes[0].set_title(f'Module-wise Activation Loss (Step {step_idx})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Parameter Quantization Loss')
            axes[1].set_title(f'Module-wise Parameter Loss (Step {step_idx})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is not None:
                plot_path = os.path.join(save_path, f"cayley_module_wise_step_{step_idx}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Module-wise loss plot (step {step_idx}) saved to: {plot_path}")
            plt.close()
        
        for step_idx in range(num_steps):
            step_key = str(step_idx) if step_keys and isinstance(step_keys[0], str) else step_idx
            layers_in_step = set()
            for iter_data in layer_wise_loss:
                if step_key in iter_data:
                    layers_in_step.update(iter_data[step_key].keys())
            
            if not layers_in_step:
                continue
            
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            
            for layer_name in list(layers_in_step)[:10]:
                layer_errors = []
                for iter_data in layer_wise_loss:
                    if step_key in iter_data and layer_name in iter_data[step_key]:
                        layer_errors.append(iter_data[step_key][layer_name])
                    else:
                        layer_errors.append(0.0)
                axes.plot(layer_errors, label=layer_name, alpha=0.7)
            
            axes.set_xlabel('Iteration')
            axes.set_ylabel('Layer Loss')
            axes.set_title(f'Layer-wise Loss (Step {step_idx})')
            axes.legend()
            axes.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is not None:
                plot_path = os.path.join(save_path, f"cayley_layer_wise_step_{step_idx}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Layer-wise loss plot (step {step_idx}) saved to: {plot_path}")
            plt.close()
        
        for step_idx in range(num_steps):
            step_key = str(step_idx) if step_keys and isinstance(step_keys[0], str) else step_idx
            step_errors = []
            for iter_data in step_wise_loss:
                if step_key in iter_data:
                    step_errors.append(iter_data[step_key])
                else:
                    step_errors.append(0.0)
            
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.plot(step_errors, label=f'Step {step_idx}')
            axes.set_xticks(range(len(step_errors)))
            axes.set_xlabel('Iteration')
            axes.set_ylabel('Step Loss')
            # axes.set_yscale('log')
            axes.set_title(f'Step-wise Loss (Step {step_idx})')
            axes.legend()
            axes.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is not None:
                plot_path = os.path.join(save_path, f"cayley_step_wise_step_{step_idx}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Step-wise loss plot (step {step_idx}) saved to: {plot_path}")
            plt.close()

    def compute_distillation_loss(self, ref_model, batch_data, criterion, num_steps, loss_fn=F.mse_loss, **kwargs):
        """
        Compute the distillation loss between the reference model and the current model under different criterion.
        """
        timesteps = torch.linspace(0, 1, num_steps + 1, device=self.device)
        dt = 1.0 / num_steps
        loss_acm_act = torch.tensor(0.0, device=self.device)
        loss_acm_param = torch.tensor(0.0, device=self.device)
        loss_acm_layer_wise = torch.tensor(0.0, device=self.device)
        loss_acm_step_wise = torch.tensor(0.0, device=self.device)
        error_info = {}
        
        hidden_states = batch_data.get('x')
        encoder_hidden_states = batch_data.get('encoder_hidden_states')
        latent_target = hidden_states
        latent_ref = hidden_states.clone().detach()

        for i in range(num_steps):

            step_wise_module_loss_dict = {}
            step_wise_layer_loss_dict = {}  
            
            if isinstance(self.scheduler, FlowMatchingScheduler):
                t_batch = timesteps[i].expand(latent_target.shape[0])
                output_target = self.transformer(
                    latent_target, t_batch, encoder_hidden_states, 
                    # quantization_error_info=kwargs.get('quantization_error_info')
                )
                with torch.no_grad():
                    output_ref = ref_model(latent_ref, t_batch, encoder_hidden_states)
                
                for name, module in self.transformer.named_modules():
                    if isinstance(module, NVFP4Linear):
                        loss_act, loss_param = module.get_differentiable_quantization_error(loss_fn)
                        step_wise_module_loss_dict[name] = {"act": loss_act.item(), "param": loss_param.item()}
                        loss_acm_act += loss_act
                        loss_acm_param += loss_param

                        ref_module = ref_model.get_submodule(name) 
                        loss_layer_wise = loss_fn(output_ref.detach(), output_target)
                        step_wise_layer_loss_dict[name] = loss_layer_wise.item()
                        loss_acm_layer_wise += loss_layer_wise
                    else:
                        # print(f"{name:<40s}: {type(module)}")
                        pass
                
                loss_step_wise = loss_fn(output_target, output_ref)
                loss_acm_step_wise += loss_step_wise
                
                error_info[i] = {
                    "module_loss": step_wise_module_loss_dict,
                    "layer_loss": step_wise_layer_loss_dict,
                    "step_loss": loss_step_wise.item(),
                }
                latent_ref = latent_ref + dt * output_ref
                latent_target = latent_target + dt * output_target
            
            elif isinstance(self.scheduler, SCMScheduler):
                sigma_data = float(self.scheduler.config.sigma_data) if hasattr(self.scheduler, 'config') else self._sigma_data
                guidance_embeds_scale = getattr(getattr(self.transformer, 'config', None), "guidance_embeds_scale", 0.1)

                guidance = torch.full([latent_target.shape[0]], kwargs.get("guidance", 4.5), device=self.device, dtype=torch.float32)
                guidance = guidance.to(self.dtype) * guidance_embeds_scale

                self.scheduler.set_timesteps(
                    num_steps, device=self.device,
                    max_timesteps=1.5708,
                    intermediate_timesteps=(1.3 if num_steps == 2 else None),
                )
                self.scheduler.set_begin_index(0)
                scm_timesteps = self.scheduler.timesteps[:-1].to(self.device).type(self.dtype)

                latent_target_scaled = latent_target * sigma_data
                latent_ref_scaled = latent_ref * sigma_data

                for step_idx, t in enumerate(scm_timesteps):
                    timestep = t.expand(latent_target_scaled.shape[0])
                    latents_model_input_target = latent_target_scaled / sigma_data
                    latents_model_input_ref = latent_ref_scaled / sigma_data

                    scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                    scm_t = scm_t.to(self.dtype)
                    scm_t_expanded = scm_t.view(-1, 1, 1, 1)
                    model_input_target = latents_model_input_target * torch.sqrt(
                        scm_t_expanded**2 + (1 - scm_t_expanded) ** 2
                    )
                    model_input_ref = latents_model_input_ref * torch.sqrt(
                        scm_t_expanded**2 + (1 - scm_t_expanded) ** 2
                    )
                    model_input_target = model_input_target.to(self.dtype)
                    model_input_ref = model_input_ref.to(self.dtype)

                    if hasattr(self.transformer, 'config'):
                        output_target = self.transformer(
                            hidden_states=model_input_target,
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=scm_t, guidance=guidance, return_dict=False,
                        )[0]
                        with torch.no_grad():
                            output_ref = ref_model(
                                hidden_states=model_input_ref,
                                encoder_hidden_states=encoder_hidden_states,
                                timestep=scm_t, guidance=guidance, return_dict=False,
                            )[0]
                    else:
                        output_target = self.transformer(
                            x=model_input_target, t=scm_t, encoder_hidden_states=encoder_hidden_states,
                        )
                        with torch.no_grad():
                            output_ref = ref_model(
                                x=model_input_ref, t=scm_t, encoder_hidden_states=encoder_hidden_states,
                            )

                    for name, module in self.transformer.named_modules():
                        if isinstance(module, NVFP4Linear):
                            loss_act, loss_param = module.get_differentiable_quantization_error(loss_fn)
                            step_wise_module_loss_dict[name] = {"act": loss_act.item(), "param": loss_param.item()}
                            loss_acm_act += loss_act
                            loss_acm_param += loss_param

                            ref_module = ref_model.get_submodule(name) 
                            loss_layer_wise = loss_fn(output_ref.detach(), output_target)
                            step_wise_layer_loss_dict[name] = loss_layer_wise.item()
                            loss_acm_layer_wise += loss_layer_wise

                    loss_step_wise = loss_fn(output_target, output_ref)
                    loss_acm_step_wise += loss_step_wise

                    error_info[i] = {
                        "module_loss": step_wise_module_loss_dict,
                        "layer_loss": step_wise_layer_loss_dict,
                        "step_loss": loss_step_wise.item(),
                    }

                    noise_pred_target = (
                        (1 - 2 * scm_t_expanded) * model_input_target
                        + (1 - 2 * scm_t_expanded + 2 * scm_t_expanded**2) * output_target
                    ) / torch.sqrt(scm_t_expanded**2 + (1 - scm_t_expanded) ** 2)
                    noise_pred_target = noise_pred_target.float() * sigma_data

                    noise_pred_ref = (
                        (1 - 2 * scm_t_expanded) * model_input_ref
                        + (1 - 2 * scm_t_expanded + 2 * scm_t_expanded**2) * output_ref
                    ) / torch.sqrt(scm_t_expanded**2 + (1 - scm_t_expanded) ** 2)
                    noise_pred_ref = noise_pred_ref.float() * sigma_data

                    latent_target_scaled, _ = self.scheduler.step(noise_pred_target, timestep, latent_target_scaled, return_dict=False)
                    with torch.no_grad():
                        latent_ref_scaled, _ = self.scheduler.step(noise_pred_ref, timestep, latent_ref_scaled, return_dict=False)

                latent_target = (latent_target_scaled / sigma_data).to(self.dtype)
                latent_ref = (latent_ref_scaled / sigma_data).detach()

            else:
                raise ValueError(f"Scheduler type {type(self.scheduler)} is not supported.")
        
        final_loss = loss_fn(latent_ref.detach(), latent_target)
        error_info['final'] = final_loss.item()

        return_dict = [latent_target, error_info]
        if criterion == "module-wise":
            total_loss = loss_acm_act + loss_acm_param
        elif criterion == "layer-wise":
            total_loss = loss_acm_layer_wise
        elif criterion == "step-wise":
            total_loss = loss_acm_step_wise
        else:
            raise ValueError(f"Criterion type {criterion} is not supported.")
        return_dict.append(total_loss)
        return return_dict

