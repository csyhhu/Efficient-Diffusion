import json
import os
from typing import List

import torch
from PIL import Image

try:
    from .modules.quantized_linear import NVFP4Linear
    from .quant_utils.activation_calibrator import calibrate_activation_rotations, make_calibration_loader
    from .quant_utils.permutation import IdentityPermutation, MagnitudeSortPermutation, PermutationBase, RandomPermutation
    from .quant_utils.rotation import CayleyRotation, HadamardRotation, IdentityRotation, RandomRotation, RotationBase
except ImportError:
    import sys
    _src = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _src)
    from modules.quantized_linear import NVFP4Linear
    from quant_utils.activation_calibrator import calibrate_activation_rotations, make_calibration_loader
    from quant_utils.permutation import IdentityPermutation, MagnitudeSortPermutation, PermutationBase, RandomPermutation
    from quant_utils.rotation import CayleyRotation, HadamardRotation, IdentityRotation, RandomRotation, RotationBase


def _make_rotation(value, block_size=16, seed=None):
    if value is None or value in ["none", "identity"]:
        return IdentityRotation(block_size=block_size)
    if isinstance(value, RotationBase):
        return value
    if value == "hadamard":
        return HadamardRotation(block_size=block_size)
    if value == "random":
        return RandomRotation(block_size=block_size, seed=seed)
    if value == "cayley":
        return CayleyRotation(block_size=block_size, seed=seed)
    raise ValueError(f"Unknown rotation: {value}")


def _make_permutation(value, block_size=16, seed=None):
    if value is None or value in ["none", "identity"]:
        return lambda in_features: IdentityPermutation(block_size=block_size)
    if isinstance(value, PermutationBase):
        import copy
        return lambda in_features: copy.deepcopy(value)
    if value == "random":
        if seed is None:
            seed = torch.seed() % (2**31 - 1)
        return lambda in_features: RandomPermutation(block_size=block_size, seed=seed)
    if value == "mag":
        return lambda in_features: MagnitudeSortPermutation(block_size=block_size)
    raise ValueError(f"Unknown permutation: {value}")


class _DummyScheduler:
    config = {"sigma_data": 1.0}
    init_noise_sigma = 1.0

    def set_timesteps(self, num_steps, device=None):
        self.timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)

    def step(self, model_output, timestep, latent):
        return type("DummyOutput", (), {"prev_sample": latent - model_output})


class ImageGeneration:
    def __init__(
        self, model_id=None, block_size=16,
        download_source="modelscope", cache_dir=None, dry_mode=False,
        dry_config=None, config_only_mode=False, device="cuda",  dtype=t"orch.bfloat16"
    ):
        self.model_id = model_id
        self.block_size = block_size
        self.download_source = download_source
        self.cache_dir = cache_dir
        self.dry_mode = dry_mode
        self.config_only_mode = config_only_mode
        self.pipe = None
        self.transformer = None
        self._dry_config = None
        self._transformer_config = None
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        """
        if dry_mode:
            self._build_dry_components(dry_config or {})
        elif config_only_mode:
            self._build_config_only_components(model_id, cache_dir)
        else:
            self._build_real_components(model_id, cache_dir)
        """

    def _resolve_local_model_path(self, model_id, cache_dir):
        if cache_dir:
            local_dir = os.path.join(cache_dir, model_id.replace("/", os.sep))
            if os.path.exists(local_dir):
                return local_dir
            raise FileNotFoundError(f"Model config not found in local cache directory: {local_dir}")
        if model_id and os.path.exists(model_id):
            return model_id
        raise FileNotFoundError(f"Model path not found: {model_id}")

    def _build_dry_components(self, dry_config):
        self._dry_config = dry_config or {}
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = _DummyScheduler()
        self._sigma_data = 1.0
        self.transformer = self._new_dry_transformer(False, None, None)

    def _build_config_only_components(self, model_id, cache_dir):
        from diffusers import SanaSprintPipeline
        load_path = self._resolve_local_model_path(model_id, cache_dir)
        print(f"[Config-only mode] Loading config and real components (tokenizer, vae, scheduler) from {load_path}...")
        pipe = SanaSprintPipeline.from_pretrained(load_path, torch_dtype=self.dtype or torch.bfloat16, local_files_only=True)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.vae = pipe.vae.to(self.device)
        self.scheduler = pipe.scheduler
        self._sigma_data = float(pipe.scheduler.config.sigma_data)
        if self.dtype is None:
            self.dtype = next(self.vae.parameters()).dtype
        with open(os.path.join(load_path, "transformer", "config.json"), "r", encoding="utf-8") as f:
            self._transformer_config = json.load(f)
        self.transformer = self._new_config_transformer(False, None, None)
        _unused = pipe.transformer
        pipe.transformer = None
        del _unused
        self.pipe = pipe
        self.pipe.transformer = self.transformer
        self.pipe = self.pipe.to(self.device)

    def _build_real_components(self, model_id, cache_dir):
        from diffusers import SanaSprintPipeline
        load_path = self._resolve_local_model_path(model_id, cache_dir)
        pipe = SanaSprintPipeline.from_pretrained(load_path, torch_dtype=self.dtype or torch.bfloat16, local_files_only=True)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.vae = pipe.vae.to(self.device)
        self.scheduler = pipe.scheduler
        self._sigma_data = float(pipe.scheduler.config.sigma_data)
        if self.dtype is None:
            self.dtype = next(self.vae.parameters()).dtype

    def _new_dry_transformer(self, use_nvfp4, rotation, permutation):
        from .models.nvfp4_quantized_Sana import NVFP4QuantizedSana
        cfg = self._dry_config or {}
        dim = cfg.get("dim", 256)
        heads = cfg.get("num_heads", 8)
        model = NVFP4QuantizedSana(
            sample_size=cfg.get("resolution", 64) // 8, patch_size=1,
            in_channels=4, out_channels=4, num_layers=cfg.get("layers", 2),
            attention_head_dim=dim // heads, num_attention_heads=heads,
            num_cross_attention_heads=heads, cross_attention_head_dim=dim // heads,
            cross_attention_dim=dim, caption_channels=cfg.get("caption_channels", 768),
            mlp_ratio=2.5, block_size=self.block_size, use_nvfp4=use_nvfp4,
            rotation=rotation, permutation=permutation)
        return model.to(self.device, dtype=self.dtype).eval()

    def _new_config_transformer(self, use_nvfp4, rotation, permutation):
        from .models.nvfp4_quantized_Sana import NVFP4QuantizedSana
        cfg = self._transformer_config or {}
        inner_dim = cfg.get("num_attention_heads", 24) * cfg.get("attention_head_dim", 64)
        model = NVFP4QuantizedSana(
            sample_size=cfg.get("sample_size", 32), patch_size=cfg.get("patch_size", 1),
            in_channels=cfg.get("in_channels", 32), out_channels=cfg.get("out_channels", 32),
            num_layers=cfg.get("num_layers", 20), attention_head_dim=cfg.get("attention_head_dim", 64),
            num_attention_heads=cfg.get("num_attention_heads", 24),
            num_cross_attention_heads=cfg.get("num_cross_attention_heads", 24),
            cross_attention_head_dim=cfg.get("cross_attention_head_dim", 64),
            cross_attention_dim=cfg.get("cross_attention_dim", inner_dim),
            caption_channels=cfg.get("caption_channels", 2304), mlp_ratio=cfg.get("mlp_ratio", 2.5),
            attention_bias=cfg.get("attention_bias", True),
            norm_elementwise_affine=cfg.get("norm_elementwise_affine", False),
            norm_eps=cfg.get("norm_eps", 1e-6), interpolation_scale=cfg.get("interpolation_scale", None),
            guidance_embeds=cfg.get("guidance_embeds", True),
            guidance_embeds_scale=cfg.get("guidance_embeds_scale", 0.1),
            qk_norm=cfg.get("qk_norm", None), block_size=self.block_size,
            use_nvfp4=use_nvfp4, rotation=rotation, permutation=permutation)
        return model.to(self.device, dtype=self.dtype).eval()

    @property
    def in_channels(self):
        return self.transformer.config.in_channels if self.transformer is not None else 32

    @property
    def resolution(self):
        return self.transformer.config.sample_size * 8 if self.transformer is not None else 1024

    def build_transformer(self, rotation=None, permutation=None, use_nvfp4=True, ref_model=None):
        from .models.nvfp4_quantized_Sana import NVFP4QuantizedSana
        rot = _make_rotation(rotation, self.block_size)
        perm = _make_permutation(permutation, self.block_size)
        if self.dry_mode:
            model = self._new_dry_transformer(use_nvfp4, rot, perm)
        elif self.config_only_mode:
            model = self._new_config_transformer(use_nvfp4, rot, perm)
        else:
            model = NVFP4QuantizedSana.from_pretrained(
                self.model_id, download_source=self.download_source, cache_dir=self.cache_dir,
                block_size=self.block_size, use_nvfp4=use_nvfp4,
                rotation=rot, permutation=perm).to(self.device).eval()
        self.transformer = model
        return model

    def set_transformer(self, transformer):
        self.transformer = transformer.to(self.device).eval()

    def encode_prompt(self, prompt, num_images_per_prompt=1):
        if self.dry_mode or self.text_encoder is None:
            channels = self.transformer.caption_projection.linear_1.in_features
            return torch.randn(num_images_per_prompt, 77, channels, device=self.device)
        max_length = getattr(self.tokenizer, "model_max_length", 77)
        if isinstance(max_length, int) and max_length > 10000:
            max_length = 77
        text_inputs = self.tokenizer(
            prompt, padding="max_length", 
            max_length=max_length, truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            embeds = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return embeds.repeat_interleave(num_images_per_prompt, dim=0).type(self.dtype)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, generator=None, latents=None):
        if latents is not None:
            return latents.to(self.device, dtype=self.dtype)
        if generator is None:
            generator = torch.Generator(device=self.device)
        return torch.randn(batch_size, num_channels_latents, height // 8, width // 8, generator=generator, device=self.device, dtype=self.dtype) * self._sigma_data

    def decode(self, latent, txt_embed, num_steps=2, guidance=4.5, return_intermediates=False):
        if self.transformer is None:
            raise RuntimeError("Transformer not set. Call build_transformer() first.")
        self.transformer.eval()
        sigma_data = float(self.scheduler.config.sigma_data) if hasattr(self.scheduler, 'config') else self._sigma_data
        guidance_embeds_scale = getattr(self.transformer.config, "guidance_embeds_scale", 0.1)
        
        guidance = torch.full([latent.shape[0]], guidance, device=self.device, dtype=self.dtype)
        guidance = guidance * guidance_embeds_scale
        
        if hasattr(self.scheduler, "set_timesteps"):
            self.scheduler.set_timesteps(
                num_steps, device=self.device,
                max_timesteps=1.5708,
                intermediate_timesteps=(1.3 if num_steps == 2 else None),
            )
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(0)
            timesteps = self.scheduler.timesteps[:-1].to(self.device).type(self.dtype)
        else:
            timesteps = torch.linspace(1.5708, 0.0, num_steps + 1, device=self.device, dtype=self.dtype)[:-1]
        
        _noise_gen = torch.Generator(device=self.device).manual_seed(1234)
        extra_step_kwargs = {}
        
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                timestep = t.expand(latent.shape[0])
                latents_model_input = latent / sigma_data
                scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                scm_te = scm_t.view(-1, 1, 1, 1)
                model_input = latents_model_input * torch.sqrt(scm_te ** 2 + (1 - scm_te) ** 2)
                # print(step_idx, latents_model_input.dtype, scm_te.dtype, model_input.dtype, txt_embed.dtype)
                output = self.transformer(
                    hidden_states=model_input, encoder_hidden_states=txt_embed,
                    timestep=scm_t, guidance=guidance, return_dict=False
                )[0]
                
                noise_pred = (
                    (1 - 2 * scm_te) * model_input
                    + (1 - 2 * scm_te + 2 * scm_te ** 2) * output
                ) / torch.sqrt(scm_te ** 2 + (1 - scm_te) ** 2)
                noise_pred = noise_pred * sigma_data
                
                latent, denoised = self.scheduler.step(
                    noise_pred, timestep, latent, **extra_step_kwargs, return_dict=False)
                
                if return_intermediates:
                    item = {
                        "step": step_idx, 
                        "latent": latent.clone().detach(), 
                        "output": output.clone().detach(),
                        "denoised": denoised.clone().detach(),
                        "intermediate_outputs": {}
                    }
                    yield item
                else:
                    yield {"step": step_idx, "latent": latent, "output": output, "denoised": denoised}

    def generate(self, prompt, seed, num_steps=2, guidance=4.5, height=None, width=None, used_origin_pipe=False):
        if used_origin_pipe and self.pipe is not None:
            return self.pipe(prompt).images[0]
        else:
            txt_embed = self.encode_prompt(prompt)
            height = height or self.resolution
            width = width or self.resolution
            generator = torch.Generator(device=self.device).manual_seed(seed)
            latent = self.prepare_latents(1, self.in_channels, height, width, generator=generator)
            denoised = None
            for step_output in self.decode(latent, txt_embed, num_steps, guidance):
                latent = step_output["latent"]
                if "denoised" in step_output:
                    denoised = step_output["denoised"]
            if denoised is not None:
                final_latent = denoised / float(self.scheduler.config.sigma_data) if hasattr(self.scheduler, 'config') else denoised / self._sigma_data
            else:
                final_latent = latent / float(self.scheduler.config.sigma_data) if hasattr(self.scheduler, 'config') else latent / self._sigma_data
            return self.latent_to_image(final_latent)

    def generate_batch(self, prompts: List[str], seeds: List[int], num_steps=2, guidance=4.5):
        return [self.generate(prompt, seed, num_steps, guidance) for prompt, seed in zip(prompts, seeds)]

    def calibrate_cayley(self, prompts=None, txt_embeds=None, n_batches=8, iters=200, lr=0.01, verbose=True):
        if self.transformer is None:
            raise RuntimeError("Transformer not set. Call build_transformer() first.")
        if txt_embeds is None:
            if prompts is None:
                raise ValueError("Either prompts or txt_embeds must be provided")
            txt_embeds = [self.encode_prompt(p).detach().cpu() for p in prompts]
        loader = make_calibration_loader(
            pipe=self, txt_embeds=txt_embeds,
            in_channels=getattr(self.transformer.config, "in_channels", 32),
            resolution=getattr(self.transformer.config, "sample_size", 32) * 8,
            device=self.device, n_batches=n_batches)
        stats = calibrate_activation_rotations(
            self.transformer, loader, forward_fn=lambda model, batch: model(**batch),
            module_class=NVFP4Linear, rotation_attr="rotation", block_size=self.block_size,
            iters=iters, lr=lr, init_from_hadamard=True, verbose=verbose)
        self.transformer.fit_all_permutations()
        return stats

    def extract_cayley_K(self):
        cache = {}
        for name, module in self.transformer.named_modules():
            if isinstance(module, NVFP4Linear) and isinstance(module.rotation, CayleyRotation) and module.rotation.K is not None:
                cache[name] = module.rotation.K.clone().detach().cpu()
        return cache

    def apply_cayley_from_cache(self, K_cache):
        for name, module in self.transformer.named_modules():
            if isinstance(module, NVFP4Linear) and name in K_cache and isinstance(module.rotation, CayleyRotation):
                module.rotation.K = K_cache[name].to(module.weight.device)
                module.rotation.invalidate()
        self.transformer.fit_all_permutations()

    def latent_to_image(self, latent):
        if self.dry_mode or self.vae is None:
            image_latent = latent[0]
            if image_latent.shape[0] == 1:
                image_latent = image_latent.repeat(3, 1, 1)
            elif image_latent.shape[0] not in (3, 4):
                image_latent = image_latent[:3]
            arr = (image_latent.permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            return Image.fromarray((arr * 255).astype("uint8"))
        latent = latent.to(device=self.device, dtype=self.dtype) / self.vae.config.scaling_factor
        image = self.vae.decode(latent, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()
        return Image.fromarray((image[0] * 255).astype("uint8"))

    def image_to_latent(self, image):
        if self.dry_mode or self.vae is None:
            raise NotImplementedError("image_to_latent not supported without VAE")
        import numpy as np
        vae_dtype = next(self.vae.parameters()).dtype
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=vae_dtype)
        image = image / 255.0 * 2 - 1
        latent = self.vae.encode(image, return_dict=False)[0].latent_dist.sample()
        return latent * self.vae.config.scaling_factor

    def eval(self):
        if self.transformer is not None:
            self.transformer.eval()

    def train(self):
        if self.transformer is not None:
            self.transformer.train()

    def __enter__(self):
        self.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
