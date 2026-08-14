"""SD3.5 image generator.

``SD3ImageGenerator`` supports ``stabilityai/stable-diffusion-3.5-medium``.
The ``_custom_generate`` method mirrors ``StableDiffusion3Pipeline.__call__`` exactly.
"""

import os
import gc
import json
import time
import inspect

import torch
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift

from src.image_generator.base import BaseImageGenerator
from src.models.nvfp4_quantized_SD3 import NVFP4QuantizedSD3
from src.modules.quantized_linear import NVFP4Linear
from src.utils import save_sample_grid


class SD3ImageGenerator(BaseImageGenerator):
    """Image generator for ``stabilityai/stable-diffusion-3.5-medium``.

    The ``_custom_generate`` method is a 1:1 reproduction of
    ``StableDiffusion3Pipeline.__call__`` without T5:

      1. Dual-CLIP prompt encoding (CLIP-L + CLIP-G) with CFG
      2. FlowMatchEulerDiscreteScheduler with dynamic shifting (``mu``)
      3. Classifier-free guidance in the denoising loop
      4. VAE decode with ``scaling_factor`` + ``shift_factor``
    """

    SD3_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

    # Defaults from StableDiffusion3Pipeline.__call__
    SD3_DEFAULT_NUM_STEPS = 28
    SD3_DEFAULT_GUIDANCE = 7.0
    SD3_DEFAULT_HEIGHT = 1024
    SD3_DEFAULT_WIDTH = 1024
    SD3_DEFAULT_MAX_SEQ_LEN = 256

    def __init__(self, model_id=None, device="cuda", dtype=torch.bfloat16, **kwargs):
        if model_id is None:
            model_id = self.SD3_MODEL_ID
        kwargs.setdefault("device", device)
        kwargs.setdefault("dtype", dtype)
        super().__init__(model_id=model_id, **kwargs)

    def encode_prompt(self, prompt, max_sequence_length=256, num_images_per_prompt=1,
                      do_classifier_free_guidance=True, negative_prompt=None):
        """Encode prompt for SD3: CLIP-L + CLIP-G concat, then T5 zero-padding.

        Mirrors ``StableDiffusion3Pipeline.encode_prompt`` without T5.

        Returns:
            (prompt_embeds, prompt_attention_mask=None, pooled_prompt_embeds)
        """
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
        

    def load_pipe(self):
        """Load StableDiffusion3Pipeline and build custom transformer.

        Memory-optimised transformer loading:
          - Pipeline loads on CPU with low_cpu_mem_usage=True.
          - The pipeline's transformer is reused as the reference for
            weight copy, avoiding a second load from disk.
          - After copy, the reference is freed, the NVFP4 transformer is
            assigned to the pipe, and the whole pipe moves to GPU.
        """
        load_path = self._resolve_local_model_path(self.model_id, self.cache_dir)
        print(f">> [{time.time()}] Load model from: {load_path}")
        cur_time = time.time()
        pipe = StableDiffusion3Pipeline.from_pretrained(
            load_path,
            torch_dtype=self.dtype,
            local_files_only=True,
            text_encoder_3=None,
            tokenizer_3=None,
            low_cpu_mem_usage=True,
        )
        print(f">> [{time.time() - cur_time:.2f}] Finish Pipeline Loading: Use origin model: {self.use_origin_model}")
        cur_time = time.time()
        if self.use_origin_model:
            self.transformer = pipe.transformer
            print(f">> [{time.time() - cur_time:.2f}] Use Origin Transformer")
        else:
            # Reuse pipeline's transformer as reference for weight copy.
            del pipe.transformer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe.transformer = None
            self.transformer = NVFP4QuantizedSD3.from_pretrained(
                self.model_id,
                download_source=self.download_source,
                cache_dir=self.cache_dir,
                block_size=self.block_size,
                use_nvfp4=self.use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
                torch_dtype=self.dtype
            )
            # Assign NVFP4 transformer to pipe and move everything to GPU
            pipe.transformer = self.transformer
            print(f">> [{time.time() - cur_time:.2f}] Finish Custom Transformer Loading")
        
        cur_time = time.time()
        self.pipe = pipe.to(self.device)
        print(f">> [{time.time() - cur_time:.2f}] Finish Pipe Loading")
        # Extract components (SD3 has two text encoders/tokenizers)
        self.tokenizer = [pipe.tokenizer, pipe.tokenizer_2]
        self.text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
        self._pooled_projection_dim = getattr(
            self.transformer.config, 'pooled_projection_dim', 2048
        )
        self._sigma_data = 1.0

        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

        # Load transformer config
        with open(os.path.join(load_path, "transformer", "config.json"), "r", encoding="utf-8") as f:
            self._transformer_config = json.load(f)

    def build_reference_model(self):
        """Build an unquantized SD3 reference model."""
        ref_model = NVFP4QuantizedSD3.from_pretrained(
            self.model_id, download_source=self.download_source, cache_dir=self.cache_dir,
            block_size=self.block_size,
            rotation=None, permutation=None, use_nvfp4=False,
            torch_dtype=self.dtype,
        )
        return ref_model.to(self.device, dtype=self.dtype)

    def _decode_latents_to_images(self, latents):
        """Decode SD3 latents: /scaling_factor + shift_factor (VAE on CPU)."""
        with torch.no_grad():
            scaling = self.vae.config.scaling_factor
            shift = getattr(self.vae.config, "shift_factor", 0)
            vae_input = (latents.detach() / scaling + shift).to(
                "cpu", dtype=self.vae.dtype
            )
            images = self.vae.decode(vae_input, return_dict=False)[0]
            return images.clamp(-1, 1).to(torch.float32)

    @torch.no_grad()
    def _custom_generate(
        self,
        prompt,
        num_samples=1,
        seed=42,
        num_steps=None,
        return_intermediates=False,
        **kwargs,
    ):
        """Custom generation mirroring StableDiffusion3Pipeline.__call__.
        """
        # ------------------------------------------------------------------
        # 0. Resolve defaults
        # ------------------------------------------------------------------
        num_steps = num_steps or self.SD3_DEFAULT_NUM_STEPS
        guidance_scale = kwargs.get("guidance_scale", self.SD3_DEFAULT_GUIDANCE)
        height = kwargs.get("height", self.SD3_DEFAULT_HEIGHT)
        width = kwargs.get("width", self.SD3_DEFAULT_WIDTH)
        max_sequence_length = kwargs.get("max_sequence_length", self.SD3_DEFAULT_MAX_SEQ_LEN)
        negative_prompt = kwargs.get("negative_prompt", None)
        # Steps at which DiT runs; None means all steps. Other steps reuse
        # the most recent DiT noise_pred (skipping the forward).
        dit_inference_steps = kwargs.get("dit_inference_steps", None)
        # if dit_inference_steps is not None:
        #     dit_inference_steps = [int(x) for x in dit_inference_steps.split(",")]

        do_cfg = guidance_scale > 1.0
        device = self.device

        # ------------------------------------------------------------------
        # 1. Encode prompt (mirrors SD3 encode_prompt with CFG)
        # ------------------------------------------------------------------
        if not isinstance(self.text_encoder, (list, tuple)):
            self.text_encoder.to(device)
        else:
            for te in self.text_encoder:
                te.to(device)

        prompt_embeds, _, pooled_prompt_embeds = self.encode_prompt(
            prompt,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )

        # Offload text encoders to CPU
        if not isinstance(self.text_encoder, (list, tuple)):
            self.text_encoder.to("cpu")
        else:
            for te in self.text_encoder:
                te.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 2. Prepare latents
        #    SD3 pipeline: dtype = prompt_embeds.dtype (= self.dtype)
        # ------------------------------------------------------------------
        num_channels_latents = self.in_channels
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor

        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None else None
        )
        latents = randn_tensor(
            (num_samples, num_channels_latents, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype,
        )

        # ------------------------------------------------------------------
        # 3. Prepare timesteps (dynamic shifting via mu)
        # ------------------------------------------------------------------
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            patch_size = self.transformer.config.patch_size
            image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu

        self.scheduler.set_timesteps(num_steps, device=device, **scheduler_kwargs)
        timesteps = self.scheduler.timesteps.to(device)

        # ------------------------------------------------------------------
        # 4. Denoising loop (mirrors SD3 __call__)
        # ------------------------------------------------------------------
        intermediates_recorder = None
        if return_intermediates:
            intermediates_recorder = {
                "dit_outputs": [],
                "noise_preds": [],
                "scheduler_outputs": [],
            }

        last_dit_output = None
        last_noise_pred = None
        for i, t in enumerate(timesteps):
            run_dit = (dit_inference_steps is None) or (i in dit_inference_steps)

            if run_dit:
                latent_model_input = (
                    torch.cat([latents] * 2, dim=0) if do_cfg else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                dit_output = self.transformer(
                    hidden_states=latent_model_input.to(dtype=self.dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(dtype=self.dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=self.dtype),
                    return_dict=False,
                )[0]
                last_dit_output = dit_output

                # Classifier-free guidance
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = dit_output.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    noise_pred = dit_output
                last_noise_pred = noise_pred
            else:
                # Skip DiT forward, reuse the most recent outputs
                dit_output = last_dit_output
                noise_pred = last_noise_pred

            if intermediates_recorder is not None:
                intermediates_recorder["dit_outputs"].append(dit_output.detach().cpu())
                intermediates_recorder["noise_preds"].append(noise_pred.detach().cpu())

            # Scheduler step
            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False,
            )[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            if intermediates_recorder is not None:
                intermediates_recorder["scheduler_outputs"].append(latents.detach().cpu())

        # ------------------------------------------------------------------
        # 5. VAE decode (latents / scaling_factor + shift_factor)
        # ------------------------------------------------------------------
        self.vae.eval()
        vae_scaling = self.vae.config.scaling_factor
        vae_shift = getattr(self.vae.config, "shift_factor", 0)
        latents = (latents / vae_scaling + vae_shift).to(dtype=self.vae.dtype)
        images = self.vae.decode(latents, return_dict=False)[0]
        images = images.clamp(-1, 1)

        if return_intermediates:
            intermediates_recorder["final_output"] = images.detach().cpu()
            intermediates_recorder["num_steps"] = len(timesteps)
            return images, intermediates_recorder

        return images

    def compute_distillation_loss(self, ref_model, batch_data, criterion, num_steps, loss_fn=F.mse_loss, single_step_mode=True, test_mode=False, **kwargs):
        """Compute distillation loss for SD3 with a full FlowMatchEuler decode loop.

        No hooks — all losses come from NVFP4Linear's exposed intermediates.
        Noise is **randomly sampled** each call (not fixed) to enable proper
        diffusion training loss (``dit_loss``) computation.

        For each decode step:
          1. Sample noise fresh, mix x_t from clean_target and noise.
          2. Run the quantized ``self.transformer`` on ``x_t`` (with grad).
          3. Run the unquantized ``ref_model`` (no grad).
          4. Per-NVFP4Linear: module-wise (act+param) + layer-wise (output diff).
          5. step-wise: ``loss_fn(output_target, output_ref)``.
          6. dit_loss: standard FM velocity prediction loss
             (target = noise - x_0, MSE(output, target)).
          7. scheduler.step to update latents.
          8. Record raw per-step ``error_info[step_idx]``.

        Returns ``(latents, step_error_info, total_loss)`` where
        ``step_error_info`` is the raw per-step dict (not aggregated).
        """
        hidden_states = batch_data.get('x')
        encoder_hidden_states = batch_data.get('encoder_hidden_states')
        pooled_projections = batch_data.get('pooled_projections', None)
        encoder_attention_mask = batch_data.get('encoder_attention_mask', None)

        # Random noise each call — enables proper diffusion training loss.
        noise = torch.randn_like(hidden_states)
        # SD3 FM mixing: x_t = (1-t)*x_0 + t*noise, x_0 = hidden_states (no sigma_data scaling).
        clean_target = hidden_states

        # ---- FlowMatchEuler scheduler setup (with optional mu-shifting) -----
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            patch_size = getattr(getattr(self.transformer, 'config', None),
                                 'patch_size', None) or 2
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            latent_h = hidden_states.shape[2]
            latent_w = hidden_states.shape[3]
            image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu

        self.scheduler.set_timesteps(num_steps, device=self.device, **scheduler_kwargs)
        fm_boundary_timesteps = self.scheduler.timesteps.to(self.device)
        fm_timesteps = fm_boundary_timesteps[:-1] if len(fm_boundary_timesteps) == num_steps + 1 else fm_boundary_timesteps

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

        # ---- Initialise decode-loop latents -------------------------------------
        def _fm_mix(t_scalar):
            t_norm = (t_scalar / float(self.scheduler.config.num_train_timesteps)) \
                if hasattr(self.scheduler.config, 'num_train_timesteps') \
                else (t_scalar.float() / 1000.0)
            return t_norm.view(-1, *([1] * (hidden_states.dim() - 1)))

        if single_step_mode:
            i0 = steps_to_run[0]
            t0 = fm_boundary_timesteps[min(i0, len(fm_boundary_timesteps) - 1)]
            t_e0 = _fm_mix(t0)
            x_t_target = ((1 - t_e0) * clean_target + t_e0 * noise).to(self.dtype)
            x_t_ref = x_t_target.detach().clone()
        else:
            t_es = _fm_mix(fm_boundary_timesteps[0])
            x_t_target = ((1 - t_es) * clean_target + t_es * noise).to(self.dtype)
            x_t_ref = x_t_target.detach().clone()

        for i in steps_to_run:
            step_module_loss_dict = {}
            step_layer_loss_dict = {}

            t = fm_timesteps[i] if i < len(fm_timesteps) else fm_boundary_timesteps[-1]
            timestep = t.expand(hidden_states.shape[0])

            common_kwargs = dict(
                hidden_states=x_t_target,
                timestep=timestep.to(self.dtype),
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                return_dict=False,
            )
            if encoder_attention_mask is not None:
                common_kwargs["encoder_attention_mask"] = encoder_attention_mask

            # --- Transformer forward (quantized target, with grad) --------------
            output_target = self.transformer(**common_kwargs)[0]

            # --- Reference transformer forward (no grad) -------------------------
            with torch.no_grad():
                ref_kwargs = dict(common_kwargs)
                ref_kwargs["hidden_states"] = x_t_ref
                output_ref = ref_model(**ref_kwargs)[0]

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

            # --- dit_loss: standard FM velocity prediction loss ------------------
            # For FlowMatch: target = noise - x_0 (velocity), model predicts velocity.
            t_e = _fm_mix(t)
            target_velocity = noise.float() - clean_target.float()
            dit_loss = loss_fn(output_target.float(), target_velocity.detach())
            loss_acm_dit = loss_acm_dit + dit_loss.to(torch.float32)

            # --- Record raw per-step error_info ----------------------------------
            error_info[0 if single_step_mode else i] = {
                "module_loss": step_module_loss_dict,
                "layer_loss": step_layer_loss_dict,
                "step_loss": float(loss_step_wise.item()),
                "dit_loss": float(dit_loss.item()),
            }

            # --- Scheduler step (FlowMatchEuler) ----------------------------------
            x_t_target = self.scheduler.step(
                output_target, t, x_t_target, return_dict=False,
            )[0]
            with torch.no_grad():
                x_t_ref = self.scheduler.step(
                    output_ref, t, x_t_ref, return_dict=False,
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
