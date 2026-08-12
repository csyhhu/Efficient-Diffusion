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
            ref_transformer = pipe.transformer
            pipe.transformer = None
            self.transformer = NVFP4QuantizedSD3.from_pretrained(
                self.model_id,
                download_source=self.download_source,
                cache_dir=self.cache_dir,
                block_size=self.block_size,
                use_nvfp4=self.use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
                torch_dtype=self.dtype,
                ref_model=ref_transformer,
            )
            del ref_transformer
            gc.collect()
            # Assign NVFP4 transformer to pipe and move everything to GPU
            pipe.transformer = self.transformer
            print(f">> [{time.time() - cur_time:.2f}] Finish Custom Transformer Loading")
        self.pipe = pipe.to(self.device)

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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_reference_model(self):
        """Build an unquantized SD3 reference model."""
        ref_model = NVFP4QuantizedSD3.from_pretrained(
            self.model_id, download_source=self.download_source, cache_dir=self.cache_dir,
            block_size=self.block_size,
            rotation=None, permutation=None, use_nvfp4=False,
            torch_dtype=self.dtype,
        )
        return ref_model.to(self.device, dtype=self.dtype)

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

    def compute_distillation_loss(self, ref_model, batch_data, criterion, num_steps,
                                  loss_fn=F.mse_loss, single_step_mode=True, **kwargs):
        """Compute distillation loss for SD3 (FlowMatchEuler scheduler)."""
        loss_acm_act = torch.tensor(0.0, device=self.device)
        loss_acm_param = torch.tensor(0.0, device=self.device)
        loss_acm_layer_wise = torch.tensor(0.0, device=self.device)
        loss_acm_step_wise = torch.tensor(0.0, device=self.device)
        error_info = {}

        hidden_states = batch_data.get('x')
        encoder_hidden_states = batch_data.get('encoder_hidden_states')
        pooled_projections = batch_data.get('pooled_projections', None)
        encoder_attention_mask = batch_data.get('encoder_attention_mask', None)
        noise = torch.randn_like(hidden_states)

        if single_step_mode:
            sampled_step_idx = torch.randint(0, num_steps, (1,)).item()
            steps_to_run = [sampled_step_idx]
        else:
            steps_to_run = range(num_steps)

        # FlowMatchEuler scheduler setup
        self.scheduler.set_timesteps(num_steps, device=self.device)
        flowmatch_timesteps = self.scheduler.timesteps.to(self.device)

        for i in steps_to_run:
            step_wise_module_loss_dict = {}
            step_wise_layer_loss_dict = {}

            t = flowmatch_timesteps[i]
            timestep = t.expand(hidden_states.shape[0])

            t_norm = (t / float(self.scheduler.config.num_train_timesteps)) \
                if hasattr(self.scheduler.config, 'num_train_timesteps') else (t.float() / 1000.0)
            t_expanded = t_norm.view(-1, *([1] * (hidden_states.dim() - 1)))
            x_t = (1 - t_expanded) * hidden_states + t_expanded * noise

            model_input_target = x_t
            model_input_ref = x_t.detach()

            common_kwargs = dict(
                hidden_states=model_input_target,
                timestep=timestep.to(self.dtype),
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                return_dict=False,
            )
            if encoder_attention_mask is not None:
                common_kwargs["encoder_attention_mask"] = encoder_attention_mask

            output_target = self.transformer(**common_kwargs)[0]
            with torch.no_grad():
                ref_kwargs = dict(common_kwargs)
                ref_kwargs["hidden_states"] = model_input_ref
                output_ref = ref_model(**ref_kwargs)[0]

            for name, module in self.transformer.named_modules():
                if isinstance(module, NVFP4Linear):
                    loss_act, loss_param = module.get_differentiable_quantization_error(loss_fn)
                    step_wise_module_loss_dict[name] = {"act": loss_act.item(), "param": loss_param.item()}
                    loss_acm_act += loss_act
                    loss_acm_param += loss_param
                    loss_layer_wise = loss_fn(output_ref.detach(), output_target)
                    step_wise_layer_loss_dict[name] = loss_layer_wise.item()
                    loss_acm_layer_wise += loss_layer_wise

            loss_step_wise = loss_fn(output_target, output_ref)
            loss_acm_step_wise += loss_step_wise

            error_info[0 if single_step_mode else i] = {
                "module_loss": step_wise_module_loss_dict,
                "layer_loss": step_wise_layer_loss_dict,
                "step_loss": loss_step_wise.item(),
            }

        final_loss = loss_step_wise
        error_info['final'] = final_loss.item()

        return_dict = [hidden_states, error_info]
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
