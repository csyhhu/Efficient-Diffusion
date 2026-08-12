"""Sana image generator.

``SanaImageGenerator`` supports ``Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers``.
The ``_custom_generate`` method mirrors ``SanaSprintPipeline.__call__`` exactly.
"""

import os
import gc
import json
import time

import torch
import torch.nn.functional as F

from diffusers import SanaSprintPipeline, SCMScheduler
from diffusers.utils.torch_utils import randn_tensor

from src.image_generator.base import BaseImageGenerator
from src.models.nvfp4_quantized_Sana import NVFP4QuantizedSana
from src.modules.quantized_linear import NVFP4Linear
from src.utils import save_sample_grid


class SanaImageGenerator(BaseImageGenerator):
    """Image generator for ``Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers``.

    The ``_custom_generate`` method is a 1:1 reproduction of
    ``SanaSprintPipeline.__call__``:

      1. Gemma text encoding with CHI prefix
      2. SCMScheduler with ``max_timesteps=1.5708``, ``intermediate_timesteps=1.3``
      3. Latent initialization with ``sigma_data`` scaling
      4. SCM timestep computation and noise prediction correction
      5. VAE decode with ``scaling_factor`` only (no ``shift_factor``)
    """

    SANA_MODEL_ID = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"

    # Defaults from SanaSprintPipeline.__call__
    SANA_DEFAULT_NUM_STEPS = 2
    SANA_DEFAULT_GUIDANCE = 4.5
    SANA_DEFAULT_HEIGHT = 1024
    SANA_DEFAULT_WIDTH = 1024
    SANA_DEFAULT_MAX_SEQ_LEN = 300
    SANA_DEFAULT_MAX_TIMESTEPS = 1.5708
    SANA_DEFAULT_INTERMEDIATE_TIMESTEPS = 1.3

    def __init__(self, model_id=None, device="cuda", dtype=torch.bfloat16, **kwargs):
        if model_id is None:
            model_id = self.SANA_MODEL_ID
        kwargs.setdefault("device", device)
        kwargs.setdefault("dtype", dtype)
        super().__init__(model_id=model_id, **kwargs)

    def load_pipe(self):
        """Load SanaSprintPipeline and build custom transformer.

        Memory-optimised transformer loading:
          - Pipeline loads on CPU with low_cpu_mem_usage=True.
          - The pipeline's transformer is reused as the reference for
            weight copy, avoiding a second load from disk.
          - After copy, the reference is freed, the NVFP4 transformer is
            assigned to the pipe, and the whole pipe moves to GPU.
        """
        load_path = self._resolve_local_model_path(self.model_id, self.cache_dir)

        cur_time = time.time()
        pipe = SanaSprintPipeline.from_pretrained(
            load_path,
            torch_dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        print(f">> [{time.time() - cur_time:.2f}] Finish Pipeline Loading")
        # Reuse pipeline's transformer as reference for weight copy.
        cur_time = time.time()
        if self.use_origin_model:
            self.transformer = pipe.transformer
            print(f">> [{time.time() - cur_time:.2f}] Use Origin Transformer")
        else:
            # ref_transformer = pipe.transformer
            pipe.transformer = None
            del pipe.transformer
            self.transformer = NVFP4QuantizedSana.from_pretrained(
                self.model_id,
                download_source=self.download_source,
                cache_dir=self.cache_dir,
                block_size=self.block_size,
                use_nvfp4=self.use_nvfp4,
                rotation=self.rotation,
                permutation=self.permutation,
                torch_dtype=self.dtype,
                # ref_model=ref_transformer,
            )
            # del ref_transformer
            # Assign NVFP4 transformer to pipe and move everything to GPU
            pipe.transformer = self.transformer
            print(f">> [{time.time() - cur_time:.2f}] Finish Custom Transformer Loading")
        
        self.pipe = pipe.to(self.device)
        # Extract components
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self._sigma_data = float(getattr(pipe.scheduler.config, 'sigma_data', 1.0))

        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

        # Load transformer config
        with open(os.path.join(load_path, "transformer", "config.json"), "r", encoding="utf-8") as f:
            self._transformer_config = json.load(f)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_reference_model(self):
        """Build an unquantized Sana reference model."""
        ref_model = NVFP4QuantizedSana.from_pretrained(
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
        save_root=None,
        save_name=None,
        visual_n_row=4,
        return_intermediates=False,
        **kwargs,
    ):
        """Custom generation mirroring SanaSprintPipeline.__call__.
        """
        # ------------------------------------------------------------------
        # 0. Resolve defaults
        # ------------------------------------------------------------------
        num_steps = num_steps or self.SANA_DEFAULT_NUM_STEPS
        guidance_scale = kwargs.get("guidance_scale", self.SANA_DEFAULT_GUIDANCE)
        height = kwargs.get("height", self.SANA_DEFAULT_HEIGHT)
        width = kwargs.get("width", self.SANA_DEFAULT_WIDTH)
        max_sequence_length = kwargs.get("max_sequence_length", self.SANA_DEFAULT_MAX_SEQ_LEN)
        max_timesteps = kwargs.get("max_timesteps", self.SANA_DEFAULT_MAX_TIMESTEPS)
        intermediate_timesteps = kwargs.get("intermediate_timesteps", self.SANA_DEFAULT_INTERMEDIATE_TIMESTEPS)
        # Steps at which DiT runs; None means all steps. Other steps reuse
        # the most recent DiT noise_pred (skipping the forward).
        dit_inference_steps = kwargs.get("dit_inference_steps", None)

        device = self.device
        # ------------------------------------------------------------------
        # 1. Encode prompt
        # ------------------------------------------------------------------
        self.text_encoder.to(device)

        prompt_embeds, prompt_attention_mask, _ = self.encode_prompt(
            prompt,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_samples,
        )

        # Offload text encoder to CPU
        self.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 2. Prepare latents
        #    SanaSprintPipeline: dtype=torch.float32, then *= sigma_data
        # ------------------------------------------------------------------
        latent_channels = self.in_channels
        vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor

        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None else None
        )
        latents = randn_tensor(
            (num_samples, latent_channels, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        latents = latents * self.scheduler.config.sigma_data

        # ------------------------------------------------------------------
        # 3. Prepare guidance
        # ------------------------------------------------------------------
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]).to(prompt_embeds.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale

        # ------------------------------------------------------------------
        # 4. Prepare timesteps
        # ------------------------------------------------------------------
        self.scheduler.set_timesteps(
            num_steps,
            device=device,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timesteps,
        )
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        # SanaSprintPipeline: timesteps = timesteps[:-1]
        timesteps = self.scheduler.timesteps[:-1].to(device)

        # Prepare extra step kwargs (generator, eta)
        import inspect
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        # ------------------------------------------------------------------
        # 5. Denoising loop (mirrors SanaSprintPipeline.__call__)
        # ------------------------------------------------------------------
        intermediates_recorder = None
        if return_intermediates:
            intermediates_recorder = {
                "dit_outputs": [],
                "noise_preds": [],
                "scheduler_outputs": [],
            }

        denoised = None
        last_dit_output = None
        last_noise_pred = None
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            run_dit = (dit_inference_steps is None) or (i in dit_inference_steps)

            if run_dit:
                latents_model_input = latents / self.scheduler.config.sigma_data

                scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)
                latent_model_input = latents_model_input * torch.sqrt(
                    scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
                )

                dit_output = self.transformer(
                    latent_model_input.to(dtype=self.dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=self.dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    return_dict=False,
                )[0]
                last_dit_output = dit_output

                # Noise prediction correction (SanaSprintPipeline lines 864-868)
                noise_pred = (
                    (1 - 2 * scm_timestep_expanded) * latent_model_input
                    + (1 - 2 * scm_timestep_expanded + 2 * scm_timestep_expanded**2) * dit_output
                ) / torch.sqrt(scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2)
                noise_pred = noise_pred.float() * self.scheduler.config.sigma_data
                last_noise_pred = noise_pred
            else:
                # Skip DiT forward, reuse the most recent outputs
                dit_output = last_dit_output
                noise_pred = last_noise_pred

            if intermediates_recorder is not None:
                intermediates_recorder["dit_outputs"].append(dit_output.detach().cpu())
                intermediates_recorder["noise_preds"].append(noise_pred.detach().cpu())

            latents, denoised = self.scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False,
            )
            if intermediates_recorder is not None:
                intermediates_recorder["scheduler_outputs"].append(latents.detach().cpu())

        # ------------------------------------------------------------------
        # 6. VAE decode: denoised / sigma_data, then / scaling_factor
        # ------------------------------------------------------------------
        self.vae.eval()
        latents = denoised / self.scheduler.config.sigma_data
        vae_input = (latents / self.vae.config.scaling_factor).to(dtype=self.vae.dtype)
        images = self.vae.decode(vae_input, return_dict=False)[0]
        images = images.clamp(-1, 1)

        if return_intermediates:
            intermediates_recorder["final_output"] = images.detach().cpu()
            intermediates_recorder["num_steps"] = len(timesteps)
            return images, intermediates_recorder

        return images

    def compute_distillation_loss(self, ref_model, batch_data, criterion, num_steps,
                                  loss_fn=F.mse_loss, single_step_mode=True, **kwargs):
        """Compute distillation loss for Sana (SCM scheduler)."""
        loss_acm_act = torch.tensor(0.0, device=self.device)
        loss_acm_param = torch.tensor(0.0, device=self.device)
        loss_acm_layer_wise = torch.tensor(0.0, device=self.device)
        loss_acm_step_wise = torch.tensor(0.0, device=self.device)
        error_info = {}

        hidden_states = batch_data.get('x')
        encoder_hidden_states = batch_data.get('encoder_hidden_states')
        encoder_attention_mask = batch_data.get('encoder_attention_mask', None)
        noise = torch.randn_like(hidden_states)

        if single_step_mode:
            sampled_step_idx = torch.randint(0, num_steps, (1,)).item()
            steps_to_run = [sampled_step_idx]
        else:
            steps_to_run = range(num_steps)

        # SCM scheduler setup
        guidance_embeds_scale = getattr(
            getattr(self.transformer, 'config', None), "guidance_embeds_scale", 0.1
        )
        guidance = torch.full(
            [hidden_states.shape[0]], kwargs.get("guidance", 4.5),
            device=self.device, dtype=torch.float32,
        )
        guidance = guidance.to(self.dtype) * guidance_embeds_scale

        self.scheduler.set_timesteps(
            num_steps, device=self.device,
            max_timesteps=1.5708,
            intermediate_timesteps=(1.3 if num_steps == 2 else None),
        )
        self.scheduler.set_begin_index(0)
        scm_timesteps = self.scheduler.timesteps[:-1].to(self.device).type(self.dtype)

        for i in steps_to_run:
            step_wise_module_loss_dict = {}
            step_wise_layer_loss_dict = {}

            t = scm_timesteps[i]
            timestep = t.expand(hidden_states.shape[0])
            scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
            scm_t = scm_t.to(self.dtype)
            scm_t_expanded = scm_t.view(-1, 1, 1, 1)

            scale = torch.sqrt(scm_t_expanded**2 + (1 - scm_t_expanded)**2)
            x_t = scale * hidden_states + (1 - scale) * noise

            model_input_target = x_t.to(self.dtype)
            model_input_ref = x_t.detach().to(self.dtype)

            output_target = self.transformer(
                hidden_states=model_input_target,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=scm_t, guidance=guidance, return_dict=False,
            )[0]
            with torch.no_grad():
                output_ref = ref_model(
                    hidden_states=model_input_ref,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=scm_t, guidance=guidance, return_dict=False,
                )[0]

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
