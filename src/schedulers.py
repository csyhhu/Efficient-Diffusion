"""
Diffusion schedulers: DDPM (noise prediction), Flow Matching (velocity prediction),
and Consistency Model (data prediction, one-step / multistep).

Usage::

    from src.schedulers import (DDPMScheduler,
                                FlowMatchingScheduler,
                                ConsistencyModelScheduler)

    # DDPM
    sched = DDPMScheduler(num_timesteps=1000)
    x_t, noise = sched.add_noise(x_0, t)

    # Flow Matching
    sched = FlowMatchingScheduler()
    x_1 = sched.sample_noise(x_0)
    t   = sched.sample_time(B, device)
    x_t = sched.interpolate(x_0, x_1, t)
    v   = sched.compute_target(x_0, x_1)

    # Consistency Model
    sched = ConsistencyModelScheduler(num_discretization=40)
    # -- training --
    sigma_n, sigma_np1 = sched.sample_timestep_pair(B, device)
    x_sn, x_snp1, z = sched.add_noise_pair(x, sigma_n, sigma_np1)
    # -- sampling --
    samples = sched.multistep_sample(model, shape, device)
"""

import torch
import torch.nn as nn


class DDPMScheduler:
    """Linear beta schedule DDPM noise scheduler.

    Precomputes all coefficients needed for the forward (noising) and
    reverse (denoising) processes.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self._betas = betas
        self._alphas = alphas
        self._alpha_cumprod = alpha_cumprod
        self._sqrt_alpha_cumprod = alpha_cumprod.sqrt()
        self._sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt()
        # For sampling
        self._sqrt_recip_alphas = (1.0 / alphas).sqrt()
        self._posterior_variance = betas * (1.0 - alpha_cumprod) / (1.0 - alpha_cumprod)

    # -- Properties to access buffers on the correct device at runtime --
    def _to_device(self, attr: str, device) -> torch.Tensor:
        t = getattr(self, attr)
        if t.device != device:
            t = t.to(device)
            setattr(self, attr, t)
        return t

    def num_timesteps_on_device(self, device) -> torch.Tensor:
        return self._to_device("_betas", device).shape[0]

    @property
    def T(self) -> int:
        return int(self._betas.shape[0])

    # -- Forward diffusion (training) --
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple:
        """Forward diffusion: x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε.

        Args:
            x_0: clean image/latent, shape (B, C, H, W)
            t:   timestep index per sample, shape (B,), dtype long
            noise: optional; generated if None.

        Returns:
            (x_t, noise) — noisy sample + noise used (training target).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._sqrt_alpha_cumprod.to(x_0.device)[t]
        sqrt_1m_alpha = self._sqrt_one_minus_alpha_cumprod.to(x_0.device)[t]

        while sqrt_alpha.ndim < x_0.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_1m_alpha = sqrt_1m_alpha.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_1m_alpha * noise
        return x_t, noise

    def sample_timesteps(self, batch_size: int, device: str) -> torch.Tensor:
        """Sample random discrete timesteps t ~ Uniform(0, T-1)."""
        return torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)

    # -- Posterior for DDPM sampling --
    def get_posterior_coeffs(self, t: int, device: str):
        """Return coefficients for x_{t-1} ← x_t, eps prediction."""
        if t == 0:
            return {
                "mean_coef_x_t": 1.0 / self._alphas[0].sqrt().item(),
                "mean_coef_eps": 0.0,
                "sigma": 0.0,
            }

        alpha_t = self._alphas[t].item()
        alpha_bar_t = self._alpha_cumprod[t].item()
        beta_t = self._betas[t].item()
        coef = (1 - alpha_t) / (1 - alpha_bar_t) ** 0.5
        mean_coef_x_t = 1.0 / (alpha_t ** 0.5)
        mean_coef_eps = coef / (alpha_t ** 0.5)
        sigma = beta_t ** 0.5
        return {"mean_coef_x_t": mean_coef_x_t, "mean_coef_eps": mean_coef_eps, "sigma": sigma}


class FlowMatchingScheduler:
    """Scheduler for conditional Flow Matching with straight-line paths.

    Given data x_0 and noise x_1 ~ N(0,I), the conditional path is:
        x_t = (1-t)·x_0 + t·x_1       (interpolation)
        v_t = x_1 - x_0               (target velocity)

    The model v_θ(x_t, t, c) is trained to regress v_t.
    """

    def __init__(self):
        pass

    def sample_noise(self, x_0: torch.Tensor) -> torch.Tensor:
        """x_1 ~ N(0, I), same shape as x_0."""
        return torch.randn_like(x_0)

    def sample_time(self, batch_size: int, device: str) -> torch.Tensor:
        """t ~ Uniform(0, 1), shape (B,), float32."""
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """x_t = (1-t)·x_0 + t·x_1."""
        t_ = t
        for _ in range(x_0.ndim - 1):
            t_ = t_.unsqueeze(-1)
        return (1 - t_) * x_0 + t_ * x_1

    def compute_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """v_t = x_1 - x_0."""
        return x_1 - x_0

    def step(self, noise_pred, timestep, sample, return_dict=False):
        """Euler step for flow matching sampling.
        
        Args:
            noise_pred: Predicted velocity (or noise) from the model
            timestep: Current timestep
            sample: Current sample (x_t)
            return_dict: Whether to return a dict or tuple
            
        Returns:
            (next_sample, denoised) tuple or dict
        """
        dt = -1.0 / 50
        next_sample = sample + noise_pred * dt
        return (next_sample, next_sample) if not return_dict else {"prev_sample": next_sample, "denoised": next_sample}


# ============================================================================
# Consistency Model (CM) — EDM noise schedule, data-prediction
# ============================================================================

class ConsistencyModelScheduler:
    """Scheduler for Consistency Models (Song et al., 2023).

    The model f_θ(x_t, σ) is trained to map **any** point on a PF-ODE
    trajectory directly to the clean data **x_0** (data prediction), enforcing
    self-consistency::

        f_θ(x_{σ_n}, σ_n) ≈ f_θ(x_{σ_{n+1}}, σ_{n+1})

    Uses the EDM (Karras et al.) noise schedule:

    * Training:  sample adjacent σ levels, add shared noise z.
    * Sampling:  iterative multistep (few steps → high quality) or one-step.

    Preconditioning
    ---------------
    The model is parameterized with EDM boundary-aware coefficients:

        f_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

    where ``c_skip(0)=1, c_out(0)=0`` so that f_θ(x, 0) = x.

    Parameters
    ----------
    sigma_min, sigma_max:
        noise scale range (default 0.002 / 80.0 as in EDM).
    rho:
        Karras schedule exponent (higher = denser near σ_min).
    sigma_data:
        data std-dev (0.5 for images in [-1, 1]).
    num_discretization:
        number of σ levels for the discretized schedule.
    p_mean, p_std:
        log-normal parameters for continuous σ sampling during training.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        num_discretization: int = 40,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

        # Precompute discretized σ sequence (ascending)
        self._sigmas = self._build_sigma_sequence(num_discretization)

    # ── Sigma discretization (Karras) ───────────────────────────────────────

    def _build_sigma_sequence(self, N: int) -> torch.Tensor:
        """σ_i = (σ_min^{1/ρ} + i/(N-1)·(σ_max^{1/ρ} - σ_min^{1/ρ}))^ρ."""
        rho_inv = 1.0 / self.rho
        indices = torch.arange(N, dtype=torch.float32)
        sigmas = (
            self.sigma_min ** rho_inv
            + indices / max(N - 1, 1)
            * (self.sigma_max ** rho_inv - self.sigma_min ** rho_inv)
        ) ** self.rho
        return sigmas  # (N,), ascending: σ_0 < σ_1 < ... < σ_{N-1}

    def reset_sigmas(self, N: int):
        """Rebuild σ sequence (e.g. to trade off quality vs speed at sampling time)."""
        self._sigmas = self._build_sigma_sequence(N)

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    @property
    def N(self) -> int:
        """Number of discretization levels."""
        return len(self._sigmas)

    @property
    def T(self) -> int:
        """Alias for DDPM compatibility (returns N)."""
        return self.N

    # ── Training: timestep / sigma sampling ─────────────────────────────────

    def sample_timestep_pair(
        self,
        batch_size: int,
        device: str,
    ) -> tuple:
        """Sample a pair of adjacent σ indices ``(σ_n, σ_{n+1})``.

        σ_{n+1} is the **noisier** level (further from data).  The model is
        trained so that both map to the same x_0 estimate.

        Returns
        -------
        sigma_n   : (B,) float — less noisy level
        sigma_np1 : (B,) float — more noisy level
        """
        n = torch.randint(0, self.N - 1, (batch_size,), device=device, dtype=torch.long)
        sigmas = self._sigmas.to(device)
        return sigmas[n], sigmas[n + 1]

    def sample_continuous_sigma(
        self,
        batch_size: int,
        device: str,
    ) -> torch.Tensor:
        """Sample σ ~ LogNormal(p_mean, p_std²) clamped to [σ_min, σ_max].

        This matches the EDM continuous-time training distribution, useful
        when training without a fixed discretization grid.
        """
        rnd = torch.randn(batch_size, device=device)
        sigma = (rnd * self.p_std + self.p_mean).exp()
        return sigma.clamp(self.sigma_min, self.sigma_max)

    # ── Training: add noise ─────────────────────────────────────────────────

    def sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Sample Gaussian noise z ~ N(0, I) with the same shape as *x*."""
        return torch.randn_like(x)

    def add_noise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple:
        """Add scaled noise: ``x_σ = x + σ·z``.

        Args:
            x:     clean data, shape (B, C, H, W).
            sigma: noise scale per sample, shape (B,).
            noise: optional pre-sampled z (reuse same z for multiple σ).

        Returns:
            (x_sigma, z) — noisy sample and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x)
        sigma_ = sigma
        for _ in range(x.ndim - 1):
            sigma_ = sigma_.unsqueeze(-1)
        return x + sigma_ * noise, noise

    def add_noise_pair(
        self,
        x: torch.Tensor,
        sigma_n: torch.Tensor,
        sigma_np1: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple:
        """Add noise at **two** σ levels sharing the same z.

        Returns ``(x_{σ_n}, x_{σ_{n+1}}, z)`` — used for CM training where
        both noisy samples must lie on the same PF-ODE trajectory.

        x_{σ_n}   is the less-noisy sample (used as target source).
        x_{σ_{n+1}} is the more-noisy sample (used as prediction input).
        """
        if noise is None:
            noise = torch.randn_like(x)

        sigma_n_ = sigma_n
        sigma_np1_ = sigma_np1
        for _ in range(x.ndim - 1):
            sigma_n_ = sigma_n_.unsqueeze(-1)
            sigma_np1_ = sigma_np1_.unsqueeze(-1)

        x_sn = x + sigma_n_ * noise
        x_snp1 = x + sigma_np1_ * noise
        return x_sn, x_snp1, noise

    # ── Preconditioning (EDM) ───────────────────────────────────────────────

    def get_preconditioning(self, sigma: torch.Tensor) -> dict:
        """EDM boundary-aware preconditioning coefficients.

        Returns a dict with keys ``c_skip, c_out, c_in, c_noise``, each of
        shape compatible with the input tensor.

        Usage::

            coeffs = scheduler.get_preconditioning(sigma)
            model_input = coeffs["c_in"] * x
            F = model(model_input, coeffs["c_noise"])
            x0 = coeffs["c_skip"] * x + coeffs["c_out"] * F

        where ``F`` is the raw network output.
        """
        sigma_data = self.sigma_data
        sigma_data_sq = sigma_data ** 2

        # Reshape sigma for broadcasting
        sigma_ = sigma
        while sigma_.ndim < 4:
            sigma_ = sigma_.unsqueeze(-1)

        sigma_sq = sigma_ ** 2

        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma_ * sigma_data / (sigma_sq + sigma_data_sq).sqrt()
        c_in = 1.0 / (sigma_sq + sigma_data_sq).sqrt()
        c_noise = 0.25 * sigma_.log()  # ¼·ln(σ)

        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
            "c_noise": c_noise,
        }

    # ── Loss ────────────────────────────────────────────────────────────────

    @staticmethod
    def pseudo_huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        c: float = 0.00054,
    ) -> torch.Tensor:
        """Pseudo-Huber loss: ``√(‖pred - target‖² + c²) - c``.

        Smooth L2-like metric with L1 behaviour for large residuals.
        Returns a scalar loss averaged over the batch (not reduced to scalar).
        """
        diff = pred - target
        diff_sq = diff.reshape(diff.shape[0], -1).square().sum(dim=1)
        return ((diff_sq + c ** 2).sqrt() - c).mean()

    @staticmethod
    def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample L1 loss, averaged over batch."""
        return pred.sub(target).abs().reshape(pred.shape[0], -1).mean(dim=1).mean()

    @staticmethod
    def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample L2 loss (MSE), averaged over batch."""
        return pred.sub(target).square().reshape(pred.shape[0], -1).mean(dim=1).mean()

    # ── Preconditioned forward helper ───────────────────────────────────────

    def preconditioned_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply EDM preconditioning and run model forward pass.

        ``x`` is the noisy input; ``sigma`` gives the noise level per sample.

        Returns the data prediction f_θ(x, σ) = x̂₀.
        """
        coeffs = self.get_preconditioning(sigma)
        model_in = coeffs["c_in"] * x
        # Flatten c_noise to 1D for time conditioning
        c_noise_1d = coeffs["c_noise"].squeeze(-1).squeeze(-1).squeeze(-1)
        F = model(model_in, c_noise_1d)
        return coeffs["c_skip"] * x + coeffs["c_out"] * F

    # ── Sampling ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def multistep_sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: str,
        use_preconditioning: bool = True,
    ) -> torch.Tensor:
        """Multistep consistency sampling (high quality, few steps).

        Iterates through decreasing σ levels, at each step predicting x₀
        then re-noising to the next lower level::

            x̂₀ = f_θ(x_{k+1}, σ_{k+1})
            x_k = x̂₀ + σ_k · z    (z ~ N(0,I))

        Args:
            model:  consistency function f_θ.
            shape:  (B, C, H, W).
            device: torch device.
            use_preconditioning:
                If True, applies c_skip / c_out / c_in (model F_θ output).
                If False, model is called as ``model(x, sigma)`` directly.

        Returns:
            x₀ tensor in [-1, 1].
        """
        model.eval()
        B = shape[0]
        sigmas = self._sigmas.to(device)
        N = len(sigmas)

        # Start from noise: x_{N-1} ~ N(0, σ_max²·I)
        x = sigmas[-1] * torch.randn(shape, device=device)

        for k in range(N - 2, -1, -1):
            sigma_k = sigmas[k]
            sigma_kp1 = sigmas[k + 1]

            # Predict x₀ from current noisy state
            if use_preconditioning:
                x0 = self.preconditioned_forward(model, x, sigma_kp1.expand(B))
            else:
                x0 = model(x, sigma_kp1.expand(B))

            # Re-noise to next level (unless at final step)
            if k > 0:
                z = torch.randn(shape, device=device)
                x = x0 + sigma_k * z
            else:
                x = x0

        return x.clamp(-1, 1)

    @torch.no_grad()
    def onestep_sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: str,
        use_preconditioning: bool = True,
    ) -> torch.Tensor:
        """One-step generation: ``x₀ = f_θ(x_T, σ_max)``.

        Fastest mode — maps pure noise directly to data in a single forward pass.
        Quality depends heavily on training.

        Args:
            model:  consistency function f_θ.
            shape:  (B, C, H, W).
            device: torch device.
            use_preconditioning: if True, uses EDM preconditioning wrapper.

        Returns:
            x₀ tensor in [-1, 1].
        """
        model.eval()
        B = shape[0]
        sigma_max = self._sigmas[-1].to(device)
        x = sigma_max * torch.randn(shape, device=device)

        if use_preconditioning:
            x0 = self.preconditioned_forward(model, x, sigma_max.expand(B))
        else:
            x0 = model(x, sigma_max.expand(B))

        return x0.clamp(-1, 1)
