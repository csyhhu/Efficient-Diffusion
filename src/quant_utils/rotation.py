import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationBase:
    """Base class for orthogonal transforms applied on the contraction (feature) dim.

    A rotation R is an [n, n] orthogonal matrix (R @ R^T = I) where n is the
    working dimension (>= in_features, padded to a multiple of block_size, and
    for Hadamard also a power of two). It is applied identically (right-multiply)
    to BOTH the weight (W_rot = W @ R) and the activation (x_rot = x @ R), so
    that before quantization:

        y = x_rot @ W_rot^T = x @ R @ R^T @ W^T = x @ W^T

    holds exactly. The numerical benefit only appears once NVFP4 quantization is
    applied in the rotated space; mathematically the transform is lossless.
    """

    def __init__(self, block_size=16, seed=None):
        self.block_size = block_size
        self.seed = seed
        self._cache = {}  # (n, device, dtype) -> R

    # ---- methods to override ----
    def _padded_dim(self, in_features):
        raise NotImplementedError

    def _build_matrix(self, n, device, dtype):
        raise NotImplementedError

    # ---- shared machinery ----
    def padded_dim(self, in_features):
        return self._padded_dim(in_features)

    def _ensure_matrix(self, in_features, device, dtype):
        n = self._padded_dim(in_features)
        key = (n, str(device), str(dtype))
        if key not in self._cache:
            self._cache[key] = self._build_matrix(n, device, dtype)
        return self._cache[key]

    def rotate_weight(self, W):
        """W: [out, in] -> W_rot: [out, n] (padded feature dim)."""
        n = self._padded_dim(W.shape[-1])
        R = self._ensure_matrix(W.shape[-1], W.device, W.dtype)
        Wp = F.pad(W, (0, n - W.shape[-1]))
        return Wp @ R

    def rotate_activation(self, x):
        """x: [..., in] -> x_rot: [..., n] (padded feature dim)."""
        in_features = x.shape[-1]
        n = self._padded_dim(in_features)
        R = self._ensure_matrix(in_features, x.device, x.dtype)
        xp = F.pad(x, (0, n - in_features))
        return xp @ R

    def invalidate(self):
        """Drop any cached rotation matrix (e.g. after learning its parameters)."""
        self._cache = {}


class IdentityRotation(RotationBase):
    """Identity rotation (baseline). R = I_n, where n is padded to a multiple
    of block_size. This is equivalent to no rotation but keeps the code path
    consistent with other rotations.
    """

    def _padded_dim(self, in_features):
        return ((in_features + self.block_size - 1) // self.block_size) * self.block_size

    def _build_matrix(self, n, device, dtype):
        return torch.eye(n, device=device, dtype=dtype)


class HadamardRotation(RotationBase):
    """Data-free Walsh-Hadamard rotation.

    R = H_n / sqrt(n), where n is the smallest power of two that is also a
    multiple of block_size and >= in_features. The dense matrix is built here
    for clarity; for large dims a fast Walsh-Hadamard transform (O(n log n))
    can replace the explicit matmul.
    """

    def _padded_dim(self, in_features):
        n = ((in_features + self.block_size - 1) // self.block_size) * self.block_size
        pow2 = 1
        while pow2 < n:
            pow2 <<= 1
        return pow2

    def _build_matrix(self, n, device, dtype):
        H = torch.ones(1, 1, dtype=torch.float64)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = H.to(device=device, dtype=dtype)
        return H / math.sqrt(n)


class RandomRotation(RotationBase):
    """Random orthogonal rotation via QR of a Gaussian matrix.

    Data-free and deterministic given `seed`. Provides a strong "stress" baseline
    for ablation studies.
    """

    def _padded_dim(self, in_features):
        return ((in_features + self.block_size - 1) // self.block_size) * self.block_size

    def _build_matrix(self, n, device, dtype):
        g = torch.randn(
            n, n,
            generator=torch.Generator(device=device).manual_seed(self.seed or 0),
            device=device, dtype=dtype,
        )
        q, _ = torch.linalg.qr(g)
        # fix the sign of each column for deterministic output
        s = torch.sign(torch.diag(q))
        s[s == 0] = 1.0
        q = q * s.unsqueeze(0)
        return q.to(device=device, dtype=dtype)


class CayleyRotation(RotationBase):
    """Learnable orthogonal rotation parameterized by a skew-symmetric matrix K:

        R(K) = (I + K) (I - K)^{-1}

    Because K is skew-symmetric, R is strictly orthogonal for ANY K, so the
    transform stays lossless during and after optimization.

    For activation calibration, it is strongly recommended to initialize from a
    Hadamard rotation via ``fit_activation(..., init_from_matrix=H)``.  Under
    the hood this uses a residual formulation ``R_eff = R_init @ R(K)`` where
    ``R(K)`` is the (learnable) Cayley residual starting from K=0 (identity).
    This avoids the numerical singularity of the inverse Cayley transform on
    Hadamard (which has eigenvalues -1 making R+I singular).
    """

    def __init__(self, block_size=16, seed=None):
        super().__init__(block_size, seed)
        self.K = None          # skew-symmetric [n, n] tensor
        self._R_init = None    # optional [n, n] orthogonal init matrix (residual)

    def _padded_dim(self, in_features):
        return self._padded_dim_static(in_features, self.block_size)

    @staticmethod
    def _padded_dim_static(in_features, block_size):
        return ((in_features + block_size - 1) // block_size) * block_size

    def set_init_matrix(self, R):
        """Set a fixed initial rotation matrix (e.g. Hadamard).

        The effective rotation becomes ``R_eff = R_init @ R(K)`` where
        ``R(K)`` is the Cayley transform of the learnable skew-symmetric K.
        When K=0, ``R_eff = R_init`` (identity residual).
        """
        self._R_init = R
        self.invalidate()

    def _build_matrix(self, n, device, dtype):
        if self.K is None or self.K.shape[0] != n \
                or self.K.device != device or self.K.dtype != dtype:
            self.K = torch.zeros(n, n, device=device, dtype=dtype)
        K = self.K - self.K.T  # enforce skew-symmetry
        I = torch.eye(n, device=device, dtype=dtype)
        R_cayley = torch.linalg.solve((I - K).float(), (I + K).float()).type(dtype)  # Cayley residual
        if self._R_init is not None:
            R_init = self._R_init.to(device=device, dtype=dtype)
            return R_init @ R_cayley
        return R_cayley

    def fit(self, weights, iters=200, lr=1e-2, optimizer=None,
            init_from_matrix=None):
        """Optimize K to minimize NVFP4 reconstruction error over calibration weights.

        Args:
            weights: list of tensors [out_i, in_features] (un-padded originals).
            init_from_matrix: optional [n, n] orthogonal matrix to use as
                              initial rotation (e.g. Hadamard).  K starts at
                              0 so the effective rotation is R_init @ I = R_init.
        """
        from .quantization import NVFP4Quantization

        n = self._padded_dim(weights[0].shape[-1])
        device, dtype = weights[0].device, weights[0].dtype
        self.K = nn.Parameter(torch.zeros(n, n, device=device, dtype=dtype))
        if init_from_matrix is not None:
            self.set_init_matrix(init_from_matrix.to(device=device, dtype=dtype))
        self.invalidate()
        opt = optimizer or torch.optim.Adam([self.K], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            self.invalidate()  # ensure R reflects the current K
            loss = torch.zeros((), device=device, dtype=dtype)
            for w in weights:
                w_rot = self.rotate_weight(w)
                w_q = NVFP4Quantization.apply(w_rot, self.block_size)
                # detach target to avoid gradient cancellation (same tensor used
                # as both quantize-input and MSE-target)
                loss = loss + F.mse_loss(w_q, w_rot.detach())
            loss.backward()
            opt.step()
        self.K = self.K.detach()
        self.invalidate()

    def fit_activation(self, activations, iters=200, lr=1e-2, optimizer=None,
                        init_from_matrix=None):
        """Optimize K to minimize NVFP4 *activation* quantization error.

        Unlike ``fit()`` which optimizes for weight reconstruction, this method
        minimizes the quantization error of the rotated activations:

            MSE( x @ R(K),  quantize_activation(x @ R(K)) )

        **Important**: It is strongly recommended to initialize from a Hadamard
        matrix via ``init_from_matrix``.  This uses the residual formulation
        ``R_eff = R_init @ R(K)`` where K starts at 0 (identity residual),
        which avoids the numerical singularity that the inverse Cayley
        transform would encounter with Hadamard (eigenvalues -1).

        Args:
            activations: list of activation tensors (any shape ending with
                         [..., in_features]).
            iters:       number of optimization iterations.
            lr:          learning rate for Adam optimizer.
            optimizer:   optional pre-configured optimizer (overrides lr).
            init_from_matrix: optional [n, n] orthogonal matrix (e.g. Hadamard).
                              K is initialized at 0 so the effective initial
                              rotation is exactly this matrix.
        """
        from .quantization import NVFP4ActivationQuantization

        in_features = activations[0].shape[-1]
        n = self._padded_dim(in_features)
        device = activations[0].device
        dtype = activations[0].dtype
        self.K = nn.Parameter(torch.zeros(n, n, device=device, dtype=dtype))
        if init_from_matrix is not None:
            self.set_init_matrix(init_from_matrix.to(device=device, dtype=dtype))
        self.invalidate()
        opt = optimizer or torch.optim.Adam([self.K], lr=lr)

        # Pre-process activations: flatten all leading dims to [tokens, in_features]
        act_flat = []
        for a in activations:
            a_f = a.to(device=device, dtype=torch.float32).reshape(-1, in_features)
            act_flat.append(a_f)
        all_act = torch.cat(act_flat, dim=0)  # [total_tokens, in_features]

        for _ in range(iters):
            opt.zero_grad()
            self.invalidate()  # ensure R reflects the current K

            all_rot = self.rotate_activation(all_act)  # [total_tokens, n]

            # Quantize: NVFP4ActivationQuantization expects 3D [bs, n_seq, dim]
            all_rot_3d = all_rot.unsqueeze(1)  # [total_tokens, 1, n]
            all_q = NVFP4ActivationQuantization.apply(all_rot_3d, self.block_size)
            all_q = all_q.squeeze(1)  # [total_tokens, n]

            # IMPORTANT: detach the target so the gradient only flows through
            # the quantize path, not through the target.  Using all_rot as both
            # input-to-quantize AND MSE-target causes gradient cancellation:
            #   d(Loss)/dx_rot = 2*(x_q - x_rot) + (-2*(x_q - x_rot)) = 0
            loss = F.mse_loss(all_q, all_rot.detach())
            loss.backward()
            opt.step()

        self.K = self.K.detach()
        self.invalidate()


if __name__ == "__main__":

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, cls in [("hadamard", HadamardRotation), ("random", RandomRotation), ("cayley", CayleyRotation)]:
        rot = cls()
        R = rot._ensure_matrix(20, device, torch.float64)
        err = (R @ R.T - torch.eye(R.shape[0], dtype=torch.float64)).abs().max().item()
        print(f"{name}: R orthogonal, max deviation = {err:.2e}")
