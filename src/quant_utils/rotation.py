import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys, os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class RotationBase(nn.Module):
    """Base class for orthogonal transforms applied on the contraction (feature) dim.

    A rotation R is an [n, n] orthogonal matrix (R @ R^T = I) where n is the
    working dimension. It is applied identically (right-multiply) to BOTH the
    weight (W_rot = W @ R) and the activation (x_rot = x @ R), so that before
    quantization:

        y = x_rot @ W_rot^T = x @ R @ R^T @ W^T = x @ W^T

    holds exactly. The numerical benefit only appears once NVFP4 quantization is
    applied in the rotated space; mathematically the transform is lossless.

    The rotation matrix is created on CPU during initialization. When rotate_*
    methods are called, the rotation matrix is automatically moved to match
    the input tensor's device and dtype.
    """

    def __init__(self, in_features, seed=None):
        super().__init__()
        self.in_features = in_features
        self.seed = seed
        self.fit()

    def fit(self):
        raise NotImplementedError

    def rotate_weight(self, W):
        R = self.rotation.to(device=W.device, dtype=W.dtype)
        return W @ R

    def rotate_activation(self, x):
        R = self.rotation.to(device=x.device, dtype=x.dtype)
        return x @ R


class IdentityRotation(RotationBase):
    """Identity rotation (baseline). R = I_n.

    Optimised: does NOT store the n×n identity matrix, and rotate_weight /
    rotate_activation return their inputs directly without a matrix multiply.
    This saves ~8 GB of GPU memory for SD3.5-medium (288 Linear layers).
    """

    def fit(self):
        pass  # No buffer needed — identity is a no-op.

    @property
    def rotation(self):
        # Lazily create the matrix only if someone explicitly requests it
        # (e.g. for serialisation).  Not stored as a buffer to avoid the
        # GPU memory overhead.
        return torch.eye(self.in_features)

    def rotate_weight(self, W):
        return W  # W @ I = W

    def rotate_activation(self, x):
        return x  # x @ I = x


class HadamardRotation(RotationBase):
    """Data-free Walsh-Hadamard rotation.

    R = H_n / sqrt(n), where n = in_features. If in_features is not a power of two,
    we construct the next power-of-two Hadamard matrix, take the top-left
    in_features x in_features block, and orthogonalize it via QR decomposition
    to ensure strict orthogonality.
    """

    def _padded_dim(self):
        n = self.in_features
        pow2 = 1
        while pow2 < n:
            pow2 <<= 1
        return pow2

    def fit(self):
        n = self._padded_dim()
        H = torch.ones(1, 1, dtype=torch.float64)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = H / math.sqrt(n)
        
        if n == self.in_features:
            self.register_buffer('_rotation', H)
        else:
            H_block = H[:self.in_features, :self.in_features]
            q, _ = torch.linalg.qr(H_block.float())
            s = torch.sign(torch.diag(q))
            s[s == 0] = 1.0
            self.register_buffer('_rotation', (q * s.unsqueeze(0)).type(H.dtype))
    
    @property
    def rotation(self):
        return self._rotation


class RandomRotation(RotationBase):
    """Random orthogonal rotation via QR of a Gaussian matrix.

    Data-free and deterministic given `seed`. Provides a strong "stress" baseline
    for ablation studies.
    """

    def fit(self):
        g = torch.randn(
            self.in_features, self.in_features,
            generator=torch.Generator().manual_seed(self.seed or 0),
        )
        q, _ = torch.linalg.qr(g.float())
        s = torch.sign(torch.diag(q))
        s[s == 0] = 1.0
        q = q * s.unsqueeze(0)
        self.register_buffer('_rotation', q)
    
    @property
    def rotation(self):
        return self._rotation


class CayleyRotation(RotationBase):
    """Learnable orthogonal rotation parameterized by a skew-symmetric matrix K:

        R = R_init @ (I + K) (I - K)^{-1}

    fit() initializes an orthogonal rotation matrix R_init (identity by default).
    K is a learnable skew-symmetric parameter that updates the rotation via
    Cayley transform, ensuring strict orthogonality during optimization.

    The effective rotation matrix is computed dynamically via the `rotation` property,
    which enables automatic gradient flow through K during training.

    For activation calibration, initialize from a Hadamard rotation via
    ``set_init_matrix(H)``. The residual formulation avoids numerical singularity
    of the inverse Cayley transform on Hadamard (eigenvalues -1 make R+I singular).

    Computation details:
        R_cayley = (I + K) @ (I - K)^{-1}  [Cayley transform]
        R = R_init @ R_cayley                [residual formulation]
        
        Since K is skew-symmetric (K^T = -K), R is guaranteed orthogonal:
        R @ R^T = R_init @ (I+K)(I-K)^{-1} @ ((I+K)(I-K)^{-1})^T @ R_init^T
                = R_init @ (I+K)(I-K)^{-1} @ ((I-K)^{-1})^T @ (I+K)^T @ R_init^T
                = R_init @ (I+K)(I-K)^{-1} @ (I+K)^{-1} @ (I-K) @ R_init^T
                = R_init @ (I+K)((I-K)(I+K))^{-1} @ (I-K) @ R_init^T
                = R_init @ (I+K)(I-K^2)^{-1} @ (I-K) @ R_init^T
                = R_init @ (I-K^2)^{-1}(I+K)(I-K) @ R_init^T
                = R_init @ (I-K^2)^{-1}(I-K^2) @ R_init^T
                = R_init @ R_init^T = I
    """

    def __init__(self, in_features, seed=None):
        super().__init__(in_features, seed)

    def fit(self):
        self.register_buffer('_R_init', torch.eye(self.in_features))
        self.K = nn.Parameter(torch.zeros(self.in_features, self.in_features))
        self.init_k()

    def init_k(self):
        if self._R_init is None:
            raise RuntimeError("fit() must be called before init_k()")
        rng = torch.Generator().manual_seed(self.seed or 0)
        K = torch.randn(self.in_features, self.in_features, generator=rng) * 0.01
        K = K - K.T
        self.K.data = K

    def set_init_matrix(self, R_init):
        self._R_init = R_init

    @property
    def rotation(self):
        K = self.K - self.K.T
        I = torch.eye(self.in_features, device=self.K.device, dtype=self.K.dtype)
        R_cayley = torch.linalg.solve((I - K).float(), (I + K).float()).type(self.K.dtype)
        return self._R_init.to(device=self.K.device, dtype=self.K.dtype) @ R_cayley


def make_rotation(rotation_type, in_features, seed=None):
    """Create a per-layer rotation factory.

    Returns a callable that creates a RotationBase instance when called with in_features.
    Each call produces a fresh instance, enabling per-layer independent rotations.

    Args:
        value: rotation type or RotationBase instance
        seed: random seed for reproducibility

    Returns:
        Callable: factory function (in_features) -> RotationBase
    """
    if rotation_type is None or rotation_type in ["none"]:
        return None
    if rotation_type in ["identity"]:
        return IdentityRotation(in_features)
    elif rotation_type == "hadamard":
        return HadamardRotation(in_features)
    elif rotation_type == "random":
        return RandomRotation(in_features, seed=seed)
    elif rotation_type == "cayley":
        return CayleyRotation(in_features, seed=seed)
    else:
        raise ValueError(f"Unknown rotation: {rotation_type}")


if __name__ == "__main__":

    import torch.nn.functional as F

    bs = 2
    dim_1 = 6
    dim_2 = 5
    w = torch.randn(dim_2, dim_1)
    x = torch.randn(bs, dim_1)
    o = F.linear(x, w, None)

    for name, cls in [
        # ("identity", IdentityRotation),
        ("hadamard", HadamardRotation),
        # ("random", RandomRotation),
        # ("cayley", CayleyRotation)
    ]:
        p = cls(dim_1)
        pw = p.rotate_weight(w)
        px = p.rotate_activation(x)
        po = F.linear(px, pw, None)
        print(p.rotation)
        print(f"{name}: MSE = {torch.mean(torch.pow(o - po, 2)):.6f}")