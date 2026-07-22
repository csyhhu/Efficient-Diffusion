import torch

class PermutationBase:
    """Base class for column permutations applied on the contraction (feature) dim.

    A permutation reorders the feature columns of BOTH the weight
    (W_perm = W[:, order]) and the activation (x_perm = x[:, order]). Because a
    permutation matrix satisfies P @ P^T = I, before quantization:

        y = x_perm @ W_perm^T = x @ W^T

    holds exactly. The permutation is data-dependent (typically derived from the
    weight column magnitudes) so that adjacent NVFP4 blocks end up with similar
    magnitude, tightening each block's amax and improving block-scale usage.
    """

    def __init__(self):
        self.permutation = None

    def fit(self, input_):
        raise NotImplementedError

    def transform_weight(self, W):
        # self.permutation may have been fitted while the weight lived on a different
        # device (e.g. CPU during from_pretrained); cast it to match W so
        # index_select runs on a single device.
        permutation = self.permutation.to(W.device)
        return W.index_select(-1, permutation)

    def transform_activation(self, x):
        permutation = self.permutation.to(x.device)
        return x.index_select(-1, permutation)


class IdentityPermutation(PermutationBase):
    """No-op permutation (baseline)."""

    def fit(self, input_):
        in_features = input_.shape[-1]
        self.permutation = torch.arange(in_features, device=input_.device, dtype=torch.int64)


class RandomPermutation(PermutationBase):
    """Random permutation (baseline / ablation)."""

    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed

    def fit(self, input_):
        in_features = input_.shape[-1]
        self.permutation = torch.randperm(
            in_features,
            generator=torch.Generator(device=input_.device).manual_seed(self.seed or 0),
            device=input_.device,
            dtype=torch.int64
        )


class MagnitudeSortPermutation(PermutationBase):
    """Sort feature columns by per-column magnitude so that columns of similar
    magnitude become adjacent -> each NVFP4 block has a tighter amax.

    Args:
        metric: 'norm' (L2 over the out dim) or 'max' (max abs over the out dim).
        order:  'asc' (small -> large) or 'desc' (large -> small).
    """

    def __init__(self, metric='norm', order='asc'):
        super().__init__()
        self.metric = metric
        self.sort_order = order

    def fit(self, input_):
        in_features = input_.shape[-1]
        w = input_.detach().float()
        if self.metric == 'max':
            col_mag = w.abs().max(dim=0).values
        else:
            col_mag = w.norm(dim=0)
        descending = (self.sort_order == 'desc')
        self.permutation = torch.argsort(col_mag, descending=descending).to(input_.device, dtype=torch.int64)


def make_permutation(value, seed=None):
    """Create a permutation factory function based on the given value.
    
    Args:
        value: permutation type or PermutationBase instance
        seed: random seed for reproducibility
    
    Returns:
        Callable: function that creates a permutation instance
    """
    if value is None:
        return None
    elif value == "identity":
        return IdentityPermutation()
    elif value == "random":
        return RandomPermutation(seed=seed)
    elif value == "mag":
        return MagnitudeSortPermutation()
    else:
        raise ValueError(f"Unknown permutation: {value}")


if __name__ == "__main__":
    
    import torch
    import torch.nn.functional as F

    bs = 2
    dim_1 = 4
    dim_2 = 5
    w = torch.randn(dim_2, dim_1)
    x = torch.randn(bs, dim_1)
    o = F.linear(x, w, None)
    for name, cls in [
        ("identity", IdentityPermutation),
        ("random", RandomPermutation),
        ("mag", MagnitudeSortPermutation)
    ]:
        p = cls()
        # p.fit(w)
        p.fit(x)
        pw = p.transform_weight(w)
        px = p.transform_activation(x)
        po = F.linear(px, pw, None)
        print(p.permutation)
        print(torch.mean(torch.pow(o - po, 2))) # Should be similar to 0
        # print(p.permutation)
