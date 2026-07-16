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

    def __init__(self, block_size=16):
        self.block_size = block_size
        self.order = None  # LongTensor [in_features] of new column indices

    def compute_order(self, weight):
        raise NotImplementedError

    def fit(self, weight):
        # Ensure the permutation index lives on the same device as the weight,
        # otherwise downstream index_select fails on a CUDA model.
        self.order = self.compute_order(weight).to(weight.device)
        return self

    def transform_weight(self, W):
        # self.order may have been fitted while the weight lived on a different
        # device (e.g. CPU during from_pretrained); cast it to match W so
        # index_select runs on a single device.
        order = self.order.to(W.device)
        return W.index_select(-1, order)

    def transform_activation(self, x):
        order = self.order.to(x.device)
        return x.index_select(-1, order)


class IdentityPermutation(PermutationBase):
    """No-op permutation (baseline)."""

    def compute_order(self, weight):
        in_features = weight.shape[-1]
        return torch.arange(in_features, device=weight.device)


class RandomPermutation(PermutationBase):
    """Random permutation (baseline / ablation)."""

    def __init__(self, block_size=16, seed=0):
        super().__init__(block_size)
        self.seed = seed

    def compute_order(self, weight):
        in_features = weight.shape[-1]
        return torch.randperm(
            in_features,
            generator=torch.Generator(device=weight.device).manual_seed(self.seed),
            device=weight.device,
        )


class MagnitudeSortPermutation(PermutationBase):
    """Sort feature columns by per-column magnitude so that columns of similar
    magnitude become adjacent -> each NVFP4 block has a tighter amax.

    Args:
        metric: 'norm' (L2 over the out dim) or 'max' (max abs over the out dim).
        order:  'asc' (small -> large) or 'desc' (large -> small).
    """

    def __init__(self, block_size=16, metric='norm', order='asc'):
        super().__init__(block_size)
        self.metric = metric
        self.sort_order = order

    def compute_order(self, weight):
        in_features = weight.shape[-1]
        w = weight.detach().float()
        if self.metric == 'max':
            col_mag = w.abs().max(dim=0).values
        else:
            col_mag = w.norm(dim=0)
        descending = (self.sort_order == 'desc')
        return torch.argsort(col_mag, descending=descending)


if __name__ == "__main__":
    import torch

    W = torch.randn(8, 10)
    for name, cls in [("identity", IdentityPermutation),
                      ("random", RandomPermutation),
                      ("mag", MagnitudeSortPermutation)]:
        p = cls()
        p.fit(W)
        # verify it is a valid permutation of the columns
        assert torch.equal(torch.sort(p.order).values, torch.arange(10))
        assert torch.allclose(W.index_select(-1, p.order)[:, p.order], W)
        print(f"{name}: valid permutation of 10 columns")
