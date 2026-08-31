"""
This sript computes the similarity between two computation diff by different metrics, find out whether different metric leads to different skipping results.
"""

import torch

from analysis.computation_diff.utils import load_samples, samples_to_avg


def _layer_sort_key(layer_name):
    """Sort key for layer names like 'block.0.', 'block.10.' (numeric, not lexicographic)."""
    parts = layer_name.split(".")
    for p in parts:
        if p.isdigit():
            return (int(p), layer_name)
    return (999, layer_name)


def _step_sort_key(step_idx):
    """Sort key for step indices (handle int and string)."""
    try:
        return (0, int(step_idx))
    except (ValueError, TypeError):
        return (1, str(step_idx))


def arrange_to_global_metric(_diff_dict):
    """Arrange per-layer step-wise diff into a global [n_steps, n_tokens] tensor.

    IMPORTANT: Both outer (layer_name) and inner (step_idx) keys are sorted
    deterministically so that different metric dicts (cos vs l2) produce
    row-aligned tensors for valid comparison.

    Skips layers whose token count differs from the first layer (e.g. the last
    layer that only has text tokens).
    """
    global_metric = []
    """
    for outer_key, outer_value in _diff_dict.items():
        for inner_key, _diff in outer_value.items():
            if len(global_metric) > 0 and _diff.shape[1] != global_metric[-1].shape[1]:
                continue
            global_metric.append(_diff)
    """
    # """
    n_tokens = None
    # Sort outer keys (layer names) numerically by block index
    for outer_key in sorted(_diff_dict.keys(), key=_layer_sort_key):
        outer_value = _diff_dict[outer_key]
        # Sort inner keys (step indices) numerically
        for inner_key in sorted(outer_value.keys(), key=_step_sort_key):
            _diff = outer_value[inner_key]
            if n_tokens is None:
                n_tokens = _diff.shape[1]
            if _diff.shape[1] != n_tokens:
                continue
            global_metric.append(_diff)
    # """
    return torch.cat(global_metric, dim=0)


def compare_metric(_metric_A, _metric_B, name_A="cos", name_B="l2"):
    """Compare two skip-cost metrics by their global rankings.

    _metric_A and _metric_B share the same shape: [n, d]. Since L2 and cosine
    have different scales, we compare *rankings* rather than raw values:
      1. Flatten both to 1D and compute global ranks.
      2. Spearman rank correlation (overall rank agreement).
      3. Top-K overlap: what fraction of the top-K most-skippable tokens are
         shared between the two metrics, for several K values.

    Returns a dict of comparison statistics.
    """
    assert _metric_A.shape == _metric_B.shape, (
        f"Shape mismatch: {_metric_A.shape} vs {_metric_B.shape}"
    )

    flat_A = _metric_A.flatten().float()
    flat_B = _metric_B.flatten().float()
    n_total = flat_A.shape[0]

    # NOTE: Both cos and L2 here are *distance* metrics (lower = more stable
    # = more skippable). So we rank ascending: rank 1 = lowest distance.
    # Compute ranks (lower value = more skippable = rank 1)
    rank_A = torch.argsort(torch.argsort(flat_A)).float()
    rank_B = torch.argsort(torch.argsort(flat_B)).float()

    # --- Spearman rank correlation ---
    # Pearson correlation of the ranks equals Spearman correlation
    rank_A_c = rank_A - rank_A.mean()
    rank_B_c = rank_B - rank_B.mean()
    spearman = (rank_A_c * rank_B_c).sum() / (
        torch.sqrt((rank_A_c ** 2).sum()) * torch.sqrt((rank_B_c ** 2).sum())
    )

    # --- Top-K overlap for several K values ---
    # Top-K = K tokens with LOWEST distance = most skippable
    k_ratios = [0.05, 0.1, 0.2, 0.3, 0.5]
    overlaps = {}
    print(f"\n{'='*60}")
    print(f"  Compare: {name_A} vs {name_B}")
    print(f"{'='*60}")
    print(f"  Total tokens: {n_total}")
    print(f"  Spearman rank correlation: {spearman.item():.4f}")
    print(f"\n  {'Top-K':>12} | {'Overlap':>12} | {'Overlap Ratio':>14}")
    print(f"  {'-'*12} | {'-'*12} | {'-'*14}")

    for ratio in k_ratios:
        k = max(1, int(n_total * ratio))
        # Lowest distance = most skippable => use largest=False (ascending)
        top_A = set(torch.topk(flat_A, k, largest=False).indices.tolist())
        top_B = set(torch.topk(flat_B, k, largest=False).indices.tolist())
        overlap = len(top_A & top_B)
        overlap_ratio = overlap / k
        overlaps[ratio] = {
            "k": k, "overlap": overlap, "overlap_ratio": overlap_ratio
        }
        print(f"  {k:>12} | {overlap:>12} | {overlap_ratio:>14.4f}")

    # --- Quantitative stats of raw value agreement ---
    # Normalize both to [0, 1] to compare distributions
    A_norm = (flat_A - flat_A.min()) / (flat_A.max() - flat_A.min() + 1e-8)
    B_norm = (flat_B - flat_B.min()) / (flat_B.max() - flat_B.min() + 1e-8)
    mae = (A_norm - B_norm).abs().mean().item()
    corr = torch.corrcoef(torch.stack([flat_A, flat_B]))[0, 1].item()

    print(f"\n  Normalized MAE: {mae:.4f}")
    print(f"  Pearson correlation (raw): {corr:.4f}")

    return {
        "spearman": spearman.item(),
        "topk_overlap": overlaps,
        "normalized_mae": mae,
        "pearson": corr,
    }
            

computation_diff_root = "G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K"
cos_path = "full-steps-cos"
l2_path = "full-steps-L2"

cos_metric, _ = samples_to_avg(load_samples(computation_diff_root, postfix=cos_path), orientation="layer", concate_img_txt=True)
l2_metric, _ = samples_to_avg(load_samples(computation_diff_root, postfix=l2_path), orientation="layer", concate_img_txt=True)

cos_metric = arrange_to_global_metric(cos_metric)
l2_metric = arrange_to_global_metric(l2_metric)

# Verify alignment
assert cos_metric.shape == l2_metric.shape, (
    f"Shape mismatch after arrange: {cos_metric.shape} vs {l2_metric.shape}"
)
print(f"[align] cos_metric shape: {cos_metric.shape}")
print(f"[align] l2_metric shape: {l2_metric.shape}")
# Spot-check: both should be distance metrics in [0, 1] range roughly
print(f"[align] cos value range: [{cos_metric.min():.4f}, {cos_metric.max():.4f}]")
print(f"[align] l2 value range: [{l2_metric.min():.4f}, {l2_metric.max():.4f}]")

compare_metric(cos_metric, cos_metric, name_A="cos", name_B="cos (self)")
compare_metric(cos_metric, l2_metric, name_A="cos", name_B="l2")