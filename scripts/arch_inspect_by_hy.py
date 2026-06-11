"""
UNet architecture inspector: analyze parameter distribution when slicing
along dim=0 (out_channels) for per-channel quantization.

Usage:
    python scripts/arch_inspect.py
    python scripts/arch_inspect.py --load_weights   # also show weight histograms
    python scripts/arch_inspect.py --model sdxl
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_loader import load_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _layer_tag(param_name: str) -> str:
    """Return a compact layer-type tag from a parameter name."""
    p = param_name.lower()
    if "conv1" in p or "conv2" in p or "conv_shortcut" in p or ".conv." in p or p.endswith(".conv.weight"):
        return "Conv"
    if "proj" in p or "to_q" in p or "to_k" in p or "to_v" in p or "to_out" in p or "linear" in p:
        return "Linear"
    if "norm" in p:
        return "GroupNorm"
    return ""


def extract_module_name(param_name: str) -> str:
    """Return a concise module path for grouping.

    Format: "block.submodule"  (e.g. "down_blocks.0.resnets.0")
    Falls back to "block" if no submodule, or the bare prefix for top-level params.
    """
    # Identify the block prefix
    m = re.match(r"(down_blocks|up_blocks)\.(\d+)", param_name)
    if m:
        block = f"{m.group(1)}.{m.group(2)}"
        # Find deepest submodule inside this block
        sub = re.search(
            r"(resnets\.\d+|attentions\.\d+|downsamplers\.\d+|"
            r"upsamplers\.\d+|transformer_blocks\.\d+|ff\.net\.\d+)",
            param_name,
        )
        if sub:
            return f"{block}.{sub.group(1)}"
        return block

    if param_name.startswith("mid_block"):
        sub = re.search(r"(attentions\.\d+|resnets\.\d+)", param_name)
        if sub:
            return f"mid_block.{sub.group(1)}"
        return "mid_block"

    # top-level params: conv_in, time_embedding, conv_out, etc.
    return param_name.split(".")[0]


def per_slice_stats(values: np.ndarray, dim_idx: int = 0):
    """For a 1D or 2D tensor, split along dim_idx.

    Returns (slice_vals, stds) where:
        slice_vals: list of 1D arrays, one per slice
        stds: std of each slice (within-slice variance)
    """
    if values.ndim == 1:
        return [values], [float(np.std(values))]
    if dim_idx == 0:
        slices = [values[i] for i in range(values.shape[0])]
    else:
        slices = [values[:, i] for i in range(values.shape[1])]
    stds = [float(np.std(s)) for s in slices]
    return slices, stds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UNet architecture inspector")
    parser.add_argument("--model", type=str, default="sd", choices=["sd", "sdxl"])
    parser.add_argument("--mirror", type=str, default="https://hf-mirror.com")
    parser.add_argument("--load_weights", action="store_true", default=False,
                        help="Load actual weights for histogram analysis (slow, ~3.4GB RAM)")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Load UNet
    # -------------------------------------------------------------------
    if args.load_weights:
        print("[0] Loading pipeline with weights (may take a while) ...")
        pipe, device = load_model(model_name=args.model, mirror=args.mirror)
        unet = pipe.unet
        print(f"    Model loaded (device={device})\n")
    else:
        print("[0] Building UNet from config (no weights, fast) ...")
        from diffusers import UNet2DConditionModel

        model_id = "runwayml/stable-diffusion-v1-5" if args.model == "sd" else "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(model_id, subfolder="unet", local_files_only=True)
        )
        print(f"    UNet built from config (type={type(unet).__name__})\n")

    # ===================================================================
    # 1. Top-level module overview
    # ===================================================================
    print("=" * 80)
    print("1. Top-level UNet modules")
    print("=" * 80)
    total_params = 0
    for name, module in unet.named_children():
        n = sum(p.numel() for p in module.parameters())
        total_params += n
        print(f"  {name:<20s} {type(module).__name__:<30s} params={n:>12,}")
    print(f"  {'─' * 60}")
    print(f"  {'Total':<20s} {'':<30s} params={total_params:>12,}")

    # ===================================================================
    # 2. All parameters: name, shape, numel
    # ===================================================================
    print()
    print("=" * 80)
    print("2. All parameters (named_parameters)")
    print("=" * 80)
    print(f"  {'Parameter':<68s} {'Shape':<22s} {'Params':>12s}")
    print(f"  {'─' * 102}")

    ndim_ge_2_params = 0
    ndim_1_params = 0

    for name, param in unet.named_parameters():
        shape = tuple(param.shape)
        n = param.numel()
        if param.ndim >= 2:
            ndim_ge_2_params += n
        else:
            ndim_1_params += n
        print(f"  {name:<68s} {str(shape):<22s} {n:>12,}")

    print(f"  {'─' * 102}")
    print(f"  {'Total':<68s} {'':<22s} {total_params:>12,}")
    print(f"  {'  ndim >= 2 (quantizable)':<68s} {'':<22s} {ndim_ge_2_params:>12,}  ({ndim_ge_2_params/total_params*100:.1f}%)")
    print(f"  {'  ndim == 1 (skipped)':<68s} {'':<22s} {ndim_1_params:>12,}  ({ndim_1_params/total_params*100:.1f}%)")

    # ===================================================================
    # 3. dim=0 slicing analysis with module names.
    #
    #    For each unique out_channel size s:
    #      - collect all tensors with shape[0] == s
    #      - attach module names to each group
    #      - compute std of param counts within the group
    #      - #scales = s * n_tensors
    # ===================================================================
    print()
    print("=" * 80)
    print("3. dim=0 (per-out-channel) slicing analysis")
    print("=" * 80)

    dim0_groups = defaultdict(list)   # out_ch -> [(name, numel, module), ...]
    for name, param in unet.named_parameters():
        if param.ndim <= 0:
            continue
        sz = param.shape[0]
        module = extract_module_name(name)
        layer_tag = _layer_tag(name)
        dim0_groups[sz].append((name, param.numel(), module, layer_tag))

    group_stats = []
    for sz, items in dim0_groups.items():
        n_tensors = len(items)
        param_counts = np.array([n for _, n, _, _ in items], dtype=np.float64)
        # deduplicate: keep unique (module, layer_tag) pairs, order preserved
        seen = set()
        modules = []
        for _, _, m, t in items:
            key = (m, t)
            if key not in seen:
                seen.add(key)
                modules.append(f"{m}/{t}" if t else m)
        mean_p = float(np.mean(param_counts))
        std_p = float(np.std(param_counts)) if n_tensors >= 2 else 0.0
        total_p = int(np.sum(param_counts))
        n_scales = sz * n_tensors
        group_stats.append({
            "out_ch": sz,
            "n_tensors": n_tensors,
            "total_params": total_p,
            "mean": mean_p,
            "std": std_p,
            "n_scales": n_scales,
            "param_counts": param_counts,
            "names": [nm for nm, _, _, _ in items],
            "modules": modules,
        })

    group_stats.sort(key=lambda x: x["out_ch"])

    all_stds = [g["std"] for g in group_stats if g["n_tensors"] >= 2]
    total_scales = sum(g["n_scales"] for g in group_stats)
    avg_std = float(np.mean(all_stds)) if all_stds else 0.0
    min_std = min(all_stds) if all_stds else 0.0
    max_std = max(all_stds) if all_stds else 0.0

    print(f"\n  Total groups (unique out_ch sizes): {len(group_stats)}")
    print(f"  Total quantization scales needed  : {total_scales:,}")
    print(f"  Avg within-group std              : {avg_std:,.2f}")
    print(f"  Min within-group std              : {min_std:,.2f}")
    print(f"  Max within-group std              : {max_std:,.2f}")
    print()
    # print(f"  {'out_ch':<10s} {'#tensors':<10s} {'total params':<16s} {'avg/tensor':<14s} {'std':<14s} {'#scales':<10s}  modules")
    print(f"  {'out_ch':<10s} {'#tensors':<10s} {'total params':<16s} {'avg/tensor':<14s} {'std':<14s} {'#scales':<10s}")
    print(f"  {'─' * 100}")
    for g in group_stats:
        mods = ", ".join(g["modules"])
        # print(f"  {g['out_ch']:<10d} {g['n_tensors']:<10d} {g['total_params']:>14,}  {g['mean']:>12,.2f}  {g['std']:>12,.2f}  {g['n_scales']:>8,}    {mods}")
        print(f"  {g['out_ch']:<10d} {g['n_tensors']:<10d} {g['total_params']:>14,}  {g['mean']:>12,.2f}  {g['std']:>12,.2f}  {g['n_scales']:>8,}")

    # ===================================================================
    # 4. Weight histogram analysis for large tensors.
    #
    #    Part 1: single largest tensor
    #      - overall histogram
    #      - after dim=0 slicing: min-std slice vs max-std slice
    #
    #    Part 2: top-N large tensors together, dim=0 sliced
    #      - min-std group vs max-std group
    # ===================================================================
    print()
    print("=" * 80)
    print("4. Large-tensor weight histogram analysis (requires --load_weights)")
    print("=" * 80)

    if not args.load_weights:
        print("  Skipped (run with --load_weights to enable)")
    else:
        # Collect all 2D+ tensors sorted by numel
        all_tensors = [
            (name, param.detach().cpu().numpy())
            for name, param in unet.named_parameters()
            if param.ndim >= 2
        ]
        all_tensors.sort(key=lambda x: x[1].size, reverse=True)

        largest_name, largest_vals = all_tensors[0]
        top_n = 20
        top_tensors = all_tensors[:top_n]  # top-N by param count

        # ── Part 1: single largest tensor ──────────────────────────────────
        print(f"\n  Part 1: Single largest tensor")
        print(f"  {largest_name}")
        print(f"  shape={largest_vals.shape}, numel={largest_vals.size:,}")

        # overall
        slices_all, stds_all = per_slice_stats(largest_vals, dim_idx=0)
        # group by out_ch size
        dim0_slices_by_outch = defaultdict(list)
        for name, vals in [(largest_name, largest_vals)]:
            if vals.ndim < 2:
                continue
            out_ch = vals.shape[0]
            sl, st = per_slice_stats(vals, dim_idx=0)
            dim0_slices_by_outch[out_ch].append((sl, st))

        # flatten per out_ch
        all_slice_vals = []
        all_slice_stds = []
        for out_ch, records in dim0_slices_by_outch.items():
            for sl, st in records:
                all_slice_vals.extend(sl)
                all_slice_stds.extend(st)

        # std_by_outch maps out_ch -> (list of stds across tensors)
        std_by_outch = defaultdict(list)
        for out_ch, records in dim0_slices_by_outch.items():
            for sl, st in records:
                std_by_outch[out_ch].extend(st)

        # pick best/worst out_ch group (by avg std within that group)
        if std_by_outch:
            best_outch = min(std_by_outch, key=lambda oc: np.mean(std_by_outch[oc]))
            worst_outch = max(std_by_outch, key=lambda oc: np.mean(std_by_outch[oc]))
        else:
            best_outch = worst_outch = None

        print(f"  dim=0: {len(dim0_slices_by_outch)} unique out_ch sizes, "
              f"best group out_ch={best_outch} (avg std={np.mean(std_by_outch.get(best_outch, [0])):,.4f}), "
              f"worst group out_ch={worst_outch} (avg std={np.mean(std_by_outch.get(worst_outch, [0])):,.4f})")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # overall histogram
        axes[0].hist(largest_vals.ravel(), bins=80, color="gray", edgecolor="none", alpha=0.8)
        axes[0].set_title(f"Overall (all {largest_vals.size:,} values)")
        axes[0].set_xlabel("weight value")
        axes[0].set_ylabel("count")

        # min-std slice
        if best_outch is not None:
            best_slices = dim0_slices_by_outch[best_outch][0][0]
            vals_best = np.concatenate([s.ravel() for s in best_slices])
            axes[1].hist(vals_best, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
            axes[1].set_title(f"Min-std slice: out_ch={best_outch}\n(n={len(best_slices)}, mean std={np.mean([np.std(s) for s in best_slices]):.4f})")
            axes[1].set_xlabel("weight value")
            axes[1].set_ylabel("count")

        # max-std slice
        if worst_outch is not None:
            worst_slices = dim0_slices_by_outch[worst_outch][0][0]
            vals_worst = np.concatenate([s.ravel() for s in worst_slices])
            axes[2].hist(vals_worst, bins=80, color="coral", edgecolor="none", alpha=0.8)
            axes[2].set_title(f"Max-std slice: out_ch={worst_outch}\n(n={len(worst_slices)}, mean std={np.mean([np.std(s) for s in worst_slices]):.4f})")
            axes[2].set_xlabel("weight value")
            axes[2].set_ylabel("count")

        plt.suptitle(f"Part 1: {largest_name}  shape={largest_vals.shape}", fontsize=11)
        plt.tight_layout()
        out_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "outputs", "part1_single_largest_hist.png")
        os.makedirs(os.path.dirname(out_path1), exist_ok=True)
        plt.savefig(out_path1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Part 1 saved: {os.path.abspath(out_path1)}")

        # ── Part 2: top-N tensors, dim=0 sliced ───────────────────────────
        print(f"\n  Part 2: Top-{top_n} largest tensors, dim=0 sliced")

        # Build dim0 groups across all top tensors
        top_dim0_groups = defaultdict(list)  # out_ch -> list of (name, vals, slice_stds)
        for name, vals in top_tensors:
            if vals.ndim < 2:
                continue
            out_ch = vals.shape[0]
            sl, st = per_slice_stats(vals, dim_idx=0)
            top_dim0_groups[out_ch].append((name, vals, sl, st))

        # Per-group std
        group_stds_part2 = {}
        for oc, records in top_dim0_groups.items():
            all_st = []
            for _, _, _, st in records:
                all_st.extend(st)
            group_stds_part2[oc] = all_st

        if group_stds_part2:
            best_oc2 = min(group_stds_part2, key=lambda oc: np.mean(group_stds_part2[oc]))
            worst_oc2 = max(group_stds_part2, key=lambda oc: np.mean(group_stds_part2[oc]))
        else:
            best_oc2 = worst_oc2 = None

        print(f"  dim=0: {len(top_dim0_groups)} unique out_ch sizes, "
              f"best group out_ch={best_oc2} (avg std={np.mean(group_stds_part2.get(best_oc2, [0])):,.4f}), "
              f"worst group out_ch={worst_oc2} (avg std={np.mean(group_stds_part2.get(worst_oc2, [0])):,.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if best_oc2 is not None:
            best_vals = []
            for _, _, sl, st in top_dim0_groups[best_oc2]:
                best_vals.extend([s.ravel() for s in sl])
            best_combined = np.concatenate(best_vals)
            axes[0].hist(best_combined, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
            axes[0].set_title(
                f"Min-std group: out_ch={best_oc2}\n"
                f"(n_tensors={len(top_dim0_groups[best_oc2])}, slices={len(best_combined):,}, avg std={np.mean(group_stds_part2[best_oc2]):.4f})"
            )
            axes[0].set_xlabel("weight value")
            axes[0].set_ylabel("count")

        if worst_oc2 is not None:
            worst_vals = []
            for _, _, sl, st in top_dim0_groups[worst_oc2]:
                worst_vals.extend([s.ravel() for s in sl])
            worst_combined = np.concatenate(worst_vals)
            axes[1].hist(worst_combined, bins=80, color="coral", edgecolor="none", alpha=0.8)
            axes[1].set_title(
                f"Max-std group: out_ch={worst_oc2}\n"
                f"(n_tensors={len(top_dim0_groups[worst_oc2])}, slices={len(worst_combined):,}, avg std={np.mean(group_stds_part2[worst_oc2]):.4f})"
            )
            axes[1].set_xlabel("weight value")
            axes[1].set_ylabel("count")

        plt.suptitle(f"Part 2: Top-{top_n} tensors, dim=0 sliced", fontsize=11)
        plt.tight_layout()
        out_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "outputs", "part2_topn_dim0_hist.png")
        plt.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Part 2 saved: {os.path.abspath(out_path2)}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
