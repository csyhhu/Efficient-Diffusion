"""
Analyze step-wise difference for DiT outputs.

Three core intermediate values (uniformly recorded for all schedulers):
  - dit_outputs:       raw DiT forward output (before scheduler remap)
  - noise_preds:       post-remap noise / velocity prediction (after SCM remap)
  - scheduler_outputs: new latent produced by scheduler.step() at each step

Modules:
  A: Pairwise similarity matrix (relative L2 distance) — multi-sample mean + single-sample
  B: Step-wise L2 norm curve — multi-sample mean with std (Module E: std shows cross-sample consistency)
  C: Accumulation error (||N*output_i - sum(output_j)|| / ||sum(output_j)||) — multi-sample mean + single-sample

Analysis is performed on each of the three core intermediate value fields above.

Usage:
  python -m analysis.step_wise_difference.analyze_step_wise_difference `
    --input_dir G:\Outputs\Efficient-Diffusion\step_wise_output\SD3-origin-MJHQ30K
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Data loading
# ============================================================================

def load_samples(input_dir):
    """Load all sample_*.pt files from input_dir, sorted by index."""
    sample_files = sorted(glob.glob(os.path.join(input_dir, "sample_*.pt")))
    if not sample_files:
        raise FileNotFoundError(f"No sample_*.pt files found in {input_dir}")
    samples = []
    for f in sample_files:
        samples.append(torch.load(f, map_location="cpu", weights_only=False))
    print(f"[load] {len(samples)} samples from {input_dir}")
    return samples


def extract_outputs(samples, key):
    """Extract per-step output lists from all samples.

    Returns:
        list of list of tensors: [n_samples][n_steps],
        or None if the key does not exist or its first entry is None.
    """
    all_outputs = []
    for s in samples:
        if key not in s:
            return None
        steps = s[key]
        if steps and steps[0] is None:
            return None
        all_outputs.append([t.clone() for t in steps])
    return all_outputs


# ============================================================================
# Metric computation
# ============================================================================

def compute_pairwise_rel_l2(outputs):
    """Compute NxN pairwise relative L2 distance matrix.

    rel_l2(i, j) = ||o_i - o_j|| / ||o_i||
    """
    N = len(outputs)
    flat = torch.stack([o.float().flatten() for o in outputs])  # [N, D]
    norms = flat.norm(dim=1)  # [N]
    dot = flat @ flat.T  # [N, N]
    dist_sq = norms.unsqueeze(0) ** 2 + norms.unsqueeze(1) ** 2 - 2 * dot
    dist = dist_sq.clamp(min=0).sqrt()
    rel_dist = dist / (norms.unsqueeze(1) + 1e-8)
    return rel_dist.numpy()


def compute_step_norms(outputs):
    """Compute L2 norm for each step's output."""
    return np.array([o.float().norm().item() for o in outputs])


def compute_accumulation_errors(outputs):
    """For each step i, compute relative error of using output_i for all N steps.

    error_i = ||N * o_i - sum(o_j)|| / ||sum(o_j)||
    """
    N = len(outputs)
    total_sum = sum(outputs).float()
    total_norm = total_sum.norm().item()
    errors = []
    for i in range(N):
        approx = N * outputs[i].float()
        diff = (approx - total_sum).norm().item()
        errors.append(diff / (total_norm + 1e-8))
    return np.array(errors)


# ============================================================================
# Plot functions
# ============================================================================

def plot_heatmap(matrix, title, save_path, vmin=None, vmax=None, annotate=False):
    """Plot a heatmap of a 2D matrix."""
    N = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f'{i}' for i in range(N)])
    ax.set_yticklabels([f'{i}' for i in range(N)])
    ax.set_title(title)
    if annotate:
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                        fontsize=8, color='black' if matrix[i, j] < 0.5 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] saved to {save_path}")


def plot_step_norms(mean_vals, std_vals, title, save_path):
    """Plot step-wise L2 norm with mean line and std error band."""
    N = len(mean_vals)
    steps = np.arange(N)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, mean_vals, 'b-o', label='Mean', markersize=8)
    ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                    alpha=0.3, color='blue', label='±1 std')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 Norm')
    ax.set_title(title)
    ax.set_xticks(steps)
    ax.set_xticklabels([f'Step {i}' for i in steps])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] saved to {save_path}")


def plot_accumulation_error(mean_vals, std_vals, title, save_path, single_vals=None):
    """Plot accumulation error as bar chart with error bars.

    Args:
        mean_vals: array of length N (mean error per step)
        std_vals: array of length N (std error per step)
        single_vals: optional array of length N (single-sample error for overlay)
    """
    N = len(mean_vals)
    steps = np.arange(N)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(steps, mean_vals, yerr=std_vals, capsize=5, color='steelblue',
                  alpha=0.8, label='Mean ± std')
    best_idx = np.argmin(mean_vals)
    bars[best_idx].set_color('green')
    ax.set_xlabel('Step (which output is used for accumulation)')
    ax.set_ylabel('Relative Error')
    ax.set_title(title)
    ax.set_xticks(steps)
    ax.set_xticklabels([f'Step {i}' for i in steps])
    if single_vals is not None:
        ax.plot(steps, single_vals, 'r--o', label='Single sample', markersize=6)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] saved to {save_path}")


# ============================================================================
# DP for skip-plan selection
# ============================================================================

def _dp_select_steps(matrix, M):
    """Select M additional steps (beyond the mandatory step 0) to minimize skip cost.

    Step 0 is always selected (first DiT run, fixed). From the remaining
    N-1 steps (1..N-1), select M additional steps. Total selected = M+1.

    Constraint: M < N-1 (at least one step is always skipped).

    Cost model (accounts for ALL skipped steps):
      - No prefix: step 0 is always run, nothing before it.
      - Between:  steps j+1..i-1 reuse j's output  -> sum(matrix[j, j+1:i])
      - Suffix:   steps after last selected reuse it -> sum(matrix[last, last+1:])

    DP:
      dp[1][0] = 0  (step 0 fixed, no prefix cost)
      dp[m][i] = min_{0 <= j < i} ( dp[m-1][j] + sum(matrix[j, j+1:i]) )
      answer:   min_i ( dp[M+1][i] + sum(matrix[i, i+1:]) )

    Args:
        matrix: NxN pairwise rel_l2 mean matrix (noise_preds)
        M: number of additional steps to select beyond step 0 (1 <= M <= N-2)

    Returns:
        (list of step indices including step 0, total DP cost)
    """
    N = matrix.shape[0]
    if M < 1 or M > N - 2:
        raise ValueError(f"M must be in [1, {N - 2}], got {M}")

    INF = float("inf")
    total_steps = M + 1  # including the fixed step 0

    dp = [[INF] * N for _ in range(total_steps + 1)]
    prev = [[-1] * N for _ in range(total_steps + 1)]

    # Base case: step 0 is always the first selected, no prefix cost
    dp[1][0] = 0.0

    for m in range(2, total_steps + 1):
        for i in range(m - 1, N):  # need at least m-1 steps before i
            for j in range(m - 2, i):  # previous run step
                # cost(j, i): skip steps j+1..i-1, reuse step j's output
                cost = float(matrix[j, j + 1:i].sum())
                cand = dp[m - 1][j] + cost
                if cand < dp[m][i]:
                    dp[m][i] = cand
                    prev[m][i] = j

    # Find best last step, adding suffix cost
    best_i = -1
    best_total = INF
    for i in range(total_steps - 1, N):
        suffix_cost = float(matrix[i, i + 1:].sum())
        total = dp[total_steps][i] + suffix_cost
        if total < best_total:
            best_total = total
            best_i = i

    # Backtrack
    steps = []
    i = best_i
    m = total_steps
    while m >= 1:
        steps.append(i)
        j = prev[m][i]
        m -= 1
        i = j
    steps.reverse()

    return steps, best_total


# ============================================================================
# Main analysis
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze step-wise difference")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing sample_*.pt files from save_step_wise_output.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save analysis plots. Defaults to input_dir/analysis")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Specific sample index for single-sample plots. If None, only multi-sample mean plots.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load samples ----
    samples = load_samples(args.input_dir)
    n_samples = len(samples)
    num_steps = samples[0]["num_steps"]

    # Load metadata if available
    metadata_path = os.path.join(args.input_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    scheduler_type = metadata.get("scheduler_type", "unknown")
    print(f"[info] scheduler={scheduler_type}, num_steps={num_steps}, n_samples={n_samples}")

    # Three core fields (uniform across schedulers)
    data_fields = [
        ("dit_outputs", "DiT raw output (pre-scheduler)"),
        ("noise_preds", "Noise/velocity pred (post-scheduler-remap)"),
        ("scheduler_outputs", "Latent after scheduler.step()"),
    ]

    summary = {
        "n_samples": n_samples,
        "num_steps": num_steps,
        "scheduler_type": scheduler_type,
        "sample_idx": args.sample_idx,
    }

    # ---- Analyze each of the three core fields ----
    for data_key, data_label in data_fields:

        print(f"\n{'='*60}")
        print(f"  Analyzing: {data_key} — {data_label}")
        print(f"{'='*60}")

        all_outputs = extract_outputs(samples, data_key)
        if all_outputs is None:
            print(f"  [skip] {data_key} missing or None in samples")
            continue

        # ---- Module A: Pairwise similarity matrix ----
        print(f"\n[Module A] Pairwise relative L2 distance matrix")
        pairwise_matrices = [compute_pairwise_rel_l2(o) for o in all_outputs]
        mean_matrix = np.mean(pairwise_matrices, axis=0)
        std_matrix = np.std(pairwise_matrices, axis=0)
        
        # plot_heatmap(
        #     mean_matrix,
        #     f"Pairwise Rel L2 (Mean over {n_samples})\n[{data_key}]",
        #     os.path.join(output_dir, f"pairwise_{data_key}_mean.png"),
        #     annotate=False
        # )
        # # Std heatmap = cross-sample consistency
        # plot_heatmap(
        #     std_matrix,
        #     f"Pairwise Rel L2 (Std over {n_samples})\n[{data_key}]",
        #     os.path.join(output_dir, f"pairwise_{data_key}_std.png"),
        #     annotate=False
        # )
        # if args.sample_idx is not None and args.sample_idx < n_samples:
        #     plot_heatmap(
        #         pairwise_matrices[args.sample_idx],
        #         f"Pairwise Rel L2 (Sample {args.sample_idx})\n[{data_key}]",
        #         os.path.join(output_dir, f"pairwise_{data_key}_sample{args.sample_idx}.png"),
        #     )

        """
        # ---- Module B: Step-wise L2 norm ----
        print(f"\n[Module B] Step-wise L2 norm")
        step_norms = np.array([compute_step_norms(o) for o in all_outputs])
        norm_mean = step_norms.mean(axis=0)
        norm_std = step_norms.std(axis=0)

        plot_step_norms(
            norm_mean, norm_std,
            f"Step-wise L2 Norm (Mean ± Std, {n_samples})\n[{data_key}]",
            os.path.join(output_dir, f"step_norms_{data_key}.png"),
        )

        # ---- Module C: Accumulation error ----
        
        print(f"\n[Module C] Accumulation error (single-step vs multi-step)")
        acc_errors = np.array([compute_accumulation_errors(o) for o in all_outputs])
        acc_mean = acc_errors.mean(axis=0)
        acc_std = acc_errors.std(axis=0)
        best_step = int(np.argmin(acc_mean))

        single_vals = acc_errors[args.sample_idx] if (args.sample_idx is not None and args.sample_idx < n_samples) else None
        plot_accumulation_error(
            acc_mean, acc_std,
            f"Accumulation Error (Mean ± Std, {n_samples})\n"
            f"[{data_key}] Best step={best_step} (err={acc_mean[best_step]:.4f})",
            os.path.join(output_dir, f"accumulation_{data_key}_mean.png"),
            single_vals=single_vals,
        )
        if args.sample_idx is not None and args.sample_idx < n_samples:
            single_best = int(np.argmin(acc_errors[args.sample_idx]))
            plot_accumulation_error(
                acc_errors[args.sample_idx], np.zeros(num_steps),
                f"Accumulation Error (Sample {args.sample_idx})\n"
                f"[{data_key}] Best step={single_best}",
                os.path.join(output_dir, f"accumulation_{data_key}_sample{args.sample_idx}.png"),
            )
        """
        # ---- Save summary metrics ----
        summary[data_key] = {
            "label": data_label,
            "pairwise_rel_l2_mean": mean_matrix.tolist(),
            "pairwise_rel_l2_std": std_matrix.tolist(),
            # "step_norms_mean": norm_mean.tolist(),
            # "step_norms_std": norm_std.tolist(),
            # "accumulation_error_mean": acc_mean.tolist(),
            # "accumulation_error_std": acc_std.tolist(),
            # "best_step": best_step,
            # "best_step_error": float(acc_mean[best_step]),
        }

    # ---- Save summary.json ----
    """
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] saved to {summary_path}")

    print(f"\n[done] all analysis plots saved to {output_dir}")
    """

    # ---- Recommended skip plans (DP on noise_preds) ----
    noise_matrix = np.array(summary["noise_preds"]["pairwise_rel_l2_mean"])
    recommended = {}
    # M = additional steps beyond step 0; M ranges 1..N-2 (M < N-1)
    for M in range(1, num_steps - 1):
        steps, dp_cost = _dp_select_steps(noise_matrix, M)
        recommended[f"M{M}"] = {
            "dit_inference_steps": steps,
            "n_dit_steps": len(steps),
            "dp_cost": float(dp_cost),
        }

    plans_path = os.path.join(output_dir, "recommended_skip_plans.json")
    with open(plans_path, "w") as f:
        json.dump({
            "source": "noise_preds.pairwise_rel_l2_mean",
            "num_steps": num_steps,
            "plans": recommended,
        }, f, indent=2)
    print(f"\n[skip-plan] recommended plans saved to {plans_path}")
