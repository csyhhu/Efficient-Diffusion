"""
python -m analysis.computation_diff.save_analyze_computation_diff
run analysis/computation_diff/save_analyze_computation_diff.py

python -m analysis.computation_diff.save_analyze_computation_diff `
    --dit_inference_steps "0,1,2,4,10,20,27"
python -m analysis.computation_diff.save_analyze_computation_diff `
    --dit_inference_steps "0,1,2,4,8,14,22,27"
run analysis/computation_diff/save_analyze_computation_diff.py --dit_inference_steps "0,1,2,4,10,20,27"
run analysis/computation_diff/save_analyze_computation_diff.py --dit_inference_steps "0,1,2,4,8,14,22,27"
"""

import argparse
import json
import os
import glob
import numpy as np

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.image_generator import SanaImageGenerator, SD3ImageGenerator
from src.utils import compute_computation_diff

from analysis.computation_diff.utils import load_samples, samples_to_avg, visualize_computation_diff


def visualize_token_diff_distribution(diff_dict, save_path):
    """
    Visualize the distribution of computation diff.
    Global: visualize the distribution of all tokens across different layers and steps.
    Layer: visualize the distribution of tokens in each outer key.
    """
    all_token_diff = []
    layer_token_diff = {}
    for outer_key, inner_dict in diff_dict.items():
        layer_token_diff[outer_key] = []
        for inner_key, diff in inner_dict.items():
            diff_flat = diff.flatten().float().numpy()
            layer_token_diff[outer_key].append(diff_flat)
            all_token_diff.append(diff_flat)

    # Flatten per-layer lists into single arrays
    all_token_diff = np.concatenate(all_token_diff) if all_token_diff else np.array([])
    for k in layer_token_diff:
        layer_token_diff[k] = np.concatenate(layer_token_diff[k])

    # Layout: global spans the entire first row; per-group histograms start at row 1
    group_names = list(layer_token_diff.keys())
    group_type = "step" if group_names and group_names[0].startswith("step.") else "layer"
    n_groups = len(group_names)
    n_cols = min(5, max(1, n_groups))
    n_rows = 1 + (n_groups + n_cols - 1) // n_cols  # 1 row for global + ceil(N/n_cols) rows for groups

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.0))

    # Shared x-axis range for comparability
    global_min = float(all_token_diff.min()) if len(all_token_diff) else 0
    global_max = float(all_token_diff.max()) if len(all_token_diff) else 1
    percentiles = list(range(10, 91, 10))  # 10, 20, ..., 90

    def _draw_hist(ax, data, title):
        ax.hist(data, bins=100, color='steelblue', edgecolor='black', alpha=0.7,
                range=(global_min, global_max))
        # Draw 10-90 percentile lines; median (50th) stands out
        if len(data) > 0:
            pcts = np.percentile(data, percentiles)
            for p, v in zip(percentiles, pcts):
                if p == 50:
                    ax.axvline(v, color='red', linestyle='-', linewidth=1.2, alpha=0.8,
                               label=f'p50={v:.3f}')
                else:
                    ax.axvline(v, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Global histogram spans the entire first row
    ax_global = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols)
    _draw_hist(ax_global, all_token_diff, f"Global token diff distribution (n={len(all_token_diff)})")
    ax_global.legend(fontsize=7, loc='upper right')

    # Per-group histograms starting at row 1
    for idx, name in enumerate(group_names):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = plt.subplot2grid((n_rows, n_cols), (row, col))
        _draw_hist(ax, layer_token_diff[name], f"{group_type} {name} (n={len(layer_token_diff[name])})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[visualize] saved token diff distribution to {save_path}")


def generate_skip_plan(diff_dict, method="global", skip_ratio=0.5):
    """
    Generate token skip plan given computation dict values matrix.
    diff_dict: {layer_name: {step_idx: [n_img + n_txt]}} or {step_idx: {layer_name: [n_img + n_txt]}}
    Token skip plan: {step_idx: {layer_name: []}}
    global: Rank all tokens across different layers and steps, skip the tokens
        with the SMALLEST computation diff (most stable = most skippable).

    NOTE: computation diff values are *distance* metrics (lower = more stable
    = more skippable), NOT similarities. So we sort ascending and take the
    lowest-K values.
    """
    if method == "global":
        # Flatten all token diffs into one big tensor, while tracking the
        # (layer_name, step_idx, token_idx) location of each value via
        # offset bookkeeping.  This avoids building millions of Python tuples.
        all_values = []          # list of 1D tensors
        token_locations = []      # list of (layer_name, step_idx, n_tokens)
        total_count = 0
        for layer_name, step_dict in diff_dict.items():
            for step_idx, diff_tensor in step_dict.items():
                diff_flat = diff_tensor.flatten().float()
                n_tokens = diff_flat.shape[0]
                total_count += n_tokens
                all_values.append(diff_flat)
                token_locations.append((layer_name, step_idx, n_tokens))

        all_values = torch.cat(all_values, dim=0)  # [total_count]

        # Skip budget = skip_ratio * total tokens
        skip_budget = int(skip_ratio * total_count)
        print(f"[skip-plan] total tokens: {total_count}, "
              f"skip budget: {skip_budget} ({skip_ratio * 100:.1f}%)")

        # Select tokens with SMALLEST diff (most stable = most skippable).
        # largest=False => smallest K values.
        _, topk_indices = torch.topk(all_values, k=skip_budget, largest=False)
        # Sort for efficient two-pointer mapping to locations.
        topk_indices = sorted(topk_indices.tolist())

        # Map global index -> (layer_name, step_idx, token_idx) using a
        # two-pointer scan: O(skip_budget + total_locations), not O(n^2).
        skip_plan = {}
        offset = 0
        ptr = 0
        for layer_name, step_idx, n_tokens in token_locations:
            # Advance pointer past indices before this location's range
            while ptr < len(topk_indices) and topk_indices[ptr] < offset:
                ptr += 1
            # Collect indices within this location's range
            while ptr < len(topk_indices) and topk_indices[ptr] < offset + n_tokens:
                local_token_idx = topk_indices[ptr] - offset
                step_key = str(step_idx)
                if step_key not in skip_plan:
                    skip_plan[step_key] = {}
                if layer_name not in skip_plan[step_key]:
                    skip_plan[step_key][layer_name] = []
                skip_plan[step_key][layer_name].append(local_token_idx)
                ptr += 1
            offset += n_tokens

        # Sort token indices within each (step, layer) for deterministic ordering
        for step_key in skip_plan:
            for layer_name in skip_plan[step_key]:
                skip_plan[step_key][layer_name] = sorted(
                    skip_plan[step_key][layer_name]
                )
        # Print summary
        print(f"[skip-plan] {len(skip_plan)} steps with skips")
        for step_key in sorted(skip_plan.keys(), key=lambda x: int(x)):
            n_layers = len(skip_plan[step_key])
            total_skip = sum(len(v) for v in skip_plan[step_key].values())
            print(f"  Step {step_key}: {total_skip} tokens skipped "
                  f"across {n_layers} layers")
        return skip_plan

    elif method == "step-aggr":
        """
        Skip enter step
        diff_dict: {outer_key: {inner_key: [n_tokens]}}
        It could be: {0: {step.0: [n_token]}}
        """
        skip_cost = []
        for outer_key, inner_dict in diff_dict.items():
            for inner_key, diff in inner_dict.items():
                skip_cost.append(torch.sum(diff).item())
        # Skip budget = skip_ratio * total tokens
        total_count = len(diff_dict)
        skip_budget = int(skip_ratio * total_count)
        print(f"[skip-plan] total tokens: {total_count}, skip budget: {skip_budget} ({skip_ratio * 100:.1f}%)")
        skip_token = np.argsort(skip_cost)[:skip_budget]
        keep_token = np.setdiff1d(np.arange(total_count), skip_token).tolist()
        print(keep_token, skip_cost)
        return keep_token
    
    else:
        raise ValueError(f"Unknown skip plan method: {method}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Save step-wise dit_outputs / noise_preds for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_id", type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
    )
    parser.add_argument("--use_origin_model", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--dataset_name", type=str, default="mjhq30k")
    parser.add_argument("--dataset_path", type=str, default="G://datasets/MJHQ-30K")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--dit_inference_steps", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K")
    
    args = parser.parse_args()
    dit_inference_steps=[int(x) for x in args.dit_inference_steps.split(",")] if args.dit_inference_steps is not None else None
    if dit_inference_steps is not None:
        postfix = f"M{len(dit_inference_steps)}-cos"
    else:
        postfix = "full-steps-cos"

    all_step_wise_computation_diff = load_samples(args.output_dir, postfix)
    if len(all_step_wise_computation_diff) == 0:
        flag = input("Generate? (y/n)")
        if flag.lower() != "y":
            exit(0)
        ImageGenerator = SD3ImageGenerator if args.model_id == "stabilityai/stable-diffusion-3.5-medium" else SD3ImageGenerator
        gen = ImageGenerator(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=torch.bfloat16
        )
        os.makedirs(args.output_dir, exist_ok=True)
        gen.generate(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            num_samples=args.num_samples,
            save_root=args.output_dir,
            save_postfix=postfix,
            seed=args.seed,
            return_computation_diff=True,
            dit_inference_steps=dit_inference_steps,
        )
        all_step_wise_computation_diff = load_samples(args.output_dir, postfix)
    
    # Analyze diff
    ## JointTransformerBlock: block.22.img/txt[before] v.s. block.22.img/txt[after]
    ## JointAttention: 
    ### block-wise: block.22.attn.img/txt[before] v.s. block.22.attn.img/txt[after]
    ### attention: block.22.attn.attn_weights, block.22.attn.attn[before] v.s. block.22.attn.attn[after]
    ### FeedForward: block.22.ffn.before v.s. block.22.ffn.after; block.22.ff_context.before v.s. block.22.ff_context.after
    orientation = "layer"
    level = "layer"
    mean_matrix, _ = samples_to_avg(all_step_wise_computation_diff, level=level, orientation=orientation)
    """
    visualize_computation_diff(
        mean_matrix, orientation=orientation, level=level,
        save_path=f"{args.output_dir}/{orientation}_wise_{level}_aggr_mean{postfix}.png"
    )
    """
    visualize_token_diff_distribution(mean_matrix, f"{args.output_dir}/{postfix}-token-diff-distribution.png")
    """
    orientation = "layer"
    level = "step"
    mean_matrix, _ = samples_to_avg(all_step_wise_computation_diff, level=level, orientation=orientation)
    skip_ratio = 0.5
    # skip_plan = generate_skip_plan(mean_matrix, method="global", skip_ratio=skip_ratio)
    skip_plan = generate_skip_plan(mean_matrix, method="step-aggr", skip_ratio=skip_ratio)
    with open(f"{args.output_dir}/skip_plan/{postfix}-step-aggr-global-{int(100*skip_ratio):02d}.json", "w") as f:
        json.dump(skip_plan, f, indent=4)
    """
