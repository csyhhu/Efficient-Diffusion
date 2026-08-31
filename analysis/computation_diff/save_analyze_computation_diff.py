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

import torch

from src.image_generator import SanaImageGenerator, SD3ImageGenerator
from src.utils import compute_computation_diff

from analysis.computation_diff.utils import load_samples, samples_to_avg, visualize


def generate_skip_plan(diff_dict, method="global", skip_ratio=0.5):
    """
    Generate token skip plan given computation dict values matrix.
    diff_dict: {layer_name: {step_idx: [n_img + n_txt]}}
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
    parser.add_argument("--num_samples", type=int, default=10)
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
    # mean_matrix, _ = samples_to_avg(all_step_wise_computation_diff, orientation=orientation, concate_img_txt=True)
    mean_matrix, _ = samples_to_avg(all_step_wise_computation_diff, orientation=orientation)
    visualize(mean_matrix, f"{args.output_dir}/{orientation}_wise_mean{postfix}.png", orientation=orientation)
    # Retrieve diff data
    """
    # sample_idx: {step_idx: {block_name: []}} => {block_name: [step_idx: [sample_idx, n_img]]}
    layer_wise_img_diff = {} # {block_name: [step_idx: [sample_idx, n_img]]}
    layer_wise_txt_diff = {}
    attention_diff = {}
    img_diff = {}
    txt_diff = {}
    attention_weights = {}
    feed_forward_blocks_diff = {}
    for sample_idx, step_wise_computation_diff in enumerate(all_step_wise_computation_diff):
        sample_wise_computation_diff = {}
        for step_idx, computation_diff in step_wise_computation_diff.items():
            for key, value in computation_diff.items():
                keys = key.split(".")
                # For the whole joint block: block.22.img/txt: [bs, n_img, dim] => [n_img]
                if len(keys) == 3:
                    if keys[-1] == "img":
                        if key not in layer_wise_img_diff:
                            layer_wise_img_diff[key] = {}
                        if step_idx not in layer_wise_img_diff[key]:
                            layer_wise_img_diff[key][step_idx] = []
                        layer_wise_img_diff[key][step_idx].append(value.unsqueeze(0)) # [n_img] => [step_idx, n_img]
                        # print(f"{key}: {value.shape}")
                    elif keys[-1] == "txt":
                        if key not in layer_wise_txt_diff:
                            layer_wise_txt_diff[key] = {}
                        if step_idx not in layer_wise_txt_diff[key]:
                            layer_wise_txt_diff[key][step_idx] = []
                        layer_wise_txt_diff[key][step_idx].append(value.unsqueeze(0)) # [n_txt] => [step_idx, n_txt]
                        # print(f"{key}: {value.shape}")
                    else:
                        raise ValueError(f"Unknown key: {key}")
                # Within joint block: block.22.attn.img/txt/attn/attn_weights
                else:
                    print(f"Unsupported key: {key}")
    
    # Average diff data
    img_mean_matrix = {} # {block_name: [step_idx, n_img]}
    img_std_matrix = {}
    txt_mean_matrix = {} # {block_name: [step_idx, n_txt]}
    txt_std_matrix = {}
    for prefix, mean_matrix, std_matrix, total_layer_wise_diff in (
        ("img", img_mean_matrix, img_std_matrix, layer_wise_img_diff),
        ("txt", txt_mean_matrix, txt_std_matrix, layer_wise_txt_diff),
    ):
        for block_name, layer_wise_diff in total_layer_wise_diff.items():
            if block_name not in mean_matrix:
                mean_matrix[block_name] = {}
            if block_name not in std_matrix:
                std_matrix[block_name] = {}
            for step_idx, step_wise_diff in layer_wise_diff.items():
                if step_idx not in mean_matrix[block_name]:
                    mean_matrix[block_name][step_idx] = []
                if step_idx not in std_matrix[block_name]:
                    std_matrix[block_name][step_idx] = []
                mean_matrix[block_name][step_idx] = torch.stack(step_wise_diff, dim=0).mean(0) # [sample_idx, n_img] => [n_img]
                std_matrix[block_name][step_idx] = torch.stack(step_wise_diff, dim=0).std(0) # [sample_idx, n_img]
        # visualize(mean_matrix, f"{args.output_dir}/layer_wise_{prefix}_mean{postfix}.png")
        # visualize(std_matrix, f"{args.output_dir}/layer_wise_{prefix}_std{postfix}.png")  
    
    # Concat for better analysis
    mean_matrix = {}
    for block_name, block_wise_mean in img_mean_matrix.items():
        layer_name = block_name[:-4]
        if layer_name not in mean_matrix:
            mean_matrix[layer_name] = {}
        for step_idx, step_wise_mean in block_wise_mean.items():
            txt_block_name = block_name.replace("img", "txt")
            if txt_block_name in txt_mean_matrix and step_idx in txt_mean_matrix[txt_block_name]:
                mean_matrix[layer_name][step_idx] = torch.cat([step_wise_mean, txt_mean_matrix[txt_block_name][step_idx]], dim=1)
                # print(f"{block_name}, {step_idx}, {mean_matrix[layer_name][step_idx].shape}")
            else:
                mean_matrix[layer_name][step_idx] = step_wise_mean
                # print(f"block_name: {block_name}, step_idx: {step_idx} not found in txt_mean_matrix")
    visualize(mean_matrix, f"{args.output_dir}/layer_wise_mean_{postfix}.png")
    """
    """
    skip_ratio = 0.1
    skip_plan = generate_skip_plan(mean_matrix, method="global", skip_ratio=skip_ratio)
    with open(f"{args.output_dir}/token_skip_plan_{postfix}_global_{int(100*skip_ratio):02d}.json", "w") as f:
        json.dump(skip_plan, f, indent=4)
    """
