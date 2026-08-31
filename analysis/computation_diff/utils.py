import glob
import os
import torch

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_samples(input_dir, postfix=""):
    """Load all *.pt files from input_dir, sorted by index."""
    save_root = f"{input_dir}/diff_dict/{postfix}"
    sample_files = sorted(glob.glob(os.path.join(save_root, f"*.pt")))
    # print(f"sample_files: {sample_files}")
    if not sample_files:
        print(f"No .pt files found in {save_root}")
        return []
    samples = []
    for f in sample_files:
        samples.append(torch.load(f, map_location="cpu", weights_only=False))
    print(f"[load] {len(samples)} computation diff from {save_root}")
    return samples


def samples_to_avg(samples, level="layer", orientation="step", concate_img_txt=False):
    """Compute the average of samples.
    level: "layer", "module"
    orientation: "step", "layer"

    samples: [{step_idx: {layer_name: n_token}}, {}, ...]
    orientation == "step", return {step_idx, {layer_name: [n_token]]}}
    orientation == "layer", return {layer_name, {step_idx: [n_token]]}}
    """
    avg_dict = {}
    std_dict = {}
    for sample_idx, sample in enumerate(samples):
        for step_idx, step_wise_diff in sample.items():
            for layer_name, module_wise_diff in step_wise_diff.items():
                module_wise_diff = module_wise_diff.reshape(1, -1)
                keys = layer_name.split(".")
                if len(keys) == 3 and level == "layer" or len(keys) == 4 and level == "module":
                    if orientation == "step":
                        if step_idx not in avg_dict:
                            avg_dict[step_idx] = {}
                        if layer_name not in avg_dict[step_idx]:
                            avg_dict[step_idx][layer_name] = [module_wise_diff]
                        else:
                            avg_dict[step_idx][layer_name].append(module_wise_diff)
                    elif orientation == "layer":
                        if layer_name not in avg_dict:
                            avg_dict[layer_name] = {}
                        if step_idx not in avg_dict[layer_name]:
                            avg_dict[layer_name][step_idx] = [module_wise_diff]
                        else:
                            avg_dict[layer_name][step_idx].append(module_wise_diff)
                else:
                    print(f"{layer_name} is not suppored in {level}")

    for outer_key, outer_dict in avg_dict.items():
        for inner_key, inner_dict in outer_dict.items():
            avg_dict[outer_key][inner_key] = torch.mean(torch.stack(inner_dict, dim=0), dim=0)

    if concate_img_txt:
        concat_avg_dict = {}
        concat_std_dict = {}
        if orientation == "step":
            for step_idx, step_wise_diff in avg_dict.items():
                if step_idx not in concat_avg_dict:
                    concat_avg_dict[step_idx] = {}
                for layer_name, layer_wise_diff in step_wise_diff.items():
                    layer_prefix = layer_name.replace(".img", "").replace(".txt", "")
                    # print(f"{layer_name} -> {layer_prefix}")
                    if layer_prefix not in concat_avg_dict[step_idx]:
                        concat_avg_dict[step_idx][layer_prefix] = avg_dict[step_idx][layer_name]
                    else:
                        # print(f"[{layer_name}|{layer_prefix}] Concat {layer_wise_diff.shape} -> {concat_avg_dict[step_idx][layer_prefix].shape}")
                        concat_avg_dict[step_idx][layer_prefix] = torch.cat([concat_avg_dict[step_idx][layer_prefix], layer_wise_diff], dim=1)
        elif orientation == "layer":
            for layer_name, layer_wise_diff in avg_dict.items():
                layer_prefix = layer_name.replace(".img", "").replace(".txt", "")
                if layer_prefix not in concat_avg_dict:
                    concat_avg_dict[layer_prefix] = {}
                for step_idx, step_wise_diff in layer_wise_diff.items():
                    if step_idx not in concat_avg_dict[layer_prefix]:
                        concat_avg_dict[layer_prefix][step_idx] = avg_dict[layer_name][step_idx]
                    else:
                        concat_avg_dict[layer_prefix][step_idx] = torch.cat([concat_avg_dict[layer_prefix][step_idx], step_wise_diff], dim=1)
        else:
            raise ValueError(f"orientation {orientation} is not suppored")
        return concat_avg_dict, concat_std_dict

    return avg_dict, std_dict


def visualize(_diff_dict, save_path, orientation="step"):
    n_vis = len(_diff_dict)
    if n_vis == 0:
        return
    n_row = int(math.sqrt(n_vis))
    n_col = max(1, n_vis // n_row)
    if n_vis % n_row != 0:
        n_row += 1
    cmap = 'RdYlGn_r'
    fig, axes = plt.subplots(n_row, n_col, figsize=(16, 16), squeeze=False)
    axes = axes.flatten()

    prev_step_idx = -1
    prev_layer_idx = -1
    for idx, (outer_key, outer_diff_dict) in enumerate(_diff_dict.items()):
        # {step_idx: {layer_idx: [n_token]}} => step_idx x [n_layer, n_token] or 
        # {layer_name: {step_idx: [n_token]}} => layer_name x [n_step, n_token]

        # Ensure step_idx / layer_idx is in ascending order
        if orientation == "step":
            _step_idx = int(outer_key)
            assert _step_idx >= prev_step_idx
            prev_step_idx = _step_idx
            prev_layer_idx = -1
            for inner_key, _diff in outer_diff_dict.items():
                _layer_idx = int(inner_key.split(".")[1])
                assert _layer_idx >= prev_layer_idx
                prev_layer_idx = _layer_idx
        else:
            _layer_idx = int(outer_key.split(".")[1])
            assert _layer_idx >= prev_layer_idx
            prev_layer_idx = _layer_idx
            prev_step_idx = -1
            for inner_key, _diff in outer_diff_dict.items():
                _step_idx = int(inner_key)
                assert _step_idx >= prev_step_idx
                prev_step_idx = _step_idx

        # Zero padding for last layer, for txt is processed in last layer
        inner_diff_collect = []
        for inner_key, _diff in outer_diff_dict.items():
            _diff = _diff.reshape(1, -1)
            if len(inner_diff_collect) > 0 and _diff.shape[-1] != inner_diff_collect[-1].shape[-1]:
                # print(f"0 padding for shape mismatch in [{inner_key}]: {_diff.shape} -> {inner_diff_collect[-1].shape}")
                _diff = torch.cat([_diff, torch.zeros(1, inner_diff_collect[-1].shape[-1] - _diff.shape[-1])], dim=1)
            inner_diff_collect.append(_diff)
        _vis_diff = torch.cat(inner_diff_collect, dim=0).float().numpy()
        print(f"Visualize {outer_key} with shape {_vis_diff.shape}")
        ax = axes[idx]
        im = ax.imshow(_vis_diff, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(outer_key)
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()