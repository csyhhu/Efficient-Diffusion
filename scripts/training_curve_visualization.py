

import csv
import os
import re
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path so outputs/ can be found regardless of cwd
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ============================================================================
# Data readers
# ============================================================================

def read_loss_history(path: str) -> list:
    """Read a ``loss_history.csv`` file and return the ``train_loss`` column as a list of floats.

    Expected CSV columns: ``step,epoch,train_loss``

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        List of train_loss values in chronological order.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"loss_history.csv not found: {path}")

    loss = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss.append(float(row["train_loss"]))
    return loss


def read_quant_error_dir(dir_path: str) -> Dict[str, List[float]]:
    """Read all per-layer quantization error CSVs from a quant_error directory.

    Each CSV has columns ``step,quant_error_sum``.

    Args:
        dir_path: Path to the ``quant_error/`` directory.

    Returns:
        Dict mapping layer_name -> list of error values (aligned on step index).
        Steps are assumed contiguous starting from 1.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"quant_error directory not found: {dir_path}")

    layer_data: Dict[str, List[float]] = {}
    layer_steps: Dict[str, List[int]] = {}

    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".csv"):
            continue
        layer_name = fname[:-4]  # strip .csv
        fpath = os.path.join(dir_path, fname)
        values = []
        steps = []
        with open(fpath, "r", newline="") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                values.append(float(row["quant_error_sum"]))
        # Reindex to 0-based contiguous (step 1 -> index 0)
        if steps:
            aligned = [np.nan] * max(steps)
            for s, v in zip(steps, values):
                aligned[s - 1] = v
            layer_data[layer_name] = aligned
            layer_steps[layer_name] = steps

    return layer_data


# ============================================================================
# Anomaly detection
# ============================================================================

def find_anomalous_layers(
    quant_data: Dict[str, List[float]],
    ratio_threshold: float = 1.05,
    min_abs_value: float = 0.005,
    window_n: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Identify layers whose quantization error increases instead of decreasing.

    Anomaly criterion:
      ``last_window_avg / first_window_avg > ratio_threshold``
      AND ``last_window_avg > min_abs_value`` (skip near-zero noise).

    Args:
        quant_data:       Dict from ``read_quant_error_dir()``.
        ratio_threshold:  Ratio above which a layer is flagged.
        min_abs_value:    Minimum absolute error to consider.
        window_n:         Number of steps used for first/last average.

    Returns:
        Dict of ``layer_name -> {"ratio": float, "first_avg": float, "last_avg": float}``
        for anomalous layers only, sorted descending by ratio.
    """
    anomalies: Dict[str, Dict[str, float]] = {}
    for layer_name, values in quant_data.items():
        arr = np.array(values)
        arr = arr[~np.isnan(arr)]
        if len(arr) < window_n:
            continue
        first_avg = float(np.mean(arr[:window_n]))
        last_avg = float(np.mean(arr[-window_n:]))
        if first_avg < 1e-12 and last_avg < min_abs_value:
            continue  # both near zero, not meaningful
        ratio = last_avg / first_avg if first_avg > 1e-12 else float("inf")
        if ratio > ratio_threshold and last_avg > min_abs_value:
            anomalies[layer_name] = {
                "ratio": ratio,
                "first_avg": first_avg,
                "last_avg": last_avg,
            }
    # Return sorted by ratio descending
    return dict(sorted(anomalies.items(), key=lambda kv: -kv[1]["ratio"]))


# ============================================================================
# Quantization error visualisation
# ============================================================================

def plot_quant_errors(
    quant_data: Dict[str, List[float]],
    output_run_label: str = "",
    save_path: Optional[str] = None,
    figsize_per_subplot: float = 5.0,
) -> plt.Figure:
    """Plot every layer-level quantization error in a grid of per-block subplots.

    One subplot per transformer block (e.g. ``block_0`` … ``block_5``).
    Each subplot draws one coloured line per component so that all individual
    layers are visible.  A shared legend appears on each subplot.

    Args:
        quant_data:   Dict from ``read_quant_error_dir()``.
        output_run_label: Label used in the suptitle (e.g. model name).
        save_path:    If given, save the figure to this path.
        figsize_per_subplot: Base size in inches per subplot dimension
                             (final width = base * n_cols, final height = base * n_rows).

    Returns:
        The matplotlib Figure, or None if no block data found.
    """
    # Group layer names by block
    block_groups: Dict[str, Dict[str, List[float]]] = {}
    for key, values in quant_data.items():
        match = re.match(r"(block_\d+)", key)
        if not match:
            continue
        block = match.group(1)
        block_groups.setdefault(block, {})[key] = values

    if not block_groups:
        print("[quant-error-viz] No block_* layers found in quant_data.")
        return None

    blocks_sorted = sorted(block_groups.keys(), key=lambda b: int(b.split("_")[1]))
    n_blocks = len(blocks_sorted)
    n_cols = min(3, n_blocks)
    n_rows = (n_blocks + n_cols - 1) // n_cols

    # Taller subplots to accommodate the legend
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot * n_cols, figsize_per_subplot * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    # Build a fixed colour palette so the same component gets the same colour
    # across all blocks (makes comparison easier).
    all_component_names: set = set()
    for layers in block_groups.values():
        for name in layers:
            all_component_names.add(name)
    all_component_names = sorted(all_component_names)
    n_colours = len(all_component_names)
    cmap = plt.cm.tab20 if n_colours <= 20 else plt.cm.tab20b
    colour_map = {name: cmap(i % 20) for i, name in enumerate(all_component_names)}

    for idx, block in enumerate(blocks_sorted):
        ax = axes[idx]
        layers = block_groups[block]

        for layer_name, values in layers.items():
            short_label = layer_name[len(block) + 1:]  # strip "block_N_"
            color = colour_map.get(layer_name, "C0")
            ax.plot(values,
                    color=color, alpha=0.8, linewidth=0.7,
                    label=short_label)

        ax.set_title(block.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel("Quant Error Sum")
        ax.legend(fontsize=5.5, loc="upper left", ncol=2, framealpha=0.5)
        ax.grid(alpha=0.3, linestyle="--")

    # Hide unused axes
    for j in range(n_blocks, len(axes)):
        axes[j].set_visible(False)

    suptitle = "Quantization Error per Layer (all components)"
    if output_run_label:
        suptitle = f"{output_run_label} — {suptitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[quant-error-viz] Saved to {save_path}")

    return fig


def plot_anomalous_quant_errors(
    quant_data: Dict[str, List[float]],
    anomalous: Dict[str, Dict[str, float]],
    output_run_label: str = "",
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot only layers whose quantization error is **increasing** (anomalous).

    Layers are grouped into subplots by functional category so the chart
    stays readable even with many anomalous layers.

    Args:
        quant_data:  Dict from ``read_quant_error_dir()``.
        anomalous:   Dict from ``find_anomalous_layers()``.
        output_run_label: Label used in the suptitle.
        save_path:   If given, save the figure to this path.

    Returns:
        The matplotlib Figure, or None if no anomaly data.
    """
    if not anomalous:
        print("[quant-error-viz] No anomalous layers found — skipping anomaly plot.")
        return None

    # ---- Group anomalous layers by functional category ----
    groups: Dict[str, List[str]] = {}
    for layer_name in anomalous:
        # Categorise: attn.*_input, mlp.*_input, other
        if re.search(r"attn_(query|key|value)_input", layer_name):
            cat = "attn Q/K/V inputs"
        elif re.search(r"attn", layer_name):
            cat = "attn others"
        elif re.search(r"mlp_\d+_input", layer_name):
            cat = "mlp inputs"
        elif "_weight" in layer_name:
            cat = "weights"
        elif "_input" in layer_name:
            cat = "other inputs"
        else:
            cat = "others"
        groups.setdefault(cat, []).append(layer_name)

    # Remove empty groups and sort names within each
    groups = {k: sorted(v) for k, v in groups.items() if v}

    n_groups = len(groups)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    group_names = sorted(groups.keys())

    for idx, cat in enumerate(group_names):
        ax = axes[idx]
        layers = groups[cat]
        n_in_group = len(layers)
        cmap = plt.cm.tab10 if n_in_group <= 10 else plt.cm.tab20

        for j, layer_name in enumerate(layers):
            values = quant_data[layer_name]
            info = anomalous[layer_name]
            color = cmap(j % (10 if n_in_group <= 10 else 20))
            # Shorten label: remove "block_X_" prefix
            # short = re.sub(r"^block_\d+_", "", layer_name)
            # label = f"{short}  (r={info['ratio']:.2f})"
            label = layer_name
            ax.plot(values, color=color, alpha=0.85, linewidth=1.0, label=label)

        ax.set_title(f"{cat}  ({n_in_group} layers)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Quant Error Sum")
        ax.legend(fontsize=6, loc="upper left", framealpha=0.5)
        ax.grid(alpha=0.3, linestyle="--")
        # Add a horizontal line at y=0 as reference
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")

    # Hide unused axes
    for j in range(n_groups, len(axes)):
        axes[j].set_visible(False)

    suptitle = "Anomalous Layers — Quantization Error INCREASING"
    if output_run_label:
        suptitle = f"{output_run_label} — {suptitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[quant-error-viz] Saved to {save_path}")

    # Also print a text summary
    print(f"\n[anomaly] {len(anomalous)} anomalous layers detected (error increasing):")
    for name, info in anomalous.items():
        print(f"  {name:<50}  ratio={info['ratio']:5.2f}  first={info['first_avg']:.4f}  last={info['last_avg']:.4f}")

    return fig


# ============================================================================
# Main: Loss curve comparison + optional quant-error visualisation
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configure your runs here.
    # Each entry specifies:
    #   path   – relative (from project root) or absolute path to loss_history.csv
    #   label  – legend name displayed on the plot
    #   color  – matplotlib colour string
    #   style  – matplotlib linestyle  ("-", "--", "-.", ":")
    #   smooth – optional EMA smoothing factor (0 = raw, 0.9 = heavy)
    #   quant_error_dir – optional path to quant_error/ folder for per-layer plots
    # ------------------------------------------------------------------
    loss_info_dict: dict = {
        # "ddpm-dit": {
        #     "path":  os.path.join(_project_root, "outputs", "mnist_ddpm_dit", "loss_history.csv"),
        #     "label": "DDPM (DiT)",
        #     "color": "#1f77b4",   # blue
        #     "style": "-",
        #     "smooth": 0.0,
        # },
        # "ddpm-quantized-dit-trial-0": {
        #     "path":  os.path.join(_project_root, "Results", "mnist_ddpm_quantized_dit_trial_0", "loss_history.csv"),
        #     "label": "DDPM (Quantized DiT) - Trial 0",
        #     "color": "#9467bd",   # purple
        #     "style": "-",
        #     "smooth": 0.0,
        #     "quant_error_dir": os.path.join(_project_root, "Results", "mnist_ddpm_quantized_dit", "quant_error"),
        # },
        "ddpm-quantized-dit-trial-1": {
            "path":  os.path.join(_project_root, "Results", "mnist_ddpm_quantized_dit", "loss_history.csv"),
            "label": "DDPM (Quantized DiT)",
            "color": "#1f77b4",   # blue
            "style": "-",
            "smooth": 0.0,
            "quant_error_dir": os.path.join(_project_root, "Results", "mnist_ddpm_quantized_dit", "quant_error"),
        },
        # "fm-dit": {
        #     "path":  os.path.join(_project_root, "outputs", "mnist_fm_dit", "loss_history.csv"),
        #     "label": "Flow Matching (DiT)",
        #     "color": "#d62728",   # red
        #     "style": "-",
        #     "smooth": 0.0,
        # },
        # Add more runs here, for example:
        # "fm-unet": {
        #     "path":  os.path.join(_project_root, "outputs", "mnist_fm_unet", "loss_history.csv"),
        #     "label": "Flow Matching (UNet)",
        #     "color": "#2ca02c",  # green
        #     "style": "--",
        #     "smooth": 0.0,
        # },
    }

    # ---- Figure 1: Loss curves ----
    """
    for name, info in loss_info_dict.items():
        loss = read_loss_history(info["path"])
        # Apply optional smoothing
        smooth = info.get("smooth", 0.0)
        if smooth > 0:
            smoothed = [loss[0]]
            for v in loss[1:]:
                smoothed.append(smooth * smoothed[-1] + (1 - smooth) * v)
            loss = smoothed
        plt.plot(loss,
                 label=info["label"],
                 color=info["color"],
                 linestyle=info.get("style", "-"),
                 linewidth=1.2)

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MNIST Training Curve")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    # plt.savefig("mnist_training_curve.png", dpi=150)
    plt.show()
    plt.close()
    """

    # ---- Figure 2: Quantization error per layer (all components) ----
    for name, info in loss_info_dict.items():
        quant_dir = info.get("quant_error_dir", None)
        if quant_dir is None or not os.path.isdir(quant_dir):
            continue
        print(f"[quant-error-viz] Reading {quant_dir} ...")
        quant_data = read_quant_error_dir(quant_dir)
        out_base = os.path.dirname(quant_dir)
        """
        if not quant_data:
            print(f"[quant-error-viz] No CSV files found in {quant_dir}")
            continue

        # All layers (per-block subplots)
        plot_quant_errors(
            quant_data,
            output_run_label=info["label"],
            save_path=os.path.join(out_base, "quant_error_per_block.png"),
        )
        plt.show()
        plt.close()
        """

        # ---- Figure 3: Anomalous layers (error increasing instead of decreasing) ----
        anomalous = find_anomalous_layers(quant_data)
        plot_anomalous_quant_errors(
            quant_data,
            anomalous,
            output_run_label=info["label"],
            save_path=os.path.join(out_base, "quant_error_anomalous.png"),
        )
        plt.show()
        plt.close()