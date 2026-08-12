r"""Visualize per-NVFP4Linear quantization error under different
rotation + permutation + (un)quantized configurations.

This script consumes the three JSON files produced by
``analysis/cal_rotation_permutation_quantization_error.py`` (in the sweep output dir):

  - param_quant_errors.json   {config: {layer: {weight_mse, weight_mae}}}
  - activation_errors.json    {sample: {config: {step: {layer.input: {act_mse, act_mae}}}}}
  - output_diff.json          {config: {sample: {step: {mse, mae, cosine,
                                layers: {transformer_blocks.i: {mse, mae, cosine}},
                                linear_layers: {nvfp4_linear_name: {mse, mae, cosine}}}}},
                                "final_diff": {config: {sample: {
                                  final_latents: {mse, mae, cosine},
                                  dec: {mse, mae, cosine},
                                  image: {mse, mae, cosine}}}}}
                                (the "final_diff" top-level key is reserved and is
                                NOT treated as a config by this script)

It produces TWO figures:

  (1) ``layer_error_grid.png`` -- NVFP4Linear-level grid.
      * ONE ROW per NVFP4Linear module (sorted by natural order).
      * ONE COLUMN per (error kind x step), laid out as:

            col 0           : Parameters (weight) quantization error   [step-independent]
            col 1           : Activation error at step 0
            col 2           : NVFP4Linear output (vs reference) error at step 0
            col 3           : Activation error at step 1
            col 4           : NVFP4Linear output (vs reference) error at step 1
            ...

        i.e. 1 + 2*steps columns. The example above is for a 2-step decode.

  (2) ``final_diff_grid.png`` -- block-level output + end-to-end final diffs.
      * ONE ROW per transformer block (transformer_blocks.0, ...).
      * COLUMNS: block output error (aggregated across steps), final latent,
        decoder output, image (the last 3 are layer-independent, repeated per row).

Example (Windows PowerShell):

    python analysis/visualization_rotation_permutation_quantization_error.py `
        --input-dir G:/Outputs/Efficient-Diffusion/rot_perm_compare_module_steps_quantized_dry `
        --output-dir G:/Outputs/Efficient-Diffusion/rot_perm_compare_module_steps_quantized_dry/visualization `
        --metric mse --baseline identity_identity_q --max-linear-rows 40 --max-rows-per-page 40 --logy
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def natural_key(name):
    """Sort layer names like block.0, block.1, ..., block.10 in order."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) < 2:
        return float(np.mean(vals)), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_inputs(input_dir):
    def _load(name):
        p = os.path.join(input_dir, name)
        if not os.path.isfile(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    param = _load("param_quant_errors.json") or {}
    act = _load("activation_errors.json") or {}
    out = _load("output_diff.json") or {}
    return param, act, out


def derive_configs_and_samples(param, act, out):
    configs = set()
    configs |= set(param.keys())
    for smp in act.values():
        configs |= set(smp.keys())
    configs |= {c for c in out.keys() if c != "final_diff"}
    configs = sorted(configs)

    samples = set()
    for smp in act:
        try:
            samples.add(int(smp))
        except (TypeError, ValueError):
            pass
    for cfg_name, cfg in out.items():
        if cfg_name == "final_diff":
            continue
        for smp in cfg:
            try:
                samples.add(int(smp))
            except (TypeError, ValueError):
                pass
    samples = sorted(samples)
    return configs, samples


def derive_steps(out, configs):
    """Infer the sorted list of decode-step indices present in output_diff."""
    steps = set()
    for cfg in configs:
        for smp in out.get(cfg, {}).values():
            for s in smp:
                try:
                    steps.add(int(s))
                except (TypeError, ValueError):
                    pass
    return sorted(steps)


# ---------------------------------------------------------------------------
# derive NVFP4Linear layer names (rows for layer_error_grid)
# ---------------------------------------------------------------------------
def derive_linear_layer_names(param):
    """Return sorted list of NVFP4Linear layer names from param_quant_errors.

    Uses the union of layer names across all configs. These names use the
    ``block.X`` prefix (consistent with param/act data).
    """
    names = set()
    for cfg in param.values():
        names |= set(cfg.keys())
    return sorted(names, key=natural_key)


# ---------------------------------------------------------------------------
# derive transformer block names (rows for final_diff_grid)
# ---------------------------------------------------------------------------
def derive_block_names(out):
    """Return sorted list of transformer block names from output_diff layers."""
    names = set()
    for cfg_name, cfg_val in out.items():
        if cfg_name == "final_diff":
            continue
        if not isinstance(cfg_val, dict):
            continue
        for smp_val in cfg_val.values():
            for st in smp_val.values():
                names |= set(st.get("layers", {}).keys())
    return sorted(names, key=natural_key)


# ---------------------------------------------------------------------------
# value resolvers
# ---------------------------------------------------------------------------
def make_resolver(param, act, out, samples, metric):
    """Build resolvers for param, activation, linear_output, block_output,
    and final diffs."""

    if metric in ("cos", "cosine"):
        wkey = "weight_cosine"
        akey = "act_cosine"
        mkey = "cosine"       # output_diff uses "cosine" as the key
    elif metric == "mse":
        wkey, akey, mkey = "weight_mse", "act_mse", "mse"
    else:
        wkey, akey, mkey = "weight_mae", "act_mae", "mae"

    def _param_value(config, layer):
        if config not in param:
            return None, None
        return param[config].get(layer, {}).get(wkey), None

    def _act_value(config, layer, step):
        vals = []
        for smp in samples:
            d = act.get(str(smp), {}).get(config, {}).get(str(step), {})
            v = d.get(layer + ".input", {}).get(akey)
            if v is not None:
                vals.append(v)
        return _mean_std(vals)

    def _linear_out_value(config, layer, step):
        """Read output_diff.json -> config -> sample -> step -> linear_layers.

        The ``layer`` arg uses the ``block.X`` naming (same as param/act).
        We map back to ``transformer_blocks.X`` to look up in linear_layers.
        """
        denorm = re.sub(r"^block\.", "transformer_blocks.", layer)
        vals = []
        for smp in samples:
            d = out.get(config, {}).get(str(smp), {}).get(str(step), {})
            v = d.get("linear_layers", {}).get(denorm, {}).get(mkey)
            if v is not None:
                vals.append(v)
        return _mean_std(vals)

    def _block_out_value(config, block_name, step):
        """Read output_diff.json -> config -> sample -> step -> layers[block_name]."""
        vals = []
        for smp in samples:
            d = out.get(config, {}).get(str(smp), {}).get(str(step), {})
            v = d.get("layers", {}).get(block_name, {}).get(mkey)
            if v is not None:
                vals.append(v)
        return _mean_std(vals)

    def _block_out_aggregated(config, block_name, steps_list, final_mode):
        if final_mode == "mean":
            vals = []
            for s in steps_list:
                m, _ = _block_out_value(config, block_name, s)
                if m is not None:
                    vals.append(m)
            return _mean_std(vals)
        else:  # "last"
            if steps_list:
                return _block_out_value(config, block_name, steps_list[-1])
            return None, None

    def _final_value(config, kind):
        """End-to-end final diff (latent / dec / image), step-independent."""
        fd = out.get("final_diff", {})
        if not fd:
            return None, None
        vals = []
        for smp in samples:
            d = fd.get(config, {}).get(str(smp), {}).get(kind)
            if isinstance(d, dict) and mkey in d:
                vals.append(d[mkey])
        return _mean_std(vals)

    return (_param_value, _act_value, _linear_out_value,
            _block_out_value, _block_out_aggregated, _final_value)


# ---------------------------------------------------------------------------
# column layout for layer_error_grid
# ---------------------------------------------------------------------------
def build_linear_columns(steps):
    """Return list of (kind, step, title) for the NVFP4Linear-level grid columns.

    Layout:
      col 0                   : parameters (weight) quantization error
      for each step k         : activation error at step k, then
                                NVFP4Linear output (vs reference) error at step k
    """
    cols = [("param", None, "param\n(weight)")]
    for s in steps:
        cols.append(("act", s, f"act\ns{s}"))
        cols.append(("linear_out", s, f"out\ns{s}"))
    return cols


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_grid(rows, columns, configs, out_dir, baseline, logy, title,
              args_metric, max_rows_per_page, resolve_fn, file_prefix):
    """Generic grid: ``rows`` on y-axis, ``columns`` on x-axis.

    Each cell is a bar chart across configs. ``resolve_fn(row, kind, step, c)``
    returns ``(mean, std)`` for a given (row, column-kind, step, config).
    """
    n_cols = len(columns)
    page = max_rows_per_page
    starts = list(range(0, len(rows), page))
    saved = []
    for pi, start in enumerate(starts):
        chunk = rows[start:start + page]
        nr = len(chunk)
        row_h = max(1.6, min(2.6, 60.0 / max(1, nr)))
        fig, axes = plt.subplots(
            nr, n_cols, figsize=(n_cols * 3.0, row_h * nr + 1.2),
            squeeze=False, sharey=False)
        for i, row_name in enumerate(chunk):
            for j, (kind, step, ctitle) in enumerate(columns):
                ax = axes[i][j]
                results = [resolve_fn(row_name, kind, step, c)
                           for c in configs]
                means, stds = zip(*results) if results else ([], [])
                xs = np.arange(len(configs))
                colors = ["#C44E52" if baseline in c else "#4C72B0"
                          for c in configs]
                plot_vals = [v if v is not None else 0.0 for v in means]
                
                if logy:
                    yerr_lower = []
                    yerr_upper = []
                    for m, s in zip(means, stds):
                        if m is None or s is None or s == 0:
                            yerr_lower.append(0.0)
                            yerr_upper.append(0.0)
                        else:
                            m_val = float(m)
                            s_val = float(s)
                            lower = max(m_val - s_val, m_val * 0.01)
                            yerr_lower.append(m_val - lower)
                            yerr_upper.append(s_val)
                    yerr = [yerr_lower, yerr_upper]
                else:
                    yerr = [float(s) if s is not None else 0.0 for s in stds]
                
                ax.bar(xs, plot_vals, color=colors, width=0.8, yerr=yerr,
                       capsize=2, error_kw={"elinewidth": 0.8, "capsize": 2})
                ax.set_xticks(xs)
                ax.set_xticklabels(configs, rotation=90, fontsize=4.5)
                ax.tick_params(axis="y", labelsize=5)
                if logy:
                    ax.set_yscale("log")
                # layer name as ylabel on col 0
                if j == 0:
                    ax.set_ylabel(f"{row_name}", fontsize=5.5, labelpad=2)
                # Annotate layer name in top-left corner of each cell
                ax.text(0.02, 0.95, row_name, transform=ax.transAxes,
                        fontsize=3.5, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.1',
                                  facecolor='lightyellow',
                                  edgecolor='none', alpha=0.7))
                if i == 0:
                    ax.set_title(ctitle, fontsize=8)
        # legend
        handles = [
            Patch(facecolor="#C44E52", label=f"baseline ('{baseline}')"),
            Patch(facecolor="#4C72B0", label="other configs"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=7)
        fig.suptitle(
            f"{title}  (metric={args_metric})\n"
            f"baseline bars in red", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        # Compute hspace/wspace
        fig_h = fig.get_size_inches()[1]
        fig_w = fig.get_size_inches()[0]
        mean_axes_h = fig_h / nr
        mean_axes_w = fig_w / n_cols
        hspace = min(2.0, max(1.0, 1.6 / mean_axes_h))
        wspace = min(1.0, max(0.6, 1.0 / mean_axes_w))
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        suffix = f"_p{pi + 1}" if len(starts) > 1 else ""
        path = os.path.join(out_dir, f"{file_prefix}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(os.path.basename(path))
        print(f"figure saved: {saved[-1]}")
    return saved


# ---------------------------------------------------------------------------
# layer_error_grid
# ---------------------------------------------------------------------------
def plot_layer_error_grid(
        linear_layers, param_value, act_value, linear_out_value,
        steps, configs, out_dir, baseline, logy, title, args_metric,
        max_rows_per_page):
    """NVFP4Linear-level grid: one row per NVFP4Linear module.

    Columns: [param, act_s0, out_s0, act_s1, out_s1, ...].
    """
    columns = build_linear_columns(steps)

    def resolve_linear(row, kind, step, config):
        if kind == "param":
            return param_value(config, row)
        if kind == "act":
            return act_value(config, row, step)
        if kind == "linear_out":
            return linear_out_value(config, row, step)
        return None

    print(f"[layer_error_grid] {len(linear_layers)} NVFP4Linear rows x "
          f"{len(columns)} columns, metric={args_metric}")
    return plot_grid(
        linear_layers, columns, configs, out_dir, baseline, logy,
        f"{title}  NVFP4Linear-level", args_metric, max_rows_per_page,
        resolve_linear, "layer_error_grid")


# ---------------------------------------------------------------------------
# final_diff_grid  (custom layout: row-0 = final_metrics, row-1+ = per-block)
# ---------------------------------------------------------------------------
def plot_final_diff_grid(
        block_names, block_out_aggregated, final_value,
        steps, configs, out_dir, baseline, logy, title,
        args_metric, max_rows_per_page, final_mode):
    """Block-level + end-to-end final diff grid with a clean two-section layout.

    Section A (top, single row):  final_latent  |  decoder_out  |  image
      - These are end-to-end metrics; one value per config, not block-dependent.

    Section B (below, one row per block):  transformer_blocks.X output
      - Each row shows the block's aggregated output error across configs.

    Pagination splits blocks across pages (Section A repeats on every page).
    """
    kind_map = {"latent": "final_latents", "dec": "dec", "img": "image"}
    kind_labels = {
        "latent": "latent\n(end-to-end)",
        "dec": "dec\n(end-to-end)",
        "img": "img\n(end-to-end)",
    }

    n_configs = len(configs)
    xs = np.arange(n_configs)
    colors = ["#C44E52" if baseline in c else "#4C72B0" for c in configs]

    page = max_rows_per_page
    starts = list(range(0, len(block_names), page))
    saved = []

    for pi, start in enumerate(starts):
        chunk = block_names[start:start + page]
        n_blocks = len(chunk)
        total_rows = 1 + n_blocks  # row-0 = final, rows 1..N = blocks

        # Figure: row-0 has 3 cols (final), rows 1+ span all 3 cols (blocks)
        block_h = max(1.2, min(2.0, 40.0 / max(1, n_blocks + 1)))
        fig = plt.figure(figsize=(n_configs * 1.3 + 1.5,
                                  1.8 + n_blocks * block_h))
        gs = gridspec.GridSpec(total_rows, 3, figure=fig,
                               hspace=0.55, wspace=0.35)

        def _make_yerr(means, stds, logy_mode):
            if logy_mode:
                yerr_lower = []
                yerr_upper = []
                for m, s in zip(means, stds):
                    if m is None or s is None or s == 0:
                        yerr_lower.append(0.0)
                        yerr_upper.append(0.0)
                    else:
                        m_val = float(m)
                        s_val = float(s)
                        lower = max(m_val - s_val, m_val * 0.01)
                        yerr_lower.append(m_val - lower)
                        yerr_upper.append(s_val)
                return [yerr_lower, yerr_upper]
            else:
                return [float(s) if s is not None else 0.0 for s in stds]

        # ----- Section A: final latent / dec / img (row 0) -----
        for j, kind in enumerate(["latent", "dec", "img"]):
            ax = fig.add_subplot(gs[0, j])
            results = [final_value(c, kind_map[kind]) for c in configs]
            means, stds = zip(*results) if results else ([], [])
            plot_vals = [v if v is not None else 0.0 for v in means]
            yerr = _make_yerr(means, stds, logy)
            ax.bar(xs, plot_vals, color=colors, width=0.8, yerr=yerr,
                   capsize=2, error_kw={"elinewidth": 0.8, "capsize": 2})
            ax.set_xticks(xs)
            ax.set_xticklabels(configs, rotation=90, fontsize=5)
            ax.tick_params(axis="y", labelsize=5)
            if logy:
                ax.set_yscale("log")
            ax.set_title(kind_labels[kind], fontsize=7.5)

        # ----- Section B: one row per transformer block (rows 1..) -----
        for i, block_name in enumerate(chunk):
            ax = fig.add_subplot(gs[1 + i, :])  # span all 3 columns
            results = [block_out_aggregated(c, block_name, steps, final_mode)
                       for c in configs]
            means, stds = zip(*results) if results else ([], [])
            plot_vals = [v if v is not None else 0.0 for v in means]
            yerr = _make_yerr(means, stds, logy)
            ax.bar(xs, plot_vals, color=colors, width=0.8, yerr=yerr,
                   capsize=2, error_kw={"elinewidth": 0.8, "capsize": 2})
            ax.set_xticks(xs)
            # Only show x-labels on the very last row of each page
            if i == n_blocks - 1:
                ax.set_xticklabels(configs, rotation=90, fontsize=5)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=5)
            if logy:
                ax.set_yscale("log")
            # Block name as y-axis label
            ax.set_ylabel(block_name, fontsize=6, rotation=0,
                          labelpad=35, ha="right", va="center")
            # Also text annotation inside the plot area (top-left)
            ax.text(0.01, 0.94, block_name, transform=ax.transAxes,
                    fontsize=5.5, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="lightyellow",
                              edgecolor="none", alpha=0.7))

        # Legend
        handles = [
            Patch(facecolor="#C44E52", label=f"baseline ('{baseline}')"),
            Patch(facecolor="#4C72B0", label="other configs"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=7,
                   bbox_to_anchor=(0.98, 0.98))
        fig.suptitle(
            f"{title}  block-level + end-to-end  (metric={args_metric}, "
            f"final_mode={final_mode})", fontsize=10)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])

        suffix = f"_p{pi + 1}" if len(starts) > 1 else ""
        path = os.path.join(out_dir, f"final_diff_grid{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(os.path.basename(path))
        print(f"figure saved: {saved[-1]}")

    return saved


# ---------------------------------------------------------------------------
# error amplification analysis
#   Why does output diff increase when param+act error decreases?
# ---------------------------------------------------------------------------
def analyze_error_amplification(param, act, out, configs, samples, steps,
                                args_metric, out_dir):
    """Analyse why output_diff can grow even when per-layer param/act errors shrink.

    Key insight — for a linear layer  y = W·x:

        quantized:  ŷ = (W + δW) · (x + δx)
        output error:  ŷ - y = W·δx + δW·x + δW·δx
                                ^^^^^^   ^^^^^^   ^^^^^^^
                               term (a)  term(b)  term(c)

    The per-element MSEs compare:
        param_mse  ≈  E[δW²]           (direct weight quantization error)
        act_mse    ≈  E[δx²]           (direct activation quantization error)
        output_mse ≈  E[(W·δx)²] + E[(δW·x)²] + cross terms

    Even tiny δW and δx get AMPLIFIED by the full weight W and input x:

        E[(W·δx)²]  ≈  (||W||_F² / d_in) · act_mse
        E[(δW·x)²]  ≈  (||x||² / d_in)   · param_mse

    The amplification factors ||W||_F²/d_in and ||x||²/d_in can be >> 1,
    causing output_diff to dwarf the individual quantization errors.

    When rotation/permutation is applied, the effective weight W_eff = R·P(W)
    changes. Even if per-element δW and δx become SMALLER after rotation,
    the alignment between W_eff and δx (or δW and x) can produce LARGER
    output error because the error structure is different.
    """
    if args_metric in ("cos", "cosine"):
        wkey = "weight_cosine"
        akey = "act_cosine"
        mkey = "cosine"       # output_diff uses "cosine" as the key
    elif args_metric == "mse":
        wkey, akey, mkey = "weight_mse", "act_mse", "mse"
    else:
        wkey, akey, mkey = "weight_mae", "act_mae", "mae"

    # -------- 1. Extract per-layer per-config triple --------
    records = []  # list of {layer, config, param_err, act_err, out_err}

    for cfg in configs:
        if cfg not in param:
            continue
        for layer in param[cfg]:
            param_err = param[cfg][layer].get(wkey)
            if param_err is None:
                continue

            # activation error: average across samples and steps
            act_vals = []
            for smp in samples:
                if str(smp) not in act:
                    continue
                cfg_act = act[str(smp)].get(cfg, {})
                for step in steps:
                    d = cfg_act.get(str(step), {})
                    v = d.get(layer + ".input", {}).get(akey)
                    if v is not None:
                        act_vals.append(v)
            act_err = _mean(act_vals)
            if act_err is None:
                continue

            # output diff: average across samples and steps from linear_layers
            denorm_layer = re.sub(r"^block\.", "transformer_blocks.", layer)
            out_vals = []
            if cfg in out:
                for smp in samples:
                    smp_cfg = out[cfg].get(str(smp), {})
                    for step in steps:
                        lin = smp_cfg.get(str(step), {}).get("linear_layers", {})
                        v = lin.get(denorm_layer, {}).get(mkey)
                        if v is not None:
                            out_vals.append(v)
            out_err = _mean(out_vals)
            if out_err is None:
                continue

            records.append({
                "layer": layer,
                "config": cfg,
                "param_err": param_err,
                "act_err": act_err,
                "out_err": out_err,
            })

    if not records:
        print("[analyze] no matched triple records; skip")
        return

    # -------- 2. Compute amplification ratio --------
    for r in records:
        combined = max(r["param_err"] + r["act_err"], 1e-30)
        r["combined"] = combined
        r["ratio"] = r["out_err"] / combined

    records.sort(key=lambda r: r["ratio"], reverse=True)

    # Print top-15 most amplified layers
    print("\n" + "=" * 80)
    print(" Top-15 layers with highest output_error / (param_error + act_error)")
    print("=" * 80)
    print(f"{'layer':<55} {'config':<30} {'param':>10} {'act':>10} {'output':>10} {'ratio':>10}")
    print("-" * 130)
    for r in records[:15]:
        print(f"{r['layer']:<55} {r['config']:<30} "
              f"{r['param_err']:>10.6f} {r['act_err']:>10.6f} "
              f"{r['out_err']:>10.6f} {r['ratio']:>10.1f}")

    # -------- 3. Scatter plot: combined(param+act) vs output --------
    by_config = {}
    for r in records:
        by_config.setdefault(r["config"], []).append(r)

    total_pts = len(records)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: all points, color by config
    all_combined = [r["combined"] for r in records]
    all_out = [r["out_err"] for r in records]
    ax1.scatter(all_combined, all_out, c="steelblue", alpha=0.5, s=12,
                edgecolors="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("param_mse + act_mse  (combined per-element error)")
    ax1.set_ylabel("output_mse  (layer output vs reference)")
    ax1.set_title(f"Log-log: combined quant error vs output diff  "
                  f"(metric={args_metric}, N={total_pts})")
    # diagonal y = 100*x reference line
    xlim = ax1.get_xlim()
    ax1.plot(xlim, [x * 100 for x in xlim], "r--", alpha=0.4,
             label="y = 100x")
    ax1.legend(fontsize=8)

    # Right: per-config amplification ratio histogram
    cfg_names = sorted(by_config.keys())
    cfg_ratios = []
    cfg_combined_avg = []
    cfg_out_avg = []
    for cfg in cfg_names:
        items = by_config[cfg]
        cfg_ratios.append(np.median([r["ratio"] for r in items]))
        cfg_combined_avg.append(np.mean([r["combined"] for r in items]))
        cfg_out_avg.append(np.mean([r["out_err"] for r in items]))

    x_idx = np.arange(len(cfg_names))
    width = 0.35
    bars1 = ax2.bar(x_idx - width / 2, cfg_combined_avg, width,
                    color="#4C72B0", label="avg(param+act)")
    bars2 = ax2.bar(x_idx + width / 2, cfg_out_avg, width,
                    color="#C44E52", label="avg(output_diff)")
    ax2.set_yscale("log")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(cfg_names, rotation=90, fontsize=5)
    ax2.set_ylabel(f"error ({args_metric})")
    ax2.set_title("Per-config: avg combined error vs avg output diff")
    ax2.legend(fontsize=7)

    fig.suptitle("Error Amplification Analysis: "
                 "Why does output_diff outgrow param+act error?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "analysis_amplification.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nfigure saved: {os.path.basename(path)}")

    # -------- 4. Per-layer amplification ratio heatmap subset --------
    # Select top-N layers with highest median ratio across configs
    layers_by_ratio = {}
    for r in records:
        layers_by_ratio.setdefault(r["layer"], []).append(r["ratio"])
    top_layers = sorted(
        layers_by_ratio.items(),
        key=lambda kv: np.median(kv[1]), reverse=True)[:20]
    top_layer_names = [l for l, _ in top_layers]

    # Build matrix: rows=layers, cols=configs, value=ratio
    ratio_matrix = np.full((len(top_layer_names), len(cfg_names)), np.nan)
    for i, layer in enumerate(top_layer_names):
        for j, cfg in enumerate(cfg_names):
            vals = [r["ratio"] for r in records
                    if r["layer"] == layer and r["config"] == cfg]
            if vals:
                ratio_matrix[i, j] = np.mean(vals)

    fig2, ax = plt.subplots(figsize=(len(cfg_names) * 1.1 + 2,
                                     len(top_layer_names) * 0.45 + 1))
    im = ax.imshow(ratio_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(cfg_names)))
    ax.set_xticklabels(cfg_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(top_layer_names)))
    ax.set_yticklabels(top_layer_names, fontsize=5.5)
    ax.set_title(f"Error amplification ratio  "
                 f"output_diff / (param+act)  "
                 f"(metric={args_metric})")
    cbar = fig2.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("amplification ratio", fontsize=7)
    # Annotate values
    for i in range(len(top_layer_names)):
        for j in range(len(cfg_names)):
            v = ratio_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=4.5, color="white" if v > 500 else "black")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "analysis_amplification_heatmap.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"figure saved: {os.path.basename(path2)}")

    # -------- 5. Per-config: param acts as proxy for δW, act as δx --------
    # Highlight: for each config, find layers where input error decreased
    # but output error increased (vs the "none_identity" baseline)
    baseline_cfg = "none_identity" if "none_identity" in configs else configs[0]
    print(f"\nBaseline config: {baseline_cfg}")

    anomalous = []
    for r in records:
        if r["config"] == baseline_cfg:
            continue
        # find matching baseline record
        bl = next((b for b in records
                   if b["layer"] == r["layer"] and b["config"] == baseline_cfg),
                  None)
        if bl is None:
            continue
        r_combined = r["param_err"] + r["act_err"]
        bl_combined = bl["param_err"] + bl["act_err"]
        if r_combined < bl_combined and r["out_err"] > bl["out_err"]:
            anomalous.append({
                "layer": r["layer"],
                "config": r["config"],
                "delta_param": r["param_err"] - bl["param_err"],
                "delta_act": r["act_err"] - bl["act_err"],
                "delta_out": r["out_err"] - bl["out_err"],
                "ratio_vs_baseline": (r["out_err"] / max(bl["out_err"], 1e-30)),
                "combined_improvement": (r_combined / bl_combined),
                "output_worsening": (r["out_err"] / bl["out_err"]),
            })

    if anomalous:
        anomalous.sort(key=lambda a: a["output_worsening"], reverse=True)
        print(f"\nFound {len(anomalous)} layer×config pairs where "
              f"param+act improve but output worsens vs baseline:")
        print(f"{'layer':<50} {'config':<25} {'combined↓':>10} "
              f"{'output↑':>10} {'worstened':>10}")
        print("-" * 115)
        for a in anomalous[:20]:
            print(f"{a['layer']:<50} {a['config']:<25} "
                  f"{a['combined_improvement']:>10.3f}x "
                  f"{a['output_worsening']:>10.3f}x "
                  f"{a['delta_out']:>+10.4f}")
    else:
        print("No anomalous cases found.")

    # -------- 6. Write explanation text --------
    text = f"""Error Amplification Analysis
===============================
Output directory: {out_dir}
Metric: {args_metric}

The question:
  Why does output_diff (per NVFP4Linear output vs reference) sometimes
  GROW even when the per-element weight and activation quantization errors
  (param_mse + act_mse) are SMALLER?

The mechanism:
  For a linear layer  y = W·x:

      quantized :  ŷ = (W + δW)·(x + δx)
      output error:  ŷ − y = W·δx + δW·x + δW·δx

  The per-element MSEs measure:
      param_mse  ≈ E[δW²]     — direct weight quantization error
      act_mse    ≈ E[δx²]     — direct activation quantization error
      output_mse ≈ E[(W·δx)²] + E[(δW·x)²] + cross terms

  The key insight: W·δx and δW·x AMPLIFY the per-element errors by
  the magnitude of the full matrices:

      E[(W·δx)²] ≈ (‖W‖₂² / d_in) · act_mse
      E[(δW·x)²] ≈ (‖x‖²  / d_in)   · param_mse

  Given ‖W‖₂ ≈ O(100) and ‖x‖ ≈ O(1–100), the amplification factor
  can be 100x–10000x. Even if δW and δx become SMALLER after applying
  rotation+permutation, the output error depends on how the error
  structure aligns with W and x, not just on its per-element magnitude.

  Rotation and permutation change the effective W_eff = R·P(W) and the
  quantization error structure δW_eff. A config may produce smaller
  per-element errors but more "coherent" error patterns that compound
  more strongly through the matrix multiplication — leading to larger
  output diff despite improved param/act MSEs.

  This is especially pronounced in deeper transformer blocks, where the
  accumulated errors from previous layers already distort x, making
  δW·x larger even if δW itself is small.
"""
    txt_path = os.path.join(out_dir, "analysis_amplification.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nanalysis text saved: {os.path.basename(txt_path)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def select_rows(rows, row_filter):
    if row_filter and row_filter != "all":
        if "," in row_filter:
            wanted = {s.strip() for s in row_filter.split(",") if s.strip()}
            rows = [r for r in rows if r in wanted]
        else:
            rx = re.compile(row_filter)
            rows = [r for r in rows if rx.search(r)]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize per-NVFP4Linear quant error (param / activation / "
                    "output) across rotation+permutation+(un)quant configs.")
    parser.add_argument("--input-dir", type=str,
                        default=r"G:\Outputs\Efficient-Diffusion"
                                r"\rot_perm_compare_module_steps_quantized",
                        help="directory containing param_quant_errors.json, "
                             "activation_errors.json, output_diff.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="where to save figures (default: input-dir)")
    parser.add_argument("--metric", type=str, default="mse",
                        choices=["mse", "mae", "cos", "cosine"],
                        help="which error metric to show")
    parser.add_argument("--linear-layers", type=str, default="all",
                        help="NVFP4Linear layer filter for layer_error_grid: "
                             "'all', a regex, or comma-separated names")
    parser.add_argument("--block-layers", type=str, default="all",
                        help="transformer block filter for final_diff_grid: "
                             "'all', a regex, or comma-separated names")
    parser.add_argument("--baseline", type=str, default="none_identity",
                        help="configuration substring to highlight as the "
                             "standard comparison (drawn with red bars)")
    parser.add_argument("--max-linear-rows", type=int, default=0,
                        help="max NVFP4Linear rows to display in "
                             "layer_error_grid; 0 = show all")
    parser.add_argument("--max-block-rows", type=int, default=0,
                        help="max block rows to display in final_diff_grid; "
                             "0 = show all")
    parser.add_argument("--max-rows-per-page", type=int, default=40,
                        help="max rows per figure page; the rest are "
                             "paginated")
    parser.add_argument("--final-mode", type=str, default="last",
                        choices=["last", "mean"],
                        help="block-out column aggregation: 'last' step's "
                             "output error, or 'mean' over all steps")
    parser.add_argument("--logy", action="store_true",
                        help="use log scale on the y-axis")
    parser.add_argument("--steps", type=str, default="all",
                        help="which decode steps to include as columns: "
                             "'all' or a comma list of ints (e.g. '0,1')")
    parser.add_argument("--title", type=str,
                        default="NVFP4 quantization error",
                        help="figure title prefix")
    parser.add_argument("--analyze", action="store_true",
                        help="also run error amplification analysis: "
                             "why does output_diff outgrow param+act error?")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"input dir not found: {args.input_dir}")
    out_dir = args.output_dir or args.input_dir
    os.makedirs(out_dir, exist_ok=True)

    param, act, out = load_inputs(args.input_dir)
    print(f"[load] param configs={len(param)} act samples={len(act)} "
          f"out configs={len(out)}")

    configs, samples = derive_configs_and_samples(param, act, out)
    full_steps = derive_steps(out, configs)
    print(f"[load] configs={configs}")
    print(f"[load] samples={samples} steps(all)={full_steps}")

    # Restrict to requested steps.
    if args.steps and args.steps != "all":
        wanted = []
        for s in args.steps.split(","):
            s = s.strip()
            if s.isdigit():
                wanted.append(int(s))
        if wanted:
            avail = set(full_steps)
            missing = [s for s in wanted if s not in avail]
            if missing:
                print(f"[warn] requested steps {missing} not present; "
                      f"available={full_steps}")
            steps = [s for s in wanted if s in avail]
    else:
        steps = full_steps
    if not steps:
        raise SystemExit("no decode steps found in output_diff.json")

    # Build resolvers.
    (param_value, act_value, linear_out_value,
     block_out_value, block_out_aggregated, final_value) = make_resolver(
        param, act, out, samples, args.metric)

    # ==================================================================
    # Figure 1: layer_error_grid (NVFP4Linear-level)
    # ==================================================================
    linear_layers = derive_linear_layer_names(param)
    linear_layers = select_rows(linear_layers, args.linear_layers)
    if not linear_layers:
        print("[warn] no NVFP4Linear layers found in param_quant_errors; "
              "skipping layer_error_grid")
    else:
        if args.max_linear_rows and args.max_linear_rows > 0:
            if args.max_linear_rows < len(linear_layers):
                print(f"[plot] displaying first {args.max_linear_rows} of "
                      f"{len(linear_layers)} matched NVFP4Linear rows")
            linear_layers = linear_layers[:args.max_linear_rows]

        plot_layer_error_grid(
            linear_layers, param_value, act_value, linear_out_value,
            steps, configs, out_dir, args.baseline, args.logy,
            args.title, args.metric, args.max_rows_per_page)

    # ==================================================================
    # Figure 2: final_diff_grid (block-level + end-to-end)
    # ==================================================================
    if out.get("final_diff"):
        block_names = derive_block_names(out)
        block_names = select_rows(block_names, args.block_layers)
        if not block_names:
            print("[warn] no transformer blocks found in output_diff layers; "
                  "skipping final_diff_grid")
        else:
            if args.max_block_rows and args.max_block_rows > 0:
                if args.max_block_rows < len(block_names):
                    print(f"[plot] displaying first {args.max_block_rows} of "
                          f"{len(block_names)} matched block rows")
                block_names = block_names[:args.max_block_rows]

            plot_final_diff_grid(
                block_names, block_out_aggregated,
                final_value, steps, configs, out_dir, args.baseline,
                args.logy, args.title, args.metric,
                args.max_rows_per_page, args.final_mode)
    else:
        print("[skip] no 'final_diff' in output_diff.json; "
              "skip final_diff_grid")

    # ==================================================================
    # Optional: error amplification analysis
    # ==================================================================
    if args.analyze:
        print("\n" + "=" * 80)
        print(" Running error amplification analysis")
        print("=" * 80)
        analyze_error_amplification(
            param, act, out, configs, samples, steps,
            args.metric, out_dir)
