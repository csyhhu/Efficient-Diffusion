"""Analyze whether most NVFP4Linear layers have smaller quant error but larger output diff.

Compares each config against baseline (default: none_identity) for:
  - param error change:  weight_mse (and weight_cosine if available)
  - act error change:    act_mse    (and act_cosine if available)
  - output diff change:  output mse (and cosine if available)

Reports per-layer statistics of "improved quant error but worsened output diff".
"""

import json
import os
import re
import argparse
import numpy as np
from collections import defaultdict


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def load_all(input_dir):
    def _load(name):
        p = os.path.join(input_dir, name)
        if not os.path.isfile(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return _load("param_quant_errors.json"), _load("activation_errors.json"), _load("output_diff.json")


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze param/act quant error vs output diff")
    parser.add_argument("--input-dir", type=str,
                        default=r"G:\Outputs\Efficient-Diffusion\rot_perm_compare_module_steps_quantized")
    parser.add_argument("--baseline", type=str, default="none_identity")
    parser.add_argument("--metric", type=str, default="mse",
                        choices=["mse", "mae", "cosine"])
    parser.add_argument("--top-n", type=int, default=30,
                        help="show top N most anomalous layers")
    args = parser.parse_args()

    param, act, out = load_all(args.input_dir)
    if param is None:
        raise SystemExit(f"param_quant_errors.json not found in {args.input_dir}")

    # --- Determine key names ---
    if args.metric == "cosine":
        wkey, akey, mkey = "weight_cosine", "act_cosine", "cosine"
    elif args.metric == "mse":
        wkey, akey, mkey = "weight_mse", "act_mse", "mse"
    else:
        wkey, akey, mkey = "weight_mae", "act_mae", "mae"

    configs = sorted(set(param.keys()) | set(out.keys()) - {"final_diff"})
    if args.baseline not in configs:
        print(f"[warn] baseline '{args.baseline}' not in configs; configs={configs}")
        baseline = configs[0]
    else:
        baseline = args.baseline

    # Gather samples
    samples = set()
    for smp in act:
        try: samples.add(int(smp))
        except: pass
    for cfg_name, cfg in out.items():
        if cfg_name == "final_diff": continue
        for smp in cfg:
            try: samples.add(int(smp))
            except: pass
    samples = sorted(samples)

    # Gather steps
    steps = set()
    for cfg in configs:
        if cfg not in out: continue
        for smp in out[cfg]:
            for s in out[cfg][smp]:
                try: steps.add(int(s))
                except: pass
    steps = sorted(steps)

    print(f"Baseline: {baseline}")
    print(f"Configs: {configs}")
    print(f"Samples: {samples}")
    print(f"Steps: {steps}")
    print(f"Metric: {args.metric}  (wkey={wkey}, akey={akey}, mkey={mkey})")
    print()

    # --- Collect per-layer data ---
    # For each config, for each layer:
    #   param_err: direct from param json
    #   act_err: averaged across samples and steps
    #   out_err: averaged across samples and steps from output_diff linear_layers

    all_records = []  # dicts

    for cfg in configs:
        if cfg not in param:
            print(f"  [skip] config '{cfg}' not in param data")
            continue

        for layer in sorted(param[cfg].keys(), key=natural_key):
            p_val = param[cfg][layer].get(wkey)
            if p_val is None:
                continue

            # Activation error: average across samples, steps
            a_vals = []
            for smp in samples:
                s_cfg = act.get(str(smp), {}).get(cfg, {})
                for s in steps:
                    d = s_cfg.get(str(s), {})
                    v = d.get(layer + ".input", {}).get(akey)
                    if v is not None:
                        a_vals.append(v)
            a_val = mean(a_vals)
            if a_val is None:
                continue

            # Output diff: from linear_layers
            denorm_layer = re.sub(r"^block\.", "transformer_blocks.", layer)
            o_vals = []
            if cfg in out:
                for smp in samples:
                    smp_cfg = out[cfg].get(str(smp), {})
                    for s in steps:
                        lin = smp_cfg.get(str(s), {}).get("linear_layers", {})
                        v = lin.get(denorm_layer, {}).get(mkey)
                        if v is not None:
                            o_vals.append(v)
            o_val = mean(o_vals)
            if o_val is None:
                continue

            all_records.append({
                "layer": layer,
                "config": cfg,
                "param_err": p_val,
                "act_err": a_val,
                "out_err": o_val,
            })

    print(f"Total records (layer x config): {len(all_records)}")

    if not all_records:
        print("[error] no matched records")
        return

    # --- Build baseline lookup ---
    baseline_lookup = {}
    for r in all_records:
        if r["config"] == baseline:
            baseline_lookup[r["layer"]] = r

    # --- Compare each non-baseline record ---
    # For each (layer, config), compute delta vs baseline
    improved_but_worsened = []   # param+act improved but output worsened
    both_improved       = []     # both improved
    both_worsened       = []     # both worsened
    worsened_but_improved = []   # param+act worsened but output improved

    for r in all_records:
        if r["config"] == baseline:
            continue
        bl = baseline_lookup.get(r["layer"])
        if bl is None:
            continue

        # combined = param + act (for MSE/MAE) or (1 - cosine) for distance
        if args.metric == "cosine":
            # cosine: 1.0 = identical, so error = 1 - cosine
            r_combined = (1.0 - r["param_err"]) + (1.0 - r["act_err"])
            bl_combined = (1.0 - bl["param_err"]) + (1.0 - bl["act_err"])
        else:
            r_combined = r["param_err"] + r["act_err"]
            bl_combined = bl["param_err"] + bl["act_err"]

        d_param = r["param_err"] - bl["param_err"]
        if args.metric == "cosine":
            d_act = r["act_err"] - bl["act_err"]  # positive = improved (closer to 1)
        else:
            d_act = r["act_err"] - bl["act_err"]  # positive = worsened

        d_out = r["out_err"] - bl["out_err"]
        if args.metric == "cosine":
            d_out = r["out_err"] - bl["out_err"]  # positive = improved
        else:
            d_out = r["out_err"] - bl["out_err"]  # positive = worsened

        quant_improved = r_combined < bl_combined   # combined error decreased
        out_improved = (r["out_err"] < bl["out_err"]) if args.metric != "cosine" else (r["out_err"] > bl["out_err"])
        # For cosine: higher cosine = better, for mse/mae: lower = better
        if args.metric == "cosine":
            quant_improved = d_param + d_act > 0  # combined cosine distance decreased
            out_improved = d_out > 0

        record = {
            **r,
            "delta_param": d_param,
            "delta_act": d_act,
            "delta_out": d_out,
            "quant_improved": quant_improved,
            "out_improved": out_improved,
            "combined_ratio": (r_combined / max(bl_combined, 1e-30)),
            "out_ratio": (r["out_err"] / max(bl["out_err"], 1e-30)),
        }

        if quant_improved and not out_improved:
            improved_but_worsened.append(record)
        elif quant_improved and out_improved:
            both_improved.append(record)
        elif not quant_improved and not out_improved:
            both_worsened.append(record)
        elif not quant_improved and out_improved:
            worsened_but_improved.append(record)

    # --- Print summary ---
    total = len(improved_but_worsened) + len(both_improved) + len(both_worsened) + len(worsened_but_improved)
    print(f"\n{'='*90}")
    print(f" Summary: param/act quant error vs output diff  (metric={args.metric})")
    print(f" Baseline: {baseline}")
    print(f"{'='*90}")
    print(f"  {'Category':<50} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*66}")
    print(f"  quant IMPROVED, output WORSENED      (anomalous) -->  {len(improved_but_worsened):>8}  {len(improved_but_worsened)/total*100:>7.1f}%")
    print(f"  quant IMPROVED, output IMPROVED      (good)       -->  {len(both_improved):>8}  {len(both_improved)/total*100:>7.1f}%")
    print(f"  quant WORSENED, output WORSENED      (bad)        -->  {len(both_worsened):>8}  {len(both_worsened)/total*100:>7.1f}%")
    print(f"  quant WORSENED, output IMPROVED                    -->  {len(worsened_but_improved):>8}  {len(worsened_but_improved)/total*100:>7.1f}%")
    print(f"  {'-'*66}")
    print(f"  Total                                            -->  {total:>8}")
    print()

    # --- Per-config breakdown ---
    print(f"{'='*90}")
    print(f" Per-config breakdown: anomalous pairs (quant improved, output worsened)")
    print(f"{'='*90}")
    by_cfg = defaultdict(list)
    for r in improved_but_worsened:
        by_cfg[r["config"]].append(r)
    for cfg in sorted(by_cfg.keys()):
        items = by_cfg[cfg]
        print(f"\n  Config: {cfg}  ({len(items)} anomalous layers)")
        items.sort(key=lambda r: r["out_ratio"], reverse=True)
        print(f"  {'layer':<55} {'delta_param':>12} {'delta_act':>12} {'delta_out':>12} {'out_ratio':>10} {'comb_ratio':>10}")
        print(f"  {'-'*115}")
        for r in items[:args.top_n]:
            print(f"  {r['layer']:<55} {r['delta_param']:>12.6f} {r['delta_act']:>12.6f} {r['delta_out']:>12.6f} {r['out_ratio']:>10.3f} {r['combined_ratio']:>10.3f}")

    # --- Per-config also show how many layers improved vs worsened overall ---
    print(f"\n{'='*90}")
    print(f" Per-config: overall param/act/output trends (vs {baseline})")
    print(f"{'='*90}")
    by_cfg_all = defaultdict(list)
    for r in all_records:
        if r["config"] == baseline:
            continue
        bl = baseline_lookup.get(r["layer"])
        if bl is None:
            continue
        r_combined = r["param_err"] + r["act_err"]
        bl_combined = bl["param_err"] + bl["act_err"]
        if args.metric == "cosine":
            r_combined = (1 - r["param_err"]) + (1 - r["act_err"])
            bl_combined = (1 - bl["param_err"]) + (1 - bl["act_err"])
        by_cfg_all[r["config"]].append({
            **r,
            "quant_improved": r_combined < bl_combined,
            "out_improved": (r["out_err"] < bl["out_err"]) if args.metric != "cosine" else (r["out_err"] > bl["out_err"]),
        })

    for cfg in sorted(by_cfg_all.keys()):
        items = by_cfg_all[cfg]
        n_quant_imp = sum(1 for r in items if r["quant_improved"])
        n_out_imp = sum(1 for r in items if r["out_improved"])
        n_anomalous = sum(1 for r in items if r["quant_improved"] and not r["out_improved"])
        print(f"  {cfg:<45}  quant_imp={n_quant_imp}/{len(items)} ({n_quant_imp/len(items)*100:.0f}%)  "
              f"out_imp={n_out_imp}/{len(items)} ({n_out_imp/len(items)*100:.0f}%)  "
              f"anomalous={n_anomalous}")

    # --- Check: is the pattern "param/act IMPROVED but output worsened" the DOMINANT pattern? ---
    if len(improved_but_worsened) > len(both_improved):
        print(f"\n{'!'*90}")
        print(f" CONCLUSION: YES - anomalous cases ({len(improved_but_worsened)}) DOMINATE over")
        print(f" good cases ({len(both_improved)}). Most layers have IMPROVED quant error")
        print(f" but WORSENED output diff. This confirms the hypothesized error amplification.")
        print(f"{'!'*90}")
    elif len(improved_but_worsened) > total * 0.2:
        print(f"\n{'!'*90}")
        print(f" CONCLUSION: Anomalous cases ({len(improved_but_worsened)}, {len(improved_but_worsened)/total*100:.1f}%)")
        print(f" are SIGNIFICANT but not a majority. The error amplification effect exists but is not universal.")
        print(f"{'!'*90}")
    else:
        print(f"\n CONCLUSION: Anomalous cases ({len(improved_but_worsened)}, {len(improved_but_worsened)/total*100:.1f}%)")
        print(f" are relatively RARE. Quant error and output diff are mostly consistent in direction.")


if __name__ == "__main__":
    main()
