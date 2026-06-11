"""
INT8 vs INT4 量化对比: torchao 默认 + 自定义 alpha 扫描 (0.9 → 0.7)。

用法:
    python scripts/quant_comparison.py
    python scripts/quant_comparison.py --alpha_min 0.7 --alpha_max 0.9 --alpha_step 0.05
    python scripts/quant_comparison.py --perceptual  # 启用 LPIPS

输出:
    - outputs/quant_viz/comparison_curve.png  → INT8/INT4 双线对比曲线
    - outputs/quant_viz/baseline.png          → baseline 图像
    - outputs/quant_viz/int8_alpha_*.png      → INT8 各 alpha 图像
    - outputs/quant_viz/int4_alpha_*.png      → INT4 各 alpha 图像
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from src.model_loader import load_model
from src.pipeline_builder import (
    build_baseline,
    build_quant_int8,
    build_quant_int4,
    apply_alpha_int8,
    apply_alpha_int4,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="INT8 vs INT4 量化对比")
    parser.add_argument("--model", type=str, default="sd", choices=["sd", "sdxl"])
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/INT4_quant")
    parser.add_argument("--mirror", type=str, default="https://hf-mirror.com")
    parser.add_argument("--local_path", type=str, default=None)
    parser.add_argument("--prompt", type=str,
                        default="A serene landscape with mountains and a lake at sunset, highly detailed")
    parser.add_argument("--alpha_min", type=float, default=0.7,
                        help="alpha 扫描下限 (默认: 0.7)")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="alpha 扫描上限 (默认: 0.9)")
    parser.add_argument("--alpha_step", type=float, default=0.1,
                        help="alpha 步长 (默认: 0.05)")
    parser.add_argument("--perceptual", action="store_true", default=False,
                        help="启用 LPIPS 感知指标 (需要 pip install lpips)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_mse(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    return float(((img_ref - img_test) ** 2).mean())


def compute_ssim(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    ref_u8 = img_ref.clip(0, 255).astype(np.uint8)
    test_u8 = img_test.clip(0, 255).astype(np.uint8)
    return float(ssim(ref_u8, test_u8, channel_axis=2, data_range=255))


def compute_lpips(img_ref: np.ndarray, img_test: np.ndarray, lpips_fn) -> float:
    ref_t = torch.from_numpy(img_ref).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    test_t = torch.from_numpy(img_test).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    with torch.no_grad():
        dist = lpips_fn(ref_t, test_t)
    return float(dist.item())


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def run_single(pipe, prompt: str, steps: int, seed: int):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad():
        result = pipe(prompt, num_inference_steps=steps, generator=generator)
    return np.array(result.images[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- LPIPS ---
    lpips_fn = None
    if args.perceptual:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").eval()
            print("[LPIPS] AlexNet 模型加载成功")
        except ImportError:
            print("[警告] lpips 未安装, 跳过 LPIPS。安装: pip install lpips")
            args.perceptual = False

    # --- 加载模型 ---
    print("=" * 70)
    print("INT8 vs INT4 量化对比")
    print("=" * 70)
    pipe, device = load_model(
        model_name=args.model,
        mirror=args.mirror,
        local_path=args.local_path,
    )
    print(f"  Prompt: {args.prompt}")
    print(f"  Seed: {args.seed} | Steps: {args.num_inference_steps}")
    print()

    # --- baseline ---
    print("[1] 生成 baseline ...")
    img_baseline = run_single(pipe, args.prompt, args.num_inference_steps, args.seed)
    Image.fromarray(img_baseline.astype(np.uint8)).save(
        os.path.join(args.output_dir, "baseline.png")
    )
    print("    完成\n")

    # --- torchao 默认量化 ---
    # 注意: torchao INT4 (version=2) 需要 mslk CUDA kernel, CPU 上不可用
    # torchao INT8 已在之前的 measurement_mse_runtime.py 中评测过
    print("[2] torchao 默认量化 (已跳过, CPU 环境不可用)")
    print("-" * 60)

    default_results = {}  # {label: (mse, ssim, lpips, elapsed)}

    print()

    # --- Alpha 扫描 (INT8 + INT4) ---
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    alphas = [round(a, 4) for a in alphas]  # 避免浮点误差
    print(f"[3] Alpha 扫描 ({len(alphas)} 个点: {alphas[0]:.2f} → {alphas[-1]:.2f})")
    print("-" * 60)

    # 存储: [(alpha, mse, ssim, lpips, elapsed), ...]
    results_int8 = []   # INT8 alpha 扫描已跳过, 保留空列表用于后续 zip
    results_int4 = []

    for alpha in alphas:
        alpha = round(alpha, 2)

        # --- INT8 alpha ---
        # print(f"  INT8 alpha={alpha:.2f} ...", end=" ", flush=True)
        # pipe_q8 = apply_alpha_int8(pipe, alpha=alpha)
        # t0 = time.perf_counter()
        # img_q8 = run_single(pipe_q8, args.prompt, args.num_inference_steps, args.seed)
        # elapsed = time.perf_counter() - t0
        # mse = compute_mse(img_baseline, img_q8)
        # ssim = compute_ssim(img_baseline, img_q8)
        # lpips_val = compute_lpips(img_baseline, img_q8, lpips_fn) if lpips_fn else float("nan")
        # results_int8.append((alpha, mse, ssim, lpips_val, elapsed))
        # Image.fromarray(img_q8.astype(np.uint8)).save(
        #     os.path.join(args.output_dir, f"int8_alpha_{alpha:.2f}.png")
        # )
        # lpips_str = f"LPIPS={lpips_val:.4f}" if lpips_fn else ""
        # print(f"MSE={mse:.1f} SSIM={ssim:.4f} {lpips_str} | {elapsed:.1f}s")
        # del pipe_q8

        # --- INT4 alpha ---
        print(f"  INT4 alpha={alpha:.2f} ...", end=" ", flush=True)
        pipe_q4 = apply_alpha_int4(pipe, alpha=alpha)
        t0 = time.perf_counter()
        img_q4 = run_single(pipe_q4, args.prompt, args.num_inference_steps, args.seed)
        elapsed = time.perf_counter() - t0
        mse = compute_mse(img_baseline, img_q4)
        ssim = compute_ssim(img_baseline, img_q4)
        lpips_val = compute_lpips(img_baseline, img_q4, lpips_fn) if lpips_fn else float("nan")
        results_int4.append((alpha, mse, ssim, lpips_val, elapsed))
        Image.fromarray(img_q4.astype(np.uint8)).save(
            os.path.join(args.output_dir, f"int4_alpha_{alpha:.2f}.png")
        )
        lpips_str = f"LPIPS={lpips_val:.4f}" if lpips_fn else ""
        print(f"MSE={mse:.1f} SSIM={ssim:.4f} {lpips_str} | {elapsed:.1f}s")
        del pipe_q4

        print()

    # --- 绘制对比曲线 ---
    print("[4] 绘制对比曲线 ...")
    alphas_arr = np.array([r[0] for r in results_int4])

    n_plots = 3 if lpips_fn else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 2:
        axes = [axes[0], axes[1]]

    # 只画 INT4 (INT8 alpha 扫描已跳过)
    plot_data = [results_int4]
    labels = ["INT4"]
    colors = ["tab:orange"]
    markers = ["s"]

    for results, label, color, marker in zip(plot_data, labels, colors, markers):
        mses = np.array([r[1] for r in results])
        ssims = np.array([r[2] for r in results])

        # MSE
        axes[0].plot(alphas_arr, mses, f"{marker}-", color=color, markersize=6,
                     label=label)
        # SSIM
        axes[1].plot(alphas_arr, ssims, f"{marker}-", color=color, markersize=6,
                     label=label)

        # LPIPS
        if lpips_fn:
            lpips_vals = np.array([r[3] for r in results])
            axes[2].plot(alphas_arr, lpips_vals, f"{marker}-", color=color, markersize=6,
                         label=label)

    # 标注 torchao 默认值 (已跳过, 不画虚线)
    # 后续可补充 torchao INT8 和 INT4 的参考线

    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE vs alpha (Lower the Better)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].invert_xaxis()

    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs alpha (Higher the Better)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].invert_xaxis()

    if lpips_fn:
        axes[2].set_xlabel("alpha")
        axes[2].set_ylabel("LPIPS")
        axes[2].set_title("LPIPS vs alpha (Lower the Better)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)
        axes[2].invert_xaxis()

    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, "comparison_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    曲线已保存: {curve_path}")

    # --- 汇总表格 ---
    print()
    print("=" * 80)
    print("Alpha 扫描汇总 (INT4)")
    print("=" * 80)
    header = f"{'alpha':<8} {'MSE':<12} {'SSIM':<10} "
    if lpips_fn:
        header += f"{'LPIPS':<10} "
    header += f"{'耗时':<10}"
    print(header)
    print("-" * 80)
    for alpha, mse, ssim, lpips_val, elapsed in results_int4:
        line = f"{alpha:<8.2f} {mse:<12.1f} {ssim:<10.4f} "
        if lpips_fn:
            line += f"{lpips_val:<10.4f} "
        line += f"{elapsed:<10.1f}s"
        print(line)
    print("=" * 80)

    print(f"\n输出目录: {os.path.abspath(args.output_dir)}/")
    print("  baseline.png  torchao_int8.png  torchao_int4.png")
    print("  int8_alpha_*.png  int4_alpha_*.png  comparison_curve.png")


if __name__ == "__main__":
    main()
