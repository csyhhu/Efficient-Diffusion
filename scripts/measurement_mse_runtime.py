"""
多轮计时 & MSE 评测: 给定多个 pipeline 方案, 多轮运行并统计耗时/MSE。

用法:
    python scripts/measurement_mse_runtime.py --num_runs 5 --ddim_steps 10
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from src.model_loader import load_model
from src.pipeline_builder import build_baseline, build_quant, build_ddim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="多轮计时 & MSE 评测")
    parser.add_argument("--model", type=str, default="sd", choices=["sd", "sdxl"])
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--ddim_steps", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./outputs/images")
    parser.add_argument("--mirror", type=str, default="https://hf-mirror.com")
    parser.add_argument("--local_path", type=str, default=None)
    parser.add_argument("--prompt", type=str,
                        default="A serene landscape with mountains and a lake at sunset, highly detailed")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 推理 & 计时
# ---------------------------------------------------------------------------

def run_one(pipe, prompt: str, steps: int, seed: int, warmup: bool = True):
    """运行一次推理，返回 (image_array, elapsed_seconds)。"""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad():
        if warmup:
            _ = pipe(prompt, num_inference_steps=min(3, steps), generator=generator)

        if pipe.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = pipe(prompt, num_inference_steps=steps, generator=generator)

        if pipe.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return np.array(result.images[0], dtype=np.float32), elapsed


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def compute_stats(values):
    arr = np.array(values)
    return arr.mean(), arr.std(ddof=1), arr.min(), arr.max()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("Efficient-Diffusion — 多次计时 & MSE 对比")
    print("=" * 70)
    print(f"  模型: {args.model} | 每方案实验次数: {args.num_runs}")
    print()

    # 1. 加载模型
    pipe, device = load_model(
        model_name=args.model,
        mirror=args.mirror,
        local_path=args.local_path,
    )

    # 2. 构建方案
    print("[2] 构建方案: baseline / torchao INT8量化 / DDIM")
    methods = [
        ("baseline", build_baseline(pipe), args.num_inference_steps),
        ("quant",    build_quant(pipe),    args.num_inference_steps),
        ("ddim",     build_ddim(pipe),     args.ddim_steps),
    ]

    # 3. 多轮实验
    all_times = {name: [] for name, _, _ in methods}
    all_mses = {name: [] for name, _, _ in methods}

    for run_idx in range(args.num_runs):
        seed = 42 + run_idx
        print(f"\n--- 第 {run_idx + 1}/{args.num_runs} 轮 (seed={seed}) ---")

        round_images = {}
        for name, p, steps in methods:
            img_arr, elapsed = run_one(p, args.prompt, steps, seed)
            all_times[name].append(elapsed)
            round_images[name] = img_arr

            if name == "baseline":
                print(f"  [{name:<8}] {steps:>3} steps | {elapsed:.3f}s")
            else:
                speedup = all_times["baseline"][-1] / elapsed
                mse = ((img_arr - round_images["baseline"]) ** 2).mean()
                all_mses[name].append(mse)
                print(f"  [{name:<8}] {steps:>3} steps | {elapsed:.3f}s | 加速 {speedup:.1f}x | MSE={mse:.1f}")

    # 4. 汇总
    print()
    print("=" * 70)
    print("统计汇总")
    print("=" * 70)
    header = f"{'方案':<10} {'步数':<6} {'耗时均值':<10} {'耗时std':<10} {'耗时范围':<18} {'加速比':<8} {'MSE均值':<12} {'MSE std':<12}"
    print(header)
    print("-" * 70)

    baseline_mean_time = np.mean(all_times["baseline"])
    for name, _, steps in methods:
        times = all_times[name]
        t_mean, t_std, t_min, t_max = compute_stats(times)
        speedup = baseline_mean_time / t_mean

        if name == "baseline":
            print(f"{name:<10} {steps:<6} {t_mean:.3f}s{'':<4} {t_std:.3f}s{'':<4} [{t_min:.3f}, {t_max:.3f}]   {'--':<8} {'--':<12} {'--':<12}")
        else:
            mses = all_mses[name]
            m_mean, m_std, _, _ = compute_stats(mses)
            print(f"{name:<10} {steps:<6} {t_mean:.3f}s{'':<4} {t_std:.3f}s{'':<4} [{t_min:.3f}, {t_max:.3f}]   {speedup:<8.1f}x {m_mean:<12.1f} {m_std:<12.1f}")

    print("=" * 70)

    # 5. 保存最后一轮图像
    os.makedirs(args.output_dir, exist_ok=True)
    for name, p, steps in methods:
        img_arr, _ = run_one(p, args.prompt, steps, 42)
        Image.fromarray(img_arr.astype(np.uint8)).save(
            os.path.join(args.output_dir, f"{args.model}_{name}.png")
        )

    print(f"\n图像已保存至: {args.output_dir}/")
    for name in ["baseline", "quant", "ddim"]:
        print(f"  {args.model}_{name}.png")

    # 6. GPU 显存
    if device == "cuda":
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\nGPU 显存: 峰值分配 {allocated:.2f} GB, 峰值预留 {reserved:.2f} GB")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
