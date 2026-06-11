"""
本地验证脚本: 不依赖网络，纯本地 UNet2DModel 验证 baseline / 量化 / 算法加速。
多次实验，统计耗时与 MSE(相比 baseline) 的均值和方差。

用法:
    python scripts/demo_local.py                           # 默认 10 次实验
    python scripts/demo_local.py --num_runs 5              # 5 次实验
    python scripts/demo_local.py --ddim_steps 20           # DDIM 20 步
"""

import argparse
import copy
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, DDIMScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Local Demo (no network)")
    parser.add_argument("--ddim_steps", type=int, default=20,
                        help="DDIM 采样步数 (默认: 20, 越少越快)")
    parser.add_argument("--num_inference_steps", type=int, default=100,
                        help="baseline DDPM 推理步数 (默认: 100)")
    parser.add_argument("--image_size", type=int, default=32,
                        help="图像尺寸 (默认: 32)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="每个方案的实验次数 (默认: 10)")
    # --- alpha 扫描: scale = alpha * max(|W|) ---
    parser.add_argument("--alpha", type=float, default=None,
                        help="INT8 量化 scale 系数 alpha: scale = alpha * max(|W|) / 127")
    parser.add_argument("--alpha_scan", action="store_true", default=False,
                        help="扫描 alpha 从 1.0 到 0.1, 输出 MSE-耗时曲线")
    parser.add_argument("--alpha_steps", type=int, default=10,
                        help="alpha 扫描步数 (默认: 10, 即 1.0, 0.9, ..., 0.1)")
    return parser.parse_args()


def create_tiny_unet(image_size: int = 32):
    return UNet2DModel(
        sample_size=image_size, in_channels=3, out_channels=3,
        layers_per_block=2, block_out_channels=(64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


# ---------------------------------------------------------------------------
# 加速方案实现
# ---------------------------------------------------------------------------

def apply_baseline(unet):
    """不做任何优化，标准 DDPM"""
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipe = DDPMPipeline(unet=unet, scheduler=scheduler)
    return pipe


def apply_quantization(unet, alpha=1.0):
    """torchao INT8 权重量化 (alpha=1.0 即默认)"""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    quantize_(unet, Int8WeightOnlyConfig())
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipe = DDPMPipeline(unet=unet, scheduler=scheduler)
    return pipe


def apply_alpha_quantization(unet, alpha: float):
    """自定义 per-channel INT8 量化: scale = alpha * max(|W|) / 127

    alpha = 1.0  → 等价于 torchao 默认 (无裁剪)
    alpha < 1.0  → scale 更小, INT8 范围覆盖更多离群值被裁剪

    返回新的 UNet2DModel (深拷贝后量化), 不修改原模型。
    """
    unet_q = UNet2DModel(
        sample_size=unet.config.sample_size,
        in_channels=unet.config.in_channels,
        out_channels=unet.config.out_channels,
        layers_per_block=unet.config.layers_per_block,
        block_out_channels=unet.config.block_out_channels,
        down_block_types=unet.config.down_block_types,
        up_block_types=unet.config.up_block_types,
    )

    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(unet.named_parameters(), unet_q.named_parameters()):
            w = p1.data.float()
            if w.ndim < 2:
                # bias / 1D 参数不量化, 直接复制
                p2.copy_(p1)
                continue
            # per-channel: 沿 dim=0 对每个输出通道独立计算 max(|W|)
            w_abs_max = w.abs().amax(dim=list(range(1, w.ndim)), keepdim=True)
            # scale = alpha * max(|W|) / 127
            scale = alpha * w_abs_max / 127.0
            scale = scale.clamp(min=1e-8)
            # 量化 + 反量化
            w_q = torch.round(w / scale).clamp(-128, 127) * scale
            p2.copy_(w_q.to(p1.dtype))

    return unet_q


def apply_ddim(unet):
    """DDIM 算法加速"""
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    pipe = DDPMPipeline(unet=unet, scheduler=scheduler)
    return pipe


# ---------------------------------------------------------------------------
# 推理 & 计时
# ---------------------------------------------------------------------------

def run_one(pipe, steps: int, seed: int, warmup: bool = True):
    """运行一次推理，返回 (image_array, elapsed_seconds)"""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if warmup:
            _ = pipe(batch_size=1, num_inference_steps=min(5, steps), generator=generator)
        t0 = time.perf_counter()
        result = pipe(batch_size=1, num_inference_steps=steps, generator=generator)
        elapsed = time.perf_counter() - t0
    return np.array(result.images[0], dtype=np.float32), elapsed


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def compute_stats(values):
    """返回 (mean, std, min, max)"""
    arr = np.array(values)
    return arr.mean(), arr.std(ddof=1), arr.min(), arr.max()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. 创建模型
    unet = create_tiny_unet(args.image_size)
    n_params = sum(p.numel() for p in unet.parameters()) / 1e6

    print("=" * 70)
    print("Efficient-Diffusion — 多次对比实验")
    print("=" * 70)
    print(f"  UNet 参数量: {n_params:.2f}M | 图像尺寸: {args.image_size}x{args.image_size}")
    print(f"  每方案实验次数: {args.num_runs}")
    print()

    # 2. 构建 baseline 和 ddim
    pipe_baseline = apply_baseline(unet)
    pipe_ddim = apply_ddim(unet)
    ddim_steps = args.ddim_steps

    # ---- alpha 扫描模式 ----
    if args.alpha_scan:
        alphas = np.linspace(1.0, 0.1, args.alpha_steps)
        print(f"Alpha 扫描: {alphas}")
        print()

        all_alpha_results = []  # [(alpha, t_mean, t_std, m_mean, m_std)]

        for alpha in alphas:
            alpha = round(alpha, 2)
            print(f"--- alpha={alpha:.2f} ---")

            unet_q = apply_alpha_quantization(unet, alpha=alpha)
            scheduler = DDPMScheduler(num_train_timesteps=1000)
            pipe_quant = DDPMPipeline(unet=unet_q, scheduler=scheduler)

            times = []
            mses = []
            for run_idx in range(args.num_runs):
                seed = 42 + run_idx
                img_base, _ = run_one(pipe_baseline, args.num_inference_steps, seed)
                img_q, elapsed = run_one(pipe_quant, args.num_inference_steps, seed)
                times.append(elapsed)
                mse = ((img_q - img_base) ** 2).mean()
                mses.append(mse)

            t_mean, t_std, _, _ = compute_stats(times)
            m_mean, m_std, _, _ = compute_stats(mses)
            all_alpha_results.append((alpha, t_mean, t_std, m_mean, m_std))

            print(f"  耗时: {t_mean:.3f}s ± {t_std:.3f}s | MSE: {m_mean:.1f} ± {m_std:.1f}")

        # 汇总 alpha 扫描结果
        print()
        print("=" * 70)
        print("Alpha 扫描汇总")
        print("=" * 70)
        print(f"{'alpha':<8} {'耗时均值':<10} {'耗时std':<10} {'MSE均值':<12} {'MSE std':<12}")
        print("-" * 70)
        for alpha, t_mean, t_std, m_mean, m_std in all_alpha_results:
            print(f"{alpha:<8.2f} {t_mean:.3f}s{'':<4} {t_std:.3f}s{'':<4} {m_mean:<12.1f} {m_std:<12.1f}")
        print("=" * 70)
        return

    # ---- 单 alpha 模式 / 默认三方案对比 ----
    if args.alpha is not None:
        # 单 alpha: 自定义量化
        unet_q = apply_alpha_quantization(unet, alpha=args.alpha)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        pipe_quant = DDPMPipeline(unet=unet_q, scheduler=scheduler)
        quant_label = f"quant(a={args.alpha:.2f})"
    else:
        # 默认: torchao 量化
        unet_quant = copy.deepcopy(unet)
        pipe_quant = apply_quantization(unet_quant)
        quant_label = "quant"

    methods = [
        ("baseline", pipe_baseline, args.num_inference_steps),
        (quant_label, pipe_quant, args.num_inference_steps),
        ("ddim",     pipe_ddim,     ddim_steps),
    ]

    # 3. 多轮实验
    all_times = {name: [] for name, _, _ in methods}
    all_mses = {name: [] for name, _, _ in methods}

    for run_idx in range(args.num_runs):
        seed = 42 + run_idx
        print(f"--- 第 {run_idx + 1}/{args.num_runs} 轮 (seed={seed}) ---")

        round_images = {}
        for name, pipe, steps in methods:
            img_arr, elapsed = run_one(pipe, steps, seed)
            all_times[name].append(elapsed)
            round_images[name] = img_arr

            if name == "baseline":
                print(f"  [{name:<8}] {steps:>3} steps | {elapsed:.3f}s")
            else:
                speedup = all_times["baseline"][-1] / elapsed
                mse = ((img_arr - round_images["baseline"]) ** 2).mean()
                all_mses[name].append(mse)
                print(f"  [{name:<8}] {steps:>3} steps | {elapsed:.3f}s | 加速 {speedup:.1f}x | MSE={mse:.1f}")

    # 4. 汇总统计
    print()
    print("=" * 70)
    print("统计汇总")
    print("=" * 70)

    header = f"{'方案':<12} {'步数':<6} {'耗时均值':<10} {'耗时std':<10} {'耗时范围':<16} {'加速比':<8} {'MSE均值':<12} {'MSE std':<12}"
    print(header)
    print("-" * 70)

    baseline_mean_time = np.mean(all_times["baseline"])
    for name, _, steps in methods:
        times = all_times[name]
        t_mean, t_std, t_min, t_max = compute_stats(times)
        speedup = baseline_mean_time / t_mean

        if name == "baseline":
            print(f"{name:<12} {steps:<6} {t_mean:.3f}s{'':<4} {t_std:.3f}s{'':<4} [{t_min:.3f}, {t_max:.3f}] {'--':<8} {'--':<12} {'--':<12}")
        else:
            mses = all_mses[name]
            m_mean, m_std, _, _ = compute_stats(mses)
            print(f"{name:<12} {steps:<6} {t_mean:.3f}s{'':<4} {t_std:.3f}s{'':<4} [{t_min:.3f}, {t_max:.3f}] {speedup:<8.1f}x {m_mean:<12.1f} {m_std:<12.1f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
