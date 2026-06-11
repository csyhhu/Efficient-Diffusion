"""
单 Setting 量化可视化: 对指定的量化方案生成图像并对比 baseline。

用法:
    # torchao INT8 (默认)
    python scripts/quant_visualization.py

    # torchao INT4
    python scripts/quant_visualization.py --method torchao_int4

    # 自定义 alpha INT8
    python scripts/quant_visualization.py --method alpha_int8 --alpha 0.8

    # 自定义 alpha INT4
    python scripts/quant_visualization.py --method alpha_int4 --alpha 0.8

    # DDIM 加速
    python scripts/quant_visualization.py --method ddim --num_inference_steps 10

可选 method:
    torchao_int8   torchao INT8 权重量化
    torchao_int4   torchao INT4 权重量化 (需要 mslk CUDA kernel)
    alpha_int8     自定义 alpha INT8 (需传 --alpha)
    alpha_int4     自定义 alpha INT4 (需传 --alpha)
    ddim           DDIM 加速 (需传 --num_inference_steps)

输出:
    outputs/quant_viz/{method}/baseline.png   baseline 图像
    outputs/quant_viz/{method}/quant.png     量化后图像
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
from src.pipeline_builder import (
    build_baseline,
    build_quant_int8,
    build_quant_int4,
    apply_alpha_int8,
    apply_alpha_int4,
    build_ddim,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="单 Setting 量化可视化")
    parser.add_argument("--model", type=str, default="sd", choices=["sd", "sdxl"])
    parser.add_argument("--method", type=str, default="torchao_int8",
                        choices=["torchao_int8", "torchao_int4", "alpha_int8",
                                 "alpha_int4", "ddim"],
                        help="量化方案 (default: torchao_int8)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="alpha 值 (仅 alpha_int8/alpha_int4 有效)")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="推理步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (保证可比性)")
    parser.add_argument("--output_dir", type=str, default="./outputs/INT8_quant",
                        help="输出根目录")
    parser.add_argument("--mirror", type=str, default="https://hf-mirror.com")
    parser.add_argument("--local_path", type=str, default=None)
    parser.add_argument("--prompt", type=str,
                        default="A serene landscape with mountains and a lake at sunset, highly detailed")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def compute_mse(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    return float(((img_ref - img_test) ** 2).mean())


def compute_ssim(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    ref_u8 = img_ref.clip(0, 255).astype(np.uint8)
    test_u8 = img_test.clip(0, 255).astype(np.uint8)
    return float(ssim(ref_u8, test_u8, channel_axis=2, data_range=255))


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

    # 按 method 创建子目录, 方便区分不同实验
    if args.method in ("alpha_int8", "alpha_int4"):
        subdir = f"{args.method}_alpha_{args.alpha:.2f}"
    elif args.method == "ddim":
        subdir = f"{args.method}_steps_{args.num_inference_steps}"
    else:
        subdir = args.method

    out_dir = os.path.join(args.output_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # --- 加载模型 ---
    print("=" * 70)
    print(f"量化可视化: {args.method}")
    print("=" * 70)
    pipe, device = load_model(
        model_name=args.model,
        mirror=args.mirror,
        local_path=args.local_path,
    )

    # --- 打印运行设置 (确保可比性) ---
    print()
    print("运行设置:")
    print(f"  Model:      {args.model}")
    print(f"  Method:     {args.method}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Seed:       {args.seed}")
    print(f"  Steps:      {args.num_inference_steps}")
    if args.method in ("alpha_int8", "alpha_int4"):
        print(f"  Alpha:      {args.alpha}")
    print(f"  Output:     {os.path.abspath(out_dir)}/")
    print()

    # --- [1] Baseline ---
    # print("[1] 生成 baseline ...", end=" ", flush=True)
    # img_baseline = run_single(pipe, args.prompt, args.num_inference_steps, args.seed)
    # Image.fromarray(img_baseline.astype(np.uint8)).save(
    #     os.path.join(out_dir, "baseline.png")
    # )
    # print("完成")

    # --- [2] 量化生成 ---
    print(f"[2] 量化方案: {args.method} ...", end=" ", flush=True)

    t0 = time.perf_counter()

    if args.method == "torchao_int8":
        pipe_q = build_quant_int8(pipe)
    elif args.method == "torchao_int4":
        pipe_q = build_quant_int4(pipe)
    elif args.method == "alpha_int8":
        pipe_q = apply_alpha_int8(pipe, alpha=args.alpha)
    elif args.method == "alpha_int4":
        pipe_q = apply_alpha_int4(pipe, alpha=args.alpha)
    elif args.method == "ddim":
        pipe_q = build_ddim(pipe)
    else:
        raise ValueError(f"未知 method: {args.method}")

    img_quant = run_single(pipe_q, args.prompt, args.num_inference_steps, args.seed)
    elapsed = time.perf_counter() - t0

    Image.fromarray(img_quant.astype(np.uint8)).save(
        os.path.join(out_dir, "quant.png")
    )

    # --- [3] 计算指标 ---
    mse = compute_mse(img_baseline, img_quant)
    ssim = compute_ssim(img_baseline, img_quant)

    print(f"完成")
    print()
    print("=" * 70)
    print("结果")
    print("=" * 70)
    print(f"  MSE:     {mse:.1f}")
    print(f"  SSIM:    {ssim:.4f}")
    print(f"  耗时:     {elapsed:.1f}s")
    print(f"  输出:     {os.path.abspath(out_dir)}/")
    print(f"            ├── baseline.png")
    print(f"            └── quant.png")
    print(f"  Seed:     {args.seed}  (与之前实验保持一致以确保可比)")
    print("=" * 70)

    del pipe_q


if __name__ == "__main__":
    main()
