"""
Pipeline 构建模块: baseline / torchao INT8 / torchao INT4 / 自定义 alpha INT8 / 自定义 alpha INT4 / DDIM。

用法:
    from src.pipeline_builder import (
        build_baseline, build_quant_int8, build_quant_int4,
        build_ddim, apply_alpha_int8, apply_alpha_int4,
    )
"""

import copy
import torch
from diffusers import DDIMScheduler

from quant_utils import symmetric_quantize, asymmetric_quantize


# ---------------------------------------------------------------------------
# 基础方案
# ---------------------------------------------------------------------------

def build_baseline(pipe):
    """不做优化, 返回原始 pipeline (不拷贝, 原地复用)。"""
    return pipe


def build_quant_int8(pipe):
    """torchao INT8 权重量化 (深拷贝后量化, 不影响原 pipe)。"""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    pipe_q = copy.deepcopy(pipe)
    target = getattr(pipe_q, "unet", None)
    if target is not None:
        quantize_(target, Int8WeightOnlyConfig())
    return pipe_q


def build_quant_int4(pipe):
    """torchao INT4 权重量化 (深拷贝后量化, 不影响原 pipe)。

    使用 version=1 避免 mslk 依赖 (CPU 环境无 CUDA kernel)。
    """
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    pipe_q = copy.deepcopy(pipe)
    target = getattr(pipe_q, "unet", None)
    if target is not None:
        quantize_(target, Int4WeightOnlyConfig(version=1))
    return pipe_q


def build_ddim(pipe):
    """DDIM 算法加速 (深拷贝后替换 scheduler, 不影响原 pipe)。"""
    pipe_d = copy.deepcopy(pipe)
    pipe_d.scheduler = DDIMScheduler.from_config(pipe_d.scheduler.config)
    return pipe_d


# 向后兼容别名
build_quant = build_quant_int8


# ---------------------------------------------------------------------------
# 自定义 alpha 量化: scale = alpha * max(|W|) / qmax
# ---------------------------------------------------------------------------

def _apply_alpha_quant(pipe, alpha: float, qmax: int):
    """通用 alpha 量化核心: scale = alpha * max(|W|) / qmax。

    qmax=127 → INT8, qmax=7 → INT4
    """
    pipe_q = copy.deepcopy(pipe)
    unet = getattr(pipe_q, "unet", None)
    if unet is None:
        raise RuntimeError("Pipeline 中没有 unet 模块")

    # print(f"\n{'─' * 100}")
    # print(f"  alpha={alpha}, qmax={qmax} (INT{qmax.bit_length()})")
    # print(f"  {'Parameter':<58s} {'W shape':<20s} {'scale shape':<20s} {'scale min':>10s} {'scale max':>10s}")
    # print(f"  {'─' * 100}")

    with torch.no_grad():
        for name, param in unet.named_parameters():
            if param.ndim < 2:
                continue
            w = param.data.float()
            w_abs_max = w.abs().amax(dim=list(range(1, w.ndim)), keepdim=True)
            scale = (alpha * w_abs_max / qmax).clamp(min=1e-8)
            # scale_min = scale.min().item()
            # scale_max = scale.max().item()
            # print(f"  {name:<78s} {str(tuple(w.shape)):<20s} {str(tuple(w_abs_max.shape)):<20s} {scale_min:>10.6f} {scale_max:>10.6f}")
            w_q = torch.round(w / scale).clamp(-qmax, qmax) * scale
            param.data.copy_(w_q.to(param.dtype))

    return pipe_q


# ---
# alpha / beta for min / max threshold
# ---
def _apply_ajust_threshold_quant(pipe, bitW=8):
    
    pipe_q = copy.deepcopy(pipe)
    unet = getattr(pipe_q, "unet", None)

    def cal_error(_mask, _method=0):
        return _mask.abs().mean()

    with torch.no_grad():
        for name, param in unet.named_parameters():
            if param.ndim < 2:
                continue
            w = param.data.float()
            w_min = w.amin(dim=list(range(1, w.ndim)), keepdim=True)
            w_max = w.amax(dim=list(range(1, w.ndim)), keepdim=True)
            alpha, beta = w_min, w_max
            prev_error = 1e9
            lr = 1e-3
            print(f"{name}\n Quantization on Parameters ({w.size()}) using Scale ({alpha.size()})")
            while True:
                left_outlier_mask = w < alpha
                right_outlier_mask = w > beta
                in_domain_mask = ~ (left_outlier_mask | right_outlier_mask)
                quantized_w = asymmetric_quantize(w, bitW, alpha, beta)
                quantization_error_mask = w - quantized_w
                in_domain_quantization_error = cal_error(quantization_error_mask[in_domain_mask]) if in_domain_mask.any() else 0
                left_outlier_quantization_error = cal_error(quantization_error_mask[left_outlier_mask]) if left_outlier_mask.any() else 0
                right_outlier_quantization_error = cal_error(quantization_error_mask[right_outlier_mask]) if right_outlier_mask.any() else 0
                quantization_error = cal_error(quantization_error_mask)
                print(f"In Domain Error: {in_domain_quantization_error}, Left Domain Error: {left_outlier_quantization_error}, Right Domain Error: {right_outlier_quantization_error}")
                print(f"Quantization error: {quantization_error}")
                next_alpha = torch.clamp(alpha - lr * left_outlier_quantization_error + lr * in_domain_quantization_error, min=w_min, max=w_max)
                next_beta = torch.clamp(beta + lr * right_outlier_quantization_error - lr * in_domain_quantization_error, min=w_min, max=w_max)
                # if prev_error < quantization_error:
                #     break
                alpha = next_alpha
                beta = next_beta
                prev_error = quantization_error
                input("Continue?")
            param.data.copy_(quantized_w.to(param.dtype))
            input("Continue?")
    return pipe_q



def apply_alpha_int8(pipe, alpha: float):
    """自定义 alpha INT8 量化: scale = alpha * max(|W|) / 127。

    alpha=1.0 → 无裁剪, 等价于 torchao 默认
    alpha<1.0 → 裁剪离群值, 量化误差增大
    """
    return _apply_alpha_quant(pipe, alpha, qmax=127)


def apply_alpha_int4(pipe, alpha: float):
    """自定义 alpha INT4 量化: scale = alpha * max(|W|) / 7。

    alpha=1.0 → 无裁剪, 等价于 torchao 默认
    alpha<1.0 → 裁剪离群值, 量化误差增大
    """
    return _apply_alpha_quant(pipe, alpha, qmax=7)


# 向后兼容别名
apply_alpha_quantization = apply_alpha_int8


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    from model_loader import load_model

    # Load full pipeline with weights
    print("Loading SD model with weights ...")
    pipe, device = load_model(model_name="sd", mirror="https://hf-mirror.com")

    # Run alpha INT8 quantization (alpha=1.0 = no clipping, standard INT8)
    # print("\nRunning _apply_alpha_quant (alpha=1.0, INT8) ...\n")
    # pipe_q = apply_alpha_int8(pipe, alpha=1.0)
    # print("\nDone!")

    pipe_q = _apply_ajust_threshold_quant(pipe)