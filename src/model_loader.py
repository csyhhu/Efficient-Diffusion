"""
模型加载模块: 支持 diffusers pipeline 的加载, 缓存优先策略。

用法:
    from src.model_loader import load_model

    pipe, device = load_model(model_name="sd", mirror="https://hf-mirror.com")
    pipe, device = load_model(model_name="sd", local_path="/path/to/model")
"""

import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


MODEL_MAP = {
    "sd":   ("runwayml/stable-diffusion-v1-5", StableDiffusionPipeline),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", StableDiffusionXLPipeline),
}


def load_model(
    model_name: str = "sd",
    mirror: str = "https://hf-mirror.com",
    local_path: str = None,
    force_dtype: torch.dtype = None,
    force_device: str = None,
):
    """加载 diffusers pipeline，支持镜像和本地路径。

    加载策略: 先尝试本地缓存 (local_files_only=True), 未命中再下载。

    Args:
        model_name: 模型名称, "sd" 或 "sdxl"
        mirror: HuggingFace 镜像地址, 传空字符串则不使用镜像
        local_path: 本地模型路径 (优先级最高)
        force_dtype: 强制指定 dtype, 默认 auto (CUDA→bfloat16, CPU→float32)
        force_device: 强制指定 device, 默认 auto (CUDA→cuda, CPU→cpu)

    Returns:
        (pipeline, device_str)
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {list(MODEL_MAP.keys())}")

    model_id, pipeline_cls = MODEL_MAP[model_name]
    dtype = force_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    device = force_device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 决定加载来源 ----
    if local_path:
        model_source = local_path
        print(f"[1] 从本地路径加载模型: {model_source}")
    elif mirror:
        os.environ["HF_ENDPOINT"] = mirror
        model_source = model_id
        print(f"[1] 镜像: {mirror} | 模型: {model_source}")
    else:
        model_source = model_id
        print(f"[1] 直连 HuggingFace | 模型: {model_source}")

    common_kwargs = {"torch_dtype": dtype}

    # --- 第一次尝试: 仅从本地缓存加载 ---
    try:
        print(f"    -> 尝试从本地缓存加载 ...", end=" ", flush=True)
        pipe = pipeline_cls.from_pretrained(
            model_source,
            local_files_only=True,
            **common_kwargs,
        ).to(device)
        print("成功 (缓存命中)")
    except Exception:
        print("未命中")
        # --- 第二次尝试: 从网络下载 ---
        print(f"    -> 从网络下载模型 ...")
        pipe = pipeline_cls.from_pretrained(
            model_source,
            local_files_only=False,
            **common_kwargs,
        ).to(device)

    print(f"    -> 加载成功 (device={device}, dtype={dtype})")
    return pipe, device


if __name__ == "__main__":

    # pipe, device = load_model("sd")
    # print(f"Pipeline 类型: {type(pipe).__name__}, Device: {device}")

    pass
