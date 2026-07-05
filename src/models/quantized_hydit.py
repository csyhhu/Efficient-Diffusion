"""
Quantized HunYuan-DiT — Download from ModelScope + Inference via Diffusers.

HunYuan-DiT (腾讯混元) is a bilingual (CN/EN) text-to-image diffusion transformer.
This script downloads the model from ModelScope and runs inference through the
official ``HunyuanDiTPipeline`` from diffusers.

Architecture overview:
  +------------------+----------+-------------------------------------------+
  | Component        | Size     | Notes                                     |
  +==================+==========+===========================================+
  | Transformer (DiT)| ~5.6 GB  | 28 layers, cross-attn, RoPE, adaLN        |
  | text_encoder     | ~1.3 GB  | Bilingual CLIP (ViT-L)                    |
  | text_encoder_2   | ~6.2 GB  | mT5 (T5-like), OPTIONAL, ~4× the memory  |
  | VAE (SDXL)       | ~319 MB  | Latent en/decoding 8× compression         |
  +------------------+----------+-------------------------------------------+
  | Total (full)     | ~13.4 GB | Disk + RAM                                |
  | Total (no T5)    | ~7.2 GB  | Skipping text_encoder_2 saves ~6.2 GB     |
  +------------------+----------+-------------------------------------------+

ModelScope repository: ``dengcao/HunyuanDiT-v1.2-Diffusers``

Usage::

    python src/models/quantized_hydit.py

Memory tips:
  - ``SKIP_T5=True``: skips the 6 GB mT5 encoder, works with ~8 GB RAM.
  - ``enable_model_cpu_offload()``: keeps components on CPU, only one on GPU at a time.
"""


if __name__ == "__main__":

    import os
    import torch

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------

    # Skip mT5 (text_encoder_2) — set True to save ~6.2 GB disk and memory.
    # When skipping T5, model_index.json is automatically patched to remove the
    # text_encoder_2 entry so that HunyuanDiTPipeline loads without it.
    SKIP_T5 = os.environ.get("SKIP_T5", "True").lower() in ("1", "true", "yes")

    # ModelScope model ID and cache directory
    MODEL_ID = "dengcao/HunyuanDiT-v1.2-Diffusers"
    CACHE_DIR = "C:/Users/Shangyu/.cache/modelscope"

    # Output directory for generated images
    OUTPUT_DIR = "outputs/images"

    # Enable CPU offload (recommended to save GPU VRAM)
    USE_CPU_OFFLOAD = True

    # ---------------------------------------------------------------------------
    # Download model from ModelScope
    # ---------------------------------------------------------------------------

    from modelscope import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    # ---- Build download patterns ----
    allow_patterns = [
        # === Transformer (required) ~5.6 GB ===
        "**/transformer/**",

        # === CLIP text_encoder (required) ~1.3 GB ===
        "**/text_encoder/**",

        # === VAE (required) ~319 MB ===
        "**/vae/**",

        # === Config & metadata (a few KB) ===
        "*.json",
        "**/*.json",
        "*.txt",
        "**/*.txt",

        # === Scheduler ===
        "**/scheduler/**",
    ]

    ignore_patterns = []

    # Optionally skip mT5 text_encoder_2 to save ~6.2 GB
    if SKIP_T5:
        ignore_patterns.append("**/text_encoder_2/**")
        print("⚠  SKIP_T5=True — skipping text_encoder_2 (mT5, ~6.2 GB).")
        print("   Prompt understanding may be slightly reduced for long Chinese text.")
    else:
        allow_patterns.append("**/text_encoder_2/**")

    print(f"Downloading {MODEL_ID} via ModelScope ...")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Estimated download: {'~7.2 GB' if SKIP_T5 else '~13.4 GB'}")

    local_path = snapshot_download(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    print(f"Model cached at: {local_path}")

    # ---- If T5 was skipped, patch model_index.json to remove text_encoder_2 ----
    if SKIP_T5:
        import json
        index_path = os.path.join(local_path, "model_index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
        if "text_encoder_2" in model_index:
            del model_index["text_encoder_2"]
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(model_index, f, indent=2)
            print("  Patched model_index.json: removed text_encoder_2 entry.")

    # -----------------------------------------------------------------------
    # Load pipeline
    # -----------------------------------------------------------------------

    print("Loading HunyuanDiTPipeline ...")
    from diffusers import HunyuanDiTPipeline

    pipe = HunyuanDiTPipeline.from_pretrained(
        local_path,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )

    # CPU offload: components stay on CPU, only moved to GPU one-at-a-time
    if USE_CPU_OFFLOAD and device == "cuda":
        print("Enabling model CPU offload (low VRAM mode) ...")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    print("Pipeline loaded successfully.")
    print(f"  Transformer params: {sum(p.numel() for p in pipe.transformer.parameters()) / 1e9:.2f}B")

    # -----------------------------------------------------------------------
    # Generate images
    # -----------------------------------------------------------------------

    prompts = [
        "一个宇航员在骑马",                                 # Chinese prompt
        "长城上的日落，水墨画风格",                           # Chinese prompt
        "A small cactus with a happy face in the Sahara desert.",  # English prompt
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")
        print("  Generating ...")

        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]

        # Save
        safe_name = f"hydit_{idx:02d}.png"
        save_path = os.path.join(OUTPUT_DIR, safe_name)
        image.save(save_path)
        print(f"  Saved: {save_path}")

    print("\nDone!")
