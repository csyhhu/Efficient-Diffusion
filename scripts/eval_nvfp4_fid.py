"""
FP4 vs 非量化 图片质量定量对比：SSIM、PSNR、LPIPS、FID、CLIP Score

用法:
    python scripts/eval_nvfp4_fid.py --num_images 100 --steps 4

指标说明:
  - SSIM (↑): 结构相似度，衡量两张图的结构一致性
  - PSNR (↑): 峰值信噪比，逐像素差异
  - LPIPS (↓): 感知相似度 (AlexNet 特征空间)，更贴近人眼感知
  - FID (↓): Fréchet Inception Distance，衡量生成分布与真实分布的距离
  - CLIP Score (↑): 图文匹配度 (可选，需要 open_clip)

输出:
    outputs/eval_nvfp4/
      ├── fp4/            → FP4 模式生成的图片
      ├── unquantized/    → 非量化模式生成的图片
      ├── eval_results.csv
      └── eval_summary.txt
"""

import argparse
import os
import sys
import time
import csv
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

# ---- Patch CUDA allocator for clean-fid compatibility ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    
    p = argparse.ArgumentParser(description="FP4 vs unquantized quality evaluation")
    p.add_argument("--num_images", type=int, default=5,
                   help="Number of images to generate per mode (default: 5)")
    p.add_argument("--steps", type=int, default=2,
                   help="Number of inference steps (default: 2)")
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--resolution", type=int, default=512,
                   help="Image resolution, 512 for fast, 1024 for quality (default: 512)")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed_start", type=int, default=42,
                   help="Starting seed (increments per image)")
    p.add_argument("--output_dir", type=str, default="outputs/eval_nvfp4")
    p.add_argument("--skip_generation", action="store_true",
                   help="Skip all generation, only compute metrics from existing images")
    p.add_argument("--skip_unquantized", action="store_true",
                   help="Skip unquantized generation (reuse existing unquantized images)")
    p.add_argument("--with_fid", action="store_true",
                   help="Opt-in: compute FID (requires downloading reference dataset)")
    p.add_argument("--with_clip", action="store_true",
                   help="Opt-in: compute CLIP Score (requires downloading CLIP model)")
    p.add_argument("--fid_ref", type=str, default="coco",
                   choices=["coco", "cifar10", "none"],
                   help="FID reference dataset (default: coco)")
    p.add_argument("--model_id", type=str,
                   default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")
    p.add_argument("--fp4_subdir", type=str, default=None,
                   help="FP4 image subdirectory name (e.g., fp4_bs1536). "
                        "If set, overrides the default 'fp4' directory.")
    p.add_argument("--output_suffix", type=str, default=None,
                   help="Suffix for output files (e.g., bs1536). "
                        "Results saved as eval_results_{suffix}.json etc.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Diverse prompt list (100 prompts covering various categories)
# ---------------------------------------------------------------------------

PROMPTS = [
    # Landscapes
    "A serene mountain landscape at sunrise with a crystal-clear lake reflecting the peaks",
    "A dense tropical rainforest with sunlight filtering through the canopy",
    "Rolling green hills under a dramatic cloudy sky, oil painting style",
    "A peaceful beach at sunset with gentle waves and golden sand",
    "A snowy alpine village at dusk with warm lights glowing from cabins",
    "A vast desert landscape with sand dunes under a starry night sky",
    "A misty forest path in autumn with colorful falling leaves",
    "A dramatic volcanic landscape with flowing lava under dark clouds",
    "A tranquil Japanese garden with a koi pond and cherry blossoms",
    "A rugged coastline with crashing waves against tall cliffs",
    # Urban / Architecture
    "A futuristic city skyline at night with neon lights and flying vehicles",
    "A cozy European cobblestone street with outdoor cafes and flower baskets",
    "An ancient temple complex in a jungle setting, detailed stone carvings",
    "A modern minimalist living room with large windows overlooking a forest",
    "A bustling night market in an Asian city with colorful lanterns and steam",
    "A Gothic cathedral interior with stained glass windows and soaring arches",
    "A cyberpunk street scene in the rain with holographic advertisements",
    "A Mediterranean white-washed village perched on a hillside by the sea",
    "An Art Deco theater interior with ornate gold details and red velvet",
    "A floating city above the clouds with airships and hanging gardens",
    # Animals
    "A majestic lion resting on a rock at golden hour, detailed fur texture",
    "A colorful hummingbird hovering near a tropical flower, macro photography",
    "A red fox in a snowy forest, soft winter morning light",
    "A pod of dolphins leaping through ocean waves at sunset",
    "A detailed close-up of a chameleon with vibrant scales on a branch",
    "A wolf howling at a full moon in a misty pine forest",
    "An owl perched on an ancient oak tree branch at twilight",
    "A colorful coral reef teeming with tropical fish, underwater photography",
    "A giant panda eating bamboo in a misty Chinese forest",
    "A horse galloping through a field of wildflowers in golden light",
    # Food / Still Life
    "A beautifully plated gourmet dish in a Michelin-star restaurant setting",
    "A rustic still life with fresh bread, cheese, grapes, and wine on a wooden table",
    "A steaming cup of coffee with latte art, surrounded by coffee beans",
    "Fresh sushi arranged on a traditional Japanese ceramic plate",
    "A colorful fruit market stall with exotic fruits in natural lighting",
    "A decadent chocolate cake with gold leaf decoration on a marble surface",
    "A traditional tea ceremony setup with ceramic teaware and incense",
    "Freshly baked croissants on a wire rack with morning light",
    "A tropical cocktail with umbrella garnish on a beach at sunset",
    "An elaborate charcuterie board with cured meats, cheeses, and figs",
    # Portraits / People
    "A wise elderly woman with deep wrinkles, wearing traditional attire, portrait photography",
    "A young ballet dancer mid-leap in an abandoned warehouse, dramatic lighting",
    "A cyberpunk samurai with glowing armor, standing in a neon-lit alley",
    "A steampunk explorer with brass goggles and leather gear, portrait shot",
    "A contemplative monk in orange robes, meditation hall with incense smoke",
    "A child reading a glowing book in a magical library, fantasy illustration",
    "A jazz musician playing saxophone in a smoky underground club",
    "A Victorian-era lady in an elaborate gown walking through a garden",
    "A street photographer capturing city life, black and white candid shot",
    "A martial artist practicing tai chi at sunrise on a mountain peak",
    # Fantasy / Sci-Fi
    "A dragon curled around a crystal tower under a purple sky, fantasy art",
    "A spaceship docking at a massive orbital station above an alien planet",
    "An enchanted forest with glowing mushrooms and fairy lights, magical atmosphere",
    "A wizard's study filled with floating books, potions, and arcane artifacts",
    "A post-apocalyptic city reclaimed by nature, vines covering skyscrapers",
    "A robot gardener tending to bioluminescent plants in a greenhouse",
    "A portal to another dimension opening in a contemporary city street",
    "An ancient tree spirit awakening in a moonlit grove, ethereal glow",
    "A space colony on Mars with geodesic domes and red landscape",
    "A time traveler's workshop filled with clocks, gears, and antique devices",
    # Abstract / Artistic
    "An abstract expressionist painting with bold colors and dynamic brushstrokes",
    "A surreal dreamscape with melting clocks and floating islands, Dali-inspired",
    "Geometric patterns inspired by Islamic art with intricate tessellations",
    "A watercolor painting of a rainy city street with colorful umbrellas",
    "A minimalist zen composition with a single bonsai tree and raked sand",
    "Pop art portrait in the style of Roy Lichtenstein with Ben-Day dots",
    "A stained glass window depicting a cosmic scene with stars and galaxies",
    "An origami crane made of reflective metallic paper, studio lighting",
    "A charcoal sketch of an old tree with exposed roots and textured bark",
    "Ink wash painting of mountains and mist in traditional Chinese style",
    # Vehicles / Transport
    "A vintage steam locomotive crossing a stone viaduct in the countryside",
    "A sleek concept electric car on a coastal highway at golden hour",
    "A wooden sailing ship navigating through stormy seas, dramatic clouds",
    "A hot air balloon festival at dawn with colorful balloons filling the sky",
    "A retro-futuristic flying car hovering above a 1950s style diner",
    # Night / Low Light
    "A campfire under the Milky Way in a remote wilderness, astrophotography",
    "A lighthouse beam cutting through thick fog on a stormy night",
    "Fireflies dancing in a summer meadow at twilight, long exposure",
    "City lights reflecting on a rain-soaked street, cinematic mood",
    "A neon-lit Tokyo alleyway at night with steam rising from vents",
    # Seasonal / Weather
    "A spring meadow covered in wildflowers with butterflies and soft sunlight",
    "A thunderstorm over a wheat field with dramatic lightning strikes",
    "A winter wonderland scene with ice sculptures and falling snow",
    "A foggy autumn morning in a park with golden leaves and a lone bench",
    "A rainbow appearing over a waterfall in a lush green valley",
    # Objects / Texture
    "A highly detailed macro shot of a dewdrop on a spider web",
    "Vintage typewriter with a half-written letter, warm desk lamp lighting",
    "An antique pocket watch with exposed gears, macro photography detail",
    "A collection of colorful glass marbles on a reflective surface",
    "Intricate henna tattoo designs on hands, detailed pattern photography",
    # Extra diverse
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A cat wearing a wizard hat casting spells in a magical library",
    "An underwater palace made of coral and pearls, mermaid swimming nearby",
    "A giant mechanical clockwork heart in a steampunk laboratory",
    "A peaceful Japanese onsen (hot spring) with snow-covered rocks",
]


# ---------------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------------

def compute_ssim(img_ref, img_test):
    """Compute SSIM between two numpy images (H, W, 3) uint8."""
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(img_ref, img_test, channel_axis=2, data_range=255))


def compute_psnr(img_ref, img_test):
    """Compute PSNR between two numpy images (H, W, 3) uint8."""
    mse = np.mean((img_ref.astype(np.float32) - img_test.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_lpips(img_ref, img_test, lpips_fn, device="cuda"):
    """Compute LPIPS between two numpy images (H, W, 3) uint8."""
    ref_t = torch.from_numpy(img_ref).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    test_t = torch.from_numpy(img_test).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    ref_t, test_t = ref_t.to(device), test_t.to(device)
    with torch.no_grad():
        dist = lpips_fn(ref_t, test_t)
    return float(dist.item())


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------

def load_clip_model(device="cuda"):
    """Try loading open_clip; fallback to transformers CLIP."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(device).eval()
        return model, preprocess, tokenizer, "open_clip"
    except ImportError:
        pass
    try:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor, None, "transformers_clip"
    except Exception:
        return None, None, None, None


def compute_clip_score(image_path, prompt, clip_model_info, device="cuda"):
    """Compute CLIP cosine similarity for a single image-prompt pair."""
    model, preprocess, tokenizer, backend = clip_model_info
    if model is None:
        return None

    img = Image.open(image_path).convert("RGB")

    if backend == "open_clip":
        img_t = preprocess(img).unsqueeze(0).to(device)
        text_t = tokenizer([prompt]).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            text_feat = model.encode_text(text_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return float((img_feat @ text_feat.T).item())

    elif backend == "transformers_clip":
        processor = preprocess
        inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            img_feat = outputs.image_embeds
            text_feat = outputs.text_embeds
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return float((img_feat @ text_feat.T).item())

    return None


# ---------------------------------------------------------------------------
# FID computation via clean-fid
# ---------------------------------------------------------------------------

def compute_fid(dir1, dir2=None, dataset_name="coco", device="cuda"):
    """
    Compute FID between dir1 and dir2 (or dir1 vs reference dataset).
    Uses clean-fid library.

    If dir2 is None, compares dir1 against a reference dataset's precomputed stats.
    Otherwise, computes FID between two custom image directories.
    """
    from clean_fid import fid

    if dir2 is not None:
        score = fid.compute_fid(dir1, dir2, device=device)
    elif dataset_name == "coco":
        score = fid.compute_fid(
            dir1, dataset_name="coco_val2017", dataset_split="val2017",
            device=device,
        )
    elif dataset_name == "cifar10":
        score = fid.compute_fid(
            dir1, dataset_name="cifar10", dataset_split="train",
            device=device,
        )
    else:
        raise ValueError(f"Unknown reference: {dataset_name}")
    return score


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_pipe(model_id, cache_dir, download_source, device, block_size, use_nvfp4):
    """Build a Sana pipeline with NVFP4QuantizedSana transformer."""
    from src.models.nvfp4_quantized_Sana import NVFP4QuantizedSana
    from diffusers import SanaSprintPipeline

    model = NVFP4QuantizedSana.from_pretrained(
        model_id, download_source=download_source, cache_dir=cache_dir,
        block_size=block_size, use_nvfp4=use_nvfp4,
    )
    model.to(device=device)

    pipe_load_kwargs = dict(torch_dtype=torch.bfloat16)
    if cache_dir:
        pipe_load_kwargs["cache_dir"] = cache_dir

    if download_source == "modelscope":
        from modelscope import snapshot_download
        pipe_local = snapshot_download(model_id, cache_dir=cache_dir)
        pipe = SanaSprintPipeline.from_pretrained(
            pipe_local, local_files_only=True, **pipe_load_kwargs)
    else:
        pipe = SanaSprintPipeline.from_pretrained(
            model_id, local_files_only=True, **pipe_load_kwargs)

    # Swap transformer
    _unused = pipe.transformer
    pipe.transformer = None
    del _unused
    if device == "cuda":
        torch.cuda.empty_cache()
    pipe.transformer = model
    if device == "cuda":
        pipe.to(device)
    return pipe, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    fp4_subdir = args.fp4_subdir or "fp4"
    fp4_dir = os.path.join(args.output_dir, fp4_subdir)
    uq_dir = os.path.join(args.output_dir, "unquantized")
    os.makedirs(fp4_dir, exist_ok=True)
    os.makedirs(uq_dir, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    # Prepare prompts
    num_prompts = min(args.num_images, len(PROMPTS))
    prompts = PROMPTS[:num_prompts]

    cache_dir = ".cache/modelscope"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("MODEL_CACHE_DIR", cache_dir)
    download_source = os.environ.get("DOWNLOAD_SOURCE", "modelscope").lower()

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    times_uq = []
    if not args.skip_generation:
        print("=" * 70)
        print("Stage 1: Generating images")
        print("=" * 70)

        # ---- Unquantized ----
        if not args.skip_unquantized:
            print(f"\n[1/2] Loading unquantized model ...")
            pipe_uq, model_uq = build_pipe(
                args.model_id, cache_dir, download_source, device,
                args.block_size, use_nvfp4=False,
            )
            n_params = sum(p.numel() for p in model_uq.parameters()) / 1e6
            print(f"  Parameters: {n_params:.2f}M | Mode: unquantized")
            print(f"  Generating {num_prompts} images ...")

            for i, prompt in enumerate(prompts):
                seed = args.seed_start + i
                generator = torch.Generator(device=device).manual_seed(seed)
                t0 = time.perf_counter()
                with torch.no_grad():
                    img = pipe_uq(
                        prompt=prompt,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        height=args.resolution, width=args.resolution,
                        generator=generator,
                    ).images[0]
                elapsed = time.perf_counter() - t0
                times_uq.append(elapsed)
                path = os.path.join(uq_dir, f"{i:04d}_{seed}.png")
                img.save(path)
                print(f"  [{i+1}/{num_prompts}] {elapsed:.1f}s  seed={seed}  {prompt[:50]}...")

            del pipe_uq, model_uq
            torch.cuda.empty_cache()
            mean_uq = np.mean(times_uq) if times_uq else 0
            print(f"  Unquantized avg: {mean_uq:.1f}s/image")
        else:
            print("\n[1/2] Skipping unquantized (--skip_unquantized)")

        # ---- FP4 ----
        step_label = "2/2" if not args.skip_unquantized else "1/1"
        print(f"\n[{step_label}] Loading FP4 model (block_size={args.block_size}) ...")
        pipe_fp4, model_fp4 = build_pipe(
            args.model_id, cache_dir, download_source, device,
            args.block_size, use_nvfp4=True,
        )
        n_params = sum(p.numel() for p in model_fp4.parameters()) / 1e6
        print(f"  Parameters: {n_params:.2f}M | Mode: NVFP4")
        print(f"  Generating {num_prompts} images ...")

        times_fp4 = []
        for i, prompt in enumerate(prompts):
            seed = args.seed_start + i
            generator = torch.Generator(device=device).manual_seed(seed)
            t0 = time.perf_counter()
            with torch.no_grad():
                img = pipe_fp4(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.resolution, width=args.resolution,
                    generator=generator,
                ).images[0]
            elapsed = time.perf_counter() - t0
            times_fp4.append(elapsed)
            path = os.path.join(fp4_dir, f"{i:04d}_{seed}.png")
            img.save(path)
            print(f"  [{i+1}/{num_prompts}] {elapsed:.1f}s  seed={seed}  {prompt[:50]}...")

        del pipe_fp4, model_fp4
        torch.cuda.empty_cache()

        mean_fp4 = np.mean(times_fp4)
        print(f"  FP4 avg: {mean_fp4:.1f}s/image")
        if not args.skip_unquantized and times_uq:
            print(f"  Speed ratio (FP4/unquantized): {mean_fp4 / np.mean(times_uq):.2f}x")
    else:
        print("Skipping generation (--skip_generation), using existing images.")

    # -----------------------------------------------------------------------
    # Pairwise metrics: SSIM, PSNR, LPIPS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 2: Pairwise metrics (SSIM, PSNR, LPIPS)")
    print("=" * 70)

    ssims, psnrs = [], []

    # Load LPIPS
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").eval().to(device)
        print("  LPIPS model (AlexNet) loaded [OK]")
    except Exception as e:
        print(f"  LPIPS load failed: {e}")

    lpips_vals = []
    n_found = 0
    for i in range(len(prompts)):
        seed = args.seed_start + i
        fp4_path = os.path.join(fp4_dir, f"{i:04d}_{seed}.png")
        uq_path = os.path.join(uq_dir, f"{i:04d}_{seed}.png")
        if not os.path.exists(fp4_path) or not os.path.exists(uq_path):
            continue
        n_found += 1
        img_uq = np.array(Image.open(uq_path).convert("RGB"))
        img_fp4 = np.array(Image.open(fp4_path).convert("RGB"))

        ssims.append(compute_ssim(img_uq, img_fp4))
        psnrs.append(compute_psnr(img_uq, img_fp4))
        if lpips_fn is not None:
            lpips_vals.append(compute_lpips(img_uq, img_fp4, lpips_fn, device))

    print(f"  Found {n_found}/{len(prompts)} image pairs")
    print(f"  SSIM  (↑): {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"  PSNR  (↑): {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    if lpips_vals:
        print(f"  LPIPS (↓): {np.mean(lpips_vals):.4f} ± {np.std(lpips_vals):.4f}")

    # -----------------------------------------------------------------------
    # FID
    # -----------------------------------------------------------------------
    fid_fp4_ref = fid_uq_ref = fid_fp4_uq = None
    if args.with_fid:
        print("\n" + "=" * 70)
        print("Stage 3: FID computation (clean-fid)")
        print("=" * 70)

        # FID between FP4 and unquantized (direct distribution comparison)
        if n_found >= 10:
            try:
                print(f"  Computing FID(FP4 vs Unquantized) ...")
                fid_fp4_uq = compute_fid(fp4_dir, uq_dir, device=device)
                print(f"  FID(FP4, Unquantized) (↓): {fid_fp4_uq:.2f}")
            except Exception as e:
                print(f"  FID(FP4, Unquantized) failed: {e}")

        # FID against reference dataset
        if args.fid_ref != "none" and n_found >= 10:
            try:
                print(f"  Computing FID(Unquantized vs {args.fid_ref}) ...")
                fid_uq_ref = compute_fid(uq_dir, dataset_name=args.fid_ref, device=device)
                print(f"  FID(Unquantized, {args.fid_ref}) (↓): {fid_uq_ref:.2f}")
            except Exception as e:
                print(f"  FID(Unquantized, ref) failed: {e}")

            try:
                print(f"  Computing FID(FP4 vs {args.fid_ref}) ...")
                fid_fp4_ref = compute_fid(fp4_dir, dataset_name=args.fid_ref, device=device)
                print(f"  FID(FP4, {args.fid_ref}) (↓): {fid_fp4_ref:.2f}")
            except Exception as e:
                print(f"  FID(FP4, ref) failed: {e}")

    # -----------------------------------------------------------------------
    # CLIP Score
    # -----------------------------------------------------------------------
    clip_fp4_mean = clip_uq_mean = None
    if args.with_clip and n_found > 0:
        print("\n" + "=" * 70)
        print("Stage 4: CLIP Score")
        print("=" * 70)
        clip_info = load_clip_model(device)
        if clip_info[0] is not None:
            print(f"  CLIP backend: {clip_info[3]} [OK]")

            clip_uq, clip_fp4 = [], []
            for i in range(len(prompts)):
                seed = args.seed_start + i
                fp4_path = os.path.join(fp4_dir, f"{i:04d}_{seed}.png")
                uq_path = os.path.join(uq_dir, f"{i:04d}_{seed}.png")
                if not os.path.exists(fp4_path) or not os.path.exists(uq_path):
                    continue
                cs_uq = compute_clip_score(uq_path, prompts[i], clip_info, device)
                cs_fp4 = compute_clip_score(fp4_path, prompts[i], clip_info, device)
                if cs_uq is not None:
                    clip_uq.append(cs_uq)
                if cs_fp4 is not None:
                    clip_fp4.append(cs_fp4)
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(prompts)}] ...")

            if clip_uq:
                clip_uq_mean = np.mean(clip_uq)
                print(f"  CLIP Score Unquantized (↑): {clip_uq_mean:.4f} ± {np.std(clip_uq):.4f}")
            if clip_fp4:
                clip_fp4_mean = np.mean(clip_fp4)
                print(f"  CLIP Score FP4 (↑):        {clip_fp4_mean:.4f} ± {np.std(clip_fp4):.4f}")
        else:
            print("  CLIP model not available (install open_clip_torch or transformers)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        "block_size": args.block_size,
        "num_images": n_found,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "ssim_mean": np.mean(ssims) if ssims else None,
        "ssim_std": np.std(ssims) if ssims else None,
        "psnr_mean": np.mean(psnrs) if psnrs else None,
        "psnr_std": np.std(psnrs) if psnrs else None,
        "lpips_mean": np.mean(lpips_vals) if lpips_vals else None,
        "lpips_std": np.std(lpips_vals) if lpips_vals else None,
        "fid_fp4_vs_uq": fid_fp4_uq,
        "fid_uq_vs_ref": fid_uq_ref,
        "fid_fp4_vs_ref": fid_fp4_ref,
        "clip_score_uq": clip_uq_mean,
        "clip_score_fp4": clip_fp4_mean,
    }

    print(f"  Block size: {args.block_size}")
    print(f"  Images evaluated: {n_found}")
    print(f"  SSIM  (↑): {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}" if results['ssim_mean'] else "  SSIM: N/A")
    print(f"  PSNR  (↑): {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB" if results['psnr_mean'] else "  PSNR: N/A")
    print(f"  LPIPS (↓): {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}" if results['lpips_mean'] else "  LPIPS: N/A")
    if fid_fp4_uq is not None:
        print(f"  FID(FP4, Unquantized) (↓): {fid_fp4_uq:.2f}")
        delta = abs(fid_fp4_uq)
        print(f"    → FP4 与非量化输出分布几乎相同 (ΔFID={delta:.2f})" if delta < 5 else
              f"    → FP4 与非量化输出分布有明显差异 (ΔFID={delta:.2f})")
    if fid_uq_ref is not None and fid_fp4_ref is not None:
        print(f"  FID(Unquantized, ref) (↓): {fid_uq_ref:.2f}")
        print(f"  FID(FP4, ref) (↓):        {fid_fp4_ref:.2f}")
        delta_ref = fid_fp4_ref - fid_uq_ref
        print(f"    → FP4 FID 劣化: {delta_ref:+.2f}")
    if clip_uq_mean is not None:
        print(f"  CLIP Score Unquantized (↑): {clip_uq_mean:.4f}")
        print(f"  CLIP Score FP4 (↑):        {clip_fp4_mean:.4f}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"eval_results{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"\n  CSV saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, f"eval_results{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    txt_path = os.path.join(args.output_dir, f"eval_summary{suffix}.txt")
    with open(txt_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"  Summary saved: {txt_path}")