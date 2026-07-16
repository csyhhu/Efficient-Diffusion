"""
为综合评测生成图片的辅助脚本。

用法:
    # 使用 Sana 模型生成评测图片
    python eval/generate_for_eval.py --model_id Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \\
        --output_dir outputs/eval_gen/Sana_0.6B --num_images 500 --steps 4
    
    # 使用本地 checkpoint
    python eval/generate_for_eval.py --model_path /path/to/checkpoint \\
        --output_dir outputs/eval_gen/my_model
    
    # 生成多个分辨率
    python eval/generate_for_eval.py --output_dir outputs/eval_gen/Sana_0.6B_512 --resolution 512

输出结构:
    outputs/eval_gen/{model_name}/
      ├── 0000_{seed}.png
      ├── 0001_{seed}.png
      ├── ...
      └── prompts.txt        # 自动保存所有使用的 prompt
"""

import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Generate images for evaluation")
    
    p.add_argument("--model_id", type=str,
                   default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                   help="HuggingFace/ModelScope model ID")
    p.add_argument("--model_path", type=str, default=None,
                   help="Local model path (overrides --model_id)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for generated images")
    p.add_argument("--num_images", type=int, default=100,
                   help="Number of images to generate")
    p.add_argument("--steps", type=int, default=4,
                   help="Number of inference steps")
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--resolution", type=int, default=1024,
                   help="Image resolution")
    p.add_argument("--seed_start", type=int, default=42,
                   help="Starting seed (increments per image)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--block_size", type=int, default=256,
                   help="Block size for NVFP4")
    p.add_argument("--use_nvfp4", action="store_true",
                   help="Use NVFP4 quantization")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Custom prompts file (one per line). Uses built-in if not provided.")
    p.add_argument("--download_source", type=str, default="modelscope",
                   choices=["modelscope", "huggingface"])
    p.add_argument("--cache_dir", type=str, default=".cache/modelscope")
    return p.parse_args()


# ---- 内置多样 prompt ----
PROMPTS = [
    # Nature & Landscapes
    "A serene mountain lake at sunrise with crystal clear water reflecting snow-capped peaks",
    "A dense tropical rainforest with sunlight filtering through the canopy, vibrant green foliage",
    "A peaceful beach at sunset with gentle waves, golden sand, and palm trees silhouetted",
    "A snowy alpine village at dusk with warm lights glowing from wooden cabins",
    "A vast desert landscape with sand dunes under a starry night sky, Milky Way visible",
    "A misty forest path in autumn with colorful falling leaves and soft morning light",
    "A dramatic volcanic landscape with flowing lava under dark stormy clouds",
    "A tranquil Japanese garden with a koi pond, cherry blossoms, and a wooden bridge",
    "A rugged coastline with crashing waves against tall cliffs, seabirds circling above",
    "Rolling green hills under a dramatic cloudy sky, oil painting style",
    # Urban & Architecture
    "A futuristic city skyline at night with neon lights, flying vehicles, and towering skyscrapers",
    "A cozy European cobblestone street with outdoor cafes, flower baskets, and warm lighting",
    "An ancient temple complex in a jungle setting with detailed stone carvings and moss",
    "A modern minimalist living room with large windows overlooking a forest",
    "A bustling night market in an Asian city with colorful lanterns, steam from food stalls",
    "A Gothic cathedral interior with stained glass windows and soaring vaulted arches",
    "A cyberpunk street scene in the rain with holographic advertisements and neon reflections",
    "A Mediterranean white-washed village perched on a hillside by the sea",
    "An Art Deco theater interior with ornate gold details and red velvet seats",
    "A floating city above the clouds with airships, hanging gardens, and waterfalls",
    # Animals
    "A majestic lion resting on a rock at golden hour, detailed fur texture, warm lighting",
    "A colorful hummingbird hovering near a tropical flower, macro photography, sharp detail",
    "A red fox in a snowy forest, soft winter morning light, alert expression",
    "A pod of dolphins leaping through ocean waves at sunset, splashing water",
    "A detailed close-up of a chameleon with vibrant scales, curled tail on a branch",
    "A wolf howling at a full moon in a misty pine forest, atmospheric",
    "An owl perched on an ancient oak tree branch at twilight, large piercing eyes",
    "A colorful coral reef teeming with tropical fish, underwater photography, sun rays",
    "A giant panda eating bamboo in a misty Chinese forest, peaceful mood",
    "A horse galloping through a field of wildflowers in golden light, dynamic pose",
    # Food & Still Life
    "A beautifully plated gourmet dish in a Michelin-star restaurant setting, elegant presentation",
    "A rustic still life with fresh bread, cheese, grapes, and red wine on a wooden table",
    "A steaming cup of coffee with intricate latte art, surrounded by coffee beans",
    "Fresh sushi arranged on a traditional Japanese ceramic plate with chopsticks",
    "A colorful fruit market stall with exotic fruits in natural sunlight",
    "A decadent chocolate cake with gold leaf decoration on a marble surface",
    "A traditional tea ceremony setup with ceramic teaware, incense, and bamboo whisk",
    "Freshly baked croissants on a wire rack with morning light streaming in",
    "A tropical cocktail with umbrella garnish on a beach at sunset",
    "An elaborate charcuterie board with cured meats, cheeses, figs, and nuts",
    # Portraits & People
    "A wise elderly woman with deep wrinkles wearing traditional attire, portrait photography",
    "A young ballet dancer mid-leap in an abandoned warehouse, dramatic lighting",
    "A cyberpunk samurai with glowing armor standing in a neon-lit alley",
    "A steampunk explorer with brass goggles and leather gear, atmospheric portrait",
    "A contemplative monk in orange robes, meditation hall with swirling incense smoke",
    "A child reading a glowing book in a magical library, fantasy illustration style",
    "A jazz musician playing saxophone in a smoky underground club, moody lighting",
    "A Victorian-era lady in an elaborate gown walking through a rose garden",
    "A street photographer capturing city life, black and white candid shot",
    "A martial artist practicing tai chi at sunrise on a mountain peak",
    # Fantasy & Sci-Fi
    "A dragon curled around a crystal tower under a purple sky, fantasy art, epic scale",
    "A spaceship docking at a massive orbital station above an alien planet with rings",
    "An enchanted forest with glowing mushrooms, fairy lights, and magical atmosphere",
    "A wizard's study filled with floating books, potions, and arcane artifacts",
    "A post-apocalyptic city reclaimed by nature, vines covering skyscrapers",
    "A robot gardener tending to bioluminescent alien plants in a glass greenhouse",
    "A portal to another dimension opening in a contemporary city street",
    "An ancient tree spirit awakening in a moonlit grove, ethereal glow, fireflies",
    "A space colony on Mars with geodesic domes against the red landscape",
    "A time traveler's workshop filled with clocks, gears, and antique devices",
    # Artistic & Abstract
    "An abstract expressionist painting with bold colors and dynamic brushstrokes",
    "A surreal dreamscape with melting clocks and floating islands, Dali-inspired",
    "Geometric patterns inspired by Islamic art with intricate tessellations and gold",
    "A watercolor painting of a rainy city street with colorful umbrellas",
    "A minimalist zen composition with a single bonsai tree and raked sand garden",
    "Pop art portrait in the style of Roy Lichtenstein with Ben-Day dots",
    "A stained glass window depicting a cosmic scene with stars, galaxies, and nebulas",
    "An origami crane made of reflective metallic paper, studio lighting",
    "A charcoal sketch of an old tree with exposed roots and textured bark",
    "Ink wash painting of mountains and mist in traditional Chinese style",
    # Vehicles & Transport
    "A vintage steam locomotive crossing a stone viaduct in the countryside, billowing smoke",
    "A sleek concept electric car on a coastal highway at golden hour",
    "A wooden sailing ship navigating through stormy seas, dramatic clouds and lightning",
    "A hot air balloon festival at dawn with dozens of colorful balloons filling the sky",
    "A retro-futuristic flying car hovering above a 1950s style diner at night",
    # Night & Low Light
    "A campfire under the Milky Way in a remote wilderness, astrophotography style",
    "A lighthouse beam cutting through thick fog on a stormy night, dramatic lighting",
    "Fireflies dancing in a summer meadow at twilight, long exposure effect",
    "City lights reflecting on a rain-soaked street, cinematic mood, wet pavement",
    "A neon-lit Tokyo alleyway at night with steam rising from vents, atmospheric",
    # Color tests (for GenEval-style color evaluation)
    "a red car", "a blue bird", "a green apple", "a yellow flower", "a white dog",
    "a black cat", "a purple butterfly", "an orange fish", "a pink rose", "a brown horse",
    # Object tests (for GenEval-style object evaluation)
    "A photo of a single dog", "A photo of a single cat", "A photo of a single chair",
    "A photo of a single book", "A photo of a single cup", "A dog and a cat together",
    "A chair and a table", "Two dogs playing", "Three cats sleeping", "Four chairs around a table",
    # Text rendering (for DPG evaluation)
    "A sign that says 'Welcome' on a wooden board",
    "A neon sign with the word 'OPEN' glowing in a shop window",
    "A book cover with the title 'The Great Adventure'",
]

def load_prompts(prompts_file=None):
    """Load prompts from file or use built-in."""
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return PROMPTS


def build_pipe(model_id, cache_dir, download_source, device, block_size, use_nvfp4):
    """Build a Sana pipeline."""
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

    pipe.transformer = model
    if device == "cuda":
        pipe.to(device)
    return pipe


if __name__ == "__main__":

    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.prompts_file)
    num_images = min(args.num_images, len(prompts))
    prompts = prompts[:num_images]

    # Save prompts
    prompts_file = os.path.join(args.output_dir, "prompts.txt")
    with open(prompts_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"Prompts saved: {prompts_file}")

    # Save metadata
    metadata = {
        "model_id": args.model_id,
        "model_path": args.model_path,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed_start": args.seed_start,
        "num_images": num_images,
        "use_nvfp4": args.use_nvfp4,
        "block_size": args.block_size,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Setup
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("MODEL_CACHE_DIR", cache_dir)
    download_source = args.download_source

    # Build pipeline
    print(f"\nLoading model: {args.model_id}")
    pipe = build_pipe(
        args.model_id, cache_dir, download_source, device,
        args.block_size, args.use_nvfp4,
    )
    mode = "NVFP4" if args.use_nvfp4 else "unquantized"
    print(f"Mode: {mode} | Resolution: {args.resolution} | Steps: {args.steps}")

    # Generate
    print(f"\nGenerating {num_images} images...")
    times = []
    for i, prompt in enumerate(prompts):
        seed = args.seed_start + i
        generator = torch.Generator(device=device).manual_seed(seed)

        t0 = time.perf_counter()
        with torch.no_grad():
            img = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            ).images[0]
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        path = os.path.join(args.output_dir, f"{i:04d}_{seed}.png")
        img.save(path)

        remaining = (num_images - i - 1) * np.mean(times)
        print(f"  [{i+1}/{num_images}] {elapsed:.1f}s | ETA {remaining:.0f}s | "
              f"seed={seed} | {prompt[:60]}...")

    # Summary
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Total time: {sum(times):.0f}s")
    print(f"  Avg time:   {np.mean(times):.1f}s/image")
    print(f"  Images:     {num_images}")
    print(f"  Output:     {args.output_dir}")
    print(f"\nNow run evaluation:")
    print(f"  python eval/eval_comprehensive.py --image_dir {args.output_dir} "
          f"--fid --mjhq_path /path/to/mjhq30k --clip")
    print(f"{'='*60}")
