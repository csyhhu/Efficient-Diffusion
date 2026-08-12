r"""
Test image_generator package: verify 4 generation modes produce identical images.

Four modes (all with seed=42):
  1. _origin_pipe:       Original diffusers pipeline directly
  2. _origin_pipe_im:    ImageGenerator with used_origin_pipe=True
  3. _origin_model_im:   ImageGenerator custom generate + original transformer
  4. _im:                ImageGenerator custom generate + NVFP4 transformer

Run as:
python test/test_model_download_generation.py `
    --model_id "stabilityai/stable-diffusion-3.5-medium" `
    --prompt "A cute cat" `
    --seed 42 `
    --output_dir "G://Outputs//Efficient-Diffusion//generation//SD3" `
    --filename cat

python test/test_model_download_generation.py `
    --model_id "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" `
    --prompt "A cute cat" `
    --seed 42 `
    --output_dir "G://Outputs//Efficient-Diffusion//generation//Sana" `
    --filename cat `
    --num_steps 2
"""

import os
import sys
import argparse
import time
import torch

# Make "from src.xxx" work regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from diffusers import StableDiffusion3Pipeline, SanaSprintPipeline
from src.image_generator import SanaImageGenerator, SD3ImageGenerator

from src.image_generation import ImageGeneration
from src.SD3_image_generator import SD3ImageGenerator


def resolve_or_download(model_id: str, cache_dir: str) -> str:
    """Resolve a local cached path or download from ModelScope."""
    print(f"\n{'=' * 60}")
    print(f"[Download] {model_id}  from ModelScope -> {cache_dir}")
    print(f"{'=' * 60}")

    from modelscope import snapshot_download

    t0 = time.time()
    local_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "**/*.safetensors",
            "**/*.json",
            "*.json",
            "*.txt",
            "**/*.txt",
            "**/tokenizer.json",
            "**/tokenizer_config.json",
            "**/special_tokens_map.json",
            "**/vocab.json",
            "**/merges.txt",
        ],
        ignore_patterns=[
            "**/text_encoder_3/**",
            "**/tokenizer_3/**",
        ],
    )
    dt = time.time() - t0
    print(f"  Download finished in {dt/60:.1f} min -> {local_path}")
    return local_path


def load_pipeline(model_id: str, local_path: str, device: str, dtype: torch.dtype):
    """Load the original diffusers pipeline from local cache."""
    print(f"\n[Pipeline] Loading {model_id} from {local_path}")
    t0 = time.time()
    if model_id == "stabilityai/stable-diffusion-3.5-medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            local_path,
            torch_dtype=dtype,
            local_files_only=True,
            text_encoder_3=None,
            tokenizer_3=None,
        )
    elif model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
        pipe = SanaSprintPipeline.from_pretrained(
            local_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    else:
        raise ValueError(f"model_id {model_id} not supported")
    pipe = pipe.to(device)
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")
    return pipe


def generate_origin_pipe(model_id, pipe, prompt, output_dir, filename, seed, device):
    """Mode 1: Generate using the original pipeline directly."""
    print(f"\n[1/4] _origin_pipe: original pipeline")
    t0 = time.time()
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(prompt, generator=generator).images[0]
    save_path = os.path.join(output_dir, f"{filename}_origin_pipe_2.png")
    os.makedirs(output_dir, exist_ok=True)
    image.save(save_path)
    print(f"  Generated in {time.time() - t0:.1f}s -> {save_path}")
    return save_path



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test 4 generation modes for consistency")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--cache_dir", type=str, default="G://models")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--prompt", type=str,
                        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override num_steps (None = model default)")
    parser.add_argument("--output_dir", type=str,
                        default="G://Outputs//Efficient-Diffusion//generation//test")
    parser.add_argument("--filename", type=str, default="cat")
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"\nModel ID   : {args.model_id}")
    print(f"Cache      : {args.cache_dir}")
    print(f"Device     : {args.device}")
    print(f"Dtype      : {dtype}")
    print(f"Seed       : {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    # ------------------------------------------------------------------
    # 1. Load original pipeline and generate _origin_pipe
    # ------------------------------------------------------------------
    """
    local_path = os.path.join(args.cache_dir, args.model_id)
    if not os.path.exists(local_path):
        print(f"[Warning] Local path not found: {local_path}")
        local_path = resolve_or_download(args.model_id, args.cache_dir)

    pipe = load_pipeline(args.model_id, local_path, args.device, dtype)

    generate_origin_pipe(
        args.model_id, pipe, args.prompt,
        args.output_dir, args.filename, args.seed, args.device,
    )

    # Free pipeline to save GPU memory before loading ImageGenerator
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    """
    # ------------------------------------------------------------------
    # 2. Load ImageGenerator
    # ------------------------------------------------------------------
    # """
    if args.model_id == "stabilityai/stable-diffusion-3.5-medium":
        gen = SD3ImageGenerator(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=dtype,
            use_origin_model=True
        )
    elif args.model_id == "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers":
        gen = SanaImageGenerator(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"model_id {args.model_id} not supported")
    # """
    # ------------------------------------------------------------------
    # 3. Generate _origin_pipe_im (used_origin_pipe=True)
    # ------------------------------------------------------------------
    """
    print(f"\n[2/4] _origin_pipe_im: ImageGenerator + used_origin_pipe=True")
    gen.generate(
        prompt=args.prompt,
        num_samples=1,
        visual_n_row=1,
        seed=args.seed,
        num_steps=args.num_steps,
        used_origin_pipe=True,
        save_root=args.output_dir,
        save_name=f"{args.filename}_origin_pipe_im.png",
    )
    # print(f"  Saved -> {os.path.join(args.output_dir, args.filename + '_origin_pipe_im.png')}")
    """
    # ------------------------------------------------------------------
    # 4. Generate _origin_model_im (custom generate + original transformer)
    # ------------------------------------------------------------------
    print(f"\n[3/4] _origin_model_im: ImageGenerator custom generate + origin transformer")
    gen.generate(
        prompt=args.prompt,
        num_samples=1,
        visual_n_row=1,
        seed=args.seed,
        num_steps=args.num_steps,
        used_origin_pipe=False,
        save_root=args.output_dir,
        save_name=f"{args.filename}_origin_model_im.png",
    )
    print(f"  Saved -> {os.path.join(args.output_dir, args.filename + '_origin_model_im.png')}")
    # """
    # ------------------------------------------------------------------
    # 5. Generate _im (custom generate + NVFP4 transformer)
    # ------------------------------------------------------------------
    """
    print(f"\n[4/4] _im: ImageGenerator custom generate + NVFP4 transformer")
    gen.generate(
        prompt=args.prompt,
        num_samples=1,
        visual_n_row=1,
        seed=args.seed,
        num_steps=args.num_steps,
        used_origin_pipe=False,
        use_origin_model=False,
        save_root=args.output_dir,
        save_name=f"{args.filename}_im_2.png",
    )
    print(f"  Saved -> {os.path.join(args.output_dir, args.filename + '_im.png')}")
    """
    # gen = SD3ImageGenerator(
    #     model_id=args.model_id,
    #     cache_dir=args.cache_dir,
    #     device=args.device,
    #     dtype=dtype,
    # )
    """
    gen = ImageGeneration(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=dtype,
    )
    gen.generate(
        prompt=args.prompt,
        num_samples=1,
        visual_n_row=1,
        seed=args.seed,
        num_steps=args.num_steps,
        save_root=args.output_dir,
        save_name=f"{args.filename}_im_old.png",
    )
    """
