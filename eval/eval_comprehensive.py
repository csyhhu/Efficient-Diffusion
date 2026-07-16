"""
Comprehensive Evaluation Script: Reproduce the 5 metrics from the Sana paper
===========================================================================

Supported metrics:
  - FID (↓):       Fréchet Inception Distance — distance between generated and real distributions
  - CLIP Score (↑): Image-text alignment
  - GenEval (↑):     Text-image alignment (6 sub-dimensions: single/two objects, counting, colors, position, color attribution)
  - DPG-Bench (↑):   Dense prompt alignment (5 sub-dimensions: Global/Entity/Attribute/Relation/Other)
  - ImageReward (↑): Human preference alignment

Datasets:
  - FID / CLIP Score: MJHQ-30K (30K Midjourney images)
  - GenEval:          533 test prompts
  - DPG-Bench:        1065 test prompts
  - ImageReward:      100 test prompts

Usage:

    # Pre-compute FID reference stats (random subset of 3000 by default)
    python eval/eval_comprehensive.py --precompute_fid_stats --mjhq_path G:/datasets/MJHQ-30K --fid_sample 3000 --batch_size 64
    # ...or a deterministic sequential subset (first N images in sorted order):
    python eval/eval_comprehensive.py --precompute_fid_stats --mjhq_path G:/datasets/MJHQ-30K --fid_sample 30 --fid_sampling sequential --fid_ref_stats "G:/datasets/MJHQ-30K-sequential-30_fid_stats.npz"

    # Run all metrics
    python eval/eval_comprehensive.py --image_dir outputs/my_model/ --all

    # Run only selected metrics
    python eval/eval_comprehensive.py --image_dir outputs/my_model/ --fid --clip --geneval

    # Compute FID using PRE-COMPUTED reference stats (fast — no raw MJHQ images needed)
    # This is the recommended path once you have a *_fid_stats.npz (e.g. from precompute above)
    python eval/eval_comprehensive.py --image_dir G:/Outputs/Efficient-Diffusion/eval_gen/Sana_nvfp4_bs16_had_mag --output_dir G:/Outputs/Efficient-Diffusion//eval_gen/Sana_nvfp4_bs16_had_mag/eval_comprehensive --fid --fid_ref_stats "G:/datasets/MJHQ-30K_fid_stats.npz"

    # Or compute FID live from raw MJHQ-30K images (slow, only the very first run)
    python eval/eval_comprehensive.py --image_dir outputs/eval_gen/Sana_origin --fid --mjhq_path G:/datasets/MJHQ-30K

    # Evaluate multiple model directories
    python eval/eval_comprehensive.py --image_dirs model_a/ model_b/ --all

Output:
    outputs/eval_comprehensive/
      ├── eval_results.csv
      ├── eval_results.json
      └── eval_summary.txt

Dependencies:
    pip install lpips open_clip_torch pycocotools
    # GenEval: download prompt files and evaluation code from official repo
    # ImageReward: pip install image-reward
    # DPG-Bench: download official prompt files and evaluation code
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive evaluation for text-to-image models")
    
    # --- Input ---
    p.add_argument("--image_dir", type=str, default=None,
                   help="Single directory of generated images (format: {idx:04d}.png + prompts.txt)")
    p.add_argument("--image_dirs", type=str, nargs="+", default=None,
                   help="Multiple directories of generated images for comparison")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Path to prompts file (one per line). Auto-detect from image_dir if not given.")
    
    # --- Metric flags ---
    p.add_argument("--all", action="store_true", help="Run all available metrics")
    p.add_argument("--fid", action="store_true", help="Compute FID")
    p.add_argument("--clip", action="store_true", help="Compute CLIP Score")
    p.add_argument("--geneval", action="store_true", help="Compute GenEval score")
    p.add_argument("--dpg", action="store_true", help="Compute DPG-Bench score")
    p.add_argument("--imagereward", action="store_true", help="Compute ImageReward score")
    
    # --- Common parameters ---
    p.add_argument("--output_dir", type=str, default="outputs/eval_comprehensive")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of DataLoader workers for image loading (default: 4). "
                        "Use 0 for single-process, higher values accelerate I/O-bound loading.")
    p.add_argument("--seed", type=int, default=42)
    
    # --- FID parameters ---
    p.add_argument("--mjhq_path", type=str, default=None,
                   help="Path to MJHQ-30K images directory (for FID reference). "
                        "First time: all images loaded, stats saved as .npz. "
                        "Thereafter: only .npz needed — no images required.")
    p.add_argument("--fid_ref_stats", type=str, default=None,
                   help="Path to pre-computed FID reference stats (.npz). "
                        "Use this to skip image loading (recommended after first run).")
    p.add_argument("--precompute_fid_stats", action="store_true",
                   help="Only precompute and save FID reference stats from --mjhq_path, then exit.")
    p.add_argument("--fid_sample", type=int, default=None,
                   help="Max number of reference images to use for FID stats precomputation "
                        "(randomly sampled unless --fid_sampling sequential). Useful for quick "
                        "testing. Default: all images.")
    p.add_argument("--fid_sampling", type=str, default="sequential",
                   choices=["random", "sequential"],
                   help="Sampling strategy for --fid_sample when precomputing FID reference "
                        "stats. 'random': fixed-seed random subset (default, matches old "
                        "behavior). 'sequential': first N images in sorted directory order.")
    p.add_argument("--fid_image_size", type=int, default=299,
                   help="Resize images for InceptionV3 (default: 299)")
    
    # --- CLIP parameters ---
    p.add_argument("--clip_model", type=str, default="ViT-B-32",
                   help="CLIP model name for open_clip (default: ViT-B-32)")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k",
                   help="CLIP pretrained weights")
    
    # --- GenEval parameters ---
    p.add_argument("--geneval_dir", type=str, default=None,
                   help="Path to GenEval benchmark data directory")
    
    # --- DPG-Bench parameters ---
    p.add_argument("--dpg_dir", type=str, default=None,
                   help="Path to DPG-Bench data directory")
    
    # --- ImageReward parameters ---
    p.add_argument("--imagereward_model", type=str, default="ImageReward-v1.0",
                   help="ImageReward model name or path")
    
    # --- Output format ---
    p.add_argument("--output_suffix", type=str, default=None,
                   help="Suffix for output filenames")
    
    return p.parse_args()


# ============================================================================
# Utility functions
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_images_from_dir(dir_path: str, image_size: int = 299) -> torch.Tensor:
    """Load all images from a directory (recursive) as uint8 tensor (N, 3, H, W)."""
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    paths = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(root, f))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in {dir_path}")
    
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((image_size, image_size), Image.BICUBIC)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        images.append(img_t)
    return torch.cat(images, dim=0)


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts, one per line."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def auto_detect_prompts(image_dir: str) -> List[str]:
    """Try to auto-detect prompts from prompts.txt in image_dir."""
    prompts_file = os.path.join(image_dir, "prompts.txt")
    if os.path.exists(prompts_file):
        return load_prompts(prompts_file)
    # Fallback: use image files only
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(f)
    paths = sorted(paths)
    return [f"image {i}" for i in range(len(paths))]


def get_image_paths(image_dir: str) -> List[str]:
    """Get sorted list of image file paths from a directory (recursive)."""
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


class _FIDImageDataset(torch.utils.data.Dataset):
    """Dataset for FID feature extraction (module-level so it is picklable on Windows spawn)."""

    def __init__(self, paths: List[str], img_size: int):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1)


# ============================================================================
# 1. FID — Fréchet Inception Distance
# ============================================================================

class FIDEvaluator:
    """
    FID evaluator.

    Principle:
        FID measures the distribution distance between generated and real images
        in the InceptionV3 feature space.
        Formula: FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

        where μ, Σ are the feature mean and covariance of generated/reference images.
        Features come from InceptionV3's pool3 layer (2048-dim).

    Key optimization — Reference statistics caching:
        The reference set μ and Σ are fixed! After the first computation, save as .npz.
        Subsequent evaluations only need to load these two matrices — no raw images needed.

        Workflow:
        Step 1 (one-time): precompute_and_save_ref_stats(ref_dir, save_path)
            → iterate all reference images → extract features → compute μ, Σ → save .npz
        Step 2 (each eval): compute_fid(gen_dir, ref_stats_path=save_path)
            → load .npz → only extract generated image features → compute FID directly

    Dataset:
        Sana uses MJHQ-30K as the reference set (30K high-quality Midjourney images).
        COCO 2017 validation set can also be used as an alternative.
    """

    def __init__(self, device: str = "cuda", image_size: int = 299):
        self.device = device
        self.image_size = image_size
        self._inception = None

    def _get_inception(self):
        """Lazy-load InceptionV3 feature extractor."""
        if self._inception is None:
            from torchvision import models
            inception = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT,
                transform_input=False,
            )
            inception.fc = torch.nn.Identity()  # pool3 → 2048 features
            inception.eval().to(self.device)
            self._inception = inception
        return self._inception

    def _extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract InceptionV3 pool3 features.

        Args:
            imgs: (N, 3, H, W) uint8 in [0, 255]
        Returns:
            features: (N, 2048)
        """
        from torchvision.transforms import functional as F_t

        inception = self._get_inception()
        # Resize to 299×299, normalize with ImageNet stats
        imgs_resized = F_t.resize(imgs.float(), [299, 299])
        imgs_norm = F_t.normalize(
            imgs_resized / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        with torch.no_grad():
            return inception(imgs_norm).cpu()

    def _extract_features_from_dir(
        self, dir_path: str, batch_size: int = 32,
        sample: Optional[int] = None, num_workers: int = 4,
        sampling: str = "random",
    ) -> torch.Tensor:
        """Extract features for images in a directory (with optional random sampling).

        Uses DataLoader with multiple workers to parallelize CPU-bound image loading,
        while InceptionV3 inference runs on GPU.

        Args:
            dir_path: Path to image directory
            batch_size: InceptionV3 inference batch size
            sample: If set, randomly sample at most this many images
            num_workers: Number of dataloader workers for image loading (0 = main process)
        Returns:
            features: (N, 2048)
        """
        import random as _random
        from torch.utils.data import DataLoader

        # Collect image paths recursively
        exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        all_paths = []
        for root, _, files in os.walk(dir_path):
            for f in files:
                if os.path.splitext(f.lower())[1] in exts:
                    all_paths.append(os.path.join(root, f))
        all_paths = sorted(all_paths)
        total = len(all_paths)

        # Sampling
        if sample is not None and sample < total:
            if sampling == "sequential":
                # all_paths is already sorted (see above), take the first N
                selected = all_paths[:sample]
                print(f"  FID: using {sample}/{total} images (sequential)")
            else:
                _random.seed(42)
                selected = _random.sample(all_paths, sample)
                print(f"  FID: using {sample}/{total} images (randomly sampled)")
        else:
            selected = all_paths
            print(f"  FID: using all {total} images")

        n_selected = len(selected)

        # --- Dataset for DataLoader (module-level class, picklable on Windows) ---
        dataset = _FIDImageDataset(selected, self.image_size)
        effective_workers = num_workers if num_workers > 0 else 0
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=effective_workers, pin_memory=(self.device == "cuda"),
            prefetch_factor=2 if effective_workers > 0 else None,
        )

        # Load images in batches with progress (speed + ETA)
        feats = []
        t_start = time.time()
        step_start = time.time()

        for i, imgs in enumerate(loader):
            imgs = imgs.to(self.device, dtype=torch.uint8, non_blocking=True)
            feats.append(self._extract_features(imgs))

            # Progress indicator with speed and ETA
            done = min((i + 1) * batch_size, n_selected)
            pct = done / n_selected * 100
            elapsed = time.time() - t_start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (n_selected - done) / speed if speed > 0 else 0
            elapsed_str = self._fmt_time(elapsed)
            eta_str = self._fmt_time(eta)
            print(f"\r  FID: [{done:>6d}/{n_selected}] {pct:5.1f}% | "
                  f"{speed:6.1f} img/s | elapsed {elapsed_str} | ETA {eta_str}",
                  end="", flush=True)

        print()  # newline after progress
        total_elapsed = time.time() - t_start
        print(f"  FID: feature extraction done in {self._fmt_time(total_elapsed)} "
              f"({n_selected / total_elapsed:.1f} img/s avg)")
        return torch.cat(feats, dim=0)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as hh:mm:ss or mm:ss."""
        seconds = int(seconds)
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # ---- Reference statistics cache ---- 

    def precompute_and_save_ref_stats(
        self, ref_dir: str, save_path: str, batch_size: int = 32,
        sample: Optional[int] = None, num_workers: int = 4,
        sampling: str = "random",
    ) -> dict:
        """
        Iterate over all reference images, compute μ and Σ, save as .npz.

        This is the most time-consuming step in FID evaluation — but only once!
        All subsequent evaluations just load the .npz directly.

        Args:
            ref_dir:   Reference image directory
            save_path: Output .npz file path
            batch_size: InceptionV3 inference batch size
            sample: Max number of images to sample (None = all)
            num_workers: DataLoader workers for parallel image loading
            sampling: "random" (fixed-seed) or "sequential" (first N in sorted order)
        Returns:
            {"mu": np.ndarray (2048,), "sigma": np.ndarray (2048,2048), "n": int}
        """
        print(f"  [Precompute] Extracting features from {ref_dir} ...")
        feats = self._extract_features_from_dir(
            ref_dir, batch_size, sample=sample, num_workers=num_workers,
            sampling=sampling,
        )
        n = feats.shape[0]
        print(f"  [Precompute] {n} images processed, computing statistics ...")

        mu = feats.mean(dim=0).cpu().numpy()       # (2048,)
        sigma = self._cov(feats).cpu().numpy()      # (2048, 2048)
        stats = {"mu": mu, "sigma": sigma, "n": n}

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez(save_path, **stats)
        print(f"  [Precompute] Stats saved to {save_path}")
        print(f"  [Precompute] μ shape={mu.shape}, Σ shape={sigma.shape}")
        return stats

    @staticmethod
    def load_ref_stats(npz_path: str) -> dict:
        """
        Load pre-computed reference statistics.

        Args:
            npz_path: .npz file generated by precompute_and_save_ref_stats
        Returns:
            {"mu": np.ndarray (2048,), "sigma": np.ndarray (2048,2048), "n": int}
        """
        data = np.load(npz_path)
        return {
            "mu": data["mu"],
            "sigma": data["sigma"],
            "n": int(data["n"]) if "n" in data else None,
        }

    # ---- FID computation ----

    def compute_fid(
        self,
        gen_dir: str,
        ref_dir: str = None,
        ref_stats: dict = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> float:
        """
        Compute FID. Supports two reference input modes:

        (A) ref_dir       — Traditional: load all reference images → extract features → compute μ, Σ (slow)
        (B) ref_stats     — Recommended: pass pre-computed μ, Σ directly (fast, no raw images needed)
        (C) ref_stats_path — Pass .npz path (fast, auto-loaded)

        Formula: FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

        Args:
            gen_dir:   Generated image directory
            ref_dir:   Reference image directory (first use, slow)
            ref_stats: Pre-computed statistics dict with mu, sigma (recommended, fast)
            batch_size: Feature extraction batch size
        Returns:
            FID score (lower is better)
        """
        # ---- Step 1: Extract generated image features & statistics ----
        print(f"  FID: loading generated images from {gen_dir}")
        gen_feats = self._extract_features_from_dir(gen_dir, batch_size, num_workers=num_workers)
        mu1 = gen_feats.mean(dim=0)
        sigma1 = self._cov(gen_feats)

        # ---- Step 2: Get reference statistics ----
        if ref_stats is not None:
            # Mode B: Use pre-computed statistics directly (fastest)
            # Keep on CPU to match gen_feats, which is returned on CPU by
            # _extract_features_from_dir (mixing devices raises an error).
            mu2 = torch.from_numpy(ref_stats["mu"]).float()
            sigma2 = torch.from_numpy(ref_stats["sigma"]).float()
            n_ref = ref_stats.get("n", "?")
            print(f"  FID: using pre-computed stats (n={n_ref})")
        elif ref_dir is not None:
            # Mode A: Compute from raw images (slow)
            print(f"  FID: loading reference images from {ref_dir}")
            ref_feats = self._extract_features_from_dir(ref_dir, batch_size, num_workers=num_workers)
            mu2 = ref_feats.mean(dim=0)
            sigma2 = self._cov(ref_feats)
        else:
            raise ValueError("Either ref_dir or ref_stats must be provided.")

        print(f"  FID: computing score ({gen_feats.shape[0]} gen vs reference)")

        # ---- Step 3: Compute FID ----
        diff = mu1 - mu2
        mean_term = (diff * diff).sum()

        def _sqrtm(mat):
            eigvals, eigvecs = torch.linalg.eigh(mat)
            eigvals = eigvals.clamp(min=0)
            return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

        sigma1_sqrt = _sqrtm(sigma1)
        cross = torch.linalg.eigvals(sigma1_sqrt @ sigma2 @ sigma1_sqrt)
        cross_term = cross.real.clamp(min=0).sqrt().sum()
        trace_term = torch.trace(sigma1) + torch.trace(sigma2) - 2 * cross_term

        score = float((mean_term + trace_term).item())
        print(f"  FID = {score:.2f}")
        return score

    @staticmethod
    def _cov(feats: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        return torch.cov(feats.T)


# ============================================================================
# 2. CLIP Score — Image-text alignment
# ============================================================================

class CLIPScoreEvaluator:
    """
    CLIP Score evaluator.

    Principle:
        Uses CLIP model to encode images and text separately, computing cosine similarity.
        CLIP Score = mean(cos_sim(image_features, text_features))
        Higher values indicate better image-text alignment.

    Dataset:
        Sana evaluates CLIP Score on MJHQ-30K.
        Each image is typically scored with its corresponding prompt.

    Supports both open_clip and transformers CLIP backends.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self._setup_model()

    def _setup_model(self):
        """Load CLIP model."""
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained,
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model = self.model.to(self.device).eval()
            self.backend = "open_clip"
            print(f"  CLIP: loaded {self.model_name} via open_clip [OK]")
        except ImportError:
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).to(self.device).eval()
                self.preprocess = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self.tokenizer = None
                self.backend = "transformers"
                print("  CLIP: loaded via transformers [OK]")
            except ImportError:
                raise ImportError(
                    "CLIP requires open_clip_torch or transformers. "
                    "Install: pip install open_clip_torch"
                )

    def compute_single(self, image_path: str, prompt: str) -> float:
        """Compute CLIP score for a single image-prompt pair."""
        img = Image.open(image_path).convert("RGB")

        if self.backend == "open_clip":
            img_t = self.preprocess(img).unsqueeze(0).to(self.device)
            text_t = self.tokenizer([prompt]).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(img_t)
                text_feat = self.model.encode_text(text_t)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            return float((img_feat @ text_feat.T).item())
        else:
            inputs = self.preprocess(
                text=[prompt], images=img, return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                img_feat = outputs.image_embeds
                text_feat = outputs.text_embeds
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            return float((img_feat @ text_feat.T).item())

    def compute_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
    ) -> Dict[str, float]:
        """Compute CLIP score for a batch of images."""
        scores = []
        for i, (img_path, prompt) in enumerate(zip(image_paths, prompts)):
            scores.append(self.compute_single(img_path, prompt))
            if (i + 1) % 100 == 0:
                print(f"  CLIP: [{i+1}/{len(image_paths)}] avg={np.mean(scores):.4f}")

        return {
            "clip_mean": float(np.mean(scores)),
            "clip_std": float(np.std(scores)),
            "clip_min": float(np.min(scores)),
            "clip_max": float(np.max(scores)),
            "clip_num_samples": len(scores),
        }


# ============================================================================
# 3. GenEval — Text-image alignment evaluation
# ============================================================================

class GenEvalEvaluator:
    """
    GenEval evaluator.

    Principle:
        GenEval (Ghosh et al., 2024) verifies whether generated images satisfy
        object, attribute, relationship, and spatial layout constraints described
        in the prompt via object detection.

    6 sub-dimensions:
        - Single Object:     Can it generate a single specified object
        - Two Objects:       Can it generate two specified objects
        - Counting:          Can it generate the correct number of objects
        - Colors:            Can it generate objects with correct colors
        - Position:          Can it correctly place object spatial positions
        - Color Attribution: Can it generate objects with correct color attributes

    Dataset:
        533 test prompts from the official GenEval benchmark.
        Download: https://github.com/djghosh13/geneval

    Implementation:
        Calls official evaluation scripts or uses local detection logic.
        Uses GroundingDINO or predefined detection models for verification.
    """

    # GenEval 6 sub-tasks
    SUB_TASKS = [
        "single_object",
        "two_objects", 
        "counting",
        "colors",
        "position",
        "color_attribution",
    ]

    def __init__(
        self,
        device: str = "cuda",
        geneval_dir: Optional[str] = None,
    ):
        self.device = device
        self.geneval_dir = geneval_dir
        self._prompts = None
        self._detector = None

    def _load_prompts(self) -> Dict[str, List[Dict]]:
        """
        Load GenEval prompts.

        If no local data is available, use built-in prompt subset.
        Full 533 prompts require downloading from the official GenEval repo.
        """
        if self.geneval_dir and os.path.isdir(self.geneval_dir):
            return self._load_from_official_repo()
        else:
            print("  GenEval: using built-in prompt subset (not official benchmark)")
            print("  GenEval: download full benchmark from https://github.com/djghosh13/geneval")
            return self._get_builtin_prompts()

    def _load_from_official_repo(self) -> Dict[str, List[Dict]]:
        """Load prompts from official GenEval repo structure."""
        import json as _json
        metadata_file = os.path.join(self.geneval_dir, "metadata.jsonl")
        if os.path.exists(metadata_file):
            prompts = []
            with open(metadata_file, "r") as f:
                for line in f:
                    prompts.append(_json.loads(line.strip()))
            # Group by task
            grouped = {}
            for p in prompts:
                task = p.get("task", "unknown")
                grouped.setdefault(task, []).append(p)
            return grouped
        return self._get_builtin_prompts()

    def _get_builtin_prompts(self) -> Dict[str, List[Dict]]:
        """
        Built-in GenEval-style prompts (simplified subset for quick testing).

        These are representative prompts, not the full 533 official ones.
        For comparable scores reported in the paper, use the complete official benchmark.
        """
        prompts = {
            "single_object": [
                {"prompt": "a photo of a dog", "object": "dog"},
                {"prompt": "a photo of a cat", "object": "cat"},
                {"prompt": "a photo of a car", "object": "car"},
                {"prompt": "a photo of a bicycle", "object": "bicycle"},
                {"prompt": "a photo of an apple", "object": "apple"},
                {"prompt": "a photo of a chair", "object": "chair"},
                {"prompt": "a photo of a book", "object": "book"},
                {"prompt": "a photo of a cup", "object": "cup"},
                {"prompt": "a photo of a bird", "object": "bird"},
                {"prompt": "a photo of a clock", "object": "clock"},
            ],
            "two_objects": [
                {"prompt": "a dog and a cat", "objects": ["dog", "cat"]},
                {"prompt": "a car and a bicycle", "objects": ["car", "bicycle"]},
                {"prompt": "a chair and a table", "objects": ["chair", "table"]},
                {"prompt": "a book and a cup", "objects": ["book", "cup"]},
                {"prompt": "a person and a dog", "objects": ["person", "dog"]},
            ],
            "counting": [
                {"prompt": "two dogs", "objects": ["dog"], "count": 2},
                {"prompt": "three cats", "objects": ["cat"], "count": 3},
                {"prompt": "four chairs", "objects": ["chair"], "count": 4},
                {"prompt": "five apples", "objects": ["apple"], "count": 5},
                {"prompt": "two cars", "objects": ["car"], "count": 2},
            ],
            "colors": [
                {"prompt": "a red car", "objects": ["car", "red car"]},
                {"prompt": "a blue bird", "objects": ["bird", "blue bird"]},
                {"prompt": "a green apple", "objects": ["apple", "green apple"]},
                {"prompt": "a yellow flower", "objects": ["flower", "yellow flower"]},
                {"prompt": "a white dog", "objects": ["dog", "white dog"]},
            ],
            "position": [
                {"prompt": "a dog on the left of a cat", "objects": ["dog", "cat"]},
                {"prompt": "a car below a bird", "objects": ["car", "bird"]},
                {"prompt": "a chair to the right of a table", "objects": ["chair", "table"]},
                {"prompt": "a book on top of a cup", "objects": ["book", "cup"]},
                {"prompt": "a person behind a dog", "objects": ["person", "dog"]},
            ],
            "color_attribution": [
                {"prompt": "a red car and a blue bicycle", "objects": ["red car", "blue bicycle"]},
                {"prompt": "a green apple and a yellow banana", "objects": ["green apple", "yellow banana"]},
                {"prompt": "a white dog and a black cat", "objects": ["white dog", "black cat"]},
                {"prompt": "a blue bird and a red flower", "objects": ["blue bird", "red flower"]},
                {"prompt": "a brown chair and a white table", "objects": ["brown chair", "white table"]},
            ],
        }
        return prompts

    def evaluate(self, image_paths: List[str], prompts_file: str) -> Dict:
        """
        Evaluate GenEval score.

        Note: Full GenEval evaluation requires object detection models (e.g., GroundingDINO).
        This implementation provides the framework interface; users can plug in official evaluation logic.

        Args:
            image_paths: List of generated image paths
            prompts_file: GenEval prompts file path
        Returns:
            Dict with overall and sub-task scores
        """
        print("\n" + "=" * 60)
        print("  GenEval Evaluation")
        print("=" * 60)
        print("  WARNING: Full GenEval requires object detection models.")
        print("  This provides the framework. Use official code for paper-level results.")
        print("  Official repo: https://github.com/djghosh13/geneval")

        prompts_data = self._load_prompts()

        # Basic structure compatibility check
        total_prompts = sum(len(v) for v in prompts_data.values())
        print(f"  GenEval prompts loaded: {total_prompts} across {len(prompts_data)} tasks")

        # Return framework structure
        return {
            "geneval_overall": None,
            "geneval_num_prompts_total": total_prompts,
            "_geneval_note": "Full evaluation requires official GenEval benchmark + detection model",
            "_geneval_framework_ready": True,
        }


# ============================================================================
# 4. DPG-Bench — Dense Prompt Graph Benchmark
# ============================================================================

class DPGBenchEvaluator:
    """
    DPG-Bench (Dense Prompt Graph Benchmark) evaluator.

    Principle:
        Proposed by Hu et al. (2024, ELLA), uses prompts with dense attribute descriptions
        to evaluate a model's ability to follow complex text descriptions.

    5 sub-dimensions:
        - Global:     Global description (scene, style, etc.)
        - Entity:     Entity description
        - Attribute:  Attribute description (color, material, shape, etc.)
        - Relation:   Relationship description (spatial, action relations, etc.)
        - Other:      Other descriptions

    Dataset:
        1,065 test prompts, each containing 4-5 dense attribute descriptions.
        Uses multimodal LLMs (e.g., DSG or GPT-4V) to score generated images.

    Official repo: https://github.com/PRIS-CV/DPG-Bench
    """

    CATEGORIES = ["Global", "Entity", "Attribute", "Relation", "Other"]

    def __init__(self, device: str = "cuda", dpg_dir: Optional[str] = None):
        self.device = device
        self.dpg_dir = dpg_dir
        self._prompts = None

    def _load_prompts(self) -> List[Dict]:
        """Load DPG-Bench prompts."""
        import json as _json
        
        if self.dpg_dir and os.path.isdir(self.dpg_dir):
            prompt_file = os.path.join(self.dpg_dir, "dpg_prompts.json")
            if os.path.exists(prompt_file):
                with open(prompt_file, "r") as f:
                    return _json.load(f)
        
        # Built-in subset
        print("  DPG-Bench: using built-in prompt subset")
        print("  DPG-Bench: download full benchmark from https://github.com/PRIS-CV/DPG-Bench")
        return self._get_builtin_prompts()

    def _get_builtin_prompts(self) -> List[Dict]:
        """
        Built-in DPG-Bench-style prompts (not the official full set).

        Each prompt contains dense attribute descriptions, simulating DPG-Bench format.
        """
        prompts = [
            {
                "prompt": "A majestic golden lion with a flowing mane, standing proudly on a rocky cliff at sunset, with warm orange and purple sky in the background, photorealistic style",
                "attributes": ["golden lion", "flowing mane", "rocky cliff", "sunset", "warm sky", "photorealistic"],
            },
            {
                "prompt": "A sleek silver sports car with red racing stripes, parked on a rain-soaked city street at night, neon lights reflecting on the wet pavement",
                "attributes": ["silver car", "red stripes", "rain-soaked street", "night", "neon lights", "wet pavement"],
            },
            {
                "prompt": "An old wooden sailboat with white sails, peacefully floating on calm turquoise water near a tropical island with palm trees",
                "attributes": ["wooden sailboat", "white sails", "turquoise water", "tropical island", "palm trees"],
            },
            {
                "prompt": "A small blue bird with bright yellow chest feathers, perched on a cherry blossom branch with pink petals falling gently",
                "attributes": ["blue bird", "yellow chest", "cherry blossom", "pink petals"],
            },
            {
                "prompt": "A cozy rustic kitchen with exposed wooden beams, a large stone fireplace, copper pots hanging from the ceiling, and warm candlelight",
                "attributes": ["rustic kitchen", "wooden beams", "stone fireplace", "copper pots", "candlelight"],
            },
            {
                "prompt": "A young woman with long red hair wearing a flowing white dress, reading a book under a large oak tree in a sunlit meadow",
                "attributes": ["young woman", "red hair", "white dress", "oak tree", "sunlit meadow"],
            },
            {
                "prompt": "A futuristic robot with glowing blue eyes and chrome armor, standing in a high-tech laboratory with holographic displays",
                "attributes": ["robot", "glowing blue eyes", "chrome armor", "high-tech lab", "holographic displays"],
            },
            {
                "prompt": "A steaming cup of black coffee with intricate latte art, placed on a rustic wooden table next to an open leather-bound journal",
                "attributes": ["coffee cup", "latte art", "wooden table", "leather journal"],
            },
            {
                "prompt": "A crystal clear mountain lake surrounded by snow-capped peaks and pine trees, reflection of the mountains perfectly mirrored in the still water",
                "attributes": ["mountain lake", "snow-capped peaks", "pine trees", "reflection"],
            },
            {
                "prompt": "A vintage red telephone booth on a foggy London street corner, gas lamps casting warm light on the cobblestone road",
                "attributes": ["red telephone booth", "foggy street", "London", "gas lamps", "cobblestone"],
            },
        ]
        return prompts

    def evaluate(self, image_paths: List[str]) -> Dict:
        """
        Evaluate DPG-Bench score.

        Note: Full DPG-Bench evaluation requires multimodal LLM (DSG/GPT-4V) scoring.
        This implementation provides the framework interface.

        Returns:
            Dict with overall and sub-dimension scores
        """
        prompts_data = self._load_prompts()
        
        print("\n" + "=" * 60)
        print("  DPG-Bench Evaluation")
        print("=" * 60)
        print(f"  DPG-Bench prompts loaded: {len(prompts_data)}")
        print("  WARNING: Full DPG-Bench requires VLM (DSG/GPT-4V) for scoring.")
        print("  This provides the framework. Use official code for paper-level results.")
        print("  Official repo: https://github.com/PRIS-CV/DPG-Bench")

        return {
            "dpg_overall": None,
            "dpg_num_prompts": len(prompts_data),
            "_dpg_note": "Full evaluation requires official DPG-Bench benchmark + VLM scorer",
            "_dpg_framework_ready": True,
        }


# ============================================================================
# 5. ImageReward — Human preference alignment score
# ============================================================================

class ImageRewardEvaluator:
    """
    ImageReward evaluator.

    Principle:
        A reward model based on human preference training proposed by Xu et al. (2024).
        Given an image and its corresponding prompt, the model predicts a scalar reward value
        reflecting human preference for that image. Higher scores are better.

    Model architecture:
        Based on BLIP (ViT-L) as the image encoder,
        with an MLP reward head trained on preference data.

    Dataset:
        100 test prompts from the ImageReward benchmark.

    Installation:
        pip install image-reward
    """

    def __init__(self, device: str = "cuda", model_name: str = "ImageReward-v1.0"):
        self.device = device
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load ImageReward model."""
        if self._model is not None:
            return
        try:
            import ImageReward as RM
            self._model = RM.load(self.model_name, device=self.device)
            print(f"  ImageReward: loaded {self.model_name} [OK]")
        except ImportError:
            print("  ImageReward: image-reward package not installed.")
            print("  Install: pip install image-reward")
            self._model = None

    def _get_builtin_prompts(self) -> List[str]:
        """Built-in ImageReward style prompts."""
        return [
            "A serene mountain lake at sunrise with crystal clear water reflecting snow-capped peaks",
            "A cute orange tabby cat sleeping peacefully on a cozy windowsill with rain outside",
            "A bustling night market in Tokyo with colorful lanterns, steam rising from food stalls",
            "An elegant ballerina performing on a moonlit stage with a spotlight creating dramatic shadows",
            "A majestic eagle soaring above a vast canyon with a golden sunset in the background",
            "A cozy reading nook with a comfortable armchair, bookshelf, and warm fireplace",
            "A field of lavender stretching to the horizon under a purple and orange sunset sky",
            "A cyberpunk city street in the rain with neon signs reflecting on wet pavement -- not a city, only the ground",  # Counter-example / negative case
            "A beautiful cascading waterfall in a lush tropical rainforest with colorful birds flying nearby",
            "A medieval castle on a hill at twilight with glowing windows and a crescent moon above",
        ]

    def compute_single(self, image_path: str, prompt: str) -> Optional[float]:
        """Compute ImageReward score for a single image-prompt pair."""
        if self._model is None:
            return None
        try:
            score = self._model.score(prompt, image_path)
            return float(score)
        except Exception as e:
            print(f"  ImageReward: error scoring {image_path}: {e}")
            return None

    def evaluate(self, image_paths: List[str], prompts: Optional[List[str]] = None) -> Dict:
        """
        Evaluate ImageReward score.

        Args:
            image_paths: List of image paths
            prompts: Optional list of prompts (uses built-in if not provided)
        Returns:
            Dict with mean, std, min, max scores
        """
        self._load_model()

        if prompts is None:
            prompts = self._get_builtin_prompts()

        n = min(len(image_paths), len(prompts))
        
        print("\n" + "=" * 60)
        print("  ImageReward Evaluation")
        print("=" * 60)
        print(f"  Evaluating {n} image-prompt pairs...")

        scores = []
        for i in range(n):
            score = self.compute_single(image_paths[i], prompts[i])
            if score is not None:
                scores.append(score)
            if (i + 1) % 20 == 0:
                print(f"  ImageReward: [{i+1}/{n}]")

        if not scores:
            return {
                "imagereward_mean": None,
                "_imagereward_note": "No valid scores computed",
            }

        result = {
            "imagereward_mean": float(np.mean(scores)),
            "imagereward_std": float(np.std(scores)),
            "imagereward_min": float(np.min(scores)),
            "imagereward_max": float(np.max(scores)),
            "imagereward_num_samples": len(scores),
        }
        print(f"  ImageReward: mean={result['imagereward_mean']:.4f} ± {result['imagereward_std']:.4f}")
        return result


# ============================================================================
# Comprehensive Evaluator
# ============================================================================

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that integrates all 5 metrics.
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.results = {}

    def run(self):
        """Run selected evaluations."""
        args = self.args

        # --- Precompute-only mode: compute + save FID stats, then exit ---
        if args.precompute_fid_stats:
            ref_dir = args.mjhq_path
            if not ref_dir or not os.path.isdir(ref_dir):
                print("ERROR: --precompute_fid_stats requires --mjhq_path pointing to a valid directory")
                return
            save_path = args.fid_ref_stats or os.path.join(
                os.path.dirname(ref_dir.rstrip("/\\")),
                os.path.basename(ref_dir.rstrip("/\\")) + "_fid_stats.npz",
            )
            evaluator = FIDEvaluator(
                device=self.device, image_size=args.fid_image_size,
            )
            evaluator.precompute_and_save_ref_stats(
                ref_dir, save_path, batch_size=args.batch_size,
                sample=args.fid_sample, num_workers=args.num_workers,
                sampling=args.fid_sampling,
            )
            print(f"\nDone! Now run FID without images:")
            print(f"  --fid --fid_ref_stats {save_path}")
            return

        # Determine which metrics to run
        run_fid = args.all or args.fid
        run_clip = args.all or args.clip
        run_geneval = args.all or args.geneval
        run_dpg = args.all or args.dpg
        run_imagereward = args.all or args.imagereward

        # Determine image directories to evaluate
        image_dirs = self._get_image_dirs()
        
        if not image_dirs:
            print("ERROR: No image directories specified. Use --image_dir or --image_dirs.")
            return

        # Setup output
        os.makedirs(args.output_dir, exist_ok=True)
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""

        all_results = {}

        for dir_path in image_dirs:
            dir_name = os.path.basename(dir_path.rstrip("/\\"))
            print(f"\n{'='*70}")
            print(f"Evaluating: {dir_name}")
            print(f"Path: {dir_path}")
            print(f"{'='*70}")

            image_paths = get_image_paths(dir_path)
            if not image_paths:
                print(f"  WARNING: No images found in {dir_path}")
                continue
            
            print(f"  Found {len(image_paths)} images")

            # Auto-detect prompts
            prompts = auto_detect_prompts(dir_path)
            
            dir_results = {"image_dir": dir_path, "num_images": len(image_paths)}

            # 1. FID
            if run_fid:
                print("\n--- FID ---")
                evaluator = FIDEvaluator(
                    device=self.device, image_size=args.fid_image_size,
                )

                # Determine reference source (priority: stats > images)
                ref_stats = None
                ref_dir = None

                # Try loading pre-computed stats first
                if args.fid_ref_stats and os.path.isfile(args.fid_ref_stats):
                    ref_stats = FIDEvaluator.load_ref_stats(args.fid_ref_stats)
                    print(f"  FID: loaded pre-computed stats from {args.fid_ref_stats}")
                elif args.mjhq_path and os.path.isdir(args.mjhq_path):
                    # Auto-check for cached stats alongside image dir
                    auto_stats = args.mjhq_path.rstrip("/\\") + "_fid_stats.npz"
                    if os.path.isfile(auto_stats):
                        ref_stats = FIDEvaluator.load_ref_stats(auto_stats)
                        print(f"  FID: loaded cached stats from {auto_stats}")
                    else:
                        # First run — will compute and auto-cache
                        ref_dir = args.mjhq_path
                        print("  FID: first run on this reference set (will auto-cache stats)")

                if ref_stats is not None or ref_dir is not None:
                    try:
                        fid_score = evaluator.compute_fid(
                            dir_path, ref_dir=ref_dir, ref_stats=ref_stats,
                            num_workers=args.num_workers,
                        )
                        dir_results["fid"] = fid_score

                        # Auto-cache: if we used ref_dir, save stats for next time
                        if ref_dir is not None and ref_stats is None:
                            auto_save = ref_dir.rstrip("/\\") + "_fid_stats.npz"
                            evaluator.precompute_and_save_ref_stats(
                                ref_dir, auto_save, num_workers=args.num_workers,
                                sampling=args.fid_sampling,
                            )
                            print(f"  FID: stats auto-cached to {auto_save} (next run will be fast)")
                    except Exception as e:
                        print(f"  FID failed: {e}")
                        dir_results["fid"] = None
                else:
                    print("  FID SKIPPED: need --mjhq_path OR --fid_ref_stats")
                    print("  First-time: --mjhq_path <images_dir>")
                    print("  Thereafter: --fid_ref_stats <path_to_stats.npz>")
                    dir_results["fid"] = None

            # 2. CLIP Score
            if run_clip:
                print("\n--- CLIP Score ---")
                try:
                    evaluator = CLIPScoreEvaluator(
                        device=self.device,
                        model_name=args.clip_model,
                        pretrained=args.clip_pretrained,
                    )
                    n = min(len(image_paths), len(prompts))
                    clip_results = evaluator.compute_batch(
                        image_paths[:n], prompts[:n],
                    )
                    dir_results.update(clip_results)
                except Exception as e:
                    print(f"  CLIP Score failed: {e}")
                    dir_results["clip_mean"] = None

            # 3. GenEval
            if run_geneval:
                print("\n--- GenEval ---")
                evaluator = GenEvalEvaluator(
                    device=self.device,
                    geneval_dir=args.geneval_dir,
                )
                geneval_results = evaluator.evaluate(
                    image_paths, 
                    os.path.join(dir_path, "prompts.txt"),
                )
                dir_results.update(geneval_results)

            # 4. DPG-Bench
            if run_dpg:
                print("\n--- DPG-Bench ---")
                evaluator = DPGBenchEvaluator(
                    device=self.device,
                    dpg_dir=args.dpg_dir,
                )
                dpg_results = evaluator.evaluate(image_paths)
                dir_results.update(dpg_results)

            # 5. ImageReward
            if run_imagereward:
                print("\n--- ImageReward ---")
                evaluator = ImageRewardEvaluator(
                    device=self.device,
                    model_name=args.imagereward_model,
                )
                n = min(len(image_paths), len(prompts))
                ir_results = evaluator.evaluate(image_paths[:n], prompts[:n])
                dir_results.update(ir_results)

            all_results[dir_name] = dir_results

            # Cleanup
            torch.cuda.empty_cache()

        # ---- Save Results ----
        self._save_results(all_results, suffix)

    def _get_image_dirs(self) -> List[str]:
        """Get list of image directories to evaluate."""
        dirs = []
        if self.args.image_dir:
            dirs.append(self.args.image_dir)
        if self.args.image_dirs:
            dirs.extend(self.args.image_dirs)
        return dirs

    def _save_results(self, all_results: Dict, suffix: str):
        """Save evaluation results to files."""
        output_dir = self.args.output_dir

        # CSV
        csv_path = os.path.join(output_dir, f"eval_results{suffix}.csv")
        # Flatten all results
        flat_results = []
        for dir_name, metrics in all_results.items():
            row = {"model": dir_name}
            row.update({k: v for k, v in metrics.items() if k != "image_dir"})
            flat_results.append(row)
        
        if flat_results:
            keys = flat_results[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flat_results)
            print(f"\nCSV saved: {csv_path}")

        # JSON
        json_path = os.path.join(output_dir, f"eval_results{suffix}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"JSON saved: {json_path}")

        # Summary txt
        txt_path = os.path.join(output_dir, f"eval_summary{suffix}.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")
            for dir_name, metrics in all_results.items():
                f.write(f"Model: {dir_name}\n")
                f.write(f"  Images: {metrics.get('num_images', 'N/A')}\n")
                if metrics.get("fid") is not None:
                    f.write(f"  FID:        {metrics['fid']:.2f}\n")
                if metrics.get("clip_mean") is not None:
                    f.write(f"  CLIP Score: {metrics['clip_mean']:.4f} ± {metrics['clip_std']:.4f}\n")
                if metrics.get("imagereward_mean") is not None:
                    f.write(f"  ImageReward:{metrics['imagereward_mean']:.4f} ± {metrics['imagereward_std']:.4f}\n")
                f.write("\n")
        print(f"Summary saved: {txt_path}")


# ============================================================================
# Convenience function: Benchmark download instructions
# ============================================================================

def print_download_instructions():
    """Print instructions for downloading benchmark resources."""
    print("""
    ==========================================================================
    Benchmark Dataset Download Guide
    ==========================================================================
    
    1. MJHQ-30K (FID / CLIP Score reference set):
       Official URL: https://huggingface.co/datasets/playgroundai/MJHQ-30K
       Download and extract locally, specify path via --mjhq_path.
    
    2. GenEval Benchmark (533 prompts):
       Official repo: https://github.com/djghosh13/geneval
       git clone https://github.com/djghosh13/geneval.git
       Specify clone directory via --geneval_dir.
    
    3. DPG-Bench (1,065 prompts):
       Official repo: https://github.com/PRIS-CV/DPG-Bench
       git clone https://github.com/PRIS-CV/DPG-Bench.git
       Specify clone directory via --dpg_dir.
    
    4. ImageReward (100 prompts + model):
       Install: pip install image-reward
       Model will be downloaded automatically.
    
    ==========================================================================
    """)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Check if any metric is enabled (or precompute-only mode)
    any_enabled = args.all or args.fid or args.clip or args.geneval or args.dpg or args.imagereward or args.precompute_fid_stats
    if not any_enabled:
        print("No metrics selected. Use --all to run all metrics, or specify individual flags.")
        print("\nExample usage:")
        print("  python eval/eval_comprehensive.py --image_dir outputs/my_model/ --fid --clip")
        print("  python eval/eval_comprehensive.py --image_dir outputs/my_model/ --all")
        print("  python eval/eval_comprehensive.py --precompute_fid_stats --mjhq_path G://datasets/MJHQ-30K")
        print_download_instructions()
        sys.exit(1)

    evaluator = ComprehensiveEvaluator(args)
    evaluator.run()

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    # print_download_instructions()
