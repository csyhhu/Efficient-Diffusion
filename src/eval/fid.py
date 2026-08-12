"""FID (Frechet Inception Distance) evaluation.

Supports reference datasets: mjhq-30k, coco2017, cifar100, mnist.
For path-based datasets (mjhq, coco2017) image file paths are obtained
via ``src.data_loader`` functions. For in-memory datasets (cifar100,
mnist) the DataLoader from ``src.data_loader.get_dataloader`` is iterated
directly.

Usage::

    from src.eval.fid import FID

    # Precompute reference stats
    fid = FID("coco2017", dataset_path="G:/datasets/COCO2017")
    fid.precompute_ref_stats("G:/datasets/coco2017_fid_stats.npz")

    # Compute FID using precomputed stats
    fid = FID("coco2017")
    score = fid.compute_fid("outputs/gen_images/",
                            ref_stats_path="G:/datasets/coco2017_fid_stats.npz")
"""

import os
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level Dataset for path-based feature extraction (picklable on Windows)
# ---------------------------------------------------------------------------

class _FIDImageDataset(Dataset):
    """Dataset for FID feature extraction from image file paths."""

    def __init__(self, paths: List[str], img_size: int):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1)


class FID:
    """Frechet Inception Distance evaluator.

    Supports reference datasets via ``src.data_loader``:
      - ``mjhq-30k`` / ``mjhq``    -- path-based (images on disk)
      - ``coco2017``               -- path-based (val2017 images on disk)
      - ``cifar100`` / ``cifar``   -- loader-based (torchvision binary)
      - ``mnist``                  -- loader-based (torchvision binary, grayscale)

    Workflow:
      1. ``precompute_ref_stats(save_path)`` -- one-time: extract InceptionV3
         features from reference images, compute mu/sigma, save .npz.
      2. ``compute_fid(gen_dir, ref_stats_path)`` -- extract features from
         generated images, compute FID vs precomputed reference stats.
    """

    # Datasets that provide image file paths
    _PATH_DATASETS = {"mjhq30k", "coco2017"}
    # Datasets that require DataLoader iteration
    _LOADER_DATASETS = {"cifar100", "mnist"}

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 299,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        key = dataset_name.lower().replace("-", "").replace("_", "")
        if key in ("mjhq30k", "mjhq"):
            self.dataset_key = "mjhq30k"
        elif key in ("coco2017", "coco"):
            self.dataset_key = "coco2017"
        elif key in ("cifar100", "cifar"):
            self.dataset_key = "cifar100"
        elif key == "mnist":
            self.dataset_key = "mnist"
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Supported: mjhq-30k, coco2017, cifar100, mnist"
            )

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._inception = None

    # ------------------------------------------------------------------
    # InceptionV3
    # ------------------------------------------------------------------

    def _get_inception(self):
        """Lazy-load InceptionV3 feature extractor (pool3 -> 2048-dim)."""
        if self._inception is None:
            from torchvision import models
            inception = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT,
                transform_input=False,
            )
            inception.fc = torch.nn.Identity()
            inception.eval().to(self.device)
            self._inception = inception
        return self._inception

    def _extract_features(self, imgs_uint8: torch.Tensor) -> torch.Tensor:
        """Extract InceptionV3 pool3 features from a batch of uint8 images.

        Args:
            imgs_uint8: (N, 3, H, W) uint8 in [0, 255]
        Returns:
            features: (N, 2048) on CPU
        """
        from torchvision.transforms import functional as F_t

        inception = self._get_inception()
        imgs_resized = F_t.resize(imgs_uint8.float(), [299, 299])
        imgs_norm = F_t.normalize(
            imgs_resized / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        with torch.no_grad():
            return inception(imgs_norm).cpu()

    # ------------------------------------------------------------------
    # Reference image collection
    # ------------------------------------------------------------------

    def _get_ref_image_paths(self, n_sample: Optional[int] = None) -> List[str]:
        """Get image file paths for path-based datasets (mjhq30k, coco2017)."""
        max_total = n_sample if n_sample and n_sample > 0 else 10 ** 9

        if self.dataset_key == "mjhq30k":
            from src.data_loader import _build_paths_and_captions
            image_paths, _, _ = _build_paths_and_captions(
                "mjhq30k", max_total=max_total, dataset_path=self.dataset_path,
            )
            return image_paths

        if self.dataset_key == "coco2017":
            from src.data_loader import get_dataloader
            config = {
                "batch_size": self.batch_size,
                "data_dir": self.dataset_path,
                "image_size": self.image_size,
                "max_samples": n_sample if n_sample and n_sample > 0 else -1,
                "num_workers": self.num_workers,
                "pin_memory": False,
            }
            # train=True gives 90% split (4500 images); enough for FID
            train_loader, _ = get_dataloader("coco2017", config)
            return train_loader.dataset.image_paths

        return None

    def _get_ref_loader(self):
        """Get DataLoader for loader-based datasets (cifar100, mnist).

        Returns:
            (loader, is_grayscale) tuple
        """
        from src.data_loader import get_dataloader

        if self.dataset_key == "cifar100":
            config = {
                "batch_size": self.batch_size,
                "data_dir": self.dataset_path or "./data",
                "num_workers": self.num_workers,
                "pin_memory": False,
            }
            _, val_loader = get_dataloader("cifar100", config)
            return val_loader, False

        if self.dataset_key == "mnist":
            config = {
                "batch_size": self.batch_size,
                "data_dir": self.dataset_path or "./data",
                "num_workers": self.num_workers,
                "pin_memory": False,
            }
            _, val_loader = get_dataloader("mnist", config)
            return val_loader, True

        return None, False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features_from_paths(
        self, paths: List[str], n_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from image file paths via DataLoader."""
        # Apply sequential sampling
        if n_sample and n_sample > 0 and n_sample < len(paths):
            paths = paths[:n_sample]
            print(f"  FID: using {n_sample}/{len(paths) + n_sample} images (sequential)")
        print(f"  FID: processing {len(paths)} images from paths")

        dataset = _FIDImageDataset(paths, self.image_size)
        effective_workers = self.num_workers if self.num_workers > 0 else 0
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=effective_workers, pin_memory=(self.device == "cuda"),
            prefetch_factor=2 if effective_workers > 0 else None,
        )
        return self._extract_features_from_loader_impl(loader, n_total=len(paths))

    def _extract_features_from_loader_impl(
        self, loader: DataLoader, n_total: Optional[int] = None,
    ) -> torch.Tensor:
        """Run InceptionV3 feature extraction on a DataLoader with progress."""
        if n_total is None:
            n_total = len(loader.dataset)
        feats = []
        t_start = time.time()
        for i, batch in enumerate(loader):
            # batch may be (image,) or (image, label, prompt) etc.
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(self.device, dtype=torch.uint8, non_blocking=True)
            feats.append(self._extract_features(imgs))
            done = min((i + 1) * self.batch_size, n_total)
            pct = done / n_total * 100
            elapsed = time.time() - t_start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / speed if speed > 0 else 0
            print(f"\r  FID: [{done:>6d}/{n_total}] {pct:5.1f}% | "
                  f"{speed:6.1f} img/s | ETA {self._fmt_time(eta)}", end="", flush=True)
        print()
        return torch.cat(feats, dim=0)

    def _extract_features_from_loader(
        self, loader: DataLoader, is_grayscale: bool = False,
        n_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from a DataLoader (for cifar100, mnist).

        Converts images from [-1, 1] float to [0, 255] uint8,
        repeats grayscale to 3 channels if needed.
        """
        n_total = len(loader.dataset)
        if n_sample and n_sample > 0:
            n_total = min(n_total, n_sample)
        print(f"  FID: processing {n_total} images from DataLoader "
              f"(grayscale={is_grayscale})")

        feats = []
        count = 0
        t_start = time.time()
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            # Convert from [-1, 1] to [0, 255] uint8
            imgs = ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
            # Grayscale -> RGB by repeating channels
            if is_grayscale and imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            imgs = imgs.to(self.device)
            feats.append(self._extract_features(imgs))

            count += imgs.shape[0]
            pct = count / n_total * 100
            elapsed = time.time() - t_start
            speed = count / elapsed if elapsed > 0 else 0
            eta = (n_total - count) / speed if speed > 0 else 0
            print(f"\r  FID: [{count:>6d}/{n_total}] {pct:5.1f}% | "
                  f"{speed:6.1f} img/s | ETA {self._fmt_time(eta)}", end="", flush=True)
            if n_sample and count >= n_sample:
                break
        print()
        return torch.cat(feats, dim=0)

    def _extract_features_from_dir(self, dir_path: str) -> torch.Tensor:
        """Extract features from all images in a directory (recursive)."""
        exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        all_paths = []
        for root, _, files in os.walk(dir_path):
            for f in files:
                if os.path.splitext(f.lower())[1] in exts:
                    all_paths.append(os.path.join(root, f))
        all_paths = sorted(all_paths)
        if not all_paths:
            raise ValueError(f"No images found in {dir_path}")
        print(f"  FID: found {len(all_paths)} images in {dir_path}")
        return self._extract_features_from_paths(all_paths)

    # ------------------------------------------------------------------
    # Reference statistics
    # ------------------------------------------------------------------

    def _get_ref_features(self, n_sample: Optional[int] = None) -> torch.Tensor:
        """Get reference features, dispatching based on dataset type."""
        if self.dataset_key in self._PATH_DATASETS:
            paths = self._get_ref_image_paths(n_sample)
            if not paths:
                raise ValueError(f"No reference images found for {self.dataset_key}")
            return self._extract_features_from_paths(paths, n_sample=n_sample)
        else:
            loader, is_grayscale = self._get_ref_loader()
            return self._extract_features_from_loader(
                loader, is_grayscale=is_grayscale, n_sample=n_sample,
            )

    def precompute_ref_stats(
        self, save_path: str, n_sample: Optional[int] = None,
    ) -> dict:
        """Precompute reference FID statistics (mu, sigma) and save as .npz.

        Args:
            save_path: Output .npz file path.
            n_sample: Max images to use (None = all).
        Returns:
            {"mu": np.ndarray (2048,), "sigma": np.ndarray (2048, 2048), "n": int}
        """
        print(f"\n{'=' * 60}")
        print(f"  FID: precomputing reference stats for '{self.dataset_key}'")
        print(f"{'=' * 60}")

        feats = self._get_ref_features(n_sample)
        n = feats.shape[0]
        print(f"  FID: {n} images processed, computing statistics...")

        mu = feats.mean(dim=0).cpu().numpy()
        sigma = torch.cov(feats.T).cpu().numpy()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(save_path, mu=mu, sigma=sigma, n=n)
        print(f"  FID: stats saved to {save_path}")
        print(f"  FID: mu shape={mu.shape}, sigma shape={sigma.shape}")
        return {"mu": mu, "sigma": sigma, "n": n}

    @staticmethod
    def load_ref_stats(npz_path: str) -> dict:
        """Load pre-computed reference statistics from .npz file."""
        data = np.load(npz_path)
        return {
            "mu": data["mu"],
            "sigma": data["sigma"],
            "n": int(data["n"]) if "n" in data else None,
        }

    # ------------------------------------------------------------------
    # FID computation
    # ------------------------------------------------------------------

    def compute_fid(
        self,
        gen_dir: str,
        ref_stats_path: Optional[str] = None,
    ) -> float:
        """Compute FID score between generated images and reference stats.

        Args:
            gen_dir: Directory of generated images.
            ref_stats_path: Path to pre-computed .npz reference stats.
        Returns:
            FID score (lower is better).
        """
        if ref_stats_path is None:
            raise ValueError("ref_stats_path is required (use precompute_ref_stats first)")

        print(f"\n{'=' * 60}")
        print(f"  FID: computing score")
        print(f"{'=' * 60}")

        # Load reference stats
        ref_stats = self.load_ref_stats(ref_stats_path)
        mu2 = torch.from_numpy(ref_stats["mu"]).float()
        sigma2 = torch.from_numpy(ref_stats["sigma"]).float()
        print(f"  FID: loaded ref stats (n={ref_stats['n']}) from {ref_stats_path}")

        # Extract generated image features
        gen_feats = self._extract_features_from_dir(gen_dir)
        mu1 = gen_feats.mean(dim=0)
        sigma1 = torch.cov(gen_feats.T)

        # FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 * sigma2))
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

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = int(seconds)
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
