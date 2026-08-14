"""Comprehensive evaluation entry point for text-to-image models.

Currently supports:
  - FID (Frechet Inception Distance)

Usage:

    # Precompute FID reference stats for coco2017
    python -m eval.main --dataset_name coco2017 --dataset_path G:/datasets/COCO2017 \
        --precompute_fid_stats --fid_ref_stats G:/datasets/coco2017_fid_stats.npz

    # Compute FID for generated images
    python -m eval.main --fid --dataset_name coco2017 \
        --input_dir G:/Outputs/Efficient-Diffusion/eval_gen/sana-mjhq30k \
        --fid_ref_stats G:/datasets/coco2017_fid_stats.npz
"""

import argparse
import json
import os

import torch

from src.eval.fid import FID


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for text-to-image models",
    )
    # Metric flags
    parser.add_argument("--all", action="store_true", help="Run all available metrics")
    parser.add_argument("--fid", action="store_true", help="Compute FID")
    parser.add_argument("--clip", action="store_true", help="Compute CLIP Score")
    parser.add_argument("--geneval", action="store_true", help="Compute GenEval score")
    parser.add_argument("--dpg", action="store_true", help="Compute DPG-Bench score")
    parser.add_argument("--imagereward", action="store_true", help="Compute ImageReward score")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="coco2017",
                        help="Reference dataset name: coco2017, mjhq-30k, cifar100, mnist")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset (for mjhq-30k or local coco2017). "
                             "If None, coco2017/mnist/cifar100 download automatically.")

    # FID
    parser.add_argument("--precompute_fid_stats", action="store_true",
                        help="Only precompute and save FID reference stats, then exit.")
    parser.add_argument("--fid_sample", type=int, default=None,
                        help="Max number of reference images for FID stats (sequential sampling)")
    parser.add_argument("--fid_ref_stats", type=str, default=None,
                        help="Path to pre-computed FID reference stats (.npz)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process, avoids "
                             "multiprocessing spawn issues with eval package name)")

    # I/O
    parser.add_argument("--input_dir", type=str,
                        default="G://Outputs//Efficient-Diffusion//eval_gen//Sana_origin_mqjh",
                        help="Directory of generated images (for FID computation)")
    parser.add_argument("--output_dir", type=str,
                        default="G://Outputs//Efficient-Diffusion//Sana_origin_mqjh",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # ---- FID ----
    if args.fid or args.all:
        fid = FID(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Mode 1: precompute reference stats only
        if args.precompute_fid_stats:
            if args.fid_ref_stats is None:
                # Auto-generate path based on dataset name
                stats_name = f"{args.dataset_name}_fid_stats.npz"
                args.fid_ref_stats = os.path.join(
                    os.path.dirname(args.output_dir.rstrip("/\\")), stats_name,
                )
            fid.precompute_ref_stats(args.fid_ref_stats, n_sample=args.fid_sample)
            print(f"\n[done] FID reference stats saved to {args.fid_ref_stats}")
            print(f"  Now compute FID with:")
            print(f"  python -m eval.main --fid --dataset_name {args.dataset_name} "
                  f"--input_dir <gen_dir> --fid_ref_stats {args.fid_ref_stats}")
        else:
            # Mode 2: compute FID
            if args.fid_ref_stats is None:
                print("ERROR: --fid_ref_stats is required for FID computation.")
                print("  First precompute: --precompute_fid_stats --fid_ref_stats <path.npz>")
                exit(1)

            score = fid.compute_fid(args.input_dir, ref_stats_path=args.fid_ref_stats)

            os.makedirs(args.output_dir, exist_ok=True)
            results = {
                "dataset_name": args.dataset_name,
                "input_dir": args.input_dir,
                "fid_ref_stats": args.fid_ref_stats,
                "fid": score,
            }
            results_path = os.path.join(args.output_dir, "fid_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n[done] FID results saved to {results_path}")
