"""
Standalone evaluation script.

Evaluates a directory of 16-bit PNG depth predictions against the
KITTI val_selection_cropped ground truth.

Usage
-----
    python -m scripts.evaluate --pred-dir data/output/nn_baseline/val/predictions
    python -m scripts.evaluate --pred-dir data/output/depth_anything_v2/val/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import KITTIDepthDataset
from src.depth_utils import read_depth_safe
from src.metrics import (
    DepthMetrics,
    aggregate_metrics,
    compute_depth_metrics,
    print_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted depth maps against KITTI GT"
    )
    parser.add_argument(
        "--pred-dir", type=str, required=True,
        help="Directory containing predicted 16-bit PNG depth maps",
    )
    parser.add_argument(
        "--split", default="val", choices=["val"],
        help="GT split (only 'val' has ground truth)",
    )
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ds = KITTIDepthDataset.from_split(args.split, data_root=args.data_root)

    all_metrics: list[DepthMetrics] = []
    missing = 0

    for i in range(len(ds)):
        sample = ds[i]
        gt = sample.get("gt_depth")
        if gt is None:
            continue

        pred_path = pred_dir / f"{sample['stem']}.png"
        if not pred_path.exists():
            missing += 1
            continue

        pred = read_depth_safe(pred_path)
        m = compute_depth_metrics(gt, pred)
        all_metrics.append(m)

    if missing:
        print(f"Warning: {missing} prediction files not found in {pred_dir}")

    if all_metrics:
        agg = aggregate_metrics(all_metrics)
        print_metrics(agg)
    else:
        print("No valid samples evaluated.")


if __name__ == "__main__":
    main()
