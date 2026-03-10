"""
Nearest-Neighbour (NN) interpolation baseline for KITTI depth completion.

This is the simplest possible baseline: take the sparse Velodyne depth map
and fill every empty pixel with the depth of its nearest valid neighbour
using ``scipy.ndimage.distance_transform_edt``.

Usage
-----
    python -m scripts.run_nn_baseline            # uses defaults
    python -m scripts.run_nn_baseline --max-vis 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.data_loader import KITTIDepthDataset
from src.depth_utils import depth_to_colormap, write_depth
from src.metrics import (
    DepthMetrics,
    aggregate_metrics,
    compute_depth_metrics,
    print_metrics,
)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def nn_interpolate(sparse_depth: np.ndarray) -> np.ndarray:
    """Fill missing pixels by nearest-neighbour interpolation.

    Parameters
    ----------
    sparse_depth : np.ndarray (H, W), float32
        Sparse depth map.  0 = invalid / missing.

    Returns
    -------
    dense_depth : np.ndarray (H, W), float32
        Dense depth map with all pixels filled.
    """
    mask = sparse_depth <= 0
    _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    dense = sparse_depth[nearest_idx[0], nearest_idx[1]]
    return dense.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NN interpolation baseline for KITTI depth completion"
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "test_completion"],
        help="Dataset split to run on (default: val)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override default data root folder",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save predicted depth maps "
             "(default: data/output/nn_baseline/<split>)",
    )
    parser.add_argument(
        "--max-vis", type=int, default=10,
        help="Number of colour-mapped visualisations to save (0 = none)",
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save 16-bit PNG predictions for all samples",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "data" / "output" / "nn_baseline" / args.split
    )
    vis_dir = out_dir / "vis"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = KITTIDepthDataset.from_split(args.split, data_root=args.data_root)
    print(ds.summary())

    all_metrics: list[DepthMetrics] = []
    t0 = time.time()

    for i in range(len(ds)):
        sample = ds[i]
        velodyne = sample["velodyne"]
        if velodyne is None:
            print(f"  [{i}] {sample['stem']}  – skipped (no velodyne)")
            continue

        # Run NN interpolation
        pred = nn_interpolate(velodyne)

        # Save prediction
        if args.save_predictions:
            write_depth(pred_dir / f"{sample['stem']}.png", pred)

        # Evaluate against GT (if available)
        gt = sample.get("gt_depth")
        if gt is not None:
            m = compute_depth_metrics(gt, pred)
            all_metrics.append(m)
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i + 1}/{len(ds)}] {sample['stem']}  {m}")

        # Save visualisation
        if i < args.max_vis:
            import cv2

            h, w = sample["image"].shape[:2]
            panels = [
                sample["image"][..., ::-1],                      # RGB → BGR
                depth_to_colormap(velodyne, max_depth=80),
                depth_to_colormap(pred, max_depth=80),
            ]
            if gt is not None:
                panels.append(depth_to_colormap(gt, max_depth=80))
            # Resize panels to same height if needed
            panels = [
                cv2.resize(p, (w, h)) if p.shape[:2] != (h, w) else p
                for p in panels
            ]
            mosaic = np.concatenate(panels, axis=1)
            cv2.imwrite(str(vis_dir / f"{sample['stem']}.png"), mosaic)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(ds)} samples in {elapsed:.1f}s")

    if all_metrics:
        agg = aggregate_metrics(all_metrics)
        print_metrics(agg)

        # Save to text file
        stats_path = out_dir / "stats_depth.txt"
        with open(stats_path, "w") as f:
            for k, v in agg.items():
                f.write(f"{k}: {v}\n")
        print(f"\nResults saved to {stats_path}")


if __name__ == "__main__":
    main()
