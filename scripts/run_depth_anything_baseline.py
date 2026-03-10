"""
Depth Anything v2 baseline for KITTI depth prediction.

Runs the pre-trained Depth Anything v2 model on every image in a KITTI
split, rescales its relative predictions to metric depth using median
scaling against available GT, and evaluates using standard KITTI metrics.

Dependencies
------------
    pip install torch torchvision transformers pillow

Usage
-----
    python -m scripts.run_depth_anything_baseline                      # defaults
    python -m scripts.run_depth_anything_baseline --encoder vitl       # larger model
    python -m scripts.run_depth_anything_baseline --max-vis 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.data_loader import KITTIDepthDataset
from src.depth_utils import depth_to_colormap, valid_mask, write_depth
from src.metrics import (
    DepthMetrics,
    aggregate_metrics,
    compute_depth_metrics,
    print_metrics,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_depth_anything_v2(
    encoder: str = "vits",
    device: Optional[str] = None,
):
    """Load a pretrained Depth Anything v2 model via HuggingFace
    Transformers pipeline (``depth-estimation``).

    Parameters
    ----------
    encoder : {"vits", "vitb", "vitl"}
        Backbone size.  ``vits`` is fastest and fits in 6 GB VRAM.
    device : str, optional
        "cuda" or "cpu".  Auto-detected if ``None``.

    Returns
    -------
    pipe : transformers.Pipeline
        A HuggingFace depth-estimation pipeline.
    """
    from transformers import pipeline

    _ENCODER_TO_MODEL = {"vits": "Small", "vitb": "Base", "vitl": "Large"}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = f"depth-anything/Depth-Anything-V2-{_ENCODER_TO_MODEL[encoder]}-hf"
    print(f"Loading {model_id} on {device} ...")
    pipe = pipeline(
        "depth-estimation",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return pipe


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_relative_depth(
    pipe,
    image: np.ndarray,
) -> np.ndarray:
    """Run the pipeline on a single RGB image and return relative depth.

    Parameters
    ----------
    pipe : transformers.Pipeline
    image : np.ndarray (H, W, 3) uint8 RGB

    Returns
    -------
    rel_depth : np.ndarray (H, W) float32
        Relative / affine-invariant inverse-depth (higher = closer).
    """
    pil_img = Image.fromarray(image)
    result = pipe(pil_img)
    # The pipeline returns {"depth": PIL.Image, "predicted_depth": Tensor}
    depth_tensor = result["predicted_depth"]  # (1, H', W') or (H', W')
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.squeeze(0)
    rel_depth = depth_tensor.cpu().numpy().astype(np.float32)

    # Resize back to input resolution if needed
    h, w = image.shape[:2]
    if rel_depth.shape != (h, w):
        from PIL import Image as PILImage

        rel_depth = np.array(
            PILImage.fromarray(rel_depth).resize((w, h), PILImage.BILINEAR)
        )
    return rel_depth


def median_scale(
    rel_depth: np.ndarray,
    gt_depth: np.ndarray,
) -> np.ndarray:
    """Align relative depth to metric scale using median scaling.

    For every valid GT pixel, we compute:
        scale = median(gt / rel)
    and return ``rel * scale``.

    This is the standard "median scaling" trick from Eigen et al.
    """
    mask = valid_mask(gt_depth) & (rel_depth > 1e-6)
    if mask.sum() < 10:
        return rel_depth  # not enough valid pixels
    ratio = gt_depth[mask] / rel_depth[mask]
    scale = float(np.median(ratio))
    return rel_depth * scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth Anything v2 baseline for KITTI depth prediction"
    )
    parser.add_argument(
        "--split", default="val",
        choices=["val", "test_completion", "test_prediction"],
        help="Dataset split (default: val)",
    )
    parser.add_argument(
        "--encoder", default="vits", choices=["vits", "vitb", "vitl"],
        help="Backbone size (default: vits – fits RTX 4050 6 GB)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: data/output/depth_anything_v2/<split>)",
    )
    parser.add_argument(
        "--max-vis", type=int, default=10,
        help="Number of visualisations to save",
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save 16-bit PNG metric-depth predictions",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "data" / "output" / "depth_anything_v2" / args.split
    )
    vis_dir = out_dir / "vis"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset & model
    ds = KITTIDepthDataset.from_split(args.split, data_root=args.data_root)
    print(ds.summary())
    pipe = load_depth_anything_v2(encoder=args.encoder)

    all_metrics: list[DepthMetrics] = []
    t0 = time.time()

    for i in range(len(ds)):
        sample = ds[i]
        image = sample["image"]
        gt = sample.get("gt_depth")

        # Predict relative depth
        rel = predict_relative_depth(pipe, image)

        # Align to metric scale with median scaling (requires GT)
        if gt is not None:
            pred = median_scale(rel, gt)
        else:
            # Without GT we cannot scale; save raw relative depth
            pred = rel

        # Save prediction
        if args.save_predictions:
            write_depth(pred_dir / f"{sample['stem']}.png", pred)

        # Evaluate
        if gt is not None:
            m = compute_depth_metrics(gt, pred)
            all_metrics.append(m)
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i + 1}/{len(ds)}] {sample['stem']}  {m}")

        # Save visualisation
        if i < args.max_vis:
            import cv2

            h, w = image.shape[:2]
            panels = [
                image[..., ::-1],  # RGB → BGR
                depth_to_colormap(pred, max_depth=80),
            ]
            if gt is not None:
                panels.append(depth_to_colormap(gt, max_depth=80))
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

        stats_path = out_dir / "stats_depth.txt"
        with open(stats_path, "w") as f:
            for k, v in agg.items():
                f.write(f"{k}: {v}\n")
        print(f"\nResults saved to {stats_path}")


if __name__ == "__main__":
    main()
