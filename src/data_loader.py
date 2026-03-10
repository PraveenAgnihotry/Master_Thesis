"""
KITTI Depth dataset loader.

Supports three splits for the KITTI depth benchmark:
    - val_selection_cropped   (has GT + velodyne + image + intrinsics)
    - test_depth_completion   (image + velodyne, no GT)
    - test_depth_prediction   (image only, no GT / velodyne)

Usage
-----
>>> from src.data_loader import KITTIDepthDataset
>>> ds = KITTIDepthDataset.from_split("val")          # auto-detects paths
>>> sample = ds[0]
>>> sample["image"].shape        # (H, W, 3)  uint8  RGB
>>> sample["gt_depth"].shape     # (H, W)     float32  metres
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src.depth_utils import read_depth, read_intrinsics, valid_mask

# ---------------------------------------------------------------------------
# Default root paths (resolved relative to repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "input" / "depth_selection"

# Regex to decompose a val_selection_cropped filename into components
# e.g.  2011_09_26_drive_0002_sync_image_0000000005_image_02.png
_VAL_IMAGE_RE = re.compile(
    r"^(?P<prefix>.+_sync)_image_(?P<frame>\d{10})_(?P<cam>image_\d{2})\.png$"
)


# ---------------------------------------------------------------------------
# Single sample container
# ---------------------------------------------------------------------------

@dataclass
class DepthSample:
    """Lightweight container for a single dataset sample."""

    stem: str                                       # unique sample id
    image_path: Path                                # RGB image
    gt_depth_path: Optional[Path] = None            # ground-truth depth map
    velodyne_path: Optional[Path] = None            # sparse LiDAR depth
    intrinsics_path: Optional[Path] = None          # camera intrinsics

    # Lazy-loaded numpy arrays (populated on first access via dataset)
    _cache: Dict[str, np.ndarray] = field(
        default_factory=dict, repr=False
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class KITTIDepthDataset:
    """Iterable / indexable dataset over KITTI depth samples.

    Parameters
    ----------
    samples : list[DepthSample]
        Pre-built list of samples.
    transform : callable, optional
        Function applied to the dict returned by ``__getitem__``.
    """

    def __init__(
        self,
        samples: List[DepthSample],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_split(
        cls,
        split: str = "val",
        data_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
    ) -> "KITTIDepthDataset":
        """Build a dataset from one of the standard KITTI depth splits.

        Parameters
        ----------
        split : {"val", "test_completion", "test_prediction"}
        data_root : Path, optional
            Override the default ``data/input/depth_selection`` root.
        transform : callable, optional
        """
        root = Path(data_root) if data_root else _DATA_ROOT

        if split == "val":
            return cls._build_val(root / "val_selection_cropped", transform)
        elif split == "test_completion":
            return cls._build_test_completion(
                root / "test_depth_completion_anonymous", transform
            )
        elif split == "test_prediction":
            return cls._build_test_prediction(
                root / "test_depth_prediction_anonymous", transform
            )
        else:
            raise ValueError(
                f"Unknown split '{split}'. "
                "Choose from: val, test_completion, test_prediction"
            )

    # ---- val_selection_cropped -------------------------------------------

    @classmethod
    def _build_val(
        cls, base: Path, transform: Optional[Callable]
    ) -> "KITTIDepthDataset":
        image_dir = base / "image"
        gt_dir = base / "groundtruth_depth"
        vel_dir = base / "velodyne_raw"
        intr_dir = base / "intrinsics"

        samples: List[DepthSample] = []
        for img_path in sorted(image_dir.glob("*.png")):
            m = _VAL_IMAGE_RE.match(img_path.name)
            if m is None:
                continue
            prefix, frame, cam = m.group("prefix"), m.group("frame"), m.group("cam")
            stem = f"{prefix}_{frame}_{cam}"

            gt_path = gt_dir / f"{prefix}_groundtruth_depth_{frame}_{cam}.png"
            vel_path = vel_dir / f"{prefix}_velodyne_raw_{frame}_{cam}.png"
            intr_path = intr_dir / f"{prefix}_image_{frame}_{cam}.txt"

            samples.append(
                DepthSample(
                    stem=stem,
                    image_path=img_path,
                    gt_depth_path=gt_path if gt_path.exists() else None,
                    velodyne_path=vel_path if vel_path.exists() else None,
                    intrinsics_path=intr_path if intr_path.exists() else None,
                )
            )

        print(f"[KITTIDepthDataset] val split: {len(samples)} samples")
        return cls(samples, transform)

    # ---- test_depth_completion -------------------------------------------

    @classmethod
    def _build_test_completion(
        cls, base: Path, transform: Optional[Callable]
    ) -> "KITTIDepthDataset":
        image_dir = base / "image"
        vel_dir = base / "velodyne_raw"
        intr_dir = base / "intrinsics"

        samples: List[DepthSample] = []
        for img_path in sorted(image_dir.glob("*.png")):
            stem = img_path.stem
            vel_path = vel_dir / img_path.name
            intr_path = intr_dir / f"{stem}.txt"

            samples.append(
                DepthSample(
                    stem=stem,
                    image_path=img_path,
                    gt_depth_path=None,
                    velodyne_path=vel_path if vel_path.exists() else None,
                    intrinsics_path=intr_path if intr_path.exists() else None,
                )
            )

        print(f"[KITTIDepthDataset] test_completion split: {len(samples)} samples")
        return cls(samples, transform)

    # ---- test_depth_prediction -------------------------------------------

    @classmethod
    def _build_test_prediction(
        cls, base: Path, transform: Optional[Callable]
    ) -> "KITTIDepthDataset":
        image_dir = base / "image"
        intr_dir = base / "intrinsics"

        samples: List[DepthSample] = []
        for img_path in sorted(image_dir.glob("*.png")):
            stem = img_path.stem
            intr_path = intr_dir / f"{stem}.txt"

            samples.append(
                DepthSample(
                    stem=stem,
                    image_path=img_path,
                    gt_depth_path=None,
                    velodyne_path=None,
                    intrinsics_path=intr_path if intr_path.exists() else None,
                )
            )

        print(f"[KITTIDepthDataset] test_prediction split: {len(samples)} samples")
        return cls(samples, transform)

    # ------------------------------------------------------------------
    # __getitem__ / __len__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        out: Dict[str, object] = {"stem": sample.stem}

        # RGB image (H, W, 3) uint8
        out["image"] = np.array(Image.open(sample.image_path).convert("RGB"))

        # Ground-truth depth (H, W) float32, metres.  0 = invalid.
        if sample.gt_depth_path is not None:
            out["gt_depth"] = read_depth(sample.gt_depth_path)
        else:
            out["gt_depth"] = None

        # Sparse velodyne depth (H, W) float32, metres.  0 = invalid.
        if sample.velodyne_path is not None:
            out["velodyne"] = read_depth(sample.velodyne_path)
        else:
            out["velodyne"] = None

        # Camera intrinsics 3×3
        if sample.intrinsics_path is not None:
            out["intrinsics"] = read_intrinsics(sample.intrinsics_path)
        else:
            out["intrinsics"] = None

        if self.transform is not None:
            out = self.transform(out)

        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        has_gt = sum(s.gt_depth_path is not None for s in self.samples)
        has_vel = sum(s.velodyne_path is not None for s in self.samples)
        return (
            f"KITTIDepthDataset: {len(self.samples)} samples  |  "
            f"GT depth: {has_gt}  |  Velodyne: {has_vel}"
        )
