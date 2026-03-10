"""
Depth I/O utilities for KITTI depth benchmark.

Handles reading/writing 16-bit PNG depth maps and camera intrinsics,
following the convention from the KITTI devkit:
    depth_m = uint16_value / 256.0
    invalid pixels have value 0 in the PNG (mapped to -1 or NaN).
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Depth reading / writing
# ---------------------------------------------------------------------------

def read_depth(path: Union[str, Path]) -> np.ndarray:
    """Read a KITTI 16-bit PNG depth map and return depth in metres.

    Parameters
    ----------
    path : str or Path
        Path to a 16-bit PNG depth image.

    Returns
    -------
    depth : np.ndarray, dtype float32, shape (H, W)
        Metric depth in metres.  Invalid / missing pixels are set to 0.0.
    """
    path = Path(path)
    depth_png = np.array(Image.open(path), dtype=np.int32)
    assert depth_png.max() > 255, (
        f"{path.name} does not look like a 16-bit depth map "
        f"(max value = {depth_png.max()})"
    )
    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = 0.0          # keep 0 = invalid
    return depth


def read_depth_safe(path: Union[str, Path]) -> np.ndarray:
    """Like ``read_depth`` but skips the >255 assertion (useful for
    predictions that may contain only small depth values)."""
    depth_png = np.array(Image.open(Path(path)), dtype=np.int32)
    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = 0.0
    return depth


def write_depth(path: Union[str, Path], depth: np.ndarray) -> None:
    """Write a depth map as a KITTI-compatible 16-bit PNG.

    Parameters
    ----------
    path : str or Path
        Output file path.
    depth : np.ndarray, dtype float32, shape (H, W)
        Metric depth in metres.  Pixels <= 0 are treated as invalid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_uint16 = np.round(depth * 256.0).astype(np.uint16)
    depth_uint16[depth <= 0] = 0
    Image.fromarray(depth_uint16).save(path)


# ---------------------------------------------------------------------------
# Validity mask helpers
# ---------------------------------------------------------------------------

def valid_mask(depth: np.ndarray) -> np.ndarray:
    """Return a boolean mask where depth is valid (> 0)."""
    return depth > 0


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def read_intrinsics(path: Union[str, Path]) -> np.ndarray:
    """Read a KITTI intrinsics file (9 floats → 3×3 matrix).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    K : np.ndarray, shape (3, 3), dtype float64
        Camera intrinsic matrix.
    """
    values = np.loadtxt(path).reshape(3, 3)
    return values


# ---------------------------------------------------------------------------
# Colourised depth visualisation
# ---------------------------------------------------------------------------

def depth_to_colormap(
    depth: np.ndarray,
    max_depth: float = 80.0,
    colormap: int = 20,          # cv2.COLORMAP_MAGMA
) -> np.ndarray:
    """Convert a depth map to a false-colour RGB image for visualisation.

    Requires ``cv2`` (OpenCV).  Returns uint8 BGR image.
    """
    import cv2

    mask = depth > 0
    normalised = np.clip(depth / max_depth, 0, 1)
    grey = (normalised * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(grey, colormap)
    coloured[~mask] = 0
    return coloured
