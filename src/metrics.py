"""
KITTI depth evaluation metrics.

Mirrors the C++ ``evaluate_depth.cpp`` from the KITTI devkit, implementing
all nine standard error metrics.  The four headline metrics used on the
KITTI leaderboard are:

    * **MAE**    – Mean Absolute Error (mm)
    * **RMSE**   – Root Mean Square Error (mm)
    * **iMAE**   – inverse MAE  (1/km)
    * **iRMSE**  – inverse RMSE (1/km)

All functions expect depth maps in **metres** with invalid pixels marked
as 0 (or ≤ 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-image error struct
# ---------------------------------------------------------------------------

@dataclass
class DepthMetrics:
    """Container for the nine standard KITTI depth error metrics."""

    mae: float            # Mean Absolute Error  [mm]
    rmse: float           # Root Mean Square Error [mm]
    imae: float           # inverse MAE  [1/km]
    irmse: float          # inverse RMSE [1/km]
    log_mae: float        # log MAE
    log_rmse: float       # log RMSE
    si_log: float         # Scale-invariant log error
    abs_rel: float        # Absolute relative error
    sq_rel: float         # Squared relative error
    num_valid: int        # number of valid GT pixels used

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.1f} mm  |  RMSE={self.rmse:.1f} mm  |  "
            f"iMAE={self.imae:.1f} 1/km  |  iRMSE={self.irmse:.1f} 1/km  |  "
            f"absRel={self.abs_rel:.4f}  |  sqRel={self.sq_rel:.4f}  |  "
            f"valid={self.num_valid}"
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> DepthMetrics:
    """Compute KITTI depth error metrics between *gt* and *pred*.

    Parameters
    ----------
    gt : np.ndarray, shape (H, W), float
        Ground-truth depth in metres.  Pixels ≤ 0 are invalid.
    pred : np.ndarray, shape (H, W), float
        Predicted / interpolated depth in metres.
    min_depth : float
        Ignore GT pixels with depth < ``min_depth`` (avoids division by ≈0).
    max_depth : float
        Ignore GT pixels with depth > ``max_depth``.

    Returns
    -------
    DepthMetrics
    """
    assert gt.shape == pred.shape, (
        f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}"
    )

    # Valid mask: GT must be positive, prediction must be positive,
    # and depth must lie within [min_depth, max_depth].
    mask = (gt > min_depth) & (gt < max_depth) & (pred > 0)
    n = mask.sum()
    if n == 0:
        return DepthMetrics(*(0.0,) * 9, num_valid=0)

    g = gt[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)

    # Absolute errors
    d_err = np.abs(g - p)
    mae = float(d_err.mean())
    rmse = float(np.sqrt((d_err ** 2).mean()))

    # Inverse errors
    d_err_inv = np.abs(1.0 / g - 1.0 / p)
    imae = float(d_err_inv.mean())
    irmse = float(np.sqrt((d_err_inv ** 2).mean()))

    # Log errors
    log_diff = np.abs(np.log(g) - np.log(p))
    log_mae = float(log_diff.mean())
    log_sq_mean = float((log_diff ** 2).mean())
    log_rmse = float(np.sqrt(log_sq_mean))

    # Scale-invariant log
    log_raw = np.log(g) - np.log(p)
    si_log = float(np.sqrt(log_sq_mean - (log_raw.mean()) ** 2))

    # Relative errors
    abs_rel = float((d_err / g).mean())
    sq_rel = float(((d_err ** 2) / (g ** 2)).mean())

    # Convert to KITTI leaderboard units:
    # MAE / RMSE → millimetres,  iMAE / iRMSE → 1/km
    return DepthMetrics(
        mae=mae * 1000.0,
        rmse=rmse * 1000.0,
        imae=imae * 1000.0,
        irmse=irmse * 1000.0,
        log_mae=log_mae,
        log_rmse=log_rmse,
        si_log=si_log,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        num_valid=int(n),
    )


# ---------------------------------------------------------------------------
# Aggregate over a dataset
# ---------------------------------------------------------------------------

def aggregate_metrics(
    all_metrics: list[DepthMetrics],
) -> Dict[str, float]:
    """Average a list of per-image ``DepthMetrics`` into a single dict.

    Returns a plain dict with the same keys as ``DepthMetrics`` (minus
    ``num_valid``), plus ``total_valid`` and ``num_images``.
    """
    keys = [
        "mae", "rmse", "imae", "irmse",
        "log_mae", "log_rmse", "si_log",
        "abs_rel", "sq_rel",
    ]
    result: Dict[str, float] = {}
    for k in keys:
        vals = [getattr(m, k) for m in all_metrics if m.num_valid > 0]
        result[k] = float(np.mean(vals)) if vals else 0.0
    result["total_valid"] = sum(m.num_valid for m in all_metrics)
    result["num_images"] = len(all_metrics)
    return result


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print aggregated metrics."""
    print("=" * 65)
    print("KITTI Depth Evaluation Results")
    print("=" * 65)
    print(f"  Images evaluated : {metrics.get('num_images', '?')}")
    print(f"  Total valid px   : {metrics.get('total_valid', '?'):,}")
    print("-" * 65)
    print(f"  MAE              : {metrics['mae']:.1f} mm")
    print(f"  RMSE             : {metrics['rmse']:.1f} mm")
    print(f"  iMAE             : {metrics['imae']:.1f} 1/km")
    print(f"  iRMSE            : {metrics['irmse']:.1f} 1/km")
    print(f"  Abs Rel          : {metrics['abs_rel']:.4f}")
    print(f"  Sq Rel           : {metrics['sq_rel']:.4f}")
    print(f"  Log MAE          : {metrics['log_mae']:.4f}")
    print(f"  Log RMSE         : {metrics['log_rmse']:.4f}")
    print(f"  SI Log           : {metrics['si_log']:.4f}")
    print("=" * 65)
