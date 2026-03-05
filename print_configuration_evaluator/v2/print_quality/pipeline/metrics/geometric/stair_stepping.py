#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, Optional, Any
from print_quality.pipeline.mesh_precompute import MeshPrecompute
import numpy as np

def compute_stair_stepping(mesh_pc: MeshPrecompute, layer_height_mm: Optional[float],
                         bottom_band_mm: float = 0.3,
                         z0: Optional[float] = None) -> Dict[str, float]:
    """
    Compute a stair-stepping amplitude statistic from mesh face slopes.

    Angle convention:
    - mesh_pc.face_slope_deg is the angle (degrees) between each face normal and +Z.

    Per-face amplitude:
    - Let theta be the face slope angle from +Z (radians).
    - Define δ = |cos(theta)|.
    - Define amp_mm = layer_height_mm * δ * (1 - δ).
      This is 0 for vertical faces (theta=90°, δ=0) and exactly horizontal faces (theta=0°, δ=1),
      and is bounded by amp_mm <= layer_height_mm/4.

    Filtering:
    - Faces whose centroid z is below (z0 or mesh bbox min z) + bottom_band_mm are excluded.
    - Faces with non-finite slope/area or non-positive area are excluded.

    Statistic and outputs:
    - Computes the area-weighted 95th percentile of amp_mm.
    - Returns a dict with:
        {"layer_height_mm": h, "SSI_p95": p95, "amp_sorted": amp_sorted, "cum_w": cum_w}
      where amp_sorted is the ascending amp_mm array and cum_w is the corresponding cumulative
      area-weight in [0,1].

    Failure cases:
    - If layer_height_mm is missing/non-positive, or if no faces remain after filtering,
      returns {"SSI_mean": NaN, "SSI_p50": NaN, "SSI_p95": NaN} (legacy keys).
    """

    if not layer_height_mm or layer_height_mm <= 0:
        return {"SSI_mean": float("nan"), "SSI_p50": float("nan"), "SSI_p95": float("nan")}
    h = float(layer_height_mm)

    C = mesh_pc.face_centroids()
    zref = float(mesh_pc.bbox_min[2]) if z0 is None else float(z0)
    keep = C[:, 2] >= (zref + float(bottom_band_mm))
    if not np.any(keep):
        return {"SSI_mean": float("nan"), "SSI_p50": float("nan"), "SSI_p95": float("nan")}


    theta_deg = np.asarray(mesh_pc.face_slope_deg, dtype=float)[keep]
    A = np.asarray(mesh_pc.face_areas, dtype=float)[keep]
    finite = np.isfinite(theta_deg) & np.isfinite(A) & (A > 0)
    if not np.any(finite):
        return {"SSI_mean": float("nan"), "SSI_p50": float("nan"), "SSI_p95": float("nan")}
    theta = np.radians(theta_deg[finite])
    A = A[finite]


    delta = np.abs(np.cos(theta))
    amp = h * delta * (1.0 - delta)  # bounded in [0, h/4]


    w = A / (A.sum() + 1e-12)



    def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q):
        order = np.argsort(values)
        v = values[order]
        wts = weights[order]
        cw = np.cumsum(wts)
        if cw[-1] <= 0:
            return float("nan")
        cw /= cw[-1]
        q = float(q / 100.0)
        return float(np.interp(q, cw, v)), v, cw 

    p95, amp_sorted, cum_w = _weighted_percentile(amp, w, 95)

    return {
        "layer_height_mm": float(h),
        "SSI_p95": float(p95),
        "amp_sorted": amp_sorted,
        "cum_w": cum_w
    }




def normalize_stair_stepping_p95(
    ssi: Dict[str, Any],
    h_ref_mm: float,
) -> Dict[str, float]:

    """
    Normalize stair-stepping amplitude to a [0,1] objective using the p95 statistic.

    Inputs:
    - ssi["layer_height_mm"]: layer height h (mm)
    - ssi["SSI_p95"]: area-weighted 95th percentile of raw amplitude (mm)
    Optional (enables exact handling when h > h_ref_mm):
    - ssi["amp_sorted"]: ascending raw amplitudes (mm)
    - ssi["cum_w"]: cumulative area weights in [0,1] aligned with amp_sorted, with cum_w[-1] == 1

    Normalization:
    - Uses scale = 4 / h_ref_mm (since raw amplitude is bounded by h/4).
    - Defines clipped normalized amplitude: a_hat = min(1, scale * amp_mm).
    - Returns SSI_objective as the area-weighted 95th percentile of a_hat.

    Exactness:
    - If h <= h_ref_mm, clipping cannot occur anywhere, so SSI_objective is exactly
      clip(scale * SSI_p95, 0, 1).
    - If h > h_ref_mm, exact SSI_objective after clipping requires the distribution
      (amp_sorted, cum_w). If those are not provided, falls back to clip(scale * SSI_p95, 0, 1),
      which may underestimate the true clipped p95.

    Returns:
    - {"SSI_objective": float in [0,1], "stat": "p95", "used_exact": 1.0 or 0.0}
    """

    if not np.isfinite(h_ref_mm) or h_ref_mm <= 0:
        raise ValueError("h_ref_mm must be a positive finite float.")
    for k in ("SSI_p95", "layer_height_mm"):
        if k not in ssi or not np.isfinite(ssi[k]):
            raise ValueError(f"Missing or non-finite required key: {k}")

    h = float(ssi["layer_height_mm"])
    scale = 4.0 / float(h_ref_mm)

    if h <= h_ref_mm:
        p95_n = float(np.clip(scale * float(ssi["SSI_p95"]), 0.0, 1.0))
        return {"SSI_objective": p95_n, "stat": "p95", "used_exact": 1.0}

    if ("amp_sorted" in ssi) and ("cum_w" in ssi):
        amp_sorted = np.asarray(ssi["amp_sorted"], dtype=float)
        cum_w = np.asarray(ssi["cum_w"], dtype=float)
        if amp_sorted.ndim != 1 or cum_w.ndim != 1 or amp_sorted.size != cum_w.size:
            raise ValueError("amp_sorted and cum_w must be 1D arrays of equal length.")
        if amp_sorted.size == 0 or not np.isclose(cum_w[-1], 1.0):
            raise ValueError("cum_w must end at 1.0 and arrays must be non-empty.")

        a_hat_sorted = np.minimum(1.0, scale * amp_sorted)

        p95_n = float(np.interp(0.95, cum_w, a_hat_sorted))
        return {"SSI_objective": float(np.clip(p95_n, 0.0, 1.0)), "stat": "p95", "used_exact": 1.0}


    p95_n = float(np.clip(scale * float(ssi["SSI_p95"]), 0.0, 1.0))
    return {"SSI_objective": p95_n, "stat": "p95", "used_exact": 0.0}


def compute_stair_stepping_penalty(mesh_pc, layer_height_mm, bottom_band_mm, z0, max_layer_height):
    ss = compute_stair_stepping(mesh_pc, layer_height_mm, bottom_band_mm, z0)
    out = normalize_stair_stepping_p95(ss, max_layer_height)
    return out["SSI_objective"]
