#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import numpy as np
import math
from collections import defaultdict
from print_quality.utils.raster import Raster
from print_quality.pipeline.metrics.helpers import  _saturating_decay, _saturating_rise, _segment_length_xy, _segment_speed_mms, _segment_width_height, _get
from print_quality.utils.config_resolution import flow_caps_mm3s

from typing import Dict


def _dilate(b: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return b
    H, W = b.shape
    rr = range(-r, r + 1)
    out = b.copy()
    for dy in rr:
        y0 = max(0, dy)
        out[max(0, dy):H, :] |= b[0:H - y0, :]
    tmp = np.zeros_like(b)
    for dx in rr:
        x0 = max(0, dx)
        tmp[:, max(0, dx):W] |= out[:, 0:W - x0]
    return tmp

import numpy as np


# ----- Prusa rotation semantics  -----

def Rx(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]])


def Ry(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def Rz(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def prusa_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    return Rz(rz) @ (Ry(ry) @ Rx(rx))


def orientation_factor_for_z_bonding(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    load_direction: str,
) -> float:
    """
    Returns f_orient in [0,1] measuring alignment between the
    printer's weak Z direction (build +Z) and one or more possible load
    directions defined in the CAD frame.

    Parameters
    ----------
    rx_deg, ry_deg, rz_deg : float
        Rotation angles (degrees) used to orient the part for this build.
        prusa_rotation_matrix(rx, ry, rz) is assumed to map CAD -> printer.

    load_direction : str
        [Unverified] String specifying which CAD axes are plausible load
        directions in service.
        Allowed combinations: "x", "y", "z", "xy", "xz", "yz", "xyz".
        Each character is treated as an independent possible load axis.

    Returns
    -------
    f_orient : float
        Maximum |cos(theta)| between:
          - the printer build +Z direction, and
          - any of the rotated CAD load axes,
        in [0, 1].
    """
    axis_vecs = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    # Rotation from CAD to printer coordinates
    R = prusa_rotation_matrix(rx_deg, ry_deg, rz_deg)

    # Weak direction in printer frame: machine +Z
    z_printer = np.array([0.0, 0.0, 1.0], dtype=float)

    # Collect candidate load axes from the string
    load_axes = []
    for ch in load_direction.lower():
        if ch in axis_vecs:
            load_axes.append(axis_vecs[ch])
    if not load_axes:
        raise ValueError(
            f"[Unverified] load_direction '{load_direction}' must contain at least one of 'x', 'y', 'z'."
        )

    # Compute max |cos(theta)| between build Z and rotated load axes
    f_max = 0.0
    for axis_cad in load_axes:
        # Load direction in printer frame
        load_printer = R @ axis_cad
        # R is a rotation, so load_printer is already unit-length
        cos_theta = float(np.clip(np.dot(load_printer, z_printer), -1.0, 1.0))
        f = abs(cos_theta)
        if f > f_max:
            f_max = f

    return f_max  # in [0, 1]



def min_layer_time_part_present_total(print_job, cfg, default_layer_height: Optional[float] = None) -> Dict[str, float]:
    """
    For each layer that has ANY part extrusion, sum ALL extrusion time on that layer
    (part + supports + adhesion; travels excluded). Then compute min / p05 / p50
    across those layers. This reflects real cooling time available to the part.
    """
    segs = getattr(print_job, "segments", None) or print_job.get("segments", [])
    t_total = defaultdict(float)
    t_part  = defaultdict(float)

    for s in segs:
        # Skip travels by your flag
        if bool(s.get("is_travel", False) if isinstance(s, dict) else getattr(s, "is_travel", False)):
            continue
        li = int(s.get("layer_index", 0) if isinstance(s, dict) else getattr(s, "layer_index", 0))
        v = _segment_speed_mms(s, cfg)
        #print(s["feature"], v)
        if v is None or v <= 1e-6:
            continue
        L = _segment_length_xy(s) or 0.0
        t = L / v
        t_total[li] += t
        if not _is_support_segment(s):
            t_part[li] += t

    # Consider only layers that actually have part extrusion
    values = [t_total[li] for li, tp in t_part.items() if tp > 0.0]
    if not values:
        return {"min_s": float("nan"), "p05_s": float("nan"), "p50_s": float("nan")}

    arr = np.asarray(values, dtype=float)
    return {
        "min_s": float(arr.min()),
        "p05_s": float(np.percentile(arr, 5.0)),
        "p50_s": float(np.percentile(arr, 50.0)),
    }



def interlayer_overlap(
    layer_rasters: Dict[int, Dict[str, "Raster"]],
    *, dilate_px: int = 1, persist_prev_layers: int = 2
) -> Dict[str, float]:
    """
    Compute interlayer overlap for aligned rasters.

    Assumptions:
      - All layers share the same pixel grid shape (H, W).
      - pixel_mm is constant across all layers.
      - Origins are identical (no shifts).
    Behavior:
      - For each layer i that has a part mask, union the previous K = persist_prev_layers
        layers' (part ∪ support) masks (optionally dilated), then compute:
              overlap_i = |part_i ∩ union_prev| / |part_i|
      - Return mean overlap and 5th percentile across eligible layers.
    Returns:
      {"mean_overlap": float, "p05_overlap": float}
      If no eligible layers are found, both values are NaN.
    """
    if not layer_rasters:
        return {"mean_overlap": float("nan"), "p05_overlap": float("nan")}

    # Determine the sorted layer indices we actually have
    layer_ids = sorted(layer_rasters.keys())
    overlaps: list[float] = []

    for li in layer_ids:
        cur_part = layer_rasters.get(li, {}).get("V_part", None)
        if cur_part is None or not hasattr(cur_part, "mask"):
            continue

        cur_mask = np.asarray(cur_part.mask, dtype=bool)
        if cur_mask.ndim != 2:
            continue  # unexpected shape

        H, W = cur_mask.shape
        if H == 0 or W == 0:
            continue

        # Aggregate last-K previous unions (part ∪ support), each optionally dilated
        agg_prev = np.zeros((H, W), dtype=bool)
        has_prev = False

        start_k = max(layer_ids[0], li - persist_prev_layers)
        for k in range(start_k, li):
            prev = layer_rasters.get(k, {})
            prev_part = prev.get("V_part", None)
            prev_sup  = prev.get("V_sup",  None)

            Pk = None
            if prev_part is not None and hasattr(prev_part, "mask"):
                Pk = np.asarray(prev_part.mask, dtype=bool)
            if prev_sup is not None and hasattr(prev_sup, "mask"):
                Sk = np.asarray(prev_sup.mask, dtype=bool)
                Pk = Sk if Pk is None else (Pk | Sk)

            # Skip if nothing to use or shape mismatch with current layer
            if Pk is None or Pk.shape != (H, W):
                continue

            if dilate_px > 0:
                Pk = _dilate(Pk, dilate_px)

            agg_prev |= Pk
            has_prev = True

        if not has_prev:
            continue

        denom = int(cur_mask.sum())
        if denom <= 0:
            continue  # no current part pixels to normalize

        num = int((cur_mask & agg_prev).sum())
        overlaps.append(float(num) / float(denom))

    if not overlaps:
        return {"mean_overlap": float("nan"), "p05_overlap": float("nan")}

    arr = np.asarray(overlaps, dtype=float)
    return {
        "mean_overlap": float(arr.mean()),
        "p05_overlap": float(np.percentile(arr, 5.0)),
    }



def flow_operating_point(print_job, cfg, default_layer_height: Optional[float] = None) -> Dict[str, float]:
    segs = getattr(print_job, "segments", None) or print_job.get("segments", [])
    _, qcap = flow_caps_mm3s(cfg)
    ratios = []
    for s in segs:
        if _is_support_segment(s):
            continue
        if bool(s.get("is_travel", False) if isinstance(s, dict) else getattr(s, "is_travel", False)): continue
        v = _segment_speed_mms(s, cfg)
        if v is None or v <= 0: continue
        w, h = _segment_width_height(s, cfg, default_layer_height)
        if w is None or h is None: continue
        q = w * h * v
        if qcap is not None and qcap > 0: ratios.append(q / qcap)
    if not ratios: return {"q95_over_cap": float("nan"), "CVaR95_over_cap": float("nan"), "max_over_cap": float("nan")}
    r = np.asarray(ratios, float)
    q95 = float(np.percentile(r, 95.0)); cvar = float(r[r >= q95].mean())
    return {"q95_over_cap": q95, "CVaR95_over_cap": cvar, "max_over_cap": float(r.max())}



def _is_support_segment(s: Any) -> bool:
    """
    Treats true supports and adhesion aids (BRIM/RAFT/SKIRT) as support-like so they do not
    distort min-layer-time or interlayer feasibility statistics.
    """
    feature = s['feature']
        
    if feature in ['SUPPORT', 'SUP_IFC', 'ADHESION']:
        return True
    else:
        return False
    
def _iter_layers(print_job):
    segs = print_job["segments"]
    by_layer: Dict[int, List[Any]] = defaultdict(list)
    for s in segs:
        li = int(s.get("layer_index", 0) if isinstance(s, dict) else getattr(s, "layer_index", 0))
        by_layer[li].append(s)
    return sorted(by_layer.items(), key=lambda kv: kv[0])
    
@dataclass
class ZBondingParams:
    # Contact term weights and thresholds
    w_overlap: float = 0.6          # weight on overlap p05 vs squish
    r_good: float = 0.50            # good r = layer_height/line_width
    r_k: float = 10.0               # steepness for squish mapping
    dilate_px: int = 1              # pixel dilation
    persist_prev_layers: int = 2    # previous layers to consider

    # Time (healing) mapping
    t0_s: float = 3.0               # neutral per-layer time
    t_k: float = 1.2                # steepness

    # Fan (physics-based cooling modulation)
    use_fan_in_cooling: bool = True
    fan_cooling_gain: float = 2.0   # PLA ~[1..3]; ABS/PETG smaller
    fan_cooling_pow: float = 1.5    # convexity for high fan

    # Legacy fan mapping kept only for debug visibility (not used in product)
    fan0: float = 0.20
    fan_k: float = 8.0

    # Flow cap mapping
    flow_k: float = 10.0
    w_time: float = 1.0
    w_flow: float = 1.0

def z_bonding_proxy(pj_dict, cfg,layer_rasters):

    """
    Compute a Z-bonding proxy score and derived penalty.

    This function assembles three sub-terms (when available) and combines them into
    S_hat via a generalized mean with exponent p = -4 (emphasizes the worst term):

    1) Contact term A_contact:
       - overlap_p05 from interlayer_overlap(layer_rasters, dilate_px, persist_prev_layers)
       - r_contact from a median squish ratio r_med = (height / width) over part, non-travel segments,
         mapped by _saturating_decay with (r_good, r_k)
       - A_contact is a weighted blend of overlap_p05 and r_contact when both are finite.

    2) Healing term H_time:
       - t_s is a lower-tail layer-time statistic from min_layer_time_part_present_total(...)
       - if use_fan_in_cooling is enabled, t_eff = t_s * (1 + gain * fan_stat**pow),
         where fan_stat is a mean of per-layer p95 fan values over weld-relevant segments
         (excluding first layer, supports/adhesion, and bridge/overhang-like features)
       - H_time is mapped from t_eff using _saturating_rise with (t0_s, t_k)

    3) Flow term Flow_ok:
       - q95_over_cap from flow_operating_point(...)
       - Flow_ok = 1 if q95_over_cap <= 1, else _saturating_decay(q95_over_cap, x0=1, k=flow_k)

    Combine:
    - valid_terms are the available non-NaN terms among {A_contact, H_time, Flow_ok}
      (H_time / Flow_ok may be forced to 1.0 if their corresponding weights are disabled).
    - S_hat = generalized_mean(valid_terms, p=-4) if any valid terms exist, else NaN.
    - penalty_P_Z = clip(1 - (S_hat**gamma), 0, 1) with gamma = 3.
      If S_hat is NaN, the implementation uses 0 in the exponent term, yielding penalty_P_Z = 1.

    Returns a dict containing:
    - "S_hat"
    - "penalty_P_Z"
    - nested diagnostics: "contact", "healing", "flow", and "params".
    """

    # ---------- CONTACT ----------
    # A) Overlap on raster masks if provided
    
    default_layer_height = _get(cfg,"layer_height", default=0.2)

    ov = interlayer_overlap(layer_rasters, dilate_px=ZBondingParams.dilate_px, persist_prev_layers=ZBondingParams.persist_prev_layers)
    overlap_mean = float(ov.get("mean_overlap", float("nan")))
    overlap_p05  = float(ov.get("p05_overlap", float("nan")))

    # B) Squish ratio r = h / w (median over part segments)
    rs: List[float] = []
    segs = pj_dict["segments"]
    for s in segs:
        if _is_support_segment(s):
            continue
        is_travel = s.get("is_travel", False) if isinstance(s, dict) else getattr(s, "is_travel", False)
        if is_travel:
            continue
        w, h = _segment_width_height(s, cfg, default_layer_height)
        if w and h and w > 0 and h > 0:
            rs.append(float(h) / float(w))
    r_med = float(np.median(rs)) if rs else float("nan")
    r_contact = _saturating_decay(r_med, x0=ZBondingParams.r_good, k=ZBondingParams.r_k) if not math.isnan(r_med) else float("nan")

    # Combine contact
    if not math.isnan(overlap_p05) and not math.isnan(r_contact):
        A_contact = float(ZBondingParams.w_overlap * overlap_p05 + (1.0 - ZBondingParams.w_overlap) * r_contact)
    elif not math.isnan(overlap_p05):
        A_contact = float(overlap_p05)
    else:
        A_contact = float(r_contact) if not math.isnan(r_contact) else float("nan")

    # ---------- HEALING (time with fan-as-cooling) ----------
    mlt = min_layer_time_part_present_total(pj_dict, cfg, default_layer_height)
    t_s = float(mlt.get("p05_s", mlt.get("min_s", float("nan"))))

    # Fan statistic over weld-relevant segments: extruding, part-only, non-first-layer, non-bridge
    by_layer: Dict[int, List[float]] = defaultdict(list)
    for li, seg_layer in _iter_layers(pj_dict):
        if li == 0:
            continue
        for s in seg_layer:
            is_travel = bool(s.get("is_travel") if isinstance(s, dict) else getattr(s, "is_travel", False))
            if is_travel or _is_support_segment(s):
                continue
            feat = (s.get("feature") if isinstance(s, dict) else getattr(s, "feature", None))
            fstr = (str(feat).upper() if feat is not None else "")
            if fstr in ("BRIDGE", "P_OVERHANG", "BRIDGING"):
                continue
            fs = s.get("fan_speed", None) if isinstance(s, dict) else getattr(s, "fan_speed", None)
            if fs is None:
                continue
            try:
                fv = float(fs)
            except Exception:
                continue
            if fv > 1.0:
                fv = fv / 100.0 if fv <= 100.0 else fv / 255.0
            fv = max(0.0, min(1.0, fv))
            by_layer[li].append(fv)

    per_layer_p95 = [float(np.percentile(vs, 95.0)) for vs in by_layer.values() if vs]
    fan_stat = float(np.mean(per_layer_p95)) if per_layer_p95 else (float("nan"))

    if ZBondingParams.use_fan_in_cooling and not math.isnan(t_s):
        fstat = 0.0 if math.isnan(fan_stat) else fan_stat
        scale = 1.0 + ZBondingParams.fan_cooling_gain * (fstat ** ZBondingParams.fan_cooling_pow)
        t_eff = t_s * scale
    else:
        t_eff = t_s

    H_time = _saturating_rise(t_eff, x0=ZBondingParams.t0_s, k=ZBondingParams.t_k) if not math.isnan(t_eff) else float("nan")
    # Debug-only classic H_fan (NOT multiplied)
    H_fan = _saturating_decay(fan_stat, x0=ZBondingParams.fan0, k=ZBondingParams.fan_k) if not math.isnan(fan_stat) else float("nan")

    # ---------- FLOW ----------
    flow = flow_operating_point(pj_dict, cfg, default_layer_height)
    q95_over = float(flow.get("q95_over_cap", float("nan")))
    if not math.isnan(q95_over):
        Flow_ok = 1.0 if q95_over <= 1.0 else _saturating_decay(q95_over, x0=1.0, k=ZBondingParams.flow_k)
    else:
        Flow_ok = float("nan")

    # ---------- Combine ----------
    terms = [
        A_contact,
        H_time if ZBondingParams.w_time > 0 else 1.0,
        Flow_ok if ZBondingParams.w_flow > 0 else 1.0
    ]
    valid_terms = [t for t in terms if not (isinstance(t, float) and math.isnan(t))]
    # S_hat = float(np.prod(valid_terms)) if valid_terms else float("nan")
    # P_Z = float(np.clip(1.0 - (S_hat if not math.isnan(S_hat) else 0.0), 0.0, 1.0))
    def generalized_mean(values, p):
        return (sum(v**p for v in values)/len(values))**(1.0/p)

    S_hat = float(generalized_mean(valid_terms, -4)) if valid_terms else float("nan")
    gamma = 3
    P_Z = float(np.clip(1.0 - ((S_hat**gamma) if not math.isnan(S_hat) else 0.0), 0.0, 1.0))
    
    return {
        "S_hat": S_hat,
        "penalty_P_Z": P_Z,
        "contact": {
            "A_contact": A_contact,
            "overlap_mean": overlap_mean,
            "overlap_p05": overlap_p05,
            "r_med": r_med,
            "r_contact": r_contact,
        },
        "healing": {
            "min_layer_time_s": t_s,
            "t_eff_s": t_eff,
            "H_time": H_time,
            "fan_stat": fan_stat,
            "H_fan": H_fan,  # informational only
            "fan_cooling_gain": ZBondingParams.fan_cooling_gain,
            "fan_cooling_pow": ZBondingParams.fan_cooling_pow,
        },
        "flow": {
            "q95_over_cap": q95_over,
            "Flow_ok": Flow_ok,
        },
        "params": {
            "w_overlap": ZBondingParams.w_overlap,
            "r_good": ZBondingParams.r_good,
            "r_k": ZBondingParams.r_k,
            "dilate_px": ZBondingParams.dilate_px,
            "persist_prev_layers": ZBondingParams.persist_prev_layers,
            "t0_s": ZBondingParams.t0_s,
            "t_k": ZBondingParams.t_k,
            "use_fan_in_cooling": ZBondingParams.use_fan_in_cooling,
            "fan_cooling_gain": ZBondingParams.fan_cooling_gain,
            "fan_cooling_pow": ZBondingParams.fan_cooling_pow,
            "fan0": ZBondingParams.fan0,
            "fan_k": ZBondingParams.fan_k,
            "flow_k": ZBondingParams.flow_k,
            "w_time": ZBondingParams.w_time,
            "w_flow": ZBondingParams.w_flow,
        },
        "notes": "S_hat = A_contact * H_time(t_eff) * Flow_ok; t_eff = t_s * (1 + gain * fan^pow)."
    }

def _orientation_exponent(f_orient: float, e_safe: float = 3.0) -> float:
    """
    Map f_orient in [0,1] to an exponent e such that:
      - Near 90° from Z (f ≈ 0)  → big e (strong deflation, low penalty).
      - 60°+ closer to Z (f >= ~0.5) → small e (inflate penalty to high/very-high).
    Tuned for 30° grid and severity buckets.
    """
    f = float(max(0.0, min(1.0, f_orient)))

    if f <= 0.25:
        return e_safe   

    if f <= 0.5:
        return 0.25     
    if f <= 0.75:
        return 0.15    
    if f <= 0.9:
        return 0.10    
    return 0.05        


def directional_z_bonding_penalty(
    pj_dict,
    cfg,
    layer_rasters,
    load_bearing: bool,
    load_direction: str,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    e_safe: float = 3.0,  # exponent for "perfect" safe orientations
) -> float:
    """
    Z-bonding penalty adjusted for load orientation.

    For load_bearing=True:
      - If no load axis is close to in-plane (f_orient > ~0), the
        penalty is heavily inflated into the high/very-high/extreme
        range via a small exponent.
      - Only when a load axis is almost perfectly in-plane (f_orient ≈ 0)
        do we strongly deflate the base z-bonding penalty.
    """
    base_penalty = compute_z_bonding_proxy_penalty(pj_dict, cfg, layer_rasters)

    if not load_bearing:
        return base_penalty

    f_orient = orientation_factor_for_z_bonding(rx_deg, ry_deg, rz_deg, load_direction)
    e = _orientation_exponent(f_orient, e_safe=e_safe)

    # Edge cases
    if base_penalty <= 0.0:
        return 0.0
    if base_penalty >= 1.0:
        return 1.0

    return float(base_penalty ** e)



def compute_z_bonding_proxy_penalty(pj_dict, cfg, layer_rasters):
    zb = z_bonding_proxy(pj_dict, cfg, layer_rasters)
    #print(zb)
    return zb["penalty_P_Z"]