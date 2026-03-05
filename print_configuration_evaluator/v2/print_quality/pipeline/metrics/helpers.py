#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Literal
import math
from print_quality.utils.raster import Raster


from print_quality.utils.config_resolution import (
    channel_default_width_mm,
    channel_default_speed_mms,
    resolve_layer_height_mm,
)

def _logistic01(x: float, x0: float, k: float) -> float:
    """Standard logistic mapped so f(x0)=0.5, increasing with x."""
    try:
        return 1.0 / (1.0 + math.exp(-k*(x - x0)))
    except OverflowError:
        return 0.0 if (k*(x-x0)) < 0.0 else 1.0
    

def _saturating_decay(x: float, x0: float, k: float) -> float:
    """Monotone decay in [0,1]; returns ~1 for x << x0 and ~0 for x >> x0."""
    return 1.0 - _logistic01(x, x0=x0, k=k)

def _saturating_rise(x: float, x0: float, k: float) -> float:
    """Monotone rise in [0,1]; returns ~0 for x << x0 and ~1 for x >> x0."""
    return _logistic01(x, x0=x0, k=k)

def _soft_increase(x: float, k: float) -> float:
    """Penalty that increases with x (bounded [0,1)). k is scale in x-units."""
    try:
        xv = float(x)
    except Exception:
        xv = 0.0
    if math.isnan(xv):
        xv = 0.0
    k = max(1e-9, abs(float(k)))
    return xv / (xv + k)

def _select_reference_line_width_mm(
    cfg,
    *,
    policy: Literal["conservative", "optimistic", "external", "perimeter", "min"] = "conservative"
) -> float:
    """
    Pick the perimeter line width that should govern XY thin-wall resolvability.

    - conservative: min(external_perimeter, perimeter, extrusion_width); ignores Arachne narrowing
    - optimistic: same as conservative, then multiply by min_bead_width when perimeter_generator == "arachne"
    - external: use external_perimeter_extrusion_width
    - perimeter: use perimeter_extrusion_width
    - min: min(external_perimeter, perimeter)

    Falls back to extrusion_width, then nozzle_diameter if a choice is missing.
    Raises ValueError if nothing usable is found.
    """
    def num(x):
        if x is None: return 0.0
        s = str(x).strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)

    ext = num(_get(cfg,"external_perimeter_extrusion_width", default=0.45))
    per = num(_get(cfg,"perimeter_extrusion_width", default=0.45))
    gen = num(_get(cfg,"extrusion_width", default=0.45))
    noz = num(_get(cfg,"nozzle_diameter", default=0.4))

    if policy == "external":
        w = ext or gen or noz
    elif policy == "perimeter":
        w = per or gen or noz
    elif policy == "min":
        candidates = [x for x in (ext, per) if x > 0]
        w = min(candidates) if candidates else (gen or noz)
    else:  # conservative (default)
        candidates = [x for x in (ext, per, gen) if x > 0]
        w = min(candidates) if candidates else noz

    if w <= 0:
        raise ValueError("No usable perimeter/extrusion/nozzle width found in config.")

    if policy == "optimistic":
        if str(_get(cfg,"perimeter_generator", default = "")).strip().lower() == "arachne":
            mb = _get(cfg, "min_bead_width", 0.85)
            if mb is not None:
                f = num(mb)  # e.g., "85%" -> 0.85
                if 0.05 <= f <= 1.0:
                    w = w * f  # [Inference] approximate min achievable line width under Arachne
    return float(w)

def _soft_deficit(x: float, x_ref: float) -> float:
    """Penalty for being below a reference (0 if x >= ref)."""
    try:
        xv = float(x)
    except Exception:
        return 0.0
    if isinstance(xv, float) and math.isnan(xv):
        return 0.0
    if xv >= float(x_ref) or x_ref <= 1e-12:
        return 0.0
    return (float(x_ref) - xv) / float(x_ref)


def _soft_excess(x: float, x_ref: float) -> float:
    """Penalty for being above a reference (0 if x <= ref)."""
    try:
        xv = float(x)
    except Exception:
        return 0.0
    if isinstance(xv, float) and math.isnan(xv):
        return 0.0
    if xv <= float(x_ref):
        return 0.0
    return (xv - float(x_ref)) / (xv + float(x_ref) + 1e-9)

def _iter_polyline_points(seg: Any) -> Optional[np.ndarray]:
    if isinstance(seg, dict):
        pl = seg.get("polyline_mm") or seg.get("polyline")
        if pl is None:
            p0, p1 = seg.get("p0"), seg.get("p1")
            if p0 is not None and p1 is not None:
                pl = [p0, p1]
        return None if pl is None else np.asarray(pl, float)
    pl = getattr(seg, "polyline", None)
    return None if pl is None else np.asarray(pl, float)

def _job_bottom_origin_and_band(cfg_like, print_job, mesh_pc, default_band: float = 0.45):
    """
    Compute (z0, band_mm) in the *job frame*:
      - z0  : min Z seen in sliced segments (fallback: mesh bbox min)
      - band: first_layer_height + 1.0*layer_height, at least default_band
    """
    # origin z0 from segments if available
    z0 = float(mesh_pc.bbox_min[2])
    try:
        segs = getattr(print_job, "segments", None) or print_job.get("segments", [])
        zs = []
        for s in segs:
            pl = _iter_polyline_points(s)
            if pl is not None and len(pl) > 0:
                arr = np.asarray(pl, float)
                if arr.shape[1] >= 3:
                    zs.append(float(np.min(arr[:, 2])))
        if zs:
            z0 = float(np.min(zs))
    except Exception:
        pass
    # band thickness from config
    lh = float(_get(cfg_like, "layer_height", default = 0.20))
    h1 = float(_get(cfg_like, "first_layer_height", default = lh))
    band = max(float(default_band), float(h1 + 1.0 * lh))
    return z0, band

def _mask_at_xy(r: Raster, x: float, y: float) -> bool:
    """Check mask occupancy by converting world (x,y) to indices on this raster (uniform grid)."""
    px = float(r.pixel_xy)
    ox, oy = r.origin_xy
    col = int(math.floor((x - ox) / px))
    row = int(math.floor((y - oy) / px))
    H, W = r.mask.shape
    if 0 <= row < H and 0 <= col < W:
        return bool(r.mask[row, col])
    return False


def _layer_z_map(print_job) -> Dict[int, float]:
    """Map layer_index -> representative z (median) from segments."""
    zvals: Dict[int, List[float]] = {}
    segs = getattr(print_job, "segments", None) or print_job.get("segments", [])
    for s in segs:
        if isinstance(s, dict):
            li = int(s.get("layer_index", 0)); z = float(s.get("z_mm", s.get("z", 0.0)))
        else:
            li = int(getattr(s, "layer_index", 0)); z = float(getattr(s, "z_mm", getattr(s, "z", 0.0)))
        zvals.setdefault(li, []).append(z)
    out = {}
    for li, arr in zvals.items():
        if arr:
            out[li] = float(np.median(np.asarray(arr, dtype=float)))
    return out


def _get(cfg: dict, *keys, default=None):
    for k in keys:
        if not isinstance(cfg, dict):
            break

        # Try top-level key
        if k in cfg:
            return cfg[k]
        if k.lower() in cfg:
            return cfg[k.lower()]

        # Try inside 'extras' if it exists and is a dict
        extras = cfg.get("extras")
        if isinstance(extras, dict):
            if k in extras:
                return extras[k]
            if k.lower() in extras:
                return extras[k.lower()]

    return default


# def _segment_length_xy(seg) -> float:
#     """Planar XY length for a segment; sums polyline if present."""
#     if isinstance(seg, dict):
#         pl = getattr(seg, "polyline_mm", None) if getattr(seg, "polyline_mm", None) is not None else getattr(seg, "polyline", None)
#         p0, p1 = seg.get("p0"), seg.get("p1")
#     else:
#         pl = getattr(seg, "polyline_mm", None) or getattr(seg, "polyline", None)
#         p0, p1 = getattr(seg, "p0", None), getattr(seg, "p1", None)
#     if pl is None:
#         if p0 is None or p1 is None: return 0.0
#         p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
#         return float(np.linalg.norm((p1 - p0)[:2]))
#     P = np.asarray(pl, float)
#     if P.shape[0] < 2: return 0.0
#     d = P[1:, :2] - P[:-1, :2]
#     return float(np.linalg.norm(d, axis=1).sum())


def _segment_length_xy(s) -> float:
    """Planar XY length for a segment; sums polyline if present.

    Falls back to p0/p1 if no polyline is available.
    """
    # Prefer polyline_mm, then polyline
    poly = _get(s, "polyline_mm", None)
    if poly is None:
        poly = _get(s, "polyline", None)

    if poly is not None:
        # Normalize to a float array of shape (N, >=2)
        if isinstance(poly, np.ndarray):
            P = np.asarray(poly, dtype=float)
        else:
            P = np.asarray(list(poly), dtype=float)

        # Must have at least two points
        if P.ndim != 2 or P.shape[0] < 2:
            return 0.0

        # Use XY only (ignore Z or extra columns)
        d = P[1:, :2] - P[:-1, :2]
        # Sum Euclidean lengths of each segment
        return float(np.sqrt((d * d).sum(axis=1)).sum())

    # Fallback: direct endpoints p0/p1
    p0 = _get(s, "p0", None)
    p1 = _get(s, "p1", None)
    if p0 is None or p1 is None:
        return 0.0

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return float(np.linalg.norm((p1 - p0)[:2]))


def _segment_speed_mms(seg, cfg) -> Optional[float]:
    if isinstance(seg, dict):
        v = seg.get("speed_mms")
        F = seg.get("feedrate", seg.get("F"))
        feat = seg.get("feature", seg.get("channel", "UNKNOWN"))
    else:
        v = getattr(seg, "speed_mms", None)
        F = getattr(seg, "feedrate", getattr(seg, "F", None))
        feat = getattr(seg, "feature", getattr(seg, "channel", "UNKNOWN"))
    if v is not None:
        try: return float(v)
        except Exception: pass
    if F is not None:
        try: return float(F) / 60.0
        except Exception: pass
    try:
        return channel_default_speed_mms(cfg, feat)
    except Exception:
        return None
    
def _segment_width_height(seg, cfg, default_layer_height: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(seg, dict):
        w = seg.get("width_mm", seg.get("width_w_mm"))
        h = seg.get("height_mm", seg.get("height_h_mm"))
        feat = seg.get("feature", seg.get("channel", "UNKNOWN"))
        li = int(seg.get("layer_index", 0))
    else:
        w = getattr(seg, "width_mm", getattr(seg, "width_w_mm", None))
        h = getattr(seg, "height_mm", getattr(seg, "height_h_mm", None))
        feat = getattr(seg, "feature", getattr(seg, "channel", "UNKNOWN"))
        li = int(getattr(seg, "layer_index", 0))
    if w is None:
        try: w = channel_default_width_mm(cfg, feat)
        except Exception: w = None
    if h is None:
        h = default_layer_height
        try: h = resolve_layer_height_mm(cfg, li, h)
        except Exception: pass
    return (float(w) if w is not None else None, float(h) if h is not None else None)