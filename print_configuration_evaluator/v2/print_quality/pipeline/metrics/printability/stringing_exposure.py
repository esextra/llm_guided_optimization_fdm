#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any
import math
import numpy as np
from print_quality.pipeline.metrics.helpers import _mask_at_xy

def compute_stringing_exposure(
    print_job,
    layer_rasters: Dict[int, Dict[str, Any]],
    min_travel_mm: float = 5.0,
    sample_step_mm: float = 2.0
) -> Dict[str, float]:

    """
    Estimate “stringing exposure” as the total length of in-print travel moves that occur in air.

    A travel segment contributes to air_travel_mm only if all of the following hold:
    1) The segment is classified as travel (is_travel == True, with a legacy fallback via is_extruding).
    2) Its polyline length is >= min_travel_mm.
    3) Its segment index lies between the first and last *extruding* segments in print_job["segments"]
       (startup/teardown travels outside this window are excluded).
    4) For the segment’s layer_index, every midpoint sample along the travel polyline (sample_step_mm spacing)
       is outside BOTH the part mask (V_part) and the support mask (V_sup if present, else V_part),
       evaluated on the same layer.

    Also accumulates total extruded_mm as the sum of polyline lengths of non-travel segments.

    Returns a dict with:
    - air_travel_mm: total qualifying air-travel length (mm)
    - count: number of qualifying air-travel segments
    - extruded_mm: total length of non-travel segments (mm)
    """


    if not layer_rasters:
        return {"air_travel_mm": 0.0, "count": 0, "extruded_mm": 0.0}


    segs = print_job["segments"]
    if not segs:
        return {"air_travel_mm": 0.0, "count": 0, "extruded_mm": 0.0}

    def _get(s, name, default=None):
        return s.get(name, default) if isinstance(s, dict) else getattr(s, name, default)


    def _samples_2d(x0: float, y0: float, x1: float, y1: float, step: float):
        dx = float(x1 - x0); dy = float(y1 - y0)
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return
        n = max(1, int(math.ceil(L / max(1e-6, step))))
        for k in range(n):
            t = (k + 0.5) / float(n)
            yield (x0 + t*dx, y0 + t*dy, L / n)

    total_air_mm = 0.0
    total_extruded_mm = 0.0
    count_air = 0
    step = max(1e-3, float(sample_step_mm))  # be robust against zero/negatives


    def _seg_xy_pts(seg):

        poly = _get(seg, "polyline_mm", None)
        if poly is None:
            poly = _get(seg, "polyline", None)
        if poly is None:
            p0 = _get(seg, "p0", None)
            p1 = _get(seg, "p1", None)
            if (p0 is None) or (p1 is None):
                return []
            poly = [p0, p1]
        pts = poly.tolist() if isinstance(poly, np.ndarray) else list(poly)
        if len(pts) < 2:
            return []
        return [(float(p[0]), float(p[1])) for p in pts]

    first_ex_idx = None
    last_ex_idx = None
    for _idx, _s in enumerate(segs):
        _is_travel = _get(_s, "is_travel", None)
        if _is_travel is None:

            _is_travel = (not bool(_get(_s, "is_extruding", True)))
        if not _is_travel:  # extruding
            if first_ex_idx is None:
                first_ex_idx = _idx
            last_ex_idx = _idx

    for idx, s in enumerate(segs):

        poly = _get(s, "polyline_mm", None)
        if poly is None:
            poly = _get(s, "polyline", None)


        if poly is None:
            p0 = _get(s, "p0", None)
            p1 = _get(s, "p1", None)
            if (p0 is None) or (p1 is None):
                continue
            poly = [p0, p1]

        if isinstance(poly, np.ndarray):
            pts = poly.tolist()
        else:
            pts = list(poly)
        if len(pts) < 2:
            continue
        def _xy(p): return (float(p[0]), float(p[1]))
        xy_pts = [_xy(p) for p in pts]

        seg_lengths = [math.hypot(xy_pts[i+1][0] - xy_pts[i][0], xy_pts[i+1][1] - xy_pts[i][1])
                       for i in range(len(xy_pts) - 1)]
        poly_len = float(sum(seg_lengths))


        is_travel = _get(s, "is_travel", None)
        if is_travel is None:
            # Fallback if legacy fields are present
            is_travel = (not bool(_get(s, "is_extruding", True)))

        if not is_travel:
            if poly_len > 0.0:
                total_extruded_mm += poly_len
            continue

        if poly_len < float(min_travel_mm):
            continue

        li = int(_get(s, "layer_index", 0))
        r = layer_rasters.get(li)
        if not r:
            continue
        V_part = r.get("V_part", None)
        if V_part is None:
            continue
        V_sup = r.get("V_sup", V_part)

        air = True
        for i in range(len(xy_pts) - 1):
            x0, y0 = xy_pts[i]
            x1, y1 = xy_pts[i+1]
            for (x, y, _dl) in _samples_2d(x0, y0, x1, y1, step):

                if _mask_at_xy(V_part, x, y) or _mask_at_xy(V_sup, x, y):
                    air = False
                    break
            if not air:
                break

        if is_travel and (idx < first_ex_idx or idx > last_ex_idx):
            air = False  

        if air:
            total_air_mm += poly_len
            count_air += 1

    return {
        "air_travel_mm": float(total_air_mm),
        "count": int(count_air),
        "extruded_mm": float(max(0.0, total_extruded_mm))
    }



def normalize_air_travel_ratio(A_mm: float,
                               E_mm: float,
                               *,
                               R_SAFE: float = 0.02,
                               R_FAIL: float = 0.06) -> float:
    if not (R_FAIL > R_SAFE > 0.0):
        raise ValueError("R_FAIL must be > R_SAFE > 0.")

    if A_mm is None or E_mm is None or not (math.isfinite(A_mm) and math.isfinite(E_mm)):
        return 1.0  
    if E_mm <= 0.0 or A_mm <= 0.0:
        return 0.0  
    r = A_mm / E_mm

    _MID = 0.5 * (R_SAFE + R_FAIL)
    _K = (2.0 * math.log(9.0)) / (R_FAIL - R_SAFE)


    x = _K * (r - _MID)
    if x >= 0:
        z = math.exp(-x); s = 1.0 / (1.0 + z)
    else:
        z = math.exp(x);  s = z / (1.0 + z)

    if s < 0.0: return 0.0
    if s > 1.0: return 1.0
    return s


def compute_stringing_exposure_penalty(print_job,
                                             layer_rasters,
                                             *,
                                             R_SAFE: float = 0.02,
                                             R_FAIL: float = 0.06) -> float:

    out = compute_stringing_exposure(print_job, layer_rasters)
    A = float(out.get("air_travel_mm", 0.0))
    E = float(out.get("extruded_mm", 0.0))
    return normalize_air_travel_ratio(A, E, R_SAFE=R_SAFE, R_FAIL=R_FAIL)
