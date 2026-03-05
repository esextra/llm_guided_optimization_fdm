#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, Optional, Tuple, Any
import numpy as np
from print_quality.utils.raster import Raster
from print_quality.utils.graph import connected_components, _binary_dilate
from print_quality.pipeline.metrics.helpers import  _get
import math
from copy import deepcopy


def _Imin_and_area_from_mask(mask: np.ndarray, px_mm: float) -> Tuple[float, float]:
    npx = int(mask.sum())
    if npx <= 0:
        return 0.0, 0.0

    yy, xx = np.nonzero(mask)
    x = xx.astype(np.float64) * px_mm
    y = yy.astype(np.float64) * px_mm

    A = npx * (px_mm ** 2)
    xbar = x.mean(); ybar = y.mean()
    dx = x - xbar; dy = y - ybar
    w = (px_mm ** 2)


    Ixx = np.sum(dy ** 2) * w
    Iyy = np.sum(dx ** 2) * w
    Ixy = -np.sum(dx * dy) * w

    s = px_mm
    Ixx += npx * (s ** 4) / 12.0
    Iyy += npx * (s ** 4) / 12.0


    avg = 0.5 * (Ixx + Iyy)
    diff = 0.5 * (Ixx - Iyy)
    rad = np.sqrt(diff * diff + Ixy * Ixy)
    Imin = avg - rad
    return float(Imin), float(A)


def _minor_diameter_from_Imin_A(Imin_mm4: float, A_mm2: float) -> float:

    if A_mm2 <= 0.0 or Imin_mm4 <= 0.0:
        return 0.0
    rg_min = np.sqrt(Imin_mm4 / A_mm2)
    return 4.0 * float(rg_min)

def _geom_mask(mask: np.ndarray, px_mm: float, ss: int):
    if ss <= 1:
        return mask, px_mm

    up = np.kron(mask.astype(np.uint8), np.ones((ss, ss), dtype=np.uint8)).astype(bool)
    return up, px_mm / ss



def compute_slender_towers_aspect(
    layer_rasters: Dict[int, Dict[str, Raster]],
    *,
    channel: str = "V_part",
    support_channel: str = "V_sup",
    use_support_bracing: bool = True,
    support_contact_gap_mm: float = 0.0, 
    dilate_px: int = 1,
    noise_kill_min_px: int = 0,
    min_height_layers: int = 4,     
    supersample: int= 8,
    return_all: bool = False      
) -> Dict[str, Any]:
    """
    Track towers by component overlap (with split handling) and compute per-tower minor width.
    Returns:
        {
          "count": int,
          "towers": {tid: height_layers, ...},  # filtered unless return_all=True
          "features": {
            tid: {
              "height_layers": int,
              "d_min_mm_min": float,
              "per_layer": [{"layer": li, "d_min_mm": float}, ...],
            }, ...
          }
        }
    """
    if not layer_rasters:
        return {"count": 0, "towers": {}, "features": {}}

    layers = sorted(layer_rasters.keys())
    px_mm: float = float(next(iter(layer_rasters.values()))[channel].pixel_xy)


    sup_available = (
        len(layer_rasters) > 0
        and isinstance(next(iter(layer_rasters.values())).get(support_channel, None), Raster)
    )
    contact_dilate_px = (
        int(round(max(0.0, support_contact_gap_mm) / px_mm)) if use_support_bracing else 0
    )


    towers: Dict[int, Dict[str, Any]] = {}   
    prev_labels: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None   
    prev_id_of_label: Dict[int, int] = {}    
    next_id = 1

    for li in layers:
        cur_mask = layer_rasters[li][channel].mask.astype(bool)
        labels, n = connected_components(cur_mask, connectivity=8)


        if noise_kill_min_px > 0 and n > 0:
            keep = np.zeros_like(labels, bool)
            for k in range(1, n + 1):
                if (labels == k).sum() >= noise_kill_min_px:
                    keep |= (labels == k)
            labels = labels * keep
            present = [k for k in range(1, n + 1) if (labels == k).any()]
        else:
            present = [k for k in range(1, n + 1) if (labels == k).any()]

        cur_recs = []  # (label_k, tid)


        baseline_height = ({pl: towers[tid]["height"] for pl, tid in prev_id_of_label.items()}
                           if prev_id_of_label else {})
        baseline_hist = ({pl: deepcopy(towers[tid]["per_layer"]) for pl, tid in prev_id_of_label.items()}
                         if prev_id_of_label else {})

        baseline_seg  = ({pl: towers[tid].get("seg_h", 0) for pl, tid in prev_id_of_label.items()}
                         if prev_id_of_label else {})
        baseline_segmax = ({pl: towers[tid].get("seg_h_max", 1) for pl, tid in prev_id_of_label.items()}
                           if prev_id_of_label else {})

        claimed_children: set[int] = set()


        prev_dil = _binary_dilate(prev_mask, dilate_px) if (prev_mask is not None and dilate_px > 0) else prev_mask


        sup_mask = None
        if sup_available:
            sup_mask = layer_rasters[li][support_channel].mask.astype(bool)
            if use_support_bracing and contact_dilate_px > 0:
                sup_mask = _binary_dilate(sup_mask, contact_dilate_px)

        for k in present:
            comp = (labels == k)
            if comp.sum() == 0:
                continue

            tid: Optional[int] = None
            if prev_labels is not None:
                if prev_dil is not None and not (comp & prev_dil).any():
                    tid = None
                else:
                    idx = np.nonzero(comp)
                    prev_vals = prev_labels[idx]
                    if prev_vals.size > 0:
                        binc = np.bincount(prev_vals.ravel(), minlength=(prev_vals.max() + 1))
                        best_prev_label = int(np.argmax(binc[1:])) + 1 if binc.size > 1 else 0
                        if best_prev_label > 0 and best_prev_label in prev_id_of_label:
                            parent_tid = prev_id_of_label[best_prev_label]
                            parent_h0 = baseline_height.get(best_prev_label, towers[parent_tid]["height"])
                            parent_hist0 = baseline_hist.get(best_prev_label, towers[parent_tid]["per_layer"])
                            parent_seg0 = baseline_seg.get(best_prev_label, towers[parent_tid].get("seg_h", 0))
                            parent_segmax0 = baseline_segmax.get(best_prev_label, towers[parent_tid].get("seg_h_max", 1))


                            if best_prev_label in claimed_children:

                                tid = next_id; next_id += 1
                                towers[tid] = {
                                    "height": int(parent_h0 + 1),
                                    "seg_h": int(parent_seg0 + 1),
                                    "seg_h_max": int(parent_segmax0),
                                    "per_layer": deepcopy(parent_hist0)  

                                }
                            else:

                                tid = parent_tid
                                towers[tid]["height"] = int(parent_h0 + 1)
                                towers[tid]["per_layer"] = deepcopy(parent_hist0)
                                towers[tid]["seg_h"] = int(parent_seg0 + 1)
                                towers[tid]["seg_h_max"] = int(parent_segmax0)
                                claimed_children.add(best_prev_label)

            if tid is None:
                tid = next_id; next_id += 1
                towers[tid] = {"height": 1, "seg_h": 1, "seg_h_max": 1, "per_layer": []}

            if use_support_bracing and sup_available and (sup_mask is not None):
                if (comp & sup_mask).any():
                    towers[tid]["seg_h_max"] = max(towers[tid].get("seg_h_max", 1), towers[tid].get("seg_h", 1))
                    towers[tid]["seg_h"] = 1


            comp_for_geom, px_geom = _geom_mask(comp, px_mm, supersample)
            Imin, A = _Imin_and_area_from_mask(comp_for_geom, px_geom)
            d_min_mm = _minor_diameter_from_Imin_A(Imin, A)
            towers[tid]["per_layer"].append({"layer": li, "d_min_mm": float(d_min_mm)})

            cur_recs.append((k, tid))


        prev_labels = labels.copy()
        prev_mask = cur_mask
        prev_id_of_label = {k: tid for (k, tid) in cur_recs}

    for _tid, _rec in towers.items():
        _rec["seg_h_max"] = max(_rec.get("seg_h_max", 1), _rec.get("seg_h", 1))


    features: Dict[int, Dict[str, Any]] = {}
    for tid, rec in towers.items():
        h_layers = int(rec["height"])
        dmins = [p["d_min_mm"] for p in rec["per_layer"] if p["d_min_mm"] > 0.0]
        d_min_mm_min = float(min(dmins)) if dmins else 0.0

        features[tid] = {
            "height_layers": h_layers,
            "segment_height_layers_max": int(rec.get("seg_h_max", 1)),  # longest unbraced span
            "d_min_mm_min": d_min_mm_min,
            "per_layer": rec["per_layer"],
        }


    if return_all:
        towers_dict = {tid: int(f["height_layers"]) for tid, f in features.items()}
    else:
        towers_dict = {tid: int(f["height_layers"]) for tid, f in features.items()
                       if f["height_layers"] >= int(min_height_layers)}

    count = int(len(towers_dict))
    return {"count": count, "towers": towers_dict, "features": features}




def normalize_slender_penalty(
    result: Dict[str, Any],
    *,
    layer_height_mm: float,
    ar_safe: float = 20.0,   # see rationale below
    ar_bad: float  = 60.0,   # see rationale below
    beta: float = 2.0,       # emphasis on worst towers in probabilistic-OR
) -> float:
    """
    Map towers from compute_slender_towers_aspect(...) to a single penalty in [0, 1]
    using: (1) logistic per-tower mapping, (2) probabilistic-OR aggregation.

    Parameters
    ----------
    result : dict
        Output of compute_slender_towers_aspect(...). Expects result["features"][tid] with:
          - "height_layers" : int
          - "d_min_mm_min"  : float
    layer_height_mm : float
        Physical layer height (mm).
    ar_safe : float
        Aspect ratio L/d_min at which per-tower score ≈ 0.1 (typical is ~20 for PLA; see notes).
    ar_bad : float
        Aspect ratio L/d_min at which per-tower score ≈ 0.9 (typical is ~60 for PLA; see notes).
    beta : float
        Exponent inside the probabilistic-OR: O = 1 - Π_j (1 - s_j^beta).

    Returns
    -------
    float
        Final penalty in [0, 1].
    """
    # Defensive checks
    feats = result.get("features", {})
    if not isinstance(feats, dict) or len(feats) == 0:
        return 0.0  # no towers → no penalty
    if not math.isfinite(layer_height_mm) or layer_height_mm <= 0.0:
        raise ValueError("layer_height_mm must be a positive finite number.")
    if not (math.isfinite(ar_safe) and math.isfinite(ar_bad)) or ar_bad <= ar_safe:
        raise ValueError("ar_bad must be > ar_safe and both finite.")


    _MID = 0.5 * (ar_safe + ar_bad)
    _K   = (2.0 * math.log(9.0)) / (ar_bad - ar_safe)  
    s_vals = []

    for tid, f in feats.items():
        h_layers = int(f.get("height_layers", 0))
        d_min    = float(f.get("d_min_mm_min", 0.0))

        # No exposure → no risk (mirrors your bridge-span mapping)
        if h_layers <= 0:
            continue

        if not math.isfinite(d_min) or d_min <= 0.0:

            s = 1.0
        else:

            h_seg = int(f.get("segment_height_layers_max", h_layers))
            L_mm  = h_seg * layer_height_mm
            ar   = L_mm / d_min

            if not math.isfinite(ar) or ar <= 0.0:
                s = 0.0
            else:
                z = _K * (ar - _MID)
                s = 1.0 / (1.0 + math.exp(-z))

        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0

        s_vals.append(s)

    if not s_vals:
        return 0.0

    s_arr   = np.clip(np.asarray(s_vals, dtype=float), 0.0, 1.0) ** float(beta)
    product = float(np.prod(1.0 - s_arr))
    O = 1.0 - product

    if O < 0.0:
        return 0.0
    if O > 1.0:
        return 1.0
    return O



def compute_slender_towers_penalty(layer_rasters, cfg):
    sl = compute_slender_towers_aspect(layer_rasters)
    layer_h = _get(cfg, "layer_height", default = 0.2)
    objective = normalize_slender_penalty(sl, layer_height_mm = layer_h)
    
    return objective