#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from print_quality.utils.graph import connected_components,  _binary_dilate
from typing import Dict, Any, List, Literal
from print_quality.pipeline.metrics.helpers import _get


def compute_island_starts(
    layer_rasters,
    cfg,
    *,
    start_dilate_px: int = 3,        
    join_dilate_px: int = 0,         
    track_dilate_px: int = 0,        
    min_area_mm2: float = 1.0,
    persist_prev_layers: int = 2,
):
    """
    Detect and characterize “island starts” in a per-layer raster stack.

    Definition (start detection):
    - For each layer index li (except the first layer in the provided raster dict),
      compute 8-connected components in the part mask V_part.
    - A component is counted as a “start” if:
      (1) its area is >= min_area_mm2, and
      (2) it has zero overlap with the union of the previous persist_prev_layers
          of (V_part ∪ V_sup), after applying a robustness dilation of start_dilate_px
          pixels to that previous-union mask (when start_dilate_px > 0).

    Notes / semantics:
    - The first (lowest-index) layer in layer_rasters is never counted as a start
      (it is treated as “touches the plate”).
    - Support is included only for the “previous material” mask used to suppress
      starts; islands themselves are detected only in V_part.
    - The parameter join_dilate_px is currently unused.

    Join tracking (reported metadata):
    - For each detected start at z0=li, the code optionally follows its continuation
      upward by selecting components in higher layers that touch the accumulated island
      footprint (optionally dilated by track_dilate_px).
    - The first layer lj where that continuation overlaps previously printed *non-island*
      material (previous-union at lj, without dilation, excluding the island’s accumulated
      footprint) is recorded as z_join. If no such join is found, z_join is None.

    Returns
    -------
    Dict with:
    - total_island_starts: total count across layers (excluding the first layer)
    - per_layer: dict[layer_index -> start count]
    - islands: list of dicts with keys:
        z0, area0_mm2, perimeter0_mm, z_join, h_layers, h_mm, nearest_prev_mm
    """

    # ---- helpers ------------------------------------------------------------
    def _binary_erode(img: np.ndarray, r: int) -> np.ndarray:
        return np.logical_not(_binary_dilate(np.logical_not(img), r)) if r > 0 else img

    def _perimeter_len_mm(mask: np.ndarray, px_mm: float) -> float:
        boundary = mask & (~_binary_erode(mask, 1))
        return float(boundary.sum()) * px_mm

    def _min_chebyshev_distance_mm(src: np.ndarray, dst: np.ndarray, px_mm: float) -> float | None:
        if not dst.any() or not src.any():
            return None
        front = src.copy()
        for step in range(1, 4096):
            front = _binary_dilate(front, 1)
            if (front & dst).any():
                return step * px_mm
        return None
    # ------------------------------------------------------------------------
    layer_height_mm = float(_get(cfg,"layer_height", default=0.2))
    
    if not layer_rasters:
        return {"total_island_starts": 0, "per_layer": {}, "islands": []}

    layers = sorted(layer_rasters.keys())
    first_li = layers[0]

    px_mm = float(next(iter(layer_rasters.values()))["V_part"].pixel_xy)
    min_area_px = int(np.ceil(max(1.0, float(min_area_mm2) / (px_mm * px_mm))))


    masks = {li: layer_rasters[li]["V_part"].mask.astype(bool) for li in layers}

    H0, W0 = next(iter(masks.values())).shape
    masks_sup = {
        li: (layer_rasters[li]["V_sup"].mask.astype(bool)
             if ("V_sup" in layer_rasters[li] and layer_rasters[li]["V_sup"] is not None)
             else np.zeros((H0, W0), dtype=bool))
        for li in layers
    }


    prev_start_by_layer, prev_join_by_layer = {}, {}
    prev_buf_part, prev_buf_sup = [], []
    for li in layers:
        H, W = masks[li].shape
        prev_part = np.zeros((H, W), bool)
        prev_sup  = np.zeros((H, W), bool)
        for pm in prev_buf_part[-persist_prev_layers:]:
            prev_part |= pm
        for sm in prev_buf_sup[-persist_prev_layers:]:
            prev_sup |= sm
        prev_union = prev_part | prev_sup
        prev_start_by_layer[li] = (_binary_dilate(prev_union, start_dilate_px)
                                   if start_dilate_px > 0 else prev_union)
        prev_join_by_layer[li]  = prev_union  # NO dilation here
        prev_buf_part.append(masks[li])
        prev_buf_sup.append(masks_sup[li])
        if len(prev_buf_part) > max(1, persist_prev_layers):
            prev_buf_part.pop(0)
            prev_buf_sup.pop(0)

    total = 0
    per_layer: dict[int, int] = {}
    islands: list[dict] = []

    for li in layers:
        cur = masks[li] 

        if li == first_li:
            per_layer[li] = 0
            continue

        labels, n = connected_components(cur, connectivity=8)
        prev_start = prev_start_by_layer[li]

        starts = 0
        for k in range(1, n + 1):
            comp = (labels == k)
            if comp.sum() < min_area_px:
                continue

            if (comp & prev_start).any():
                continue


            starts += 1
            z0 = li
            area0_mm2 = float(comp.sum()) * (px_mm * px_mm)
            perimeter0_mm = _perimeter_len_mm(comp, px_mm)
            nearest_prev_mm = _min_chebyshev_distance_mm(comp, prev_join_by_layer[li], px_mm)


            island_accum = comp.copy()
            z_join = None

            for lj in (l for l in layers if l > li):
                cur_j = masks[lj]
                labels_j, n_j = connected_components(cur_j, connectivity=8)

                touch = _binary_dilate(island_accum, track_dilate_px) if track_dilate_px > 0 else island_accum
                sel = np.zeros_like(cur_j, dtype=bool)
                for kk in range(1, n_j + 1):
                    comp_j = (labels_j == kk)
                    if (comp_j & touch).any():
                        sel |= comp_j

                if not sel.any():
                    continue


                prev_join_j = prev_join_by_layer[lj]
                other_prev = prev_join_j & (~island_accum)
                if (sel & other_prev).any():
                    z_join = lj
                    island_accum |= sel
                    break

                island_accum |= sel

            h_layers = (z_join - z0) if z_join is not None else None
            h_mm = (h_layers * layer_height_mm) if (h_layers is not None and layer_height_mm is not None) else None

            islands.append(
                {
                    "z0": int(z0),
                    "area0_mm2": float(area0_mm2),
                    "perimeter0_mm": float(perimeter0_mm),
                    "z_join": (int(z_join) if z_join is not None else None),
                    "h_layers": (int(h_layers) if h_layers is not None else None),
                    "h_mm": (float(h_mm) if h_mm is not None else None),
                    "nearest_prev_mm": (float(nearest_prev_mm) if nearest_prev_mm is not None else None),
                }
            )

        total += starts
        per_layer[li] = starts

    return {
        "total_island_starts": int(total),
        "per_layer": per_layer,
        "islands": islands,
    }



def normalize(islands, nozzle_w_mm, fatal_k=4.0, fatal_h_layers=2, height_gamma=1.0):
    """
    islands: iterable of dicts with keys: 'area0_mm2', 'h_layers', 'z_join'
             (as returned by your compute_island_starts)
    Returns J in [0,1].
    """
    if nozzle_w_mm <= 0:
        raise ValueError("nozzle_w_mm must be > 0")

    w2 = nozzle_w_mm * nozzle_w_mm
    denom_area = fatal_k * w2

    prod = 1.0
    for iso in islands:
        A0 = float(iso.get("area0_mm2", 0.0))
        z_join = iso.get("z_join", None)
        h_layers = iso.get("h_layers", None)

        a = min(1.0, A0 / denom_area)

        if z_join is None:
            hnorm = 1.0
        else:
            if h_layers is None:
                hnorm = 1.0
            else:
                hnorm = min(1.0, float(h_layers) / float(fatal_h_layers))


        hterm = hnorm ** float(height_gamma)
        r_i = 1.0 - (1.0 - a) * (1.0 - hterm)

        prod *= (1.0 - r_i)

    J = 1.0 - prod
    return max(0.0, min(1.0, J))


def compute_island_starts_penalty(layer_rasters,cfg):
    isl = compute_island_starts(layer_rasters, cfg)
    
    nozzle = _get(cfg, "nozzle_diameter", default=0.4)
    objective = normalize(isl["islands"], nozzle)
    return objective

