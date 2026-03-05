from typing import Dict, Any, Optional, Tuple, List
import numpy as np


def _count_components_8c(mask: Optional[np.ndarray]) -> int:

    if mask is None or mask.size == 0 or not mask.any():
        return 0
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    count = 0
    for y in range(H):
        row = mask[y]
        if not row.any():
            continue
        for x in range(W):
            if not row[x] or visited[y, x]:
                continue

            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def interface_contact_area_mm2(
    layer_rasters: Dict[int, Dict[str, Any]],
    *,
    max_top_gap_layers: int = 6,      
    max_bottom_gap_layers: int = 6,   
    include_top: bool = True,
    include_bottom: bool = True,
    area_equiv_tol: float = 0.20,    
) -> Dict[str, float]:

    IFC_KEY = "SUP_IFC"
    PART_KEY = "V_part"

    if not layer_rasters:
        return {
            "A_contact_mm2": 0.0, "A_top_mm2": 0.0, "A_bottom_mm2": 0.0,
            "N_top_px": 0.0, "N_bottom_px": 0.0, "pixel_xy_mm": 0.0,
            "top_layers_count": 0, "bottom_layers_count": 0,
            "comp_top_count": 0, "comp_bottom_count": 0, "comp_union_count": 0,
        }

    # Fixed pixel pitch (same everywhere)
    px = None
    for r in layer_rasters.values():
        if IFC_KEY in r:
            px = float(r[IFC_KEY].pixel_xy); break
        if PART_KEY in r:
            px = float(r[PART_KEY].pixel_xy); break
    if px is None:
        raise KeyError("Could not infer pixel_xy from SUP_IFC or V_part rasters.")

    # Order layers by Z
    zs = sorted(layer_rasters.keys())
    L = len(zs)
    if L == 0:
        return {
            "A_contact_mm2": 0.0, "A_top_mm2": 0.0, "A_bottom_mm2": 0.0,
            "N_top_px": 0.0, "N_bottom_px": 0.0, "pixel_xy_mm": px,
            "top_layers_count": 0, "bottom_layers_count": 0,
            "comp_top_count": 0, "comp_bottom_count": 0, "comp_union_count": 0,
        }


    sample = None
    for z in zs:
        r = layer_rasters[z]
        if IFC_KEY in r:
            sample = np.asarray(r[IFC_KEY].mask, dtype=bool); break
        if PART_KEY in r:
            sample = np.asarray(r[PART_KEY].mask, dtype=bool); break
    if sample is None:
        return {
            "A_contact_mm2": 0.0, "A_top_mm2": 0.0, "A_bottom_mm2": 0.0,
            "N_top_px": 0.0, "N_bottom_px": 0.0, "pixel_xy_mm": px,
            "top_layers_count": 0, "bottom_layers_count": 0,
            "comp_top_count": 0, "comp_bottom_count": 0, "comp_union_count": 0,
        }
    H, W = sample.shape

    ifc = np.zeros((L, H, W), dtype=bool)
    part = np.zeros((L, H, W), dtype=bool)
    for i, z in enumerate(zs):
        r = layer_rasters[z]
        if IFC_KEY in r:
            ifc[i] = np.asarray(r[IFC_KEY].mask, dtype=bool)
        if PART_KEY in r:
            part[i] = np.asarray(r[PART_KEY].mask, dtype=bool)

    N_top_px = 0
    N_bottom_px = 0


    mask_top_contact = np.zeros((H, W), dtype=bool) if include_top else None
    mask_bottom_contact = np.zeros((H, W), dtype=bool) if include_bottom else None


    top_layer_px: List[int] = []
    bottom_layer_px: List[int] = []

    if include_top and max_top_gap_layers > 0:
        for k in range(L):
            if not ifc[k].any():
                continue

            stop_ifc  = min(L, k + max_top_gap_layers)     
            stop_part = min(L, k + max_top_gap_layers + 1)  


            any_part_above = part[k+1:stop_part].any(axis=0) if k+1 < stop_part else np.zeros((H, W), dtype=bool)


            cand_k = ifc[k] & any_part_above

            closer_contacting_ifc_above = np.zeros((H, W), dtype=bool)
            if k+1 < stop_ifc:
                for j in range(k+1, stop_ifc):
                    if not ifc[j].any():
                        continue
                    stop_part_j = min(L, j + max_top_gap_layers + 1)
                    any_part_above_j = part[j+1:stop_part_j].any(axis=0) if j+1 < stop_part_j else np.zeros((H, W), dtype=bool)
                    if any_part_above_j.any():
                        closer_contacting_ifc_above |= (ifc[j] & any_part_above_j)

            topmost_ifc = ifc[k] & ~closer_contacting_ifc_above
            contact_here = topmost_ifc & any_part_above

            if not contact_here.any():

                if cand_k.any():

                    top_layer_px.append(int(cand_k.sum()))
                continue

            s_area = int(contact_here.sum())
            N_top_px += s_area

            s_for_stack = int(cand_k.sum())
            top_layer_px.append(s_for_stack if s_for_stack > 0 else s_area)
            
            mask_top_contact |= contact_here

    if include_bottom and max_bottom_gap_layers > 0:
        for k in range(L):
            if not ifc[k].any():
                continue

            start_ifc  = max(0, k - max_bottom_gap_layers + 1)  # for "closer ifc below"
            start_part = max(0, k - max_bottom_gap_layers)      # inclusive lower bound for part search

            any_part_below = part[start_part:k].any(axis=0) if start_part < k else np.zeros((H, W), dtype=bool)


            cand_k_bottom = ifc[k] & any_part_below


            closer_contacting_ifc_below = np.zeros((H, W), dtype=bool)
            if start_ifc < k:
                for j in range(start_ifc, k):
                    if not ifc[j].any():
                        continue
                    start_part_j = max(0, j - max_bottom_gap_layers)
                    any_part_below_j = part[start_part_j:j].any(axis=0) if start_part_j < j else np.zeros((H, W), dtype=bool)
                    if any_part_below_j.any():
                        closer_contacting_ifc_below |= (ifc[j] & any_part_below_j)


            bottommost_ifc = ifc[k] & ~closer_contacting_ifc_below
            contact_here = bottommost_ifc & any_part_below

            if not contact_here.any():
                if cand_k_bottom.any():
                    bottom_layer_px.append(int(cand_k_bottom.sum()))
                continue


            s_area = int(contact_here.sum())
            N_bottom_px += s_area

            s_for_stack = int(cand_k_bottom.sum())
            bottom_layer_px.append(s_for_stack if s_for_stack > 0 else s_area)
            mask_bottom_contact |= contact_here


    A_top_mm2 = N_top_px * (px * px)
    A_bottom_mm2 = N_bottom_px * (px * px)

    def _count_similar_layers(areas: List[int], tol: float) -> int:
        if not areas:
            return 0
        amax = max(areas)
        thresh = (1.0 - max(0.0, float(tol))) * amax
        return sum(1 for a in areas if a >= thresh and a > 0)

    top_layers_count = _count_similar_layers(top_layer_px, area_equiv_tol) if include_top else 0
    bottom_layers_count = _count_similar_layers(bottom_layer_px, area_equiv_tol) if include_bottom else 0

    comp_top_count = _count_components_8c(mask_top_contact) if mask_top_contact is not None else 0
    comp_bottom_count = _count_components_8c(mask_bottom_contact) if mask_bottom_contact is not None else 0
    if mask_top_contact is not None and mask_bottom_contact is not None:
        comp_union_count = _count_components_8c(np.logical_or(mask_top_contact, mask_bottom_contact))
    else:
        comp_union_count = _count_components_8c(mask_top_contact) if mask_bottom_contact is None else _count_components_8c(mask_bottom_contact)

    return {
        "A_contact_mm2": float(A_top_mm2 + A_bottom_mm2),
        "A_top_mm2": float(A_top_mm2),
        "A_bottom_mm2": float(A_bottom_mm2),
        "N_top_px": float(N_top_px),
        "N_bottom_px": float(N_bottom_px),
        "pixel_xy_mm": float(px),
        "top_layers_count": int(top_layers_count),
        "bottom_layers_count": int(bottom_layers_count),
        "comp_top_count": int(comp_top_count),
        "comp_bottom_count": int(comp_bottom_count),
        "comp_union_count": int(comp_union_count),
    }


def compute_support_removal_penalty(
    layer_rasters: Dict[int, Dict[str, Any]],
    mesh_pc: Any,
    *,
    r0: float = 0.03,   
    k: float = 1.5,     
    s_ref: float = 1.0, 
    C_ref: float = 20.0,
    w_base: float = 0.60,
    w_stack: float = 0.25,
    w_frag: float  = 0.15,
    area_equiv_tol: float = 0.20,
    **kwargs
):
    """
    Returns a removability penalty in [0,1] without clipping.

    Components:
      r       = A_contact / A_surface_total
      f_base  = 1 - exp( - (r / r0)^k )
      s       = max(top_layers_count, bottom_layers_count) - 1  (extra similar-area layers)
      f_stack = s / (s + s_ref)
      C       = comp_union_count  (outermost-contact components)
      f_frag  = C / (C + C_ref)

    penalty = w_base*f_base + w_stack*f_stack + w_frag*f_frag  (weights renormalized)
    """
    import math  # local import to keep function drop-in safe

    sr = interface_contact_area_mm2(layer_rasters, area_equiv_tol=area_equiv_tol, **kwargs)


    A_contact = float(sr["A_contact_mm2"])
    A_surface_total = float(getattr(mesh_pc, "surface_area", 0.0))
    r = (A_contact / A_surface_total) if A_surface_total > 0.0 else 0.0
    r = max(0.0, r)


    r0 = max(1e-12, float(r0))
    k  = max(1.0, float(k))
    x  = (r / r0) ** k
    f_base = 1.0 - math.exp(-min(50.0, x))  # numeric safety

    stack_layers = max(int(sr["top_layers_count"]), int(sr["bottom_layers_count"]))
    s = max(0.0, float(stack_layers - 1))
    s_ref = max(1e-12, float(s_ref))
    f_stack = s / (s + s_ref)

    C = max(0.0, float(sr["comp_union_count"]))
    C_ref = max(1e-12, float(C_ref))
    f_frag = C / (C + C_ref)


    w_base = float(w_base); w_stack = float(w_stack); w_frag = float(w_frag)
    w_sum = w_base + w_stack + w_frag
    if w_sum <= 0.0:
        w_base, w_stack, w_frag, w_sum = 1.0, 0.0, 0.0, 1.0
    w_base /= w_sum; w_stack /= w_sum; w_frag /= w_sum

    penalty = (w_base * f_base) + (w_stack * f_stack) + (w_frag * f_frag)

    return penalty
