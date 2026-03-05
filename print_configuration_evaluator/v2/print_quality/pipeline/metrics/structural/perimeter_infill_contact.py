#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, Optional, Any
import numpy as np
from print_quality.utils.graph import contact_length_pixels

def perim_infill_contact(
    layer_rasters: Dict[int, Dict[str, Any]],
    role_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:

    """
    Compute perimeter↔infill contact length from raster masks.

    For each layer that contains both:
      (a) a perimeter mask, and
      (b) an infill-like mask,
    this function measures contact as the number of 4-neighbor shared pixel-edges
    between the two boolean masks via `contact_length_pixels(...)`. These per-layer
    contacts are summed in pixel-edges and scaled to millimeters using the raster
    pixel size `layer["V_part"].pixel_xy` (assumed constant; the last seen value is used).

    Role assignment:
    - If `role_map` is provided: only keys mapped to "P_OUTER" or "P_INNER" contribute
      to the perimeter union, and only keys mapped to "INFILL" contribute to the infill union.
      All other keys are ignored.
    - If `role_map` is None (legacy mode): the perimeter union is ("P_OUTER", "P_INNER", "P_OVERHANG"),
      and the infill-like union is ("INFILL", "BRIDGE", "SKIN_SOLID"). Other keys are ignored.

    Geometry-only opportunity (normalization denominator):
    For each layer with a valid `V_part.mask`, an opportunity value U_px is computed as the
    shared-edge length between the perimeter mask and interior geometry:
      interior_geom = V_part.mask & (~perimeter_mask)
      U_px = contact_length_pixels(perimeter_mask, interior_geom)
    If the layer dict contains a raw numpy array under "SKIN_TOP", its adjacency is subtracted
    from U_px. Layers contribute to the global opportunity sum only when U_px > 20 pixel-edges.

    Normalization:
    - If total_opportunity_px_edges > 0:
        normalized_contact = min(1, total_contact_px_edges / total_opportunity_px_edges)
        penalty = 1 - normalized_contact
    - Otherwise (no interior opportunity anywhere), normalized_contact = 1 and penalty = 0.

    Return value:
    A dict containing totals in pixel-edges and mm plus normalization fields.
    Note: if `layer_rasters` is empty, the returned dict contains only the basic length fields
    and does not include the normalization/opportunity keys.
    """
    if not layer_rasters:
        return {
            "total_mm": 0.0, "mean_mm": 0.0,
            "layers_with_contact": 0, "layers_total": 0,
            "pixel_size_mm": 0.0, "total_px_edges": 0.0,
        }

    def _role_for_key(k: str) -> Optional[str]:
        if role_map is not None:
            role = role_map.get(k, "UNKNOWN")
            return role if role in ("P_OUTER", "P_INNER", "INFILL") else None
        # Fallback to legacy exact names if no role_map is provided
        if k in ("P_OUTER", "P_INNER", "INFILL", "BRIDGE", "SKIN_SOLID", "SKIN_TOP", "P_OVERHANG"):
            return k
        return None

    total_px = 0.0
    contrib_layers = 0
    last_px = 0.0
    total_opp_px = 0.0
    opp_layers = 0
    opportunity_mm = 0.0
    normalized_contact = 1.0
    penalty = 0.0
    

    for _, r in layer_rasters.items():
        try:
            px = float(r["V_part"].pixel_xy)
        except Exception as e:
            raise TypeError("Expected V_part.pixel_xy to be a scalar (e.g., 0.25).") from e
        if px <= 0:
            raise ValueError(f"pixel size must be positive, got {px}.")
        last_px = px

        perim_mask: Optional[np.ndarray] = None
        infill_mask: Optional[np.ndarray] = None

        # Collect masks by role
        for key, rast in r.items():
            role = _role_for_key(key)
            if role is None:
                continue
            m = rast.mask  # bool numpy array
            if role in ("P_OUTER", "P_INNER", "P_OVERHANG"):
                perim_mask = m if perim_mask is None else (perim_mask | m)
            elif role in ("INFILL", "BRIDGE", "SKIN_SOLID"):
                infill_mask = m if infill_mask is None else (infill_mask | m)

        if perim_mask is None or infill_mask is None:
            continue


        Lpx = contact_length_pixels(perim_mask, infill_mask)
        total_px += float(Lpx)
        contrib_layers += 1
        
        vpart_raster = r.get("V_part", None)
        vpart = vpart_raster.mask
        if isinstance(vpart, np.ndarray):
            perim_b = perim_mask.astype(bool)

            interior_geom = vpart.astype(bool) & (~perim_b)
 

            U_px = float(contact_length_pixels(perim_b, interior_geom))
 

            skin_top = r.get("SKIN_TOP", None)
            if isinstance(skin_top, np.ndarray):
                U_px -= float(contact_length_pixels(perim_b, skin_top.astype(bool)))
            

            if U_px > 20.0: #### >0.0:  #POOR WAY TO AVOID QUANTIZATION ERRORS #maybe uncounting a little #I had several examples where one layer had like 4 or 6 pixels worth of interior space
                total_opp_px += U_px
                opp_layers += 1
    

    layers_total = len(layer_rasters)
    total_mm = total_px * last_px
    mean_mm = (total_mm / contrib_layers) if contrib_layers > 0 else 0.0
 

    if total_opp_px > 0.0:
        normalized_contact = float(min(1.0, total_px / total_opp_px))
        penalty = float(1.0 - normalized_contact)
        opportunity_mm = total_opp_px * last_px
    else:

        normalized_contact = 1.0
        penalty = 0.0
        opportunity_mm = 0.0



    return {
        "total_mm": float(total_mm),
        "mean_mm": float(mean_mm),
        "layers_with_contact": int(contrib_layers),
        "layers_total": int(layers_total),
        "pixel_size_mm": float(last_px),
        "total_px_edges": float(total_px),
        "opportunity_mm": float(opportunity_mm),
        "opportunity_px_edges": float(total_opp_px),
        "layers_with_opportunity": int(opp_layers),
        "normalized_contact": float(normalized_contact),
        "penalty": float(penalty),
    }


def compute_perim_infill_contact_penalty(layer_rasters):
    pic = perim_infill_contact(layer_rasters)
    return pic["penalty"]