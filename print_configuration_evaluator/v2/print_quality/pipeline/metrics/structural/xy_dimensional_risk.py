#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
from print_quality.utils.raster import Raster
from print_quality.pipeline.metrics.helpers import _get, _layer_z_map, _soft_excess 
from print_quality.pipeline.build_rasters import footprint_metrics_from_raster

def xy_dimensional_risk(
    cfg, layer_rasters,  print_job ,       # REQUIRED: {layer_index -> Z (mm)}
    *,
    z_cutoff_mm: float,                      # REQUIRED: include layers with Z <= z_cutoff_mm
) -> Dict[str, float]:
    """
    Compute an XY dimensional-risk penalty in [0, 1] (higher = worse), dominated by
    first-layer conditions and early-layer base geometry.

    Core drivers (first layer):
    - Squish ratio: first_layer_extrusion_width / first_layer_height, mapped by _soft_excess
      against a reference SQUISH_REF.
    - Z-offset magnitude: |z_offset| mapped by _soft_excess against a small Z_REF.
    - First-layer flow multiplier: first_layer_flow (or related aliases) mapped by _soft_excess
      against FLOW_REF.
    These are combined into a weighted driver term.

    Material/bed term (Tg-aware):
    - Tg is selected from a small material lookup keyed by filament_type/material.
    - bed_temperature contributes via bed_over = max(0, bed_t - Tg), mapped by _soft_excess.
    This bed term is used in the estimated lateral bulge used for compensation relief.

    Elefant-foot compensation relief:
    - An estimated bulge (mm) is computed from squish excess plus bed/z/flow contributions and
      first-layer width, capped at EST_BULGE_MAX.
    - Relief fraction p_comp_relief = min(1, elefant_foot_compensation / est_bulge_mm).
    - A RELIEF_CAP limits how much compensation can reduce the core penalty.

    Geometry sensitivity over early layers (Z <= z_cutoff_mm):
    - Base footprint uses V_base := (V_part OR V_adh). For each eligible layer, compute
      G = perimeter_mm / area_mm2 from the base footprint; the mean G increases risk.
    - A concealment term uses part-only footprint area growth between adjacent early layers;
      higher growth reduces risk (intended to represent chamfer concealment).
    - These form a multiplicative geometry factor clamped to a modest range.

    Inputs/assumptions:
    - print_job must allow a layer-index -> Z(mm) map via _layer_z_map.
    - layer_rasters must contain aligned Raster objects for keys "V_part" and "V_adh"
      for the layers at/below z_cutoff_mm; mismatched grids raise ValueError.
    - If no eligible layers have a valid base footprint at/below z_cutoff_mm, raises ValueError.

    Returns a dict with:
    - "penalty" plus diagnostic fields including squish, bed/Tg terms, compensation terms,
      G_mean, geom_mult, and penalty_core.
    """
    layer_z_mm = _layer_z_map(print_job)
    
    # --- First-layer ratios and core drivers ---
    w1 = float(_get(cfg, "first_layer_extrusion_width", "first_layer_line_width",
                    default=_get(cfg, "nozzle_diameter", default=0.4)) or 0.4)
    h1 = float(_get(cfg, "first_layer_height", default=_get(cfg, "layer_height", default=0.2)) or 0.2)
    bed_t = float(_get(cfg, "bed_temperature", "first_layer_bed_temperature", default=60) or 60)
    comp = abs(float(_get(cfg, "elefant_foot_compensation", default=0.0) or 0.0))

    # Squish
    SQUISH_REF = 1.9
    squish = w1 / max(1e-6, h1)
    p_squish = _soft_excess(squish, SQUISH_REF)

    # --- Material-aware bed term, anchored to Tg ---
    material = str(_get(cfg, "filament_type", "material", default="PLA") or "PLA").upper()
    TG_C = {
        "PLA": 60.0, "PETG": 80.0, "ABS": 105.0, "ASA": 100.0,
        "NYLON": 70.0, "PA": 70.0, "PC": 110.0, "PP": 60.0, "TPU": 60.0,
    }
    Tg = float(TG_C.get(material, 70.0))
    BED_OVER_REF = 5.0  # start ramp ~5 °C above Tg
    bed_over = max(0.0, bed_t - Tg)
    p_bed = _soft_excess(bed_over, BED_OVER_REF)

    # --- Z-offset and first-layer flow ---
    z_off = float(_get(cfg, "z_offset", default=0.0) or 0.0)
    flow1 = float(_get(cfg,
                       "first_layer_flow", "first_layer_flow_ratio",
                       "first_layer_flow_multiplier", "first_layer_extrusion_multiplier",
                       "extrusion_multiplier",
                       default=1.0) or 1.0)
    Z_REF = 0.03     # ~30 µm
    FLOW_REF = 1.05  # >5% over nominal
    p_z = _soft_excess(abs(z_off), Z_REF)
    p_flow = _soft_excess(flow1, FLOW_REF)

    # Combine drivers (pre-comp)
    driver = (0.60 * p_squish) + (0.25 * p_z) + (0.15 * p_flow)

    # --- Compensation tied to an estimated bulge (driver-dependent; no unconditional base) ---
    EST_BULGE_MAX = 0.30
    excess_ratio = max(0.0, (squish - SQUISH_REF) / SQUISH_REF)
    # gains (tunable): how each driver contributes to lateral bulge
    G_BULGE_SQUISH = 1.5
    G_BULGE_BED    = 0.35
    G_BULGE_Z      = 0.20
    G_BULGE_FLOW   = 0.20
    driver_magnitude = (
        G_BULGE_SQUISH * excess_ratio +
        G_BULGE_BED    * p_bed +
        G_BULGE_Z      * p_z +
        G_BULGE_FLOW   * p_flow
    )
    est_bulge_mm = min(EST_BULGE_MAX, max(0.0, driver_magnitude) * float(w1))
    p_comp_relief = 0.0 if est_bulge_mm <= 1e-9 else min(1.0, comp / est_bulge_mm)
    # apply a relief cap so compensation cannot fully cancel the risk
    RELIEF_CAP = 0.70  # tunable: 0.6–0.85
    penalty_core = driver * (1.0 - RELIEF_CAP * p_comp_relief)

    # --- Geometry sensitivity: base contact (adhesion included) and chamfer concealment (part-only) ---
    # 1) Base contact intensity from V_base := V_part OR V_adh (adhesion included) via G = P/A
    # 2) Concealment: part-only area growth across early layers (chamfers expand upward)
    L = sorted(li for li, z in layer_z_mm.items() if z is not None and z <= z_cutoff_mm)
    G_vals = []
    growth_fracs = []
    for i, li in enumerate(L):
        ras = layer_rasters[li]
        Vp = ras["V_part"]; Va = ras["V_adh"]
        if (Vp.origin_xy != Va.origin_xy) or (Vp.pixel_xy != Va.pixel_xy) or (Vp.mask.shape != Va.mask.shape):
            raise ValueError(f"Raster grid mismatch on layer {li}")
        # base footprint (adhesion included)
        base_mask = (Vp.mask | Va.mask)
        V_base = Raster(mask=base_mask, origin_xy=Vp.origin_xy, pixel_xy=Vp.pixel_xy)
        m_base = footprint_metrics_from_raster(V_base)
        A_base = float(m_base["area_mm2"]); P_base = float(m_base["perimeter_mm"])
        if A_base > 1e-12 and P_base > 1e-12:
            G_vals.append(P_base / A_base)
        # part-only growth (concealment signal)
        A_part = float(footprint_metrics_from_raster(Vp)["area_mm2"])
        if i + 1 < len(L):
            li2 = L[i + 1]
            Vp2 = layer_rasters[li2]["V_part"]
            A_part_next = float(footprint_metrics_from_raster(Vp2)["area_mm2"])
            if A_part > 1e-12 and A_part_next >= A_part:
                growth_fracs.append((A_part_next - A_part) / A_part)
    if not G_vals:
        raise ValueError("No layers with valid base footprint at or below z_cutoff_mm; check layer_z_mm and cutoff.")
    G_mean = sum(G_vals) / len(G_vals)
    A_growth_mean = (sum(growth_fracs) / len(growth_fracs)) if growth_fracs else 0.0
    G_REF = 0.12;      GEOM_GAIN = 0.60
    CHAMFER_REF = 0.02; CHAMFER_GAIN = 0.85
    p_geom    = _soft_excess(G_mean, G_REF)                 # adhesion included
    p_conceal = _soft_excess(A_growth_mean, CHAMFER_REF)    # chamfer/area growth reduces risk
    geom_mult = (1.0 + GEOM_GAIN * p_geom) * (1.0 - CHAMFER_GAIN * p_conceal)
    geom_mult = max(0.5, min(1.5, geom_mult))

    # Final clamp
    penalty = float(max(0.0, min(1.0, penalty_core * geom_mult)))

    return {
        "penalty": penalty,
        # diagnostics
        "squish_ratio": float(squish),
        "first_layer_width_mm": float(w1),
        "first_layer_height_mm": float(h1),
        "bed_temp_c": float(bed_t),
        "material": material,
        "Tg_c": float(Tg),
        "bed_over_Tg_c": float(bed_over),
        "p_squish": float(p_squish),
        "p_bed": float(p_bed),
        "p_z": float(p_z),
        "p_flow": float(p_flow),
        "est_bulge_mm": float(est_bulge_mm),
        "xy_comp_mm": float(comp),
        "p_comp_relief": float(p_comp_relief),
        "G_mean_perimeter_over_area_invmm": float(G_mean),
        "geom_mult": float(geom_mult),
        "penalty_core": float(penalty_core),
        "z_cutoff_mm": float(z_cutoff_mm),
    }

def compute_xy_dimensional_risk_penalty(cfg, layer_rasters, pj_dict, z_cutoff_mm=0.6):
    xy = xy_dimensional_risk(cfg, layer_rasters, pj_dict, z_cutoff_mm=0.6)
    return xy["penalty"]
    