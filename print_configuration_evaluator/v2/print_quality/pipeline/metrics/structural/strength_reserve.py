#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Any, Dict
from print_quality.pipeline.metrics.helpers import _get


def _to_float_percent(v: Any, default: float = 0.0) -> float:
    """Parse numbers like '15%' or '0.15' to a fraction in [0,1]."""
    if v is None:
        return float(default)
    s = str(v).strip().lower()
    try:
        if s.endswith("%"):
            return max(0.0, min(1.0, float(s[:-1]) / 100.0))
        return max(0.0, min(1.0, float(s)))
    except Exception:
        return float(default)

def strength_reserve(cfg: Any) -> Dict[str, float]:
    """
    Heuristic strength-adequacy score derived from slicer/profile configuration.

    This function builds a unitless "effective_index" in [0, 1] from:
    - infill fraction and infill pattern,
    - shell thickness from perimeters and top/bottom solid layers,
    - extrusion widths and layer height,
    - a mild material-type modifier,
    - an optional annealing modifier (if anneal temperature/time are present).

    It then converts the effective_index into a penalty in [0, 1] (larger = weaker) and
    returns a dict that includes intermediate components and parsed inputs.

    Expected cfg keys (aliases supported via `_get`):
    - Infill: "fill_density" / "infill_density" / "infill"
    - Pattern: "fill_pattern" / "infill_pattern"
    - Geometry/process: "perimeters", "nozzle_diameter", "layer_height"
    - Widths (0 treated as "auto" and replaced with ~1.125× nozzle):
      "perimeter_extrusion_width", "external_perimeter_extrusion_width", "infill_extrusion_width",
      "solid_infill_extrusion_width", "top_solid_infill_extrusion_width", "bottom_solid_infill_extrusion_width"
    - Solid layers: "top_solid_layers", "bottom_solid_layers"
    - Material: "filament_type" / "material"
    - Optional anneal fields: "anneal_temperature_c", "anneal_time_min"

    Returns
    -------
    Dict[str, Any] with keys:
    - "penalty": float in [0, 1]
    - "effective_index": float in [0, 1]
    - "components": small set of intermediate terms
    - "details": parsed inputs and bookkeeping values used in the heuristic
    """

    # --- helpers --------------------------------------------------------------
    def _as_float(x, default):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    def _width_or_nozzle(*keys, fallback):
        v = _get(cfg, *keys, default=None)
        if v is None:
            return fallback
        val = _as_float(v, fallback)
        return fallback if val == 0 else val

    def _clamp(x, lo=0.0, hi=1.0):
        return lo if x < lo else hi if x > hi else x

    # simple piecewise linear interpolation
    def _lerp(x, x0, y0, x1, y1):
        if x <= x0: return y0
        if x >= x1: return y1
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # --- inputs ---------------------------------------------------------------
    infill = _to_float_percent(_get(cfg, "fill_density", "infill_density", "infill", default="0%"))

    perims = _as_float(_get(cfg, "perimeters", default=2), 2.0)
    nozzle = _as_float(_get(cfg, "nozzle_diameter", default=0.4), 0.4)
    layer_h = _as_float(_get(cfg, "layer_height", default=0.2), 0.2)

    # widths
    w_per_int = _width_or_nozzle("perimeter_extrusion_width", "extrusion_width", fallback=1.125*nozzle)
    w_per_ext = _width_or_nozzle("external_perimeter_extrusion_width", fallback=w_per_int)
    w_inf     = _width_or_nozzle("infill_extrusion_width", "extrusion_width", fallback=1.125*nozzle)

    # solid infill widths (optional keys, fall back sanely)
    w_solid   = _width_or_nozzle("solid_infill_extrusion_width", fallback=w_inf)
    w_topsol  = _width_or_nozzle("top_solid_infill_extrusion_width", fallback=w_solid)
    w_botsol  = _width_or_nozzle("bottom_solid_infill_extrusion_width", fallback=w_solid)

    # top/bottom solid layers
    top_layers = int(_as_float(_get(cfg, "top_solid_layers", default=0), 0))
    bot_layers = int(_as_float(_get(cfg, "bottom_solid_layers", default=0), 0))

    # material & optional anneal settings (non-PS fields allowed in cfg)
    material = str(_get(cfg, "filament_type", "material", default="PLA") or "PLA").upper()
    anneal_T = _as_float(_get(cfg, "anneal_temperature_c", default=None), None)
    anneal_t = _as_float(_get(cfg, "anneal_time_min", default=None), None)

    # pattern normalization (aliases & cleanup)
    pat_raw = _get(cfg, "fill_pattern", "infill_pattern", default="gyroid")
    pattern = str(pat_raw or "gyroid").lower().replace(" ", "").replace("_", "-")
    # common aliases / normalizations
    alias = {
        "aligned-rectilinear": "alignedrectilinear",
        "3dhoneycomb": "3dhoneycomb",
        "3d-honeycomb": "3dhoneycomb",
        "octagram-spiral": "octagramspiral",
        "archimedeanchords": "archimedeanchords",
        "zig-zag": "zigzag",
    }
    pattern = alias.get(pattern, pattern)
    if pattern == "zigzag":
        pattern = "line"

    # --- pattern efficiency factors (bounded, heuristic) ----------------------
    # Based on Prusa docs: gyroid (omni-directional), honeycomb strong, 3D honeycomb has interlayer gaps,
    # supportcubic/lightning mainly for top support; triangles/stars/grid similar; concentric/hilbert/spirals mostly aesthetic/flexible.
    pattern_factor_lut = {
        "gyroid":            1.00,
        "honeycomb":         0.95,
        "3dhoneycomb":       0.88,
        "cubic":             0.90,
        "adaptivecubic":     0.90,
        "grid":              0.85,
        "triangles":         0.85,
        "stars":             0.84,
        "rectilinear":       0.80,
        "alignedrectilinear":0.80,
        "line":              0.78,
        "concentric":        0.70,
        "hilbertcurve":      0.72,
        "archimedeanchords": 0.72,
        "octagramspiral":    0.70,
        "supportcubic":      0.40,
        "lightning":         0.25,
    }
    pattern_factor = float(pattern_factor_lut.get(pattern, 0.85))

    # --- shell terms: perimeters + top/bottom skins ---------------------------
    shell_perim_thick = (w_per_ext if perims > 0 else 0.0) + max(perims - 1.0, 0.0) * w_per_int
    skin_top_thick = top_layers * layer_h
    skin_bot_thick = bot_layers * layer_h
    shell_skin_thick = skin_top_thick + skin_bot_thick

    # targets & weights (tunable but realistic)
    T_PERIM = 1.20  # mm ~ three 0.4 mm or two 0.6 mm lines
    T_SKIN  = 0.80  # mm combined top+bottom to get a "real" face sheet
    perim_score = _clamp(shell_perim_thick / T_PERIM)
    skin_score  = _clamp(shell_skin_thick  / T_SKIN)

    # perimeters dominate strength more than skins; skins still matter for bending
    shell_score = 0.75 * perim_score + 0.25 * skin_score

    # --- infill term, scaled by width & pattern --------------------------------
    # width effect: mixed evidence; allow only a mild boost/penalty around nozzle
    w_inf_ratio = _clamp(w_inf / nozzle, 0.7, 1.4)
    infill_width_factor = _clamp(w_inf_ratio ** 0.5, 0.85, 1.18)  # soften effect

    infill_component = infill * pattern_factor * infill_width_factor
    infill_component = _clamp(infill_component, 0.0, 1.0)

    # --- global modifiers: layer height, material, annealing -------------------
    # layer-height vs nozzle ratio: smaller layers -> modest boost; very tall layers (>=80% nozzle) -> penalty
    r = layer_h / max(nozzle, 1e-9)
    if r <= 0.25:        layer_factor = 1.10
    elif r <= 0.50:      layer_factor = _lerp(r, 0.25, 1.10, 0.50, 1.00)
    elif r <= 0.80:      layer_factor = _lerp(r, 0.50, 1.00, 0.80, 0.92)
    else:                layer_factor = 0.90
    layer_factor = _clamp(layer_factor, 0.85, 1.12)

    # material factor (very mild; this reflects typical interlayer behavior, not bulk datasheet strength)
    mat = material
    mat_factor = 1.00
    if "CF" in mat or "CARBON" in mat: mat_factor *= 0.95
    if "GF" in mat or "GLASS" in mat:  mat_factor *= 0.97
    if "PETG" in mat:                  mat_factor *= 1.02
    if mat in ("PC", "POLYCARBONATE"): mat_factor *= 1.05
    if mat.startswith("PA") or "NYLON" in mat: mat_factor *= 1.05
    if mat in ("ABS", "ASA"):          mat_factor *= 0.98

    # annealing: small boost in correct temperature windows (if provided)
    anneal_factor = 1.00
    if anneal_T is not None and anneal_t is not None and anneal_t >= 20:
        T = anneal_T
        if "PLA" in mat and 75 <= T <= 100:
            anneal_factor = 1.03 if anneal_t < 45 else 1.05
        elif "PETG" in mat and 55 <= T <= 80:
            anneal_factor = 1.01 if anneal_t < 45 else 1.02

    # --- combine ----------------------------------------------------------------
    # Perimeters > infill for strength per Prusa KB -> higher shell weight
    w_shell, w_infill = 0.60, 0.40
    base_effective = w_shell * shell_score + w_infill * infill_component

    global_mod = layer_factor * mat_factor * anneal_factor
    effective = _clamp(base_effective * global_mod, 0.0, 1.0)

    # adequacy threshold: target >= 0.6
    penalty = 1.0 - min(1.0, effective / 0.60)

    return {
        "penalty": float(_clamp(penalty)),
        "effective_index": float(effective),
        "components": {
            "shell_score": float(shell_score),
            "infill_component": float(infill_component),
            "layer_factor": float(layer_factor),
            "material_factor": float(mat_factor),
            "anneal_factor": float(anneal_factor),
            "infill_width_factor": float(infill_width_factor),
        },
        "details": {
            "pattern": pattern,
            "pattern_factor": float(pattern_factor),
            "infill_fraction": float(infill),
            "perimeters": float(perims),
            "shell_perimeter_thickness_mm": float(shell_perim_thick),
            "shell_skin_thickness_mm": float(shell_skin_thick),
            "layer_height_mm": float(layer_h),
            "nozzle_mm": float(nozzle),
            "perim_width_internal_mm": float(w_per_int),
            "perim_width_external_mm": float(w_per_ext),
            "infill_width_mm": float(w_inf),
            "top_solid_layers": int(top_layers),
            "bottom_solid_layers": int(bot_layers),
            "top_solid_infill_width_mm": float(w_topsol),
            "bottom_solid_infill_width_mm": float(w_botsol),
            "solid_infill_width_mm": float(w_solid),
            "material": material,
            "anneal_temperature_c": anneal_T,
            "anneal_time_min": anneal_t,
            "weights": {"shell": w_shell, "infill": w_infill},
            "targets": {"T_PERIM_mm": T_PERIM, "T_SKIN_mm": T_SKIN},
        },
    }


def compute_strength_reserve_penalty(cfg):
    sr = strength_reserve(cfg)
    return sr["penalty"]