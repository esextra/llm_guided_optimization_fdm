# print_quality/io/config_parser.py
from __future__ import annotations

import configparser
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)

# -------- Key normalization (Prusa/JSON -> internal) --------
KEY_MAP: Dict[str, str] = {
    # Geometry / process
    "nozzle_diameter": "nozzle_diameter",
    "printer_variant": "printer_variant",  # fallback for nozzle
    "layer_height": "layer_height",
    "first_layer_height": "first_layer_height",
    "seam_position": "seam_position",

    # Shells / infill
    "fill_density": "fill_density",              # keep strings like "5%"
    "infill_density": "infill_density",          # JSON synonym
    "fill_pattern": "fill_pattern",
    "infill_pattern": "fill_pattern",            # normalize
    "perimeters": "perimeters",

    # Widths (mm or % of nozzle) — post-normalized
    "extrusion_width": "extrusion_width",
    "perimeter_extrusion_width": "perimeter_extrusion_width",
    "external_perimeter_extrusion_width": "external_perimeter_extrusion_width",
    "infill_extrusion_width": "infill_extrusion_width",
    "solid_infill_extrusion_width": "solid_infill_extrusion_width",
    "top_infill_extrusion_width": "top_infill_extrusion_width",
    "first_layer_extrusion_width": "first_layer_extrusion_width",

    # Top skins
    "top_solid_layers": "top_solid_layers",

    # Temps / material
    "bed_temperature": "bed_temperature",
    "filament_type": "filament_type",
    "filament_settings_id": "filament_settings_id",

    # Speeds / timing / flow
    "max_print_speed": "max_print_speed",
    "travel_speed": "travel_speed",
    "first_layer_speed": "first_layer_speed",
    "perimeter_speed": "perimeter_speed",
    "solid_infill_speed": "solid_infill_speed",
    "top_solid_infill_speed": "top_solid_infill_speed",
    "infill_speed": "infill_speed",
    "support_material_speed": "support_material_speed",
    "support_material_interface_speed": "support_material_interface_speed",  # may be "80%"
    "min_layer_time": "min_layer_time",
    "bridge_speed": "bridge_speed",
    "bridge_flow_ratio": "bridge_flow_ratio",
    "filament_max_volumetric_speed": "filament_max_volumetric_speed",

    # Supports
    "support_material": "support_material",
    "support_material_threshold": "support_material_threshold",

    # Dimensional XY
    "elefant_foot_compensation": "elefant_foot_compensation",
}

LOWER_STRING_KEYS = {"fill_pattern", "seam_position", "filament_type", "filament_settings_id"}
DENSITY_KEYS = {"fill_density", "infill_density"}
WIDTH_KEYS = {
    "extrusion_width", "perimeter_extrusion_width", "external_perimeter_extrusion_width",
    "infill_extrusion_width", "solid_infill_extrusion_width", "top_infill_extrusion_width",
    "first_layer_extrusion_width",
}
PERCENT_SPEED_KEYS = {("support_material_interface_speed", "support_material_speed")}

def _boolish(s: str) -> bool:
    return s.strip().lower() in {"1","true","on","yes"}

def _coerce_scalar(key: str, val: str) -> Any:
    s = (val if isinstance(val, str) else str(val)).strip()

    # keep densities as strings ("5%")
    if key in DENSITY_KEYS:
        return s

    # enums as lowercase strings
    if key in LOWER_STRING_KEYS:
        return s.lower() if s else None

    # explicit booleans
    if key in {"support_material"}:
        if s.lower() in {"0","1","true","false","on","off","yes","no"}:
            return _boolish(s)

    # widths: float or "120%" (leave % for post-pass)
    if key in WIDTH_KEYS:
        try:
            return float(s)
        except Exception:
            return s if s.endswith("%") else None

    # percent speeds (e.g., "80%") — resolve later
    if key in {"support_material_interface_speed"} and s.endswith("%"):
        return s

    # try float
    try:
        return float(s)
    except Exception:
        if s.lower() in {"nil", ""}:
            return None
        return s

def _flatten_ini_loose(ini_path: Path) -> Dict[str, str]:
    """
    Read Prusa INI. If headerless (no [section]), wrap in a virtual [__root__] section.
    """
    text = ini_path.read_text(encoding="utf-8", errors="ignore")
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(text)
    except configparser.MissingSectionHeaderError:
        cp.read_string("[__root__]\n" + text)

    flat: Dict[str, str] = {}
    # defaults first
    for k, v in cp.defaults().items():
        flat[k.lower()] = v
    # then all sections (later wins)
    for sec in cp.sections():
        for k, v in cp.items(sec):
            flat[k.lower()] = v
    return flat

def _map_and_coerce(flat: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    for k, v in flat.items():
        ink = k.lower()
        outk = KEY_MAP.get(ink)
        if outk is None:
            extras[ink] = v
            continue
        kwargs[outk] = _coerce_scalar(outk, v)

    # nozzle fallback from printer_variant (e.g., "0.4")
    if not kwargs.get("nozzle_diameter"):
        pv = kwargs.get("printer_variant") or extras.get("printer_variant")
        try:
            nd = float(str(pv).split(",")[0].strip()) if pv is not None else None
        except Exception:
            nd = None
        if nd:
            kwargs["nozzle_diameter"] = nd

    return kwargs, extras

def _normalize_widths(kwargs: Dict[str, Any]) -> None:
    nz = kwargs.get("nozzle_diameter")
    try:
        nozzle = float(str(nz).split(",")[0]) if nz is not None else None
    except Exception:
        nozzle = None
    if not nozzle:
        return  # no dummy nozzle guessed

    for k in WIDTH_KEYS:
        v = kwargs.get(k)
        if isinstance(v, str) and v.endswith("%"):
            try:
                kwargs[k] = float(v.rstrip("%"))/100.0 * nozzle
            except Exception:
                pass

def _normalize_percent_speeds(kwargs: Dict[str, Any]) -> None:
    for target, base in PERCENT_SPEED_KEYS:
        tv, bv = kwargs.get(target), kwargs.get(base)
        if isinstance(tv, str) and tv.endswith("%") and isinstance(bv, (int, float)):
            try:
                kwargs[target] = float(tv.rstrip("%"))/100.0 * float(bv)
            except Exception:
                pass

def _diagnose_missing(kwargs: Dict[str, Any], extras: Dict[str, Any]) -> None:
    metric_keys = [
        # functional
        "fill_density","fill_pattern","perimeters","first_layer_extrusion_width","top_solid_layers",
        # risk / prediction
        "seam_position","support_material","support_material_threshold","filament_max_volumetric_speed",
        # general
        "nozzle_diameter","layer_height","first_layer_height","max_print_speed",
        "infill_speed","perimeter_speed",
    ]
    missing = [k for k in metric_keys if kwargs.get(k) is None]
    if missing:
        shadowed = [k for k in missing if k in extras]
        log.warning(f"[CFG] Missing critical keys after parse: {missing}"
                    + (f" (present in extras: {shadowed})" if shadowed else ""))

def parse_config_ini(path: str | Path) -> Dict[str, Any]:
    ini_path = Path(path)
    if not ini_path.exists():
        raise FileNotFoundError(f"INI not found: {ini_path}")
    flat = _flatten_ini_loose(ini_path)
    kwargs, extras = _map_and_coerce(flat)
    _normalize_widths(kwargs)
    _normalize_percent_speeds(kwargs)
    log.info(f"[CFG] Parsed INI: {ini_path.name}")
    _diagnose_missing(kwargs, extras)
    cfg = dict(kwargs)
    cfg["extras"] = extras
    return cfg

def parse_config_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    flat: Dict[str, str] = {k.lower(): (v if isinstance(v, str) else str(v)) for k, v in obj.items()}
    kwargs, extras = _map_and_coerce(flat)
    _normalize_widths(kwargs)
    _normalize_percent_speeds(kwargs)
    log.info(f"[CFG] Parsed JSON: {p.name}")
    _diagnose_missing(kwargs, extras)
    cfg = dict(kwargs)
    cfg["extras"] = extras
    return cfg
