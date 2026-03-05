# print_quality/utils/config_resolution.py
from __future__ import annotations

from typing import Iterable, Optional, Any

try:
    # These exist in your repo
    from print_quality.data.types import Config, Channel
except Exception:  # pragma: no cover
    # Type-only fallback to keep this module importable even if codegen runs before types are in place.
    from typing import TypedDict
    class Config(TypedDict, total=False):  # type: ignore
        nozzle_diameter: float
        layer_height: float
        first_layer_height: float
        external_perimeter_extrusion_width: float
        perimeter_extrusion_width: float
        infill_extrusion_width: float
        solid_infill_extrusion_width: float
        top_infill_extrusion_width: float
        external_perimeter_speed: float
        perimeter_speed: float
        infill_speed: float
        solid_infill_speed: float
        top_solid_infill_speed: float
        bridge_speed: float
        travel_speed: float
        bridge_flow_ratio: float
        filament_max_volumetric_speed: float
    class Channel:  # type: ignore
        P_OUTER="P_OUTER"; P_INNER="P_INNER"; P_OVERHANG="P_OVERHANG"; SKIN_TOP="SKIN_TOP"; SKIN_SOLID="SKIN_SOLID"; INFILL="INFILL"; BRIDGE="BRIDGE"; SUPPORT="SUPPORT"; SUP_IFC="SUP_IFC"; ADHESION="ADHESION"


# ---- helpers ----

def _cfg_get(cfg: Any, keys: Iterable[str]) -> Optional[float]:
    """
    Retrieve a float-like value from Config or dict using a list of candidate keys.
    Accepts flattened names like 'print.layer_height' or bare 'layer_height'.
    Returns None if not found or not castable to float.
    """
    for k in keys:
        # Try exact attribute first
        if hasattr(cfg, k):
            try:
                return float(getattr(cfg, k))  # type: ignore[arg-type]
            except Exception:
                pass
        # Try bare dict-like
        if isinstance(cfg, dict) and k in cfg:
            try:
                return float(cfg[k])  # type: ignore[index]
            except Exception:
                pass
        # Try with common section prefixes
        for prefix in ("print.", "printer.", "filament.", "global."):
            key = prefix + k
            if hasattr(cfg, key):
                try:
                    return float(getattr(cfg, key))
                except Exception:
                    pass
            if isinstance(cfg, dict) and key in cfg:
                try:
                    return float(cfg[key])
                except Exception:
                    pass
    # Fallback: Config.extras (INI keys not mapped into dataclass fields)
    try:
        if hasattr(cfg, "extras"):
            ext = getattr(cfg, "extras") or {}
            for cand in keys:
                if cand in ext:
                    try: return float(ext[cand])
                    except Exception: pass
                for prefix in ("print.", "printer.", "filament.", "global."):
                    candp = prefix + cand
                    if candp in ext:
                        try: return float(ext[candp])
                        except Exception: pass
    except Exception:
        pass
    return None


# ---- defaults by channel ----

def channel_default_width_mm(cfg: Config, channel: Channel) -> Optional[float]:
    """
    Resolve a sensible extrusion width (mm) per channel using config precedence.
    Fallback ultimately to nozzle diameter if per-feature widths are absent.
    """
    c = channel
    nozzle = _cfg_get(cfg, ["nozzle_diameter"])
    if c == Channel.P_OUTER.name:
        return _cfg_get(cfg, ["external_perimeter_extrusion_width", "perimeter_extrusion_width"]) or nozzle
    if c in (Channel.P_INNER.name, Channel.P_OVERHANG.name):
        return _cfg_get(cfg, ["perimeter_extrusion_width"]) or nozzle
    if c == Channel.SKIN_TOP.name:
        return _cfg_get(cfg, ["top_infill_extrusion_width", "solid_infill_extrusion_width", "infill_extrusion_width"]) or nozzle
    if c in (Channel.SKIN_SOLID.name, Channel.SUP_IFC.name, Channel.SUPPORT.name):
        return _cfg_get(cfg, ["solid_infill_extrusion_width", "infill_extrusion_width"]) or nozzle
    if c in (Channel.INFILL.name, Channel.ADHESION.name, Channel.BRIDGE.name):
        return _cfg_get(cfg, ["infill_extrusion_width"]) or nozzle
    return nozzle


def channel_default_speed_mms(cfg: Config, channel: Channel) -> Optional[float]:
    """
    Resolve a sensible default speed (mm/s) per channel.
    """
    c = channel
    if c == Channel.P_OUTER.name:
        return _cfg_get(cfg, ["external_perimeter_speed"]) or _cfg_get(cfg, ["perimeter_speed"])
    if c in (Channel.P_INNER.name, Channel.P_OVERHANG.name):
        return _cfg_get(cfg, ["perimeter_speed"])
    if c == Channel.SKIN_TOP.name:
        return _cfg_get(cfg, ["top_solid_infill_speed"]) or _cfg_get(cfg, ["solid_infill_speed"])
    if c == Channel.SKIN_SOLID.name:
        return _cfg_get(cfg, ["solid_infill_speed"])
    if c == Channel.INFILL.name:
        return _cfg_get(cfg, ["infill_speed"])
    if c == Channel.BRIDGE.name:
        return _cfg_get(cfg, ["bridge_speed"]) or _cfg_get(cfg, ["infill_speed"])
    if c in (Channel.SUPPORT.name, Channel.SUP_IFC.name):
        # Many INIs lack explicit support speeds; fall back to infill
        return _cfg_get(cfg, ["support_material_speed", "support_material_interface_speed", "infill_speed"])
    if c == Channel.ADHESION.name:
        return _cfg_get(cfg, ["first_layer_speed"]) or _cfg_get(cfg, ["perimeter_speed"])
    return None


def resolve_layer_height_mm(cfg: Config, layer_index: int, emitted_height: Optional[float]) -> Optional[float]:
    """
    Pick height for a segment: prefer emitted ;HEIGHT: comment (emitted_height) if present,
    else first_layer_height for layer 0, else layer_height.
    """
    if emitted_height is not None:
        return emitted_height
    if layer_index == 0:
        return _cfg_get(cfg, ["first_layer_height"]) or _cfg_get(cfg, ["layer_height"])
    return _cfg_get(cfg, ["layer_height"])


def flow_caps_mm3s(cfg: Config) -> tuple[Optional[float], Optional[float]]:
    """
    Return (q_min, q_max) volumetric flow caps in mm^3/s if available.
    q_max from filament_max_volumetric_speed; q_min is usually unspecified (return None).
    """
    q_max = _cfg_get(cfg, ["filament_max_volumetric_speed"])
    return (None, q_max)


def bridge_flow_ratio(cfg: Config) -> float:
    return _cfg_get(cfg, ["bridge_flow_ratio"]) or 1.0


def mm_per_min_to_mms(F: float) -> float:
    """Convert mm/min feedrate to mm/s."""
    return F / 60.0
