# print_quality/pipeline/build_rasters.py  (updated: global, fixed grid across all layers)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List, Any
import math
import numpy as np

try:
    # Prefer repo types if available
    from print_quality.data.types import PrintJob
except Exception:  # pragma: no cover
    # Minimal fallback so this module can import before types are generated
    from typing import TypedDict, List
    class Segment(TypedDict, total=False):  # type: ignore
        layer_index: int
        polyline_mm: list
        width_w_mm: float
        feature: str
        is_travel: bool
    class PrintJob(TypedDict, total=False):  # type: ignore
        segments: List[Segment]
        config: dict

from print_quality.utils.raster import (
    Raster,
    stroke_polyline_to_polygon,
    rasterize_layer_segments,
    layer_composite_masks,
    PART_CHANNELS,
    SUP_CHANNELS,
    ADH_CHANNELS,
    union_polygons,
)
from print_quality.utils.graph import connected_components
from print_quality.pipeline.metrics.helpers import  _get
from print_quality.utils.config_resolution import channel_default_width_mm

# --- helpers to derive a uniform grid and create blank rasters on it ---

def _grid_from_bounds_and_pixel(bounds: Tuple[float, float, float, float],
                                pixel_xy_mm: float,
                                bounds_margin_mm: float):
    """
    Compute (shape, origin_xy, pixel_xy) for the fixed grid that rasterize_layer_segments()
    will use when called with fixed_bounds=bounds and bounds_margin_mm=bounds_margin_mm.
    """
    minx, miny, maxx, maxy = bounds
    # apply margin the same way the rasterizer does
    ox = float(minx - bounds_margin_mm)
    oy = float(miny - bounds_margin_mm)
    width  = float((maxx - minx) + 2.0 * bounds_margin_mm)
    height = float((maxy - miny) + 2.0 * bounds_margin_mm)
    # note: nx (X/cols), ny (Y/rows)
    nx = int(np.ceil(width  / float(pixel_xy_mm)))
    ny = int(np.ceil(height / float(pixel_xy_mm)))
    return (ny, nx), (ox, oy), float(pixel_xy_mm)

def _blank_raster(shape: Tuple[int, int], origin_xy: Tuple[float, float], pixel_xy: float) -> Raster:
    """Create an empty Raster on the given grid."""
    mask = np.zeros(shape, dtype=bool)
    # Support both positional and keyword Raster constructors.
    try:
        return Raster(mask, origin_xy, pixel_xy)  # positional
    except TypeError:
        return Raster(mask=mask, origin_xy=origin_xy, pixel_xy=pixel_xy)  # keyword



@dataclass
class LayerComposite:
    V_part: Raster
    V_sup: Raster
    V_adh: Raster


def _collect_segments_by_layer(print_job: PrintJob) -> Dict[int, list]:
    by_layer: Dict[int, list] = {}
    for s in print_job["segments"]:
        li = int(s.get("layer_index", 0)) if isinstance(s, dict) else int(getattr(s, "layer_index", 0))
        by_layer.setdefault(li, []).append(s)
    return by_layer


def _include_adhesion_from_config(cfg: dict) -> bool:
    """
    Decide whether to include ADHESION in rasters.
    Per our plan: only include if brim/raft is set; skirt alone does not increase contact area.
    """
    keys = [
        "brim_width",
        "raft_layers",
    ]
    for k in keys:
        val = _get(cfg,k,default=0)
        try:
            if val is None:
                continue
            v = float(val)
            if "brim" in k and v > 0:
                return True
            if "raft" in k and v >= 1:
                return True
        except Exception:
            continue
    return False

def _polyline_open_for_stroke(poly):
    """
    Return an np.ndarray of shape (N,2). If the polyline is a closed ring,
    drop the duplicated last vertex. Never returns a plain list.
    """
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr
    # ensure we only keep XY
    if arr.shape[1] > 2:
        arr = arr[:, :2]

    is_closed = (arr.shape[0] >= 3) and np.allclose(arr[0], arr[-1], rtol=0, atol=1e-9)
    return arr[:-1] if is_closed else arr


def _compute_global_bounds(
    print_job, 
    by_layer: Dict[int, list],
    channels: Iterable[str],
    bounds_margin_mm: float = 0.0,
    *,
    polygon_cache: Optional[Dict[int, Any]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute global XY bounds across ALL layers/channels by stroking polylines (width-aware).
    If polygon_cache is provided, memoize stroked polygons keyed by id(segment) for reuse later.
    """
    # Keep your original import name exactly as Polygon (no aliasing)
    try:
        from shapely.geometry import Polygon  # type: ignore
    except Exception:
        Polygon = None  # type: ignore

    global_minx = global_miny = float("inf")
    global_maxx = global_maxy = float("-inf")
    any_poly = False

    for _, segs in by_layer.items():
        for seg in segs:
            ch = seg.get("feature")
            if ch not in channels:
                continue
            if seg.get("is_travel", False):
                continue

            width = seg.get("width_w_mm")
            if not width:
                # Prefer a channel-aware default if you have it; otherwise fall back to general extrusion width
                try:
                    ch = seg.get("feature")
                    width = channel_default_width_mm(print_job["config"], ch)  # may return None
                except Exception:
                    width = None
            
                if width is None:
                    # Absolute last resort for bounds: a conservative width (e.g., nozzle/extrusion width)
                    width = float(_get(print_job["config"], "extrusion_width", 0.45))

            polyline = seg.get("polyline_mm", None)
            if polyline is None:
                continue
            # Handle lists/tuples/ndarrays without triggering "ambiguous truth value"
            try:
                if len(polyline) == 0:
                    continue
            except TypeError:
                # If it doesn't support len(), treat as invalid
                continue

            key = id(seg)
            poly = None
            if polygon_cache is not None:
                poly = polygon_cache.get(key)

            if poly is None:
                try:
                    #pline_for_stroke = _polyline_open_for_stroke(polyline)
                    poly = stroke_polyline_to_polygon(polyline, width) 
                    #poly = stroke_polyline_to_polygon(pline_for_stroke, width) #polyline, width)
                except Exception:
                    poly = None
                if polygon_cache is not None and poly is not None:
                    polygon_cache[key] = poly  # memoize for rasterization

            if poly is None:
                continue

            any_poly = True
            minx, miny, maxx, maxy = poly.bounds
            if minx < global_minx: global_minx = minx
            if miny < global_miny: global_miny = miny
            if maxx > global_maxx: global_maxx = maxx
            if maxy > global_maxy: global_maxy = maxy

    if not any_poly:
        return None

    # Note: bounds_margin_mm is applied inside rasterization; return raw bounds here.
    return (global_minx, global_miny, global_maxx, global_maxy)


def build_layer_rasters(
    print_job: PrintJob,
    pixel_xy_mm: float = 0.20,
    channels: Optional[Iterable[str]] = None,
    include_adhesion: Optional[bool] = None,
    bounds_margin_mm: float = 0.0,
) -> Dict[int, Dict[str, Raster]]:
    """
    Rasterize channels per layer and compose V_part, V_sup, V_adh **on a single, fixed grid**.
    Returns: {layer_index: {"V_part": Raster, "V_sup": Raster, "V_adh": Raster, <per-channel rasters>...}}
    """

    by_layer: Dict[int, list] = _collect_segments_by_layer(print_job)

    # Decide channels
    if channels is None:
        channels_set = set(PART_CHANNELS) | set(SUP_CHANNELS)
        if include_adhesion is None:
            include_adhesion = _include_adhesion_from_config(print_job["config"])
        if include_adhesion:
            channels_set |= set(ADH_CHANNELS)
        channels = tuple(channels_set)
    else:
        channels = tuple(channels)

    # Build a cache and populate it during the bounds pass
    polygon_cache: Dict[int, Any] = {}
    bounds = _compute_global_bounds(
        print_job,
        by_layer,
        channels,
        bounds_margin_mm=bounds_margin_mm,
        polygon_cache=polygon_cache,  # <- memoize here
    )
    if bounds is None:
        return {}

    # Rasterize each layer, reusing cached polygons (no second stroking)
    out: Dict[int, Dict[str, Raster]] = {}

    for layer_idx, segs in by_layer.items():
    # for layer_idx in sorted(by_layer.keys()):
    #     segs = by_layer[layer_idx]
        rasters = rasterize_layer_segments(
            segs,
            pixel_xy_mm,
            channels,
            fixed_bounds=bounds,
            bounds_margin_mm=bounds_margin_mm,
            polygon_cache=polygon_cache,  # <- reuse cache here
        )
    
        # --- NEW: seed blanks on the fixed global grid for any missing channels ---
        if len(rasters) > 0:
            # Use any existing raster on this layer to capture the canonical grid.
            any_r = next(iter(rasters.values()))
            shape = tuple(any_r.mask.shape)
            origin_xy = tuple(any_r.origin_xy)
            px = float(any_r.pixel_xy)
        else:
            # This layer produced no rasters at all; derive the grid from global bounds.
            shape, origin_xy, px = _grid_from_bounds_and_pixel(
                bounds, pixel_xy_mm=pixel_xy_mm, bounds_margin_mm=bounds_margin_mm
            )
    
        # Ensure all requested channels exist on this grid (blank if absent)
        for ch in channels:
            if ch not in rasters:
                rasters[ch] = _blank_raster(shape, origin_xy, px)
        # --- END NEW ---
    
        rasters.update(layer_composite_masks(rasters))

        # ---- NEW: attach vector unions (part / adh / base) for this layer ----
        # We reuse polygon_cache entries (if present) to avoid re-stroking.
        polys_part: List[Any] = []
        polys_adh:  List[Any] = []
        for seg in segs:
            ch = seg.get("feature")
            if ch not in channels:
                continue
            if seg.get("is_travel", False):
                continue
            width = seg.get("width_w_mm")
            if not width:
                continue
            polyline = seg.get("polyline_mm", None)
            if polyline is None:
                continue
            try:
                if len(polyline) == 0:
                    continue
            except TypeError:
                continue
            poly = polygon_cache.get(id(seg)) if polygon_cache is not None else None
            if poly is None:
                # Fallback only if cache missed (should be rare since we filled it earlier)
                try:
                    poly = stroke_polyline_to_polygon(polyline, width)
                except Exception:
                    poly = None
            if poly is None:
                continue
            if ch in PART_CHANNELS:
                polys_part.append(poly)
            elif ch in ADH_CHANNELS:
                polys_adh.append(poly)

        poly_part = union_polygons(polys_part) if len(polys_part) > 0 else None
        poly_adh  = union_polygons(polys_adh)  if len(polys_adh)  > 0 else None
        if poly_part is not None and poly_adh is not None:
            poly_base = union_polygons([poly_part, poly_adh])
        else:
            poly_base = poly_part or poly_adh

        # Stash under a reserved key to avoid colliding with Raster channels.
        rasters["_VECTORS"] = {
            "part": poly_part,
            "adh":  poly_adh,
            "base": poly_base,
        }

        out[layer_idx] = rasters

    return out




def _perimeter_crofton(mask: np.ndarray, px: float) -> float:
    # Axis transitions
    h = np.sum(mask[:, 1:] ^ mask[:, :-1])
    v = np.sum(mask[1:, :] ^ mask[:-1, :])
    # Diagonal transitions (two diagonal families)
    d1 = np.sum(mask[1:, 1:] ^ mask[:-1, :-1])
    d2 = np.sum(mask[1:, :-1] ^ mask[:-1, 1:])
    # 4-direction Crofton estimate (weights follow the standard 0°,90°,45°,135° scheme)
    return float((math.pi / 8.0) * (h + v + (d1 + d2) / math.sqrt(2.0)) * px)

def footprint_metrics_from_raster(V_part: Raster) -> dict:
    mask = V_part.mask.astype(bool)
    px = float(V_part.pixel_xy)

    A_px = int(mask.sum())
    area_mm2 = A_px * (px * px)

    perimeter_mm = _perimeter_crofton(mask, px)  # <— changed
    compactness = 0.0 if perimeter_mm <= 1e-12 else float(4.0 * np.pi * area_mm2 / (perimeter_mm * perimeter_mm))

    from print_quality.utils.graph import connected_components
    labels, n = connected_components(mask, connectivity=8)

    return {"area_mm2": float(area_mm2),
            "perimeter_mm": float(perimeter_mm),
            "compactness": float(compactness),
            "island_count": int(n),
            "pixels": int(A_px)}
