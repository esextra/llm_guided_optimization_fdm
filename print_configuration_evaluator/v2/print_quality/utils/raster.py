# print_quality/utils/raster.py  (updated to support fixed bounds for all layers)
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Any

import numpy as np

try:
    from shapely.geometry import LineString, Polygon, box, Point
    from shapely.ops import unary_union
except Exception as e:  # pragma: no cover
    LineString = None
    Polygon = None
    unary_union = None
    Point = None
    


# ----- Basic geometry stroking and rasterization -----

def stroke_polyline_to_polygon(polyline_xy: np.ndarray, width_mm: float) -> "Polygon":
    """
    Stroke a 2D polyline into a Polygon by buffering a LineString by width/2.
    Requires shapely. Raises if shapely is not available.
    """
    if LineString is None:
        raise RuntimeError("shapely is required for stroke_polyline_to_polygon")
    if width_mm is None or width_mm <= 0:
        raise ValueError("width_mm must be > 0")
    ls = LineString(polyline_xy.tolist())
    # # cap_style=2 (flat), join_style=2 (mitre) works well for toolpaths
    # return ls.buffer(width_mm / 2.0, cap_style=2, join_style=2)
    # Use ROUND joins/caps so curved/faceted paths do not inflate perimeter.
    # (cap_style/join_style: 1=round, 2=flat/mitre)
    return ls.buffer(width_mm / 2.0, cap_style=1, join_style=1)



def union_polygons(polys: List["Polygon"]) -> "Polygon":
    if unary_union is None:
        raise RuntimeError("shapely is required for union_polygons")
    if len(polys) == 0:
        return Polygon()
    return unary_union(polys)


@dataclass
class Raster:
    mask: np.ndarray                 # (H,W) bool
    origin_xy: Tuple[float, float]   # lower-left world coordinates of pixel (0,0)
    pixel_xy: float                  # pixel size in mm


def _rasterize_with_bounds(poly, pixel_xy, bounds, margin=0.0):
    """
    Rasterize onto a boolean grid defined explicitly by bounds (minx, miny, maxx, maxy).
    Sampling: center-point test using poly.covers(Point(x,y)) so boundary pixels are included.
    """
    if Polygon is None or Point is None:
        raise RuntimeError("shapely is required for rasterization")
    minx, miny, maxx, maxy = bounds
    minx -= margin; miny -= margin; maxx += margin; maxy += margin
    # Grid size
    W = max(1, int(np.ceil((maxx - minx) / pixel_xy)))
    H = max(1, int(np.ceil((maxy - miny) / pixel_xy)))
    if getattr(poly, "is_empty", False):
        return Raster(mask=np.zeros((H, W), dtype=bool), origin_xy=(minx, miny), pixel_xy=pixel_xy)
    # proceed with center-point sampling on the same grid
    xs = minx + (np.arange(W) + 0.5) * pixel_xy
    ys = miny + (np.arange(H) + 0.5) * pixel_xy
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    mask = np.fromiter((poly.covers(Point(x, y)) for x, y in pts), count=pts.shape[0], dtype=bool).reshape(H, W)
    return Raster(mask=mask, origin_xy=(minx, miny), pixel_xy=pixel_xy)


def rasterize_polygon_union_to_mask(poly: "Polygon", pixel_xy: float, margin: float = 0.0) -> Raster:
    """
    Backwards-compatible helper that derives bounds from the polygon itself.
    NOTE: prefer _rasterize_with_bounds when composing multiple channels per layer.
    """
    if Polygon is None:
        raise RuntimeError("shapely is required for rasterization")
    if poly.is_empty:
        return Raster(mask=np.zeros((1, 1), dtype=bool), origin_xy=(0.0, 0.0), pixel_xy=pixel_xy)
    # Use poly's own bounds (legacy behavior)
    return _rasterize_with_bounds(poly, pixel_xy, poly.bounds, margin)


# ----- Routing helpers at layer level -----

PART_CHANNELS = {"P_OUTER","P_INNER","P_OVERHANG","SKIN_TOP","SKIN_SOLID","INFILL","BRIDGE"}
SUP_CHANNELS  = {"SUPPORT","SUP_IFC"}
ADH_CHANNELS  = {"ADHESION"}


def rasterize_layer_segments(
    segments: list,
    pixel_xy: float,
    channels: Iterable[str],
    fixed_bounds: Optional[Tuple[float, float, float, float]] = None,
    bounds_margin_mm: float = 0.0,
    polygon_cache: Optional[Dict[int, Any]] = None,
) -> Dict[str, Raster]:
    """
    Given segments for one layer, create a raster per requested channel.
    If polygon_cache is provided, reuse pre-stroked polygons keyed by id(segment).
    Sampling is delegated to _rasterize_with_bounds so margin is applied exactly once.
    """
    channels = tuple(channels)
    polys_by_channel: Dict[str, list] = {ch: [] for ch in channels}

    # 1) Collect stroked polygons per channel (reuse cache if provided)
    for seg in segments:
        ch = seg.get("feature")
        if ch not in polys_by_channel:
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
            try:
                poly = stroke_polyline_to_polygon(polyline, width)
            except Exception:
                poly = None
            if polygon_cache is not None and poly is not None:
                polygon_cache[id(seg)] = poly
        if poly is not None:
            polys_by_channel[ch].append(poly)

    if not any(polys_by_channel[ch] for ch in channels):
        return {}

    # 2) Determine bounds (raw; margin applied inside _rasterize_with_bounds)
    if fixed_bounds is None:
        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for ch in channels:
            for poly in polys_by_channel[ch]:
                bx = poly.bounds
                if bx[0] < minx: minx = bx[0]
                if bx[1] < miny: miny = bx[1]
                if bx[2] > maxx: maxx = bx[2]
                if bx[3] > maxy: maxy = bx[3]
        bounds = (minx, miny, maxx, maxy)
    else:
        bounds = fixed_bounds

    # 3) For each channel, union polygons if possible; otherwise OR per-poly rasters
    rasters: Dict[str, Raster] = {}
    for ch in channels:
        polys = polys_by_channel[ch]
        if not polys:
            continue
        try:
            union = union_polygons(polys) if len(polys) > 1 else polys[0]
            rasters[ch] = _rasterize_with_bounds(union, pixel_xy, bounds, margin=bounds_margin_mm)
        except Exception:
            # Fallback: rasterize each polygon and OR masks on the same grid
            accum = None
            template = None
            for poly in polys:
                r = _rasterize_with_bounds(poly, pixel_xy, bounds, margin=bounds_margin_mm)
                if accum is None:
                    accum = r.mask.copy(); template = r
                else:
                    accum |= r.mask
            if template is not None:
                rasters[ch] = Raster(accum, template.origin_xy, template.pixel_xy)

    return rasters



def layer_composite_masks(layer_rasters: Dict[str, Raster]) -> Dict[str, Raster]:
    """
    Compose V_part, V_sup, V_adh masks from per-channel rasters.
    Assumes all rasters share the same grid (origin & pixel size).
    """
    if not layer_rasters:
        z = Raster(mask=np.zeros((1,1), dtype=bool), origin_xy=(0.0,0.0), pixel_xy=0.2)
        return {"V_part": z, "V_sup": z, "V_adh": z}

    any_raster = next(iter(layer_rasters.values()))
    H, W = any_raster.mask.shape
    V_part = np.zeros((H, W), dtype=bool)
    V_sup  = np.zeros((H, W), dtype=bool)
    V_adh  = np.zeros((H, W), dtype=bool)

    for ch, r in layer_rasters.items():
        # Shapes must agree now; assert once to catch regressions
        if r.mask.shape != (H, W):
            raise ValueError(f"Raster grid mismatch for channel {ch}: {r.mask.shape} vs {(H,W)}")
        if ch in PART_CHANNELS:
            V_part |= r.mask
        elif ch in SUP_CHANNELS:
            V_sup |= r.mask
        elif ch in ADH_CHANNELS:
            V_adh |= r.mask

    return {
        "V_part": Raster(V_part, any_raster.origin_xy, any_raster.pixel_xy),
        "V_sup":  Raster(V_sup, any_raster.origin_xy, any_raster.pixel_xy),
        "V_adh":  Raster(V_adh, any_raster.origin_xy, any_raster.pixel_xy),
    }
