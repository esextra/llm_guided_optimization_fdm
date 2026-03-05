# print_quality/pipeline/mesh_precompute.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Dict, Any

import numpy as np

try:
    # Prefer repo types if available
    from print_quality.data.types import MeshData
except Exception:  # pragma: no cover
    # Minimal fallback so this module can import before types are generated
    @dataclass
    class MeshData:  # type: ignore
        vertices: np.ndarray
        faces: np.ndarray
        bbox_min: Optional[np.ndarray] = None
        bbox_max: Optional[np.ndarray] = None

from print_quality.utils.geometry import (
    SurfaceLocator,
    compute_face_normals,
    slope_angles_deg,
    sample_surface_points,
)

def align_mesh_xy_to_raster(md: MeshData, layer_rasters: Dict[int, Dict[str, Any]]) -> tuple[MeshData, tuple[float,float]]:
    """
    Translate mesh XY so md.bbox_min[:2] aligns with the raster grid origin.
    Uses the first available layer's composite raster (V_part or V_adh) as the reference.
    Returns (new_mesh_data, (dx, dy)).
    """
    # pick a reference raster
    if not layer_rasters:
        return md, (0.0, 0.0)
    li0 = min(layer_rasters.keys())
    r0 = layer_rasters[li0]
    base = r0.get("V_part") or r0.get("V_adh")
    if base is None:
        return md, (0.0, 0.0)
    ox, oy = map(float, base.origin_xy)
    # compute delta from mesh bbox_min to raster origin
    dx = ox - float(md.bbox_min[0])
    dy = oy - float(md.bbox_min[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return md, (0.0, 0.0)
    # apply in-place (new object so callers don't mutate their copy)
    md2 = MeshData(vertices=md.vertices.copy(),
                   faces=md.faces,
                   face_normals=md.face_normals,
                   vertex_normals=md.vertex_normals,
                   bbox_min=md.bbox_min.copy(),
                   bbox_max=md.bbox_max.copy())
    md2.vertices[:,0] += dx
    md2.vertices[:,1] += dy
    md2.bbox_min[0] += dx; md2.bbox_min[1] += dy
    md2.bbox_max[0] += dx; md2.bbox_max[1] += dy
    return md2, (dx, dy)


@dataclass
class MeshPrecompute:
    """
    Precomputed mesh queries used by prediction & risk metrics.

    Exact computations (no dummy data):
      - face normals and per-face slope angles (deg) w.r.t. +Z
      - total surface area and face areas
      - closest-point locator for distance queries (exact if trimesh.proximity is available; see note)
    """
    vertices: np.ndarray         # (N,3) float64
    faces: np.ndarray            # (M,3) int32
    face_normals: np.ndarray     # (M,3) float64 unit vectors
    face_slope_deg: np.ndarray   # (M,) float64
    face_areas: np.ndarray       # (M,) float64
    surface_area: float          # sum(face_areas)
    bbox_min: np.ndarray         # (3,)
    bbox_max: np.ndarray         # (3,)
    locator: SurfaceLocator

    # ---- Query helpers ----

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """
        Euclidean distance from query points (K,3) to the mesh surface.
        Exact if trimesh.proximity is available; otherwise uses centroid-NN approximation (see utils.geometry).
        """
        return self.locator.distance(pts)

    def sample_surface(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Area-weighted sampling of n surface points, shape (n,3)."""
        return sample_surface_points(self.vertices, self.faces, n, rng)

    def face_centroids(self) -> np.ndarray:
        """Return (M,3) array of triangle centroids."""
        v = self.vertices
        f = self.faces
        return v[f].mean(axis=1)

    def downward_face_indices(self, angle_gt: float = 45.0) -> np.ndarray:
        # downward-facing (nz < 0) and slope beyond threshold from horizontal:
        return np.nonzero(
            (self.face_normals[:, 2] < 0.0) &
            (self.face_slope_deg > (90.0 + float(angle_gt)))
        )[0]

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bbox_min, self.bbox_max


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def build_mesh_precompute(mesh: MeshData) -> MeshPrecompute:
    """
    Build MeshPrecompute from MeshData.
    No dummy data; any approximations are documented:
      - Distance queries are exact if trimesh.proximity is installed; otherwise approximate by nearest face centroids.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)

    normals = compute_face_normals(V, F)
    slopes = slope_angles_deg(normals)
    areas = _triangle_areas(V, F)
    surface_area = float(areas.sum())

    if getattr(mesh, "bbox_min", None) is not None and getattr(mesh, "bbox_max", None) is not None:
        bmin = np.asarray(mesh.bbox_min, dtype=float)
        bmax = np.asarray(mesh.bbox_max, dtype=float)
    else:
        bmin = V.min(axis=0)
        bmax = V.max(axis=0)

    locator = SurfaceLocator(V, F)

    return MeshPrecompute(
        vertices=V,
        faces=F,
        face_normals=normals,
        face_slope_deg=slopes,
        face_areas=areas,
        surface_area=surface_area,
        bbox_min=bmin,
        bbox_max=bmax,
        locator=locator,
    )
