# print_quality/utils/geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import trimesh
    from trimesh.proximity import closest_point as tm_closest_point  # type: ignore
except Exception:  # pragma: no cover
    trimesh = None
    tm_closest_point = None


@dataclass
class SurfaceLocator:
    """
    Lightweight wrapper providing closest-point and distance queries on a mesh.
    Uses trimesh.proximity if available; otherwise falls back to a KD-tree over face centroids (approximate).
    """
    vertices: np.ndarray  # (N,3)
    faces: np.ndarray     # (M,3)
    _tm: Optional["trimesh.Trimesh"] = None
    _centroids: Optional[np.ndarray] = None

    def __post_init__(self):
        if trimesh is not None:
            try:
                self._tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
            except Exception:
                self._tm = None
        if self._tm is None:
            # Approx fallback: face centroids
            v = self.vertices
            f = self.faces
            self._centroids = v[f].mean(axis=1)

    def closest_point(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (closest_points, distances) for each query point in pts (K,3).
        When trimesh is unavailable, uses centroid nearest-neighbor (approximate).
        """
        pts = np.asarray(pts, dtype=float)
        if self._tm is not None and tm_closest_point is not None:
            cp, dist, _ = tm_closest_point(self._tm, pts)
            return cp, dist
        # Approximate: nearest centroid
        C = self._centroids  # (M,3)
        if C is None or len(C) == 0:
            return np.empty((0, 3)), np.empty((0,))
        # brute-force; fast enough for moderate K
        d2 = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (K,M)
        j = d2.argmin(axis=1)
        cp = C[j]
        dist = np.sqrt(d2[np.arange(len(j)), j])
        return cp, dist

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return self.closest_point(pts)[1]


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute unit face normals for a triangle mesh.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n = n / lens
    return n


def slope_angles_deg(normals: np.ndarray) -> np.ndarray:
    """
    Angle between face normal and +Z axis, in degrees.
    0° => upward-facing horizontal (normal aligned with +Z)
    90° => vertical side wall (normal horizontal)
    180° => downward-facing horizontal (normal aligned with -Z)
    """
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    # Angle between normal and +Z
    theta = np.degrees(np.arccos(nz))
    return theta


def sample_surface_points(vertices: np.ndarray, faces: np.ndarray, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample approximately-uniform points on mesh surface using area-weighted triangle sampling (requires numpy only).
    """
    rng = rng or np.random.default_rng()
    v0 = vertices[faces[:, 0]]; v1 = vertices[faces[:, 1]]; v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        raise ValueError("Mesh has zero total area; cannot sample surface points.")
    p = areas / total
    tri_idx = rng.choice(len(faces), size=n_samples, p=p, replace=True)
    u = np.sqrt(rng.random(n_samples))
    v = rng.random(n_samples)
    a = 1 - u
    b = u * (1 - v)
    c = u * v
    pts = a[:, None] * v0[tri_idx] + b[:, None] * v1[tri_idx] + c[:, None] * v2[tri_idx]
    return pts


def approximate_thickness_along_normal(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray, normals: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Very coarse thickness estimator near sampled surface points:
      - Cast a short step along +/- normal using closest-point distances.
      - Returns NaN where not reliable (e.g., missing normals).
    This is a placeholder for a more robust medial-axis based estimator.
    """
    if normals is None:
        return np.full((len(points),), np.nan, dtype=float)
    loc = SurfaceLocator(vertices, faces)
    forward = points + 1.0 * normals  # 1 mm step
    backward = points - 1.0 * normals
    d_f = loc.distance(forward)
    d_b = loc.distance(backward)
    # crude proxy: sum of distances
    return d_f + d_b
