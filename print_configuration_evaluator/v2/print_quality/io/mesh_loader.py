from __future__ import annotations
from typing import Optional
import numpy as np
import trimesh
from ..data.types import MeshData
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

def load_mesh(path: str) -> MeshData:
    tm = trimesh.load(path, force='mesh')
    if not tm.is_watertight:
        print("Warning: Mesh is not watertight. Results may be inaccurate.")
    tm.fix_normals()
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(g for g in tm.dump().geometry.values()))

    vertices = tm.vertices.view(np.ndarray).copy()
    faces = tm.faces.view(np.ndarray).copy()
    face_normals = tm.face_normals.view(np.ndarray).copy() if tm.face_normals is not None else None
    vertex_normals = tm.vertex_normals.view(np.ndarray).copy() if tm.vertex_normals is not None else None
    bbox_min, bbox_max = tm.bounds
    bbox_min = bbox_min.astype(np.float64, copy=True)
    bbox_max = bbox_max.astype(np.float64, copy=True)

    log.info(f"Loaded mesh with {len(vertices)} vertices, {len(faces)} faces.")
    return MeshData(vertices=vertices, faces=faces,
                    face_normals=face_normals, vertex_normals=vertex_normals,
                    bbox_min=bbox_min, bbox_max=bbox_max)
