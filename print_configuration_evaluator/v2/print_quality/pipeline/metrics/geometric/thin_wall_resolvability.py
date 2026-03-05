#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any


def compute_thin_wall_resolvability(
    mesh_pc: Any,
    extrusion_width_mm: float,
    *,
    width_threshold_factor: float = 1.5,
    n_surface_samples: int = 8000,
    vertical_only: bool = True,
    vertical_tol_deg: float = 10.0,
    alignment_min_cos: float = 0.85,
    probe_mm: float = 0.02,
    boundary_eps_mm: float = 1e-4,
    rng: Any = None,
) -> Dict[str, float]:
    # not implemented in current version
    return {"thin_area_fraction": 0.0, "threshold_mm": float(width_threshold_factor) * float(extrusion_width_mm)}


def compute_thin_wall_resolvability_penalty(mesh_pc, w_ref, width_threshold_factor):
    # not implemented in current version
    return 0.0