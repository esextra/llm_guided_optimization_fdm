#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any


def compute_surface_distance(
    mesh_pc: Any,
    print_job: Any,
    *,
    bottom_band_mm: float,
    sample_step_mm: float = 1.2,
    n_mesh_samples: int = 4000,
    seed: int = 0,
    line_width_mm: float | None = None,
    layer_height_mm: float | None = None,
) -> Dict[str, float]:
    # not implemented in current version
    return {"HD95": 0.0}


def penalty_surface_distance_hd95(metrics: Dict[str, float], line_width_mm: float, layer_height_mm: float) -> float:
    # not implemented in current version
    return 0.0


def compute_surface_distance_penalty(mesh_pc, print_job, band, w_ref, layer_h, seed):
    # not implemented in current version
    return 0.0