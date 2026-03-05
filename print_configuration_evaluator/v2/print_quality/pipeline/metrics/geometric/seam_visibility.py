#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any


def compute_seam_visibility(
    mesh_pc: Any,
    print_job: Any,
    *,
    bottom_band_mm: float = 0.45,
    min_seam_gap_mm: float = 0.4,
    bottom_band_mode: str = "auto",
    bottom_band_mm_override: float | None = None,
    bottom_band_z0_mm: float | None = None,
    bottom_band_height_mm: float | None = None,
    bottom_band_skip_n_layers: int = 0,
    bottom_band_skip_mm: float | None = None,
    bottom_band_skip_mode: str = "band",
    min_lateral_area_cm2: float = 1e-9,
    bottom_band_lateral_only: bool = True,
    dispersion: str = "weighted",
) -> Dict[str, float]:
    # not implemented in current version
    return {
        "seam_count": 0,
        "lateral_area_cm2": float("nan"),
        "H_lat_mm": float("nan"),
        "seam_density_per_cm2": 0.0,
        "angular_dispersion": 0.0,
    }


def normalize_seam_visibility(
    metrics: dict,
    layer_height_mm: float,
    *,
    mode: str = "exp",
    tau: float = 1.5,
    r_bad: float = 5.0,
    gamma: float = 0.7,
) -> float:
    # not implemented in current version
    return 0.0


def compute_seam_visibility_penalty(mesh_pc, print_job, bottom_band_mm, layer_height_mm):
    # not implemented in current version
    return 0.0