#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict
import numpy as np
from print_quality.pipeline.metrics.helpers import  _get

from print_quality.pipeline.metrics.geometric.surface_distance import compute_surface_distance_penalty
from print_quality.pipeline.metrics.geometric.stair_stepping import compute_stair_stepping_penalty
from print_quality.pipeline.metrics.geometric.seam_visibility import compute_seam_visibility_penalty
from print_quality.pipeline.metrics.geometric.thin_wall_resolvability import compute_thin_wall_resolvability_penalty

from print_quality.pipeline.metrics.helpers import _job_bottom_origin_and_band, _select_reference_line_width_mm



def compute_geometric_penalties(mesh_pc, print_job, cfg) -> Dict[str, Dict[str, float]]:
    layer_h = _get(cfg, "layer_height", default=0.2)
    w_ref = _select_reference_line_width_mm(cfg, policy="conservative")   # or "optimistic"
    z0, band = _job_bottom_origin_and_band(cfg, print_job, mesh_pc, default_band=0.45)
    seed = 2023
    out = {}
    out["surface_distance"]   = 0 #compute_surface_distance_penalty(mesh_pc, print_job, band, w_ref, layer_h, seed)
    out["stair_stepping"]     = compute_stair_stepping_penalty(mesh_pc, layer_h, bottom_band_mm=band, z0=z0, max_layer_height=0.27)
    out["thin_wall"]          = 0#compute_thin_wall_resolvability_penalty(mesh_pc, w_ref, width_threshold_factor=1.5)
    out["seam_visibility"]    = 0 #compute_seam_visibility_penalty(mesh_pc, print_job, band, layer_h)
    
    return out

