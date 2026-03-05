#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Dict

    
from print_quality.pipeline.metrics.printability.island_starts import compute_island_starts_penalty
from print_quality.pipeline.metrics.printability.overhang_exposure import compute_overhang_exposure_penalty
from print_quality.pipeline.metrics.printability.bridge_exposure import compute_bridge_exposure_penalty
from print_quality.pipeline.metrics.printability.slender_towers import compute_slender_towers_penalty
from print_quality.pipeline.metrics.printability.stringing_exposure import compute_stringing_exposure_penalty
from print_quality.pipeline.metrics.printability.bed_adhesion import compute_bed_adhesion_penalty
from print_quality.pipeline.metrics.printability.warping import compute_warping_lever_penalty
from print_quality.pipeline.metrics.printability.support_removal import compute_support_removal_penalty


def compute_printability_penalties(mesh_pc, print_job, layer_rasters, pj_dict, cfg) -> Dict[str, Dict[str, float]]:
    out = {}
    out["bed_adhesion"]       = 0 #compute_bed_adhesion_penalty(layer_rasters, cfg, pj_dict["segments"])
    out["warping_lever"]      = 0 #compute_warping_lever_penalty(layer_rasters, cfg)
    out["overhang_exposure"]  = 0#compute_overhang_exposure_penalty(mesh_pc, layer_rasters, print_job)
    out["bridge_exposure"]    = 0#compute_bridge_exposure_penalty(pj_dict, layer_rasters)
    out["island_starts"]      = compute_island_starts_penalty(layer_rasters, cfg)
    out["slender_towers"]     = compute_slender_towers_penalty(layer_rasters, cfg)
    out["stringing_exposure"] = compute_stringing_exposure_penalty(pj_dict, layer_rasters)
    out["support_removal"] = compute_support_removal_penalty(layer_rasters, mesh_pc)
    
    return out