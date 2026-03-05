#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Dict, Optional

from print_quality.pipeline.metrics.structural.strength_reserve import compute_strength_reserve_penalty
from print_quality.pipeline.metrics.structural.xy_dimensional_risk import compute_xy_dimensional_risk_penalty
from print_quality.pipeline.metrics.structural.thermal_creep_risk import compute_thermal_creep_risk_penalty
from print_quality.pipeline.metrics.structural.perimeter_infill_contact import compute_perim_infill_contact_penalty
from print_quality.pipeline.metrics.structural.z_bonding import directional_z_bonding_penalty 



def compute_structural_penalties(mesh_pc, pj_dict, layer_rasters, cfg, load_bearing, load_direction, rx_deg, ry_deg, rz_deg, risk_cache: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute a set of functional metrics (all return a 'penalty' in [0,1]).
    """
    out = {}
    out["strength_reserve"]      = compute_strength_reserve_penalty(cfg)
    out["xy_dimensional_risk"]   = compute_xy_dimensional_risk_penalty(cfg, layer_rasters, pj_dict, z_cutoff_mm=0.6)
    out["thermal_creep_risk"]    = 0 #compute_thermal_creep_risk_penalty(cfg)   
    out["perim_infill_contact"]  = compute_perim_infill_contact_penalty(layer_rasters)
    out["z_bonding_proxy"] = directional_z_bonding_penalty(pj_dict, cfg, layer_rasters, load_bearing,load_direction,  rx_deg, ry_deg, rz_deg)
    
    return out
