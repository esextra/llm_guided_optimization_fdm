#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import uuid
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List, Any
import time
import numpy as np
import GPyOpt  
import argparse


                                             
from pathlib import Path
import sys

_here = Path(__file__).resolve().parent                 
_pkg_root = _here / "print_configuration_evaluator" / "v2" 

sys.path.insert(0, str(_pkg_root))

_expected_pkg = _pkg_root / "print_quality"
if not _expected_pkg.exists():
    raise RuntimeError(f"Expected package folder not found: {_expected_pkg}")

                         


from print_quality.io.config_parser import parse_config_ini
from print_quality.pipeline.metrics.helpers import _get
import trimesh
from bo_logging import BOLogger


_LOGGER, _RUN_ID = None, None
    
MODEL_STL, PROFILE_INI, OUTPUT_DIR, CONFIG_SAVE_DIR, GCODE_OUT_DIR, MESH_OUT_DIR  = None, None, None, None, None, None


OBJECTIVE_COMBINE: List[Tuple[str, float]] = [("time_normal", 0.1), ("total_filament_cost", 0.1), ("quality_J", 0.8)]
USE_COMBINED: bool = len(OBJECTIVE_COMBINE) > 0


                     
INITIAL_DESIGN: int = 32
INITIAL_DESIGN_TYPE = 'sobol' 

FAIL_PENALTY: float = float(os.environ.get("FAIL_PENALTY", "1e12"))


USE_DOMAIN_BASELINE_SCALING: bool = True    
WARMUP_QUANTILE: float = 0.50               
BLEND_GEOMEAN: bool = True                  


TIME_REF: Optional[float] = None            
COST_REF: Optional[float] = None            
QUALITY_REF: Optional[float] = None         
REFS_FROZEN: bool = False

def _squash_ratio(x: float) -> float:

    if not np.isfinite(x) or x <= 0.0:
        return 0.0

    return float(x / (1.0 + x))


def translate_stl_to_xy_center(path: str, target_x: float = 125.0, target_y: float = 105.0,
                               use_oriented_bbox: bool = False) -> None:

    m = trimesh.load(path, force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError("Loaded object is not a triangular mesh.")

    if use_oriented_bbox:

        obb = m.bounding_box_oriented  
        center_world = obb.centroid  
        cx, cy = float(center_world[0]), float(center_world[1])
    else:

        bmin, bmax = m.bounds
        cx = 0.5 * (bmin[0] + bmax[0])
        cy = 0.5 * (bmin[1] + bmax[1])

    t = np.array([target_x - cx, target_y - cy, 0.0], dtype=float)
    m.apply_translation(t)
    m.export(path)

FLATPAK_APP = "com.prusa3d.PrusaSlicer"
CLI_BINARY  = "prusa-slicer"


@dataclass(frozen=True)
class SlicingParams:
    layer_height: float           
    infill_density: float                    
    infill_pattern: str                                            
    perimeters: int
    max_print_speed: float
    filament_max_volumetric_speed: float          
    support_material: int                      
    support_material_threshold: float                  
    first_layer_height: float                 
    first_layer_extrusion_width: float        
    elefant_foot_compensation: float               
    seam_position: str                                           
    rotate_y: int                                                                
    rotate_x: int                                                                
    rotate_z: int                                                              
    brim_width: float
    
    
def _run_slicer(
    stl_path: str,
    gcode_path: str,
    profile_ini: str,
    params: SlicingParams,
    config_save_path: Optional[str] = None,
    mesh_export_path: Optional[str] = None,
) -> subprocess.CompletedProcess:

    cmd = [
        "flatpak", "run",
        "--command=" + CLI_BINARY,
        FLATPAK_APP,
        "-g", stl_path,                                
        "--load", profile_ini,                             
        "--layer-height", str(params.layer_height),
        "--fill-density", f"{params.infill_density}%",
        "--fill-pattern", params.infill_pattern,
        "--perimeters", str(params.perimeters),
        "--max-print-speed", str(params.max_print_speed),
        "--output", gcode_path,
        "--filament-max-volumetric-speed", str(params.filament_max_volumetric_speed),
        "--first-layer-height", str(params.first_layer_height),
        "--first-layer-extrusion-width", str(params.first_layer_extrusion_width),
        "--elefant-foot-compensation", str(params.elefant_foot_compensation),
        "--seam-position", params.seam_position,
        "--rotate-y", str(params.rotate_y),
        "--rotate-x", str(params.rotate_x),
        "--rotate", str(params.rotate_z),    
        "--brim-width", str(params.brim_width),
        "--loglevel", str(2),
        "--center", str(125) + ',' + str(105),                                             

    ]

    if params.support_material:

        cmd += [
            "--support-material",
            "--support-material-auto",
            "--support-material-threshold", str(params.support_material_threshold),
        ]

    if config_save_path:
        cmd += ["--save", config_save_path]

    cmdstl = cmd.copy()
    cmdstl[4] = "--export-stl"
    cmdstl[19] = mesh_export_path
    export_process = subprocess.run(cmdstl, capture_output=True, text=True)

    if export_process.returncode == 0:
        translate_stl_to_xy_center(mesh_export_path, 125, 105, False)
    
               
                  
    slice_process = subprocess.run(cmd, capture_output=True, text=True)
    
    io_dict = {
        "stl_export_stdout": export_process.stdout or "",
        "stl_export_stderr": export_process.stderr or "",
        "slice_stdout": slice_process.stdout or "",
        "slice_stderr": slice_process.stderr or "",
    }
    return slice_process, io_dict

_PATTERNS = {
    'filament_mm': re.compile(r";\s*filament used \[mm\] = ([\d\.]+)"),
    'filament_cm3': re.compile(r";\s*filament used \[cm3\] = ([\d\.]+)"),
    'filament_g': re.compile(r";\s*filament used \[g\] = ([\d\.]+)"),
    'filament_cost': re.compile(r";\s*filament cost = ([\d\.]+)"),
    'total_filament_g': re.compile(r";\s*total filament used \[g\] = ([\d\.]+)"),
    'total_filament_cost': re.compile(r";\s*total filament cost = ([\d\.]+)"),
    'total_filament_wipe_g': re.compile(r";\s*total filament used for wipe tower \[g\] = ([\d\.]+)"),
    'time_normal': re.compile(r";\s*estimated printing time \(normal mode\) = ([\dhms ]+)"),
    'time_silent': re.compile(r";\s*estimated printing time \(silent mode\) = ([\dhms ]+)"),
    'first_layer_time_normal': re.compile(r";\s*estimated first layer printing time \(normal mode\) = ([\dhms ]+)"),
    'first_layer_time_silent': re.compile(r";\s*estimated first layer printing time \(silent mode\) = ([\dhms ]+)"),
}

def parse_gcode_metrics(gcode_path: str) -> Dict[str, float]:

    results: Dict[str, Optional[str]] = {k: None for k in _PATTERNS}
    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, pat in _PATTERNS.items():
                if results[key] is None:
                    m = pat.search(line)
                    if m:
                        results[key] = m.group(1).strip()


    numeric_keys = [
        'filament_mm', 'filament_cm3', 'filament_g', 'filament_cost',
        'total_filament_g', 'total_filament_cost', 'total_filament_wipe_g'
    ]
    out: Dict[str, float] = {}
    for k, v in results.items():
        if k in numeric_keys and v is not None:
            out[k] = float(v)
        else:
            out[k] = v  
    return out

def _time_to_seconds(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    total = 0
    m = re.search(r"(\d+)h", s)
    if m: total += int(m.group(1)) * 3600
    m = re.search(r"(\d+)m", s)
    if m: total += int(m.group(1)) * 60
    m = re.search(r"(\d+)s", s)
    if m: total += int(m.group(1))
    return float(total) if total > 0 else None


def objective_from_metrics_combined(metrics: Dict[str, float], pairs: List[Tuple[str, float]]) -> float:

    vq = metrics.get("quality_J")
    if vq is not None:
        try:
            vqf = float(vq)
            if (not np.isfinite(vqf)) or (vqf >= FAIL_PENALTY):
                return float(FAIL_PENALTY)
        except Exception:
            return float(FAIL_PENALTY)
    total = 0.0

    for kind, w in pairs:
        if kind in ("time_normal", "time_silent", "first_layer_time_normal", "first_layer_time_silent"):
            seconds = _time_to_seconds(metrics.get(kind))                          
            if seconds is None:
                raise ValueError(f"Could not parse time for objective '{kind}' from metrics: {metrics.get(kind)}")
            if USE_DOMAIN_BASELINE_SCALING and REFS_FROZEN and (TIME_REF is not None):
                val = _squash_ratio(seconds / float(TIME_REF))
            else:
                val = seconds               
        elif kind == "filament_g":
            g = metrics.get("filament_g")
            if g is None:
                raise ValueError("'filament_g' not present in metrics")
                                                                   
            val = float(g)            
        elif kind in ("filament_cost", "total_filament_cost"):
            c = metrics.get(kind)
            if c is None:
                raise ValueError(f"'{kind}' not present in metrics")
            if USE_DOMAIN_BASELINE_SCALING and REFS_FROZEN and (COST_REF is not None):
                val = _squash_ratio(float(c) / float(COST_REF))
            else:
                val = float(c)               
        elif kind == "quality_J":
            v = metrics.get("quality_J")
            if v is None:
                raise ValueError("'quality_J' not present in metrics")
            q_loss = float(v)
            val = q_loss         
        else:
            v = metrics.get(kind)
            if v is None:
                raise ValueError(f"Metric '{kind}' not present in metrics")
            if isinstance(v, str):
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(f"Metric '{kind}' is non-numeric ('{v}') and not a recognized time field") from e
            val = float(v)
        total += w * val
    return float(total)


def _placeholder_quality_final_via_new_pipeline(gcode_path: str, ini_path: Optional[str], mesh_path: str, io_dict) -> float:

    q_final, meta_info = float("inf"), {}
    return q_final, meta_info
    
def _effective_param_dict_for_cache(params: SlicingParams) -> Dict[str, Any]:

    def r(x, nd=6):  
        return round(float(x), nd)

    d: Dict[str, Any] = {
        "layer_height": r(params.layer_height),
        "infill_density": r(params.infill_density),
        "infill_pattern": params.infill_pattern,
        "perimeters": int(params.perimeters),
        "max_print_speed": r(params.max_print_speed),
        "filament_max_volumetric_speed": r(params.filament_max_volumetric_speed),
        "first_layer_height": r(params.first_layer_height),
        "first_layer_extrusion_width": r(params.first_layer_extrusion_width),
        "elefant_foot_compensation": r(params.elefant_foot_compensation),
        "seam_position": params.seam_position,
        "rotate_y": int(params.rotate_y),
        "rotate_x": int(params.rotate_x),
        "rotate_z": int(params.rotate_z),
        "brim_width": r(params.brim_width),
        "support_material": bool(params.support_material),
    }
    if params.support_material:
        d["support_material_auto"] = True
        d["support_material_threshold"] = r(params.support_material_threshold)
    return d

def _cache_key(params: SlicingParams) -> str:

    eff = _effective_param_dict_for_cache(params)
    return json.dumps(eff, sort_keys=True)


_MEM_CACHE: Dict[str, Tuple[str, str, str, Dict[str, float]]] = {}                                              

def _gcode_exists_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def evaluate_params(params: "SlicingParams") -> float:


    def _mesh_path(mesh_export_path: str | None) -> str:

        return mesh_export_path if (mesh_export_path and os.path.isfile(mesh_export_path)) else MODEL_STL

    def _attach_quality(metrics: Dict, gcode_path: str, config_save_path: str, mesh_export_path: str , io_dict) -> None:

        try:
            q_final, q_info = _placeholder_quality_final_via_new_pipeline(
                gcode_path, config_save_path, _mesh_path(mesh_export_path), io_dict
            )
        except Exception as e:
                                                                     
            q_final, q_info = float("inf"), {}

        if not np.isfinite(q_final):
            q_final = FAIL_PENALTY

        metrics["quality_J"] = float(q_final)
        metrics["quality_info"] = q_info

    key = _cache_key(params)
    if key in _MEM_CACHE:
        gcode_path, config_save_path, mesh_export_path, metrics = _MEM_CACHE[key]
    else:
        uid = uuid.uuid4().hex[:8]
        gcode_path = os.path.join(GCODE_OUT_DIR, f"slice_{uid}.gcode")
        config_save_path = os.path.join(CONFIG_SAVE_DIR, f"config_{uid}.ini")
        mesh_export_path = os.path.join(MESH_OUT_DIR, f"mesh_{uid}.stl")

        proc, io_dict = _run_slicer(
            MODEL_STL, gcode_path, PROFILE_INI, params,
            config_save_path=config_save_path,
            mesh_export_path=mesh_export_path
        )
        if proc.returncode != 0:
                                                                                         
            return float("inf")

        if not _gcode_exists_nonempty(gcode_path):
                                                                      
            return float("inf")

        try:
            metrics = parse_gcode_metrics(gcode_path)
        except Exception:
                                                           
            return float("inf")

                                                          
        _attach_quality(metrics, gcode_path, config_save_path, mesh_export_path, io_dict)

        _MEM_CACHE[key] = (gcode_path, config_save_path, mesh_export_path, metrics)


    if "quality_J" not in metrics:
        try:
            _attach_quality(metrics, gcode_path, config_save_path, mesh_export_path, io_dict)
            _MEM_CACHE[key] = (gcode_path, config_save_path, mesh_export_path, metrics)
        except Exception as e:

                                                                                
            metrics["quality_J"] = FAIL_PENALTY

    try:
        if USE_COMBINED and len(OBJECTIVE_COMBINE) > 0:
            return objective_from_metrics_combined(metrics, OBJECTIVE_COMBINE)
        else:
                                                                   
            return float("inf")
    except Exception:
                                                                       
        return float("inf")


INFILL_PATTERNS: List[str] = [
    "concentric",          
    "rectilinear",         
    "grid",                
    "cubic",               
    "honeycomb",           
    "gyroid",              
]

SEAM_POSITIONS: List[str] = ["nearest", "aligned", "rear", "random"]


def _x_to_params(x: np.ndarray) -> SlicingParams:
    x = np.asarray(x).ravel()

    layer_height                  = float(x[0])
    infill_density               = float(x[1])
    pat_idx                      = int(round(x[2]))
    perimeters                   = int(round(x[3]))
    max_print_speed              = float(x[4])
    filament_max_volumetric_speed= float(x[5])
    support_material             = int(round(x[6]))       
    support_material_threshold       = float(x[7])
    first_layer_height           = float(x[8])
    first_layer_extrusion_width  = float(x[9])
    elefant_foot_compensation         = float(x[10])
    seam_idx                     = int(round(x[11]))
    rotate_y                      = int(round(x[12]))
    rotate_x                      = int(round(x[13]))
    rotate_z                    = int(round(x[14]))
    
    brim_width                 = float(x[15])

                               
    pat_idx  = max(0, min(pat_idx,  len(INFILL_PATTERNS) - 1))
    seam_idx = max(0, min(seam_idx, len(SEAM_POSITIONS) - 1))
    perimeters = max(2, min(perimeters, 6))

    rotate_y = max(0, min(360, int(5 * round(rotate_y / 5))))
    rotate_x = max(0, min(360, int(5 * round(rotate_x / 5))))
    rotate_z = max(0, min(360, int(5 * round(rotate_z / 5))))

    return SlicingParams(
        layer_height=layer_height,
        infill_density=infill_density,
        infill_pattern=INFILL_PATTERNS[pat_idx],
        perimeters=perimeters,
        max_print_speed=max_print_speed,
        filament_max_volumetric_speed=filament_max_volumetric_speed,
        support_material=support_material,
        support_material_threshold=support_material_threshold,
        first_layer_height=first_layer_height,
        first_layer_extrusion_width=first_layer_extrusion_width,
        elefant_foot_compensation=elefant_foot_compensation,
        seam_position=SEAM_POSITIONS[seam_idx],
        rotate_y =rotate_y,
        rotate_x =rotate_x,
        rotate_z = rotate_z,
        brim_width = brim_width,
    )


def _bo_objective(X: np.ndarray) -> np.ndarray:

    n = X.shape[0]
    vals = np.empty((n, 1), dtype=float)
    for i in range(n):
        params = _x_to_params(X[i])
        y = float(evaluate_params(params))
        if not np.isfinite(y):
            y = FAIL_PENALTY
        vals[i, 0] = y
    return vals

def _slice_profile_defaults_to_metrics() -> Optional[Dict[str, float]]:

    try:
        uid = uuid.uuid4().hex[:8]
        gcode_path = os.path.join(GCODE_OUT_DIR, f"baseline_{uid}.gcode")
        cmd = [
            "flatpak", "run",
            "--command=" + CLI_BINARY,
            FLATPAK_APP,
            "-g", MODEL_STL,
            "--load", PROFILE_INI,
            "--output", gcode_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        io_dict = {
            "slice_stdout": proc.stdout or "",
            "slice_stderr": proc.stderr or "",
        }
        
        if proc.returncode != 0 or not _gcode_exists_nonempty(gcode_path):
                                                                                        
            return None

        metrics = parse_gcode_metrics(gcode_path)


        try:
            q_final, q_info = _placeholder_quality_final_via_new_pipeline(gcode_path, PROFILE_INI, MODEL_STL, io_dict )
            if np.isfinite(q_final):
                metrics["quality_J"] = float(q_final)
                metrics["quality_info"] = q_info
        except Exception as e:
            print(f"[baseline] Quality computation failed: {e}")

        return metrics
    except Exception as e:
                                                               
        return None

def _metric_time_seconds(metrics: Dict[str, float]) -> Optional[float]:

    for k in ("time_normal", "time_silent", "first_layer_time_normal", "first_layer_time_silent"):
        v = metrics.get(k)
        if v:
            sec = _time_to_seconds(v)                          
            if sec is not None:
                return sec
    return None

def _metric_cost_dollars(metrics: Dict[str, float]) -> Optional[float]:

    for k in ("total_filament_cost", "filament_cost"):
        v = metrics.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None

def _gather_initial_metrics_from_cache() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    times, costs, quals = [], [], []
    for (_gcode, _cfg, _mesh, m) in _MEM_CACHE.values():
        q = m.get("quality_J")
        try:
            qf = float(q) if q is not None else float("nan")
        except Exception:
            qf = float("nan")
 
        sec = _metric_time_seconds(m)
        if sec is not None and np.isfinite(sec):
            times.append(float(sec))
 
        c = _metric_cost_dollars(m)
        if c is not None and np.isfinite(c):
            costs.append(float(c))
 
                                                 
        if np.isfinite(qf) and (0.0 <= qf <= 1.0):
            quals.append(float(qf))
    return np.array(times), np.array(costs), np.array(quals)

def _compute_and_freeze_references():

    global TIME_REF, COST_REF, QUALITY_REF, REFS_FROZEN


    base_metrics = _slice_profile_defaults_to_metrics()
    T_base = _metric_time_seconds(base_metrics) if base_metrics else None
    C_base = _metric_cost_dollars(base_metrics) if base_metrics else None


    T_data, C_data, _Q_data = _gather_initial_metrics_from_cache()
    T_med = float(np.quantile(T_data, WARMUP_QUANTILE)) if T_data.size >= 3 else None
    C_med = float(np.quantile(C_data, WARMUP_QUANTILE)) if C_data.size >= 3 else None


    def _blend(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a and b:
            return float(np.sqrt(a * b)) if BLEND_GEOMEAN else 0.5 * (a + b)
        return a or b  

    TIME_REF = _blend(T_base, T_med)
    COST_REF = _blend(C_base, C_med)


    TIME_REF = max(1e-6, float(TIME_REF))
    COST_REF = max(1e-6, float(COST_REF))

    REFS_FROZEN = True
    print(f"[scaling] TIME_REF={TIME_REF:.3f} s, COST_REF={COST_REF:.3f} $ "
          f"(baseline T={T_base}, C={C_base}; warm-up medians T={T_med}, C={C_med})")
    
def _num(val, default=None):

    if val is None:
        return default

    try:
        import numpy as _np                                                   
        bool_types = (bool, _np.bool_)
    except Exception:
        bool_types = (bool,)
    if isinstance(val, bool_types):
        return 1.0 if bool(val) else 0.0
    try:
        s = str(val).strip().lower()
        if s in {"true", "yes", "on"}:
            return 1.0
        if s in {"false", "no", "off"}:
            return 0.0
        if s.endswith("%"):
            s = s[:-1].strip()
        return float(s)
    except Exception:
        return default

def _cfg_map(cfg):

    if isinstance(cfg, dict):
        return cfg
    d = getattr(cfg, "__dict__", {}) or {}
    return d

def _get_num_from_cfg(cfg, keys, default=None):

    m = _cfg_map(cfg)
    for k in keys:
        v = _get(m, k, default=None)
        v_num = _num(v, None)
        if v_num is not None:
            return v_num
    return default


def _profile_aware_bounds(cfg) -> Dict[str, Tuple[float, float]]:

    nozzle = _get_num_from_cfg(cfg, ["nozzle_diameter"], default=0.4)

    lh0   = _get_num_from_cfg(cfg, ["layer_height"], default=0.15)
    flh0  = _get_num_from_cfg(cfg, ["first_layer_height", "layer_height"], default=0.20)
    flew0 = _get_num_from_cfg(cfg, [
        "first_layer_extrusion_width", "perimeter_extrusion_width", "infill_extrusion_width",
        "external_perimeter_extrusion_width", "solid_infill_extrusion_width",
        "top_infill_extrusion_width", "nozzle_diameter",
    ], default=nozzle)

    ild0  = _get_num_from_cfg(cfg, ["fill_density"], default=15.0)
    per0  = int(round(_get_num_from_cfg(cfg, ["perimeters"], default=2)))
    mps0  = _get_num_from_cfg(cfg, ["max_print_speed"], default=80.0)
    mvs0  = _get_num_from_cfg(cfg, ["filament_max_volumetric_speed"], default=11.5)

    saa0  = _get_num_from_cfg(cfg, ["support_material_threshold"], default=50.0)

    saa_min = max(35.0, saa0 - 15.0)
    saa_max = min(65.0, saa0 + 15.0)
    if saa_max - saa_min < 10.0:
        mid = 0.5 * (saa_min + saa_max)
        saa_min, saa_max = max(35.0, mid - 5.0), min(65.0, mid + 5.0)

    xyc0  = _get_num_from_cfg(cfg, ["elefant_foot_compensation"], default=0.0)
    xy_min, xy_max = 0.00, (0.30 if xyc0 <= 0.30 else min(0.30, max(0.15, 1.5 * xyc0)))

    lh_min = max(0.08, 0.6 * lh0, 0.25 * nozzle)
    lh_max = min(0.40, 1.8 * lh0, 0.80 * nozzle)
    min_lh = _get_num_from_cfg(cfg, ["min_layer_height"], default=None)
    max_lh = _get_num_from_cfg(cfg, ["max_layer_height"], default=None)
    if min_lh is not None: lh_min = max(lh_min, float(min_lh))
    if max_lh is not None: lh_max = min(lh_max, float(max_lh))

    flh_min = max(0.12, 0.8 * flh0, 0.20 * nozzle)
    flh_max = min(0.40, 1.6 * flh0, 0.80 * nozzle)

    flew_min = max(0.30, 0.8 * flew0, 0.80 * nozzle)
    flew_max = min(0.80, 1.5 * flew0, 1.70 * nozzle)

    ild_min = max(5.0,  ild0 - 10.0)
    ild_max = min(80.0, ild0 + 35.0)

    mps_min = max(30.0, 0.3 * mps0)
    mps_max = min(200.0, 1.0 * mps0)

    mvs_min = max(3.0,  0.5 * mvs0)
    mvs_max = min(30.0, 1.5 * mvs0)

    per_min, per_max = 2, max(5, min(6, per0 + 2))

    bw_min, bw_max = 0.0, 8.0

    return {
        "layer_height":                  (lh_min, lh_max),
        "infill_density":                (ild_min, ild_max),
        "perimeters":                    (per_min, per_max),
        "max_print_speed":               (mps_min, mps_max),
        "filament_max_volumetric_speed": (mvs_min, mvs_max),
        "support_material_threshold":    (saa_min, saa_max),
        "first_layer_height":            (flh_min, flh_max),
        "first_layer_extrusion_width":   (flew_min, flew_max),
        "elefant_foot_compensation":     (xy_min, xy_max),
        "brim_width":                    (bw_min, bw_max),
    }

def build_domain_from_profile(ini_path: str) -> List[Dict[str, Any]]:
    cfg = parse_config_ini(ini_path)
    B   = _profile_aware_bounds(cfg)
    return [
        {"name": "layer_height",      "type": "continuous", "domain": B["layer_height"]},
        {"name": "infill_density",    "type": "continuous", "domain": B["infill_density"]},
        {"name": "pattern_idx",       "type": "discrete",   "domain": tuple(range(len(INFILL_PATTERNS)))},
        {"name": "perimeters",        "type": "discrete",   "domain": tuple(range(int(B["perimeters"][0]),
                                                                                   int(B["perimeters"][1]) + 1))},
        {"name": "max_print_speed",   "type": "continuous", "domain": B["max_print_speed"]},
        {"name": "filament_max_volumetric_speed", "type": "continuous", "domain": B["filament_max_volumetric_speed"]},
        {"name": "support_material",  "type": "discrete",   "domain": (0, 1)},
        {"name": "support_material_threshold", "type": "continuous", "domain": B["support_material_threshold"]},
        {"name": "first_layer_height","type": "continuous", "domain": B["first_layer_height"]},
        {"name": "first_layer_extrusion_width","type": "continuous", "domain": B["first_layer_extrusion_width"]},
        {"name": "elefant_foot_compensation","type": "continuous", "domain": B["elefant_foot_compensation"]},
        {"name": "seam_idx",          "type": "discrete",   "domain": tuple(range(len(SEAM_POSITIONS)))},
        {"name": "rotate_y",          "type": "discrete",   "domain": tuple(range(0, 356, 15))},
        {"name": "rotate_x",          "type": "discrete",   "domain": tuple(range(0, 356, 15))},
        {"name": "rotate_z",          "type": "discrete",   "domain": tuple(range(0, 356, 15))},
        {"name": "brim_width",        "type": "continuous", "domain": B["brim_width"]},
    ]

def _params_to_dict(p: SlicingParams) -> Dict[str, Any]:

    d: Dict[str, Any] = asdict(p)
    try:
        d["infill_pattern_id"] = INFILL_PATTERNS.index(p.infill_pattern)
    except ValueError:
        d["infill_pattern_id"] = None
    try:
        d["seam_idx"] = SEAM_POSITIONS.index(p.seam_position)
    except ValueError:
        d["seam_idx"] = None

    d["seam_position_str"] = d.pop("seam_position")
    d["infill_pattern_str"] = d.pop("infill_pattern")
    return d


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2021, help="RNG seed for initial design and BO")

    p.add_argument(
        "--model_stl",
        type=str,
        default="./test_data/model/my_dovetail.stl",
        help="Path to input STL model.",
    )
    p.add_argument(
        "--profile_ini",
        type=str,
        default="./test_data/config/my_dovetail_0.2mm_PLA_MK3S_17m.ini",
        help="Path to printer/profile INI.",
    )
    p.add_argument(
        "--output_root_dir",
        type=str,
        default="./logs_my_dovetail_context_Bp7_unfreeze_at4/basic_heuristics/",
        help="Root directory where run-specific outputs are created.",
    )
    p.add_argument(
        "--refs_store",
        type=str,
        default="./logs_my_dovetail_context_Bp7_unfreeze_at4/refs_store.json",
        help="Path to refs_store.json produced by compute_refs_persist.py",
    )

    return p.parse_args()
        
def _persist_refs(store_path: str, n_warmup: int) -> None:

    if not (REFS_FROZEN and (TIME_REF is not None) and (COST_REF is not None)):
        print("[refs] Not persisting: refs are not frozen.")
        return


    T_data, C_data, _Q_data = _gather_initial_metrics_from_cache()
    T_med = float(np.quantile(T_data, WARMUP_QUANTILE)) if T_data.size >= 3 else None
    C_med = float(np.quantile(C_data, WARMUP_QUANTILE)) if C_data.size >= 3 else None


    base = _slice_profile_defaults_to_metrics()
    T_base = _metric_time_seconds(base) if base else None
    C_base = _metric_cost_dollars(base) if base else None

    key = f"{str(Path(MODEL_STL).resolve())} :: {str(Path(PROFILE_INI).resolve())}"
    rec = {
        "model_stl": str(Path(MODEL_STL).resolve()),
        "profile_ini": str(Path(PROFILE_INI).resolve()),
        "time_ref_seconds": float(TIME_REF),
        "cost_ref_dollars": float(COST_REF),
        "time_median_seconds": (float(T_med) if T_med is not None else None),
        "cost_median_dollars": (float(C_med) if C_med is not None else None),
        "time_baseline_seconds": (float(T_base) if T_base is not None else None),
        "cost_baseline_dollars": (float(C_base) if C_base is not None else None),
        "warmup_quantile": float(WARMUP_QUANTILE),
        "blend_geomean": bool(BLEND_GEOMEAN),
        "n_sobol": int(n_warmup),
    }


    sp = Path(store_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(sp.read_text()) if sp.exists() else {}
    store[key] = rec
    sp.write_text(json.dumps(store, indent=2))
                                                                   


def main():
    
    global _LOGGER, _RUN_ID
    
    global MODEL_STL, PROFILE_INI, OUTPUT_DIR, CONFIG_SAVE_DIR, GCODE_OUT_DIR, MESH_OUT_DIR           
    
                                           
    _RUN_ID = uuid.uuid4().hex
    args = _parse_args()
    seed = args.seed


    MODEL_STL = args.model_stl         
    PROFILE_INI = args.profile_ini


    OUTPUT_DIR  = os.path.join(args.output_root_dir, str(_RUN_ID)) + "/"   
                      
    CONFIG_SAVE_DIR = os.path.join(OUTPUT_DIR, "configs")   
    GCODE_OUT_DIR = os.path.join(OUTPUT_DIR, "gcodes")     
    MESH_OUT_DIR = os.path.join(OUTPUT_DIR, "meshes")
    
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    os.makedirs(GCODE_OUT_DIR, exist_ok=True)
    os.makedirs(MESH_OUT_DIR, exist_ok=True)    

    
    domain = build_domain_from_profile(PROFILE_INI)
    np.random.seed(seed)
        
    
    _LOGGER = BOLogger(
        csv_path=os.path.join(OUTPUT_DIR, f"bo_no_optimization_{_RUN_ID}.csv"),
        run_id=_RUN_ID,
        optimizer="no_optimization",
        seed=seed,
    )


    bo_warm = GPyOpt.methods.BayesianOptimization(
        f=_bo_objective,
        domain=domain,
        initial_design_numdata=INITIAL_DESIGN,
        initial_design_type=INITIAL_DESIGN_TYPE,
        acquisition_type='EI',
        exact_feval=False,
        noise_var=1e-6,
        normalize_Y=True,
        de_duplication=True
    )
    
    bo_warm.run_optimization(max_iter=1, eps=0)

    if USE_DOMAIN_BASELINE_SCALING and not REFS_FROZEN:
        _compute_and_freeze_references()

    _persist_refs(args.refs_store, n_warmup=INITIAL_DESIGN)

    X_all = np.asarray(bo_warm.X)
    X_warm = np.asarray(X_all[:INITIAL_DESIGN], dtype=float)
    for k, xw in enumerate(X_warm):
        params_w = _x_to_params(xw)
        t0 = time.time()
        y_w = float(evaluate_params(params_w))  
        t1 = time.time()
        if not np.isfinite(y_w):
            y_w = FAIL_PENALTY
    
        key_w = _cache_key(params_w)
        gcode_path, cfg_path, mesh_path, metrics = _MEM_CACHE.get(key_w, (None, None, None, {}))
        qJ = metrics.get("quality_J") if isinstance(metrics, dict) else None
        qInfo = metrics.get("quality_info") if isinstance(metrics, dict) else None
    
        def _sec_from_metrics(m):
            return _time_to_seconds(m.get("time_normal"))
        time_s = _sec_from_metrics(metrics) if isinstance(metrics, dict) else None
    
        cost = None
        if isinstance(metrics, dict):
            cost = metrics.get("total_filament_cost")
            if cost is None:
                cost = metrics.get("filament_cost")
    
        if _LOGGER is not None:
            _LOGGER.log_eval(
                iter_idx=-(len(X_warm) - k),
                objective=y_w,
                time_s=time_s,
                cost=cost,
                quality_J=qJ,
                quality_info=qInfo,
                slicer_params=_params_to_dict(params_w),          
                gcode_path=gcode_path,
                config_path=cfg_path,
                mesh_path=mesh_path,
                suggest_secs=None,
                eval_secs=(t1 - t0),
                total_secs=(t1 - t0),
                use_domain_baseline_scaling=USE_DOMAIN_BASELINE_SCALING,
                time_ref=TIME_REF,
                cost_ref=COST_REF,
                quality_ref=QUALITY_REF,
                use_combined=bool(USE_COMBINED and len(OBJECTIVE_COMBINE) > 0),
                objective_kind=("combined" if USE_COMBINED else "not specified"),
                objective_combine_pairs=(OBJECTIVE_COMBINE if USE_COMBINED else None),
            )


        
if __name__ == "__main__":
    main()


