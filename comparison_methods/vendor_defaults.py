from pathlib import Path                                                                  

import os
import re
import uuid
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List, Any, Set
from hints_categorical_to_idx import normalize_hints_categoricals_to_indices
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import trimesh
import math
import argparse
import GPy
import GPyOpt                                
import csv
from copy import deepcopy

import sys

_here = Path(__file__).resolve().parent                                      
_pkg_root = _here / "print_configuration_evaluator" / "v2"                                    

                                                                  
sys.path.insert(0, str(_pkg_root))

                                                              
_expected_pkg = _pkg_root / "print_quality"
if not _expected_pkg.exists():
    raise RuntimeError(f"Expected package folder not found: {_expected_pkg}")

from print_quality.io.mesh_loader import load_mesh
from print_quality.io.gcode_parser import parse_gcode
from print_quality.io.config_parser import parse_config_ini
from print_quality.pipeline.mesh_precompute import build_mesh_precompute, align_mesh_xy_to_raster
from print_quality.pipeline.build_rasters import build_layer_rasters
from print_quality.utils.config_resolution import channel_default_width_mm
from print_quality.data.types import Config, PrintJob                       
from print_quality.pipeline.metrics.aggregator import aggregate_for_bo
from print_quality.pipeline.metrics.helpers import _get
import trimesh
from bo_logging import BOLogger

MODEL_STL, PROFILE_INI, OUTPUT_DIR, CONFIG_SAVE_DIR, GCODE_OUT_DIR, MESH_OUT_DIR  = None, None, None, None, None, None

LOAD_BEARING = False
LOAD_DIRECTION: Optional[str] = None

                                                         
OBJECTIVE_COMBINE: List[Tuple[str, float]] = [("time_normal", 0.1), ("total_filament_cost", 0.1), ("quality_J", 0.8)]
USE_COMBINED: bool = len(OBJECTIVE_COMBINE) > 0

                                                                                  
FAIL_PENALTY: float = float(os.environ.get("FAIL_PENALTY", "1e12"))

                                                                                 
USE_DOMAIN_BASELINE_SCALING: bool = True
TIME_REF: Optional[float] = None
COST_REF: Optional[float] = None
QUALITY_REF: Optional[float] = None
REFS_FROZEN: bool = False

def _squash_ratio(x: float) -> float:
                                                                 
    if not np.isfinite(x) or x <= 0.0:
        return 0.0
    return float(x / (1.0 + x))

def _refs_key(model_stl: str, profile_ini: str) -> str:
    return f"{str(Path(model_stl).resolve())} :: {str(Path(profile_ini).resolve())}"

def _try_load_persisted_refs(store_path: str) -> bool:
                                                                                  
    global TIME_REF, COST_REF, REFS_FROZEN
    try:
        sp = Path(store_path)
        if not sp.exists():
            return False
        store = json.loads(sp.read_text())
        rec = store.get(_refs_key(MODEL_STL, PROFILE_INI))
        if not rec:
            return False
        TIME_REF = max(1e-6, float(rec["time_ref_seconds"]))
        COST_REF = max(1e-6, float(rec["cost_ref_dollars"]))
        REFS_FROZEN = True
        return True
    except Exception as e:
        print(f"[scaling] Failed to load persisted refs: {e}", file=sys.stderr)
        return False

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

_MESH_PATH_CACHED: Optional[Tuple[str, Tuple[int,int], Tuple[float,float], float]] = None                                                           
_MESH_DATA_CACHED = None
_MESH_PRECOMP_CACHED = None

FLATPAK_APP = "com.prusa3d.PrusaSlicer"
CLI_BINARY  = "prusa-slicer"


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

def _get_mesh_and_precomp(mesh_path: str, layer_rasters: Dict[int, Dict[str, Any]]):
                                                                      
    global _MESH_PATH_CACHED, _MESH_DATA_CACHED, _MESH_PRECOMP_CACHED
                                                                        
    shapes  = {(v['V_part'].mask.shape)   for v in layer_rasters.values()}
    origins = {(v['V_part'].origin_xy)    for v in layer_rasters.values()}
    pxs     = {(v['V_part'].pixel_xy)     for v in layer_rasters.values()}
    assert len(shapes)==1 and len(origins)==1 and len(pxs)==1, "Non-uniform grids detected"
    grid_shape  = next(iter(shapes))
    grid_origin = next(iter(origins))
    grid_px     = float(next(iter(pxs)))
    cache_key   = (os.path.abspath(mesh_path), grid_shape, grid_origin, grid_px)

    if _MESH_PRECOMP_CACHED is None or _MESH_PATH_CACHED != cache_key:
        mesh_data = load_mesh(mesh_path)
        mp = build_mesh_precompute(mesh_data)
        _MESH_PATH_CACHED = cache_key
        _MESH_DATA_CACHED = mesh_data
        _MESH_PRECOMP_CACHED = mp
    return _MESH_DATA_CACHED, _MESH_PRECOMP_CACHED


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

def _gcode_exists_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False

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



def _as_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}
    except Exception:
        return {}


def _close_if_perimeter_str(feature_str: str, poly):
    PERIM_FEATURES = {"P_INNER", "P_OUTER", "P_OVERHANG"}
                                                                      
    if poly is None:
        return poly
    try:
        n = len(poly)
    except Exception:
        return poly
    if n < 2 or feature_str not in PERIM_FEATURES:
        return poly

                                            
    is_np = isinstance(poly, np.ndarray)

    if is_np:
                                      
        if poly.shape[0] >= 2:
                                    
            if not (np.isclose(poly[0,0], poly[-1,0]) and np.isclose(poly[0,1], poly[-1,1])):
                                                                            
                poly = np.vstack([poly, poly[0]])
        return poly
    else:
                                        
        x0, y0 = float(poly[0][0]), float(poly[0][1])
        x1, y1 = float(poly[-1][0]), float(poly[-1][1])
        if (x0 != x1) or (y0 != y1):
            return list(poly) + [poly[0]]
        return poly


def _to_raster_segments(print_job: PrintJob | Dict[str, Any],
                        cfg: Optional[Config | Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    segs_in = getattr(print_job, "segments", None) or print_job.get("segments", [])
    out: List[Dict[str, Any]] = []
    for s in segs_in:
        def get(name: str, default=None):
            if isinstance(s, dict):
                return s.get(name, default)
            return getattr(s, name, default)
        layer_index = int(get("layer_index", 0))
        poly = get("polyline_mm", None)
        if poly is None:
            poly = get("polyline", None)
        ch = get("channel", None)
        feat = ch if (ch is not None) else (get("feature", None) or get("feature_type", None))
        feature_str = feat.name if hasattr(feat, "name") else str(feat)
        is_extruding = get("is_extruding", None)
        is_travel = get("is_travel", None)
        width = get("width_mm", None)
        height = get("height_mm",None)
        fan = get("fan_speed")
        feedrates_list = get("feedrates_mm_per_min")
        feedrate = sum(feedrates_list)/len(feedrates_list)
        if is_travel is None:
            is_travel = (is_extruding is False)
        if width is None and cfg is not None:
            try:
                width = channel_default_width_mm(cfg, ch if ch is not None else feat)
            except Exception:
                width = None
            if width is None:
                                                   
                width = float(_get(cfg,"extrusion_width", 0.45))
        if poly is None:
            p0, p1 = get("p0", None), get("p1", None)
            if p0 is not None and p1 is not None:
                poly = [p0, p1]
        if poly is None:
            continue
    
                               
        poly_closed = _close_if_perimeter_str(feature_str, poly)
        
        out.append({
            "layer_index": layer_index,
            "polyline_mm": poly_closed,
            "feature": feature_str,
            "width_w_mm": width,
            "height_h_mm": height,
            "fan_speed":fan,
            "is_travel": bool(is_travel),
            "feedrate":feedrate
        })
    return out



def _compute_quality_final_via_new_pipeline(gcode_path: str, ini_path: Optional[str], mesh_path: str, io_dict, LOAD_BEARING, LOAD_DIRECTION, rx_deg, ry_deg, rz_deg) -> float:
       
             
    cfg = parse_config_ini(ini_path)
                        

    cfg_dict = _as_dict(cfg)  
                               
    gcode_job = parse_gcode(gcode_path, cfg)
                          
                                                         
    pj_dict = {"segments": _to_raster_segments(gcode_job, cfg_dict), "config": cfg_dict}
                             
    layer_rasters = build_layer_rasters(pj_dict, pixel_xy_mm=0.25, bounds_margin_mm=0.0)
                                 
                                                                   
    mesh_data, mesh_precomp = _get_mesh_and_precomp(mesh_path, layer_rasters)
                                                     
    result = aggregate_for_bo(mesh_precomp, gcode_job, layer_rasters, pj_dict, cfg, io_dict, LOAD_BEARING, LOAD_DIRECTION, rx_deg, ry_deg, rz_deg)
    try:
        q_final = result[0]
        meta_info = {}
        meta_info["messages"] = result[-1]
        meta_info["metrics"] = result[1]
    except Exception:
        q_final, meta_info = float("inf"), {}
    return q_final, meta_info


def _slice_profile_defaults_with_warning_reruns_and_exports(
    brim_width_mm_on_warning: float = 5.0,
) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
       
    try:
        uid = uuid.uuid4().hex[:8]
        gcode_path = os.path.join(GCODE_OUT_DIR, f"baseline_warnfix_{uid}.gcode")
        mesh_path = os.path.join(MESH_OUT_DIR, f"baseline_warnfix_{uid}.STL")
        config_save_path = os.path.join(CONFIG_SAVE_DIR, f"baseline_warnfix_{uid}.ini")

                                                                 
        base_cmd = [
            "flatpak", "run",
            "--command=" + CLI_BINARY,
            FLATPAK_APP,
            "-g", MODEL_STL,
            "--load", PROFILE_INI,
            "--output", gcode_path,
        ]

        def _run(cmd: List[str]) -> subprocess.CompletedProcess:
            return subprocess.run(cmd, capture_output=True, text=True)

        def _combined_output(stdout: str, stderr: str) -> str:
            return (stdout or "") + "\n" + (stderr or "")

        def _detect_warnings(output_text: str) -> List[str]:
                                                                               
            candidates = [
                "Low bed adhesion",
                "Loose extrusions",
                "Collapsing overhang",
                "Floating object part",
                "Long bridging extrusions",
                "Floating bridge anchors",
                "Consider enabling supports",
                "Also consider enabling brim",
            ]
            return [w for w in candidates if w in output_text]

        def _insert_before_output(cmd: List[str], extra: List[str]) -> List[str]:
                                                                                      
            if not extra:
                return cmd
            if "--output" not in cmd:
                                                                                  
                return cmd + extra
            i = cmd.index("--output")
            return cmd[:i] + extra + cmd[i:]

                                                
        proc1 = _run(base_cmd)
        io_dict_attempt1 = {
            "slice_stdout": proc1.stdout or "",
            "slice_stderr": proc1.stderr or "",
        }

        out1 = _combined_output(proc1.stdout, proc1.stderr)
        warnings1 = _detect_warnings(out1)

        attempt1_ok = (proc1.returncode == 0) and _gcode_exists_nonempty(gcode_path)

                                                       
        flags_added: List[str] = []
        needs_supports = any(w in warnings1 for w in [
            "Loose extrusions",
            "Collapsing overhang",
            "Floating object part",
            "Consider enabling supports",
            "Long bridging extrusions",
            "Floating bridge anchors",
        ])
        needs_brim = any(w in warnings1 for w in [
            "Low bed adhesion",
            "Also consider enabling brim",
        ])

        if needs_supports:
                                                       
            flags_added += [
                "--support-material",
                "--support-material-auto",
                "--support-material-threshold", str(0),        
            ]

        if needs_brim:
            flags_added += [
                "--brim-width", str(5),
            ]

        io_dict_attempt2 = None
        proc2 = None

        final_slice_cmd = base_cmd
        if not attempt1_ok:
            print(
                f"[baseline_warnfix] First slicing failed or missing gcode. rc={proc1.returncode}; trying rerun",
                file=sys.stderr,
            )
                                                                                       
            if flags_added:
                final_slice_cmd = _insert_before_output(base_cmd, flags_added) + ["--save", config_save_path]
            else:
                final_slice_cmd = base_cmd + ["--save", config_save_path]

            proc2 = _run(final_slice_cmd)
            io_dict_attempt2 = {
                "slice_stdout": proc2.stdout or "",
                "slice_stderr": proc2.stderr or "",
            }
            if proc2.returncode != 0 or not _gcode_exists_nonempty(gcode_path):
                print(f"[baseline_warnfix] Second slicing failed or missing gcode. rc={proc2.returncode}", file=sys.stderr)
                return None
        elif flags_added:
                                                                
            final_slice_cmd = _insert_before_output(base_cmd, flags_added) + ["--save", config_save_path]
            proc2 = _run(final_slice_cmd)
            io_dict_attempt2 = {
                "slice_stdout": proc2.stdout or "",
                "slice_stderr": proc2.stderr or "",
            }
            if proc2.returncode != 0 or not _gcode_exists_nonempty(gcode_path):
                print(f"[baseline_warnfix] Second slicing failed or missing gcode. rc={proc2.returncode}", file=sys.stderr)
                return None
        else:
                                                                             
                                                                                                   
            final_slice_cmd = base_cmd + ["--save", config_save_path]
            proc2 = _run(final_slice_cmd)
            io_dict_attempt2 = {
                "slice_stdout": proc2.stdout or "",
                "slice_stderr": proc2.stderr or "",
            }
                                                                                     

                                                                                       
        export_cmd = final_slice_cmd.copy()
                                          
                                                                              
        if "-g" in export_cmd:
            export_cmd[export_cmd.index("-g")] = "--export-stl"
        else:
                                                                                                       
            pass

                                              
        if "--output" in export_cmd:
            oi = export_cmd.index("--output")
            if oi + 1 < len(export_cmd):
                export_cmd[oi + 1] = mesh_path

        export_proc = _run(export_cmd)
        io_dict_export = {
            "stl_export_stdout": export_proc.stdout or "",
            "stl_export_stderr": export_proc.stderr or "",
        }

        if export_proc.returncode == 0:

            try:
                translate_stl_to_xy_center(mesh_path, 125, 105, False)
            except Exception as e:
                print(f"[baseline_warnfix] STL centering failed (non-fatal): {e}", file=sys.stderr)

                                                                                   
        metrics = parse_gcode_metrics(gcode_path)

        try:
            rx_deg = 0
            ry_deg = 0
            rz_deg = 0
                                                                      
                                                                               
                                                                                               
            final_slice_io = io_dict_attempt2 if io_dict_attempt2 is not None else io_dict_attempt1
            io_dict_for_quality = {
                "slice_stdout": final_slice_io.get("slice_stdout", ""),
                "slice_stderr": final_slice_io.get("slice_stderr", ""),
                "stl_export_stdout": io_dict_export.get("stl_export_stdout", ""),
                "stl_export_stderr": io_dict_export.get("stl_export_stderr", ""),
            }
            q_final, q_info = _compute_quality_final_via_new_pipeline(
                gcode_path, PROFILE_INI, mesh_path, io_dict_for_quality,
                LOAD_BEARING, LOAD_DIRECTION, rx_deg, ry_deg, rz_deg
            )
            if not np.isfinite(q_final):
                q_final = FAIL_PENALTY
            metrics["quality_J"] = float(q_final)
            metrics["quality_info"] = q_info
        except Exception as e:
            print(f"[baseline_warnfix] Quality computation failed: {e}", file=sys.stderr)
            metrics["quality_J"] = FAIL_PENALTY
            metrics["quality_info"] = {}

        bundle: Dict[str, Any] = {
            "uid": uid,
            "paths": {
                "gcode_path": gcode_path,
                "mesh_path": mesh_path,
                "config_save_path": config_save_path,
            },
            "warnings_detected_attempt1": warnings1,
            "flags_added_for_rerun": flags_added,
            "cmds": {
                "attempt1_slice_cmd": base_cmd,
                "final_slice_cmd": final_slice_cmd,
                "stl_export_cmd": export_cmd,
            },
            "io": {
                "attempt1": io_dict_attempt1,
                "attempt2": io_dict_attempt2,
                "export": io_dict_export,
            },
            "returncodes": {
                "attempt1": proc1.returncode,
                "attempt2": (proc2.returncode if proc2 is not None else None),
                "export": export_proc.returncode,
            },
        }
        return metrics, bundle

    except Exception as e:
        print(f"[baseline_warnfix] Exception: {e}", file=sys.stderr)
        return None

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_stl", type=str, required=True, help="Path to input STL model.")
    p.add_argument("--profile_ini", type=str, required=True, help="Path to printer/profile INI.")
    p.add_argument("--output_root_dir", type=str, required=True, help="Root directory where outputs are created.")
    p.add_argument("--refs_store", type=str, default="", help="Path to refs_store.json produced elsewhere.")
    p.add_argument("--require_refs", action="store_true", help="Abort if refs_store does not contain this model/profile key.")
    p.add_argument("--load_bearing", action="store_true", help="Enable load-bearing mode in quality computation.")
    p.add_argument("--load_direction", type=str, default="z", help="String label for load direction.")
    p.add_argument("--brim_width_mm_on_warning", type=float, default=5.0, help="Brim width to apply when bed-adhesion warnings appear.")
    return p.parse_args()


def main() -> None:
    global MODEL_STL, PROFILE_INI, OUTPUT_DIR, CONFIG_SAVE_DIR, GCODE_OUT_DIR, MESH_OUT_DIR
    global LOAD_BEARING, LOAD_DIRECTION

    args = _parse_args()

    MODEL_STL = args.model_stl
    PROFILE_INI = args.profile_ini
    LOAD_BEARING = bool(args.load_bearing)
    LOAD_DIRECTION = args.load_direction

    run_id = uuid.uuid4().hex
    OUTPUT_DIR = os.path.join(args.output_root_dir, run_id)
    CONFIG_SAVE_DIR = os.path.join(OUTPUT_DIR, "configs")
    GCODE_OUT_DIR = os.path.join(OUTPUT_DIR, "gcodes")
    MESH_OUT_DIR = os.path.join(OUTPUT_DIR, "meshes")

    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    os.makedirs(GCODE_OUT_DIR, exist_ok=True)
    os.makedirs(MESH_OUT_DIR, exist_ok=True)

    if USE_DOMAIN_BASELINE_SCALING and args.refs_store:
        loaded = _try_load_persisted_refs(args.refs_store)
        if args.require_refs and not loaded:
            raise RuntimeError("Persisted refs required (--require_refs) but not found in refs_store for this model/profile key.")
    elif args.require_refs:
        raise RuntimeError("--require_refs was set but --refs_store was not provided.")

    res = _slice_profile_defaults_with_warning_reruns_and_exports(
        brim_width_mm_on_warning=float(args.brim_width_mm_on_warning),
    )

    if res is None:
        print("[vendor_defaults] Slicing failed; no objective produced.", file=sys.stderr)
                                                                                 
                                                                                
        obj = float(FAIL_PENALTY)
        result = {
            "run_id": run_id,
            "model_stl": str(Path(MODEL_STL).resolve()),
            "profile_ini": str(Path(PROFILE_INI).resolve()),
            "output_dir": str(Path(OUTPUT_DIR).resolve()),
            "objective": obj,
            "objective_combine": OBJECTIVE_COMBINE,
            "scaling": {
                "use_domain_baseline": USE_DOMAIN_BASELINE_SCALING,
                "refs_frozen": bool(REFS_FROZEN),
                "time_ref_seconds": TIME_REF,
                "cost_ref_dollars": COST_REF,
            },
            "metrics": {
                "slicing_failed": True,
                "quality_J": obj,
            },
            "artifacts": {},
            "warnings": [],
            "flags_added_for_rerun": [],
            "returncodes": {},
        }
        out_path = Path(OUTPUT_DIR) / "result.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    metrics, bundle = res


    if not USE_COMBINED or not OBJECTIVE_COMBINE:
        raise RuntimeError("OBJECTIVE_COMBINE is empty; cannot compute objective.")

    obj = objective_from_metrics_combined(metrics, OBJECTIVE_COMBINE)

    result = {
        "run_id": run_id,
        "model_stl": str(Path(MODEL_STL).resolve()),
        "profile_ini": str(Path(PROFILE_INI).resolve()),
        "output_dir": str(Path(OUTPUT_DIR).resolve()),
        "objective": float(obj),
        "objective_combine": OBJECTIVE_COMBINE,
        "scaling": {
            "use_domain_baseline": USE_DOMAIN_BASELINE_SCALING,
            "refs_frozen": bool(REFS_FROZEN),
            "time_ref_seconds": TIME_REF,
            "cost_ref_dollars": COST_REF,
        },
        "metrics": metrics,
        "artifacts": bundle.get("paths", {}),
        "warnings": bundle.get("warnings_detected_attempt1", []),
        "flags_added_for_rerun": bundle.get("flags_added_for_rerun", []),
        "returncodes": bundle.get("returncodes", {}),
    }

    out_path = Path(OUTPUT_DIR) / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
