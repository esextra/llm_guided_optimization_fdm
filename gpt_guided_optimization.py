#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path                                                                  
import os
import gc
import re
import uuid
import json
import subprocess
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List, Any, Set
from hints_categorical_to_idx import normalize_hints_categoricals_to_indices
import numpy as np
import hashlib
import trimesh
import math
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

                                    
import importlib
mod = importlib.import_module("print_quality")
# print("print_quality loaded from:", getattr(mod, "__file__", "<unknown>"))
                         
                                             
from soft_violation_from_hints import build_soft_violation                                  


from print_quality.io.mesh_loader import load_mesh
from print_quality.io.gcode_parser import parse_gcode
from print_quality.io.config_parser import parse_config_ini
from print_quality.pipeline.mesh_precompute import build_mesh_precompute
from print_quality.pipeline.build_rasters import build_layer_rasters
from print_quality.utils.config_resolution import channel_default_width_mm
from print_quality.data.types import Config, PrintJob                       
from print_quality.pipeline.metrics.aggregator import aggregate_for_bo
from print_quality.pipeline.metrics.helpers import _get

from bo_logging_for_guidance import BOLogger

import argparse

                                                                      

                                         
_LOGGER, _RUN_ID = None, None
    
MODEL_STL, PROFILE_INI, OUTPUT_DIR, CONFIG_SAVE_DIR, GCODE_OUT_DIR, MESH_OUT_DIR  = None, None, None, None, None, None


LOAD_BEARING = False
LOAD_DIRECTION: Optional[str] = None                                   

                                                       
GUIDANCE_HINTS_DIR = os.environ.get("GUIDANCE_HINTS_DIR", "./guidance_gpt5p2")

                                                                      
USE_DYNAMIC_LLM_CACHE: bool = bool(int(os.environ.get("USE_DYNAMIC_LLM_CACHE", "1")))
DYNAMIC_LLM_CACHE_PATH = os.environ.get(
    "DYNAMIC_LLM_CACHE_PATH",
    str((Path(__file__).resolve().parent / "chatgpt_5p2_index_dynamic.json"))
)

                                                                                         
                                                                    
LIVE_LLM_BACKEND = os.environ.get("LIVE_LLM_BACKEND", "openai").lower()
LIVE_LLM_MODEL = os.environ.get("LIVE_LLM_MODEL", "gpt-5.2")
LIVE_LLM_SYSTEM_FILE = os.environ.get(
    "LIVE_LLM_SYSTEM_FILE",
    str((Path(__file__).resolve().parent / "system_msg_fewshot.txt"))
)

LIVE_LLM_MAX_NEW_TOKENS = int(os.environ.get("LIVE_LLM_MAX_NEW_TOKENS", "256"))

LIVE_LLM_TEMPERATURE = float(os.environ.get("LIVE_LLM_TEMPERATURE", "0.0"))
LIVE_LLM_TOP_P = float(os.environ.get("LIVE_LLM_TOP_P", "1.0"))

_DYNAMIC_LLM_CACHE_ENTRIES = None                                  
_DYNAMIC_LLM_CACHE_LOCK = threading.Lock()

_LIVE_LLM_SYSTEM_MSG_CACHE = None
_LIVE_OPENAI_CLIENT = None

                                                         
LIVE_LLM_REASONING_EFFORT = os.environ.get("LIVE_LLM_REASONING_EFFORT", "none").strip().lower()
LIVE_LLM_TEXT_VERBOSITY = os.environ.get("LIVE_LLM_TEXT_VERBOSITY", "").strip().lower()


OBJECTIVE_COMBINE: List[Tuple[str, float]] = [("time_normal", 0.1), ("total_filament_cost", 0.1), ("quality_J", 0.8)]
USE_COMBINED: bool = len(OBJECTIVE_COMBINE) > 0


                     
MAX_ITER: int = 40
INITIAL_DESIGN: int = 16
INITIAL_DESIGN_TYPE = 'random'                    

                                                                
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
        print(f"[scaling] Loaded persisted refs: TIME_REF={TIME_REF:.3f}s, COST_REF={COST_REF:.3f}$")
        return True
    except Exception as e:
        print(f"[scaling] Failed to load persisted refs: {e}")
        return False

                                                                                
                                                                         
                                                                                      
                                                                                
ORIENTATION_UPS = None                                    
ORIENTATION_YAWS_DEG = None                               
                                                           
ORIENTATION_LUT = None                                                                   
                                                                         
ORIENT_PAIRS = None                                     


def _load_orientation_tables_npz(npz_path: str) -> None:
    import numpy as _np
    from pathlib import Path as _Path
    p = _Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Orientation tables not found: {p}")
    data = _np.load(str(p), allow_pickle=False)
    ups = data["ups"].astype(int).tolist()
    yaws = data["yaws"].astype(int).tolist()
    k = data["lut_keys"]                                   
    v = data["lut_vals"]                                                
    lut = { (int(a), int(b)): (int(c), int(d), int(e)) for (a, b), (c, d, e) in zip(k, v) }
    global ORIENTATION_UPS, ORIENTATION_YAWS_DEG, ORIENTATION_LUT
    ORIENTATION_UPS, ORIENTATION_YAWS_DEG, ORIENTATION_LUT = ups, yaws, lut
    global ORIENT_PAIRS

                                                  
    if "orient_pairs" in data.files:
        op = data["orient_pairs"].astype(int)
        pairs = [(int(a), int(b)) for a, b in op]
        if len(pairs) == 0:
            raise RuntimeError("NPZ contained 'orient_pairs' but it was empty.")
        if len(set(pairs)) != len(pairs):
            raise RuntimeError("NPZ 'orient_pairs' contains duplicates.")
        if set(pairs) != set(lut.keys()):
            raise RuntimeError("NPZ 'orient_pairs' does not match LUT keys (missing/extra pairs).")
        ORIENT_PAIRS = pairs
    else:
                                                      
        ORIENT_PAIRS = sorted(list(lut.keys()))

    if not ORIENTATION_UPS or not ORIENTATION_YAWS_DEG or not ORIENTATION_LUT:
        raise RuntimeError("Orientation tables loaded but empty or malformed.")

                                                                                  
GUIDANCE_FEATURE_ORDER = [
    "layer_height_mm",
    "infill_density_pct",
    "infill_pattern_id",
    "num_perimeters",
    "filament_max_volumetric_speed",
    "support_material",
    "first_layer_height_mm",
    "first_layer_extrusion_width_mm",
    "elefant_foot_compensation_mm",
    "seam_idx",
    "orient_pair_idx",
    "brim_width",
    "top_solid_layers",
    "bottom_solid_layers",
]
                                                        
                                                                             
TR_ENABLED     = False
TR_EXPAND      = 1.5                                        
TR_SHRINK      = 0.5                                       
TR_PATIENCE    = 3                                                        
TR_MIN_FRAC    = 0.05                                                
TR_MAX_FRAC    = 1.00                                                
TR_SIGMA_FRAC  = 0.25                                                              

                                                  
GUIDANCE_LAMBDA = float(globals().get("GUIDANCE_LAMBDA", 2.0))                          
GUIDANCE_BETA   = float(globals().get("GUIDANCE_BETA", 10.0))                        
SCALING_EPS      = 1e-9                                                                    
MAX_ABS_HAT      = 50.0                                                                   

                                                           
_guidance_hints: Optional[List[Dict[str, Any]]] = None

                                                                   
_CURRENT_DOMAIN: Optional[List[Dict[str, Any]]] = None

                                                                                             
_H_SOURCES: List[Dict[str, Any]] = []

                                                                       
_name_to_col: Dict[str, int] = {}
_cont_names: List[str] = []
_cont_idx    = np.array([], dtype=int)
_cont_bounds = np.zeros((0, 2), dtype=float)

                                         
_TR_STATE = {
    "center": None,                    
    "half":   None,                    
    "bounds": None,                      
    "names":  None,               
    "fail_streak": 0,
}

                            
_soft_violation = None
_soft_violation_batch = None
                                     
                                                            
_LAST_ACQ_DIAGNOSTICS = {
    "X": None,                          
    "H": None,                          
    "H_hat": None,                      
    "temper": None,                     
    "w_tr": None,                       
    "base_hat": None                    
}
                                           
                                                                      
_LAST_HINT_TARGETS: Set[str] = set()

# ================================
# LLM Guidance Compiler Helpers
# (action → residual clauses JSON)
# ================================

ACTION_TO_JSON: Dict[str, str] = {
    "bottom_layers_up":          "bottom_layers_up.json",
    "brim":                      "brim.json",
    "elefant_foot_up":           "elefant_foot_up.json",
    "first_layer_height_up":     "first_layer_height_up.json",
    "first_layer_width_down":    "first_layer_width_down.json",
    "first_layer_width_up":      "first_layer_width_up.json",
    "infill_density_up":         "infill_density_up.json",
    "infill_pattern_gyroid":     "infill_pattern_gyroid.json",
    "layer_height_down":         "layer_height_down.json",
    "no_supports":               "no_supports.json",
    "perimeters_up":             "perimeters_up.json",
    "reorient":                  "reorient.json",
    "supports":                  "supports.json",
    "top_layers_up":             "top_layers_up.json",
}

                                                                 
ACTION_TO_JSON.update({
    "brim": "brim.json",
    "supports": "supports.json",
    "no_supports": "no_supports.json",
    "infill_pattern_gyroid": "infill_pattern_gyroid.json",
})


_GUIDANCE_TEXT_BY_ACTION: Dict[str, str] = {
    "bottom_layers_up": "Increase the number of solid layers at the bottom to strengthen the base.",
    "top_layers_up": "Increase the number of solid layers at the top to make the top surface stronger and more solid.",
    "perimeters_up": "Make the walls stronger by increasing the number of perimeters.",
    "infill_density_up": "Increase the infill density to improve stiffness and internal support.",
    "infill_pattern_gyroid": "Switch the infill pattern to gyroid for more uniform strength.",
    "layer_height_down": "Reduce the layer height to reduce stair stepping and improve bonding.",
    "first_layer_height_up": "Increase the first layer height slightly to improve bed adhesion.",
    "first_layer_width_up": "Make the first layer lines slightly wider to improve adhesion.",
    "first_layer_width_down": "Make the first layer lines slightly narrower to reduce first-layer bulging.",
    "elefant_foot_up": "Increase elephant-foot compensation slightly to counter first-layer bulging at the base.",
    "brim": "Add a small brim to improve bed adhesion.",
    "supports": "Enable supports.",
    "no_supports": "Disable supports.",
    "reorient": "Change print orientation.",
}


def _ensure_guidance_json_exists(json_path: str, action_name: str) -> None:
    if json_path and os.path.isfile(json_path):
        return

    guidance_text = _GUIDANCE_TEXT_BY_ACTION.get(action_name)
    if guidance_text is None:
        raise RuntimeError(
            f"No guidance text is defined for action '{action_name}'. "
            f"Either pre-generate '{json_path}' offline, or add a guidance text entry."
        )

    targets_meta_path = Path(__file__).resolve().parent / "slicer_targets_meta_defaults.json"
    if not targets_meta_path.exists():
        raise RuntimeError(f"Missing targets meta file: {targets_meta_path}")

    targets_meta = json.loads(targets_meta_path.read_text())

                                                                                          
    from gpt_guidance_compiler import run_guidance_micro_reasoners

    llm_client = _get_openai_client()

    predicate_info = [{"predicate": action_name, "fired": True}]
    predicate_guidance = {action_name: guidance_text}
    guidance_hints = run_guidance_micro_reasoners(
        llm_client, predicate_info, predicate_guidance, targets_meta
    )

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(guidance_hints, f, indent=4)


# ================================
# LLM Guidance Generator Helpers
# (diagnostics → action selection)
# ================================

_DEFAULT_LLM_ACTIONS_CATALOG: List[str] = [
    "bottom_layers_up",
    "brim",
    "elefant_foot_up",
    "first_layer_height_up",
    "first_layer_width_down",
    "first_layer_width_up",
    "infill_density_up",
    "infill_pattern_gyroid",
    "layer_height_down",
    "no_supports",
    "perimeters_up",
    "reorient",
    "supports",
    "top_layers_up",
]



def _parse_severity(msg: str) -> Optional[str]:
    s = str(msg).strip().lower()
    for label in ("extreme", "very high", "high", "moderate", "mild", "negligible"):
        if s.startswith(label):
            return label
    return None

def _signature_key_from_messages(messages: Dict[str, Any]) -> Optional[str]:
       
    veto_items = []
    nonveto_items = []
    saw_veto = False
    saw_nonveto = False

    allowed = {"negligible", "mild", "moderate", "high", "very high", "extreme"}

    for k, v in messages.items():
        lab = _parse_severity(str(v))
        if lab is None:
                                                   
            if str(v).strip().upper().startswith("TRIGGERED"):
                lab = "TRIGGERED"
            else:
                return None

        if lab == "TRIGGERED":
            saw_veto = True
            veto_items.append((k, "TRIGGERED"))
        else:
            if lab not in allowed:
                return None
            saw_nonveto = True
            nonveto_items.append((k, lab))

    if saw_veto and saw_nonveto:
        return None

    if saw_veto:
        if not veto_items:
            return None
        items = sorted(veto_items)
        return "veto|" + "|".join(f"{k}={val}" for k, val in items)

    items = sorted(nonveto_items)
    return "nonveto|" + "|".join(f"{k}={lab}" for k, lab in items)
              

def _extract_inner_json_block(s: str) -> Optional[str]:
                                                                   
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0).strip() if m else None



def _load_live_system_msg() -> str:
                                                                                     
    global _LIVE_LLM_SYSTEM_MSG_CACHE
    if _LIVE_LLM_SYSTEM_MSG_CACHE is not None:
        return _LIVE_LLM_SYSTEM_MSG_CACHE

    p = Path(LIVE_LLM_SYSTEM_FILE) if LIVE_LLM_SYSTEM_FILE else None
    if p and p.exists():
        _LIVE_LLM_SYSTEM_MSG_CACHE = p.read_text(encoding="utf-8", errors="ignore")
    else:
        _LIVE_LLM_SYSTEM_MSG_CACHE = ""
    return _LIVE_LLM_SYSTEM_MSG_CACHE


def _get_dynamic_llm_cache_entries() -> Dict[str, Any]:
                                                                   
    global _DYNAMIC_LLM_CACHE_ENTRIES
    if _DYNAMIC_LLM_CACHE_ENTRIES is not None:
        return _DYNAMIC_LLM_CACHE_ENTRIES

    p = Path(DYNAMIC_LLM_CACHE_PATH)
    if not p.exists():
        _DYNAMIC_LLM_CACHE_ENTRIES = {}
        return _DYNAMIC_LLM_CACHE_ENTRIES

    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
            entries = raw["entries"]
        elif isinstance(raw, dict):
            entries = raw
        else:
            entries = {}
        _DYNAMIC_LLM_CACHE_ENTRIES = entries
        print(f"[dyn-llm-cache] loaded {len(entries)} signatures from {p}")
        return _DYNAMIC_LLM_CACHE_ENTRIES
    except Exception as e:
        print(f"[dyn-llm-cache] failed to read {p}; starting empty. {e}")
        _DYNAMIC_LLM_CACHE_ENTRIES = {}
        return _DYNAMIC_LLM_CACHE_ENTRIES


def _persist_dynamic_llm_cache_entries(entries: Dict[str, Any]) -> None:
                                                           
    p = Path(DYNAMIC_LLM_CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "created_utc": time.time(),
        "signature_space": "severity_labels_only",
        "entries": entries,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(p)


def _cache_row_to_actions(row: Any, full_messages: Dict[str, Any]) -> Set[str]:

    if isinstance(row, dict):
        acts = row.get("chosen_actions") or row.get("actions") or []
    elif isinstance(row, list):
        acts = row
    elif isinstance(row, str):
        acts = [row]
    else:
        return set()

    if isinstance(acts, str):
        acts = [acts]
    if not isinstance(acts, list):
        return set()

                                                         
    return {a for a in acts if isinstance(a, str) and a in ACTION_TO_JSON}




def _get_openai_client():
       
    global _LIVE_OPENAI_CLIENT
    if _LIVE_OPENAI_CLIENT is not None:
        return _LIVE_OPENAI_CLIENT

    try:
        from openai import OpenAI                
    except Exception as e:
        raise RuntimeError("Failed to import openai. Install it with: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it to use LIVE_LLM_BACKEND='openai'.")

    _LIVE_OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _LIVE_OPENAI_CLIENT



def _call_live_llm_for_row(messages_subset: Dict[str, Any], actions_catalog: List[str]) -> Optional[Dict[str, Any]]:
       
    system_msg = _load_live_system_msg()
    if not system_msg:
        raise RuntimeError(
            "LIVE_LLM_SYSTEM_FILE did not resolve to a readable system prompt. "
            "Set LIVE_LLM_SYSTEM_FILE or pass --live_llm_system_file."
        )

    user_obj = {"messages": messages_subset, "actions_catalog": actions_catalog}
    user_text = json.dumps(user_obj, indent=2, sort_keys=True)

    txt = ""

    openai_response_id = None

    if LIVE_LLM_BACKEND == "openai":
        client = _get_openai_client()
        try:
                                          
            openai_input = "Return a single valid JSON object only.\n\n" + user_text

            req: Dict[str, Any] = {
                "model": LIVE_LLM_MODEL,
                "instructions": system_msg,
                "input": openai_input,
                "max_output_tokens": int(LIVE_LLM_MAX_NEW_TOKENS),
            }                                                    
                                                                                                             
            text_obj: Dict[str, Any] = {"format": {"type": "json_object"}}
            if LIVE_LLM_TEXT_VERBOSITY in ("low", "medium", "high"):
                text_obj["verbosity"] = LIVE_LLM_TEXT_VERBOSITY
            req["text"] = text_obj

            eff = (LIVE_LLM_REASONING_EFFORT or "none").strip().lower()
            if eff and eff != "none":
                                                                                                    
                req["reasoning"] = {"effort": eff}
            else:
                                                              
                req["temperature"] = float(LIVE_LLM_TEMPERATURE)
                req["top_p"] = float(LIVE_LLM_TOP_P)

            response = client.responses.create(**req)
            openai_response_id = getattr(response, "id", None)
            txt = getattr(response, "output_text", None)
            if not txt:
                                                                                               
                try:
                    parts = []
                    for out_item in (getattr(response, "output", None) or []):
                        for c in (getattr(out_item, "content", None) or []):
                            if getattr(c, "type", None) == "output_text":
                                parts.append(getattr(c, "text", "") or "")
                    txt = "".join(parts)
                except Exception:
                    txt = ""
            else:
                txt = str(txt)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    else:
        raise RuntimeError(
            f"Unsupported LIVE_LLM_BACKEND={LIVE_LLM_BACKEND!r}; expected 'openai'."
        )


    jtxt = _extract_inner_json_block(txt)
    if not jtxt:
        return None
    try:
        resp = json.loads(jtxt)
    except Exception:
        return None

    if not isinstance(resp, dict):
        return None

                                                       
    acts = resp.get("chosen_actions") or resp.get("actions") or []
    if not isinstance(acts, list) or not all(isinstance(a, str) for a in acts):
        return None

                                                            
    allowed = set(actions_catalog)
    resp["chosen_actions"] = [a for a in acts if a in allowed]

                                   
    resp["_updated_utc"] = time.time()
    resp["_backend"] = LIVE_LLM_BACKEND
    resp["_model"] = LIVE_LLM_MODEL
    resp["_raw_response"] = txt
    if openai_response_id:
        resp["_openai_response_id"] = openai_response_id
    print(resp)
    return resp


def _resolve_actions_for_signature(
    signature: str,
    messages_subset: Dict[str, Any],
    full_messages: Dict[str, Any],
    actions_catalog: List[str],
) -> Tuple[Set[str], bool]:

                       
    if USE_DYNAMIC_LLM_CACHE:
        with _DYNAMIC_LLM_CACHE_LOCK:
            entries = _get_dynamic_llm_cache_entries()
            if signature in entries:
                row = entries[signature]
                return _cache_row_to_actions(row, full_messages), True
                
                                             
    if USE_DYNAMIC_LLM_CACHE and LIVE_LLM_MODEL:
        try:
            row2 = _call_live_llm_for_row(messages_subset, actions_catalog)
        except Exception as e:
            print(f"[dyn-llm] live call failed; treating as miss. {e}")
            row2 = None

        if row2 is not None:
            with _DYNAMIC_LLM_CACHE_LOCK:
                entries = _get_dynamic_llm_cache_entries()
                if signature not in entries:
                    entries[signature] = row2
                    try:
                        _persist_dynamic_llm_cache_entries(entries)
                    except Exception as e:
                        print(f"[dyn-llm-cache] failed to persist {DYNAMIC_LLM_CACHE_PATH}. {e}")
            return _cache_row_to_actions(row2, full_messages), True
            

    return set(), False



def choose_hints_json_from_info(info: Dict[str, Any]) -> List[str]:
       
    messages = (info or {}).get("messages") or {}
    if not isinstance(messages, dict) or not messages:
        return []

    actions_catalog = (info or {}).get("actions_catalog")
    if (not isinstance(actions_catalog, list)) or (not all(isinstance(a, str) for a in actions_catalog)):
        actions_catalog = list(_DEFAULT_LLM_ACTIONS_CATALOG)

                                                                                                      
    _LLM_GEOMETRIC_KEYS = ("stair_stepping",)
    _LLM_FUNCTIONAL_KEYS = ("strength_reserve", "z_bonding_proxy", "perim_infill_contact", "xy_dimensional_risk")
    _LLM_PRINTABILITY_KEYS = ("stringing_exposure", "support_removal")
    _LLM_VETO_KEYS = ("bed_adhesion", "overhang_exposure", "bridge_exposure", "island_starts", "slender_towers")
    _LLM_KEYS = set(_LLM_GEOMETRIC_KEYS + _LLM_FUNCTIONAL_KEYS + _LLM_PRINTABILITY_KEYS + _LLM_VETO_KEYS)

    pruned_for_llm: Dict[str, Any] = {k: v for k, v in messages.items() if k in _LLM_KEYS}

    def _lookup_llm(sub_messages: Dict[str, Any]) -> Tuple[Set[str], bool]:
        sig = _signature_key_from_messages(sub_messages)
        print(sig)
        if not sig:
            return set(), False

        acts, hit = _resolve_actions_for_signature(
            signature=sig,
            messages_subset=sub_messages,
            full_messages=messages,
            actions_catalog=actions_catalog,
        )
        print(acts, hit)
        return acts, hit

    actions: Set[str] = set()

    if pruned_for_llm:
        has_v = any(k in _LLM_VETO_KEYS for k in pruned_for_llm.keys())
        has_nv = any(k not in _LLM_VETO_KEYS for k in pruned_for_llm.keys())

                                                                                       
        if has_v and has_nv:
            veto_msgs = {k: v for k, v in pruned_for_llm.items() if k in _LLM_VETO_KEYS}
            nonveto_msgs = {k: v for k, v in pruned_for_llm.items() if k not in _LLM_VETO_KEYS}

            acts_v, _hit_v = _lookup_llm(veto_msgs)
            acts_nv, _hit_nv = _lookup_llm(nonveto_msgs)

            actions |= acts_v
            actions |= acts_nv
        else:
            acts, hit = _lookup_llm(pruned_for_llm)
            print("llm worked:", acts, hit)
            actions |= acts

                                               
                                                                                          
    jsons: List[str] = []
    if actions:
        for a in sorted(actions):
            j = ACTION_TO_JSON.get(a)
            if not j:
                continue
            json_path = os.path.join(GUIDANCE_HINTS_DIR, j)
            # Guidance compiler: compile action → JSON (if missing)
            _ensure_guidance_json_exists(json_path, a)
            jsons.append(json_path)

    return sorted(set(jsons))              
                                                                                
                                                                        
_HINTS_BASE_VEC: Optional[np.ndarray] = None

                                                                        
                                                                      
_RELATIVE_RESIDUAL_TYPES = {
    "inc", "dec", "diff_ge"
}

                                                                       
_ANGLE_FEATURE_PREFIX = "rotate_deg_"

                                                           
                                                       
H_BASELINE_MODE: str = globals().get("H_BASELINE_MODE", "best")
H_BASELINE_BLEND_ALPHA: float = float(globals().get("H_BASELINE_BLEND_ALPHA", 0.7))

                                                                                   
_BEST_HINTS_BASE_VEC = None
_BEST_QUALITY_INFO = None
_BEST_PARAMS = None
_BEST_OBJECTIVE_VALUE = None

def _lo_hi_from_domain_entry(d: Dict[str, Any]) -> Tuple[float, float, bool]:
                                                                                             
    t = (d or {}).get("type")
    dom = (d or {}).get("domain")
    if t == "discrete":
                                                                                   
        if isinstance(dom, (list, tuple)) and len(dom) > 0:
            vals = list(dom)
            if len(vals) == 2 and all(isinstance(v, (int, float)) for v in vals):
                lo, hi = float(vals[0]), float(vals[1])
            else:
                lo = float(min(vals))
                hi = float(max(vals))
        else:
            raise ValueError(f"Malformed discrete domain: {d}")
        return lo, hi, True
    else:
                                               
        if not (isinstance(dom, (list, tuple)) and len(dom) >= 2):
            raise ValueError(f"Malformed continuous domain: {d}")
        return float(dom[0]), float(dom[1]), False


def _at_upper_bound(val: float, lo: float, hi: float, is_discrete: bool) -> bool:
    if is_discrete:
                                                                         
        return val >= hi
    tol = 1e-9 + 1e-6 * max(1.0, abs(hi - lo))
    return (hi - val) <= tol


def _at_lower_bound(val: float, lo: float, hi: float, is_discrete: bool) -> bool:
    if is_discrete:
        return val <= lo
    tol = 1e-9 + 1e-6 * max(1.0, abs(hi - lo))
    return (val - lo) <= tol


def _hint_json_is_unactionable(hint_json_path: str,
                               domain: List[Dict[str, Any]],
                               incumbent_real: np.ndarray) -> bool:
       
    try:
        raw = _load_guidance_hints(hint_json_path)
    except Exception:
                                                                           
        return False

    blocks = raw if isinstance(raw, list) else [raw]
    if not blocks:
        return False

    name_to_dom = {d["name"]: d for d in (domain or [])}
    idx_by_name = {d["name"]: i for i, d in enumerate(domain or [])}
    h2d = dict(_G2D_NAME_MAP) if "_G2D_NAME_MAP" in globals() else {}

    saw_directional = False
    found_actionable_target = False

    for block in blocks:
        for cl in (block or {}).get("clauses", []):
            rt = str((cl or {}).get("residual_type", "")).lower().strip()
            if rt not in {"inc", "dec"}:
                continue
            saw_directional = True
            for tgt in (cl or {}).get("targets", []):
                dom_name = h2d.get(tgt, tgt)
                d = name_to_dom.get(dom_name)
                k = idx_by_name.get(dom_name)
                if (d is None) or (k is None):
                                                                               
                    found_actionable_target = True
                    continue
                lo, hi, is_disc = _lo_hi_from_domain_entry(d)
                try:
                    xk = float(incumbent_real[k])
                except Exception:
                                                             
                    return False
                if rt == "inc" and not _at_upper_bound(xk, lo, hi, is_disc):
                    found_actionable_target = True
                if rt == "dec" and not _at_lower_bound(xk, lo, hi, is_disc):
                    found_actionable_target = True

                                                                                                           
    return (saw_directional and not found_actionable_target)

def _reload_hints(guidance_json_paths, domain: List[Dict[str, Any]]):
       
    global _soft_violation, _soft_violation_batch, _guidance_hints, _H_SOURCES
    _soft_violation = None
    _soft_violation_batch = None
    _guidance_hints = None
    _H_SOURCES = []
    if build_soft_violation is None:
        print("[warn] soft_violation_from_hints not available; skipping guidance reload.")
        return
                                                    
    if not guidance_json_paths:
        print("[info] No guidance file selected; running unconstrained EI/TR.")
        return
    if isinstance(guidance_json_paths, str):
        paths = [guidance_json_paths]
    else:
        paths = [p for p in (guidance_json_paths or []) if p]
    if not paths:
        print("[info] No guidance file selected; running unconstrained EI/TR.")
        return
                                                                                  
    try:
        incumbent = globals().get("_BEST_PARAMS", None)
        if incumbent is not None:
            before = list(paths)
            arr = np.asarray(incumbent, dtype=float)
            paths = [p for p in paths if not _hint_json_is_unactionable(p, domain, arr)]
            dropped = [os.path.basename(dp) for dp in before if dp not in paths]
            for dp in dropped:
                print(f"[guidance] pruned unreachable hint at incumbent bounds: {dp}")
    except Exception as e:
        print(f"[warn] failed to prune unactionable hints: {e}")

    all_hints = []
    for idx, hp in enumerate(paths, 1):
        try:
            raw = _load_guidance_hints(hp)
            hh = normalize_hints_categoricals_to_indices(
                raw, infill_patterns=INFILL_PATTERNS, seam_positions=SEAM_POSITIONS,
            )
            row_fn, batch_fn = build_soft_violation(hh, GUIDANCE_FEATURE_ORDER)
                                                       
            name = os.path.splitext(os.path.basename(hp))[0] or f"h{idx}"
            _H_SOURCES.append({"name": name, "row": row_fn, "batch": batch_fn})
            all_hints.extend(hh if isinstance(hh, list) else [hh])
            print(f"[guidance] loaded: {hp}")
        except Exception as e:
            print(f"[warn] Failed to load guidance '{hp}': {e}")
                                                               
    if all_hints:
        _guidance_hints = all_hints
        _init_tr_from_hints_and_domain(domain)
    else:
        print("[info] No valid guidance was loaded; running unconstrained EI/TR.")

def _refresh_cont_indices():
                                                                             
    global _name_to_col, _cont_names, _cont_idx, _cont_bounds
    dom = _CURRENT_DOMAIN or []
    _name_to_col = {d["name"]: j for j, d in enumerate(dom)}
    _cont_names  = [d["name"] for d in dom if d.get("type") == "continuous"]
    if _cont_names:
        _cont_idx = np.array([_name_to_col[nm] for nm in _cont_names], dtype=int)
        _cont_bounds = np.array([[float(d["domain"][0]), float(d["domain"][1])]
                                 for d in dom if d.get("type") == "continuous"], dtype=float)
    else:
        _cont_idx    = np.array([], dtype=int)
        _cont_bounds = np.zeros((0, 2), dtype=float)
                                

_G2D_NAME_MAP = {
                                                             
    "layer_height_mm":                 "layer_height",
    "infill_density_pct":              "infill_density",
    "infill_pattern_id":               "pattern_idx",
    "num_perimeters":                  "perimeters",
    "filament_max_volumetric_speed":   "filament_max_volumetric_speed",
    "support_material":                "support_material",
    "first_layer_height_mm":           "first_layer_height",
    "first_layer_extrusion_width_mm":  "first_layer_extrusion_width",
    "elefant_foot_compensation_mm":    "elefant_foot_compensation",
    "seam_idx":                        "seam_idx",
    "orient_pair_idx":                 "orient_pair_idx",
    "brim_width":                      "brim_width",
    "top_solid_layers":                "top_solid_layers",
    "bottom_solid_layers":             "bottom_solid_layers",
}


def _targets_in_hints(hints) -> set:
                                                                              
    S = set()
    for block in hints or []:
        for cl in block.get("clauses", []):
            for t in cl.get("targets", []):
                S.add(str(t))
    return S

def _build_context_scaled(active_hints, domain_scaled, to_scaled, x_anchor_real):

    try:
        if not active_hints:
            return {}
                                               
                                                                                 
        S_h = _targets_in_hints(active_hints)
                                                                                              
        S_d = { _G2D_NAME_MAP.get(h, h) for h in S_h }
        x_anchor_scaled = to_scaled(np.atleast_2d(np.asarray(x_anchor_real, dtype=float)))[0]
        ctx = {}
        for j, d in enumerate(domain_scaled):
            name = d["name"]
            if name not in S_d:
                ctx[name] = float(x_anchor_scaled[j])
        return ctx
    except Exception:
                                                                          
        return {}




def _h_eval_only_batch(X_bo: np.ndarray) -> np.ndarray:

       
    X_bo = np.atleast_2d(X_bo)
    X_hints = _to_hints_batch(X_bo)

                                      
    if _H_SOURCES:
        H_list = []
        for s in _H_SOURCES:
            bf, rf = s.get("batch"), s.get("row")
            if callable(bf):
                h = np.asarray(bf(X_hints), dtype=float).reshape(-1, 1)
            elif callable(rf):
                h = np.asarray([rf(r) for r in X_hints], dtype=float).reshape(-1, 1)
            else:
                h = np.zeros((X_hints.shape[0], 1), dtype=float)
            H_list.append(np.clip(h, 0.0, 1.0))
            
        Hmat = np.hstack(H_list)          
        if Hmat.shape[1] == 1:
            return Hmat[:, [0]]
                                                                          
        beta = GUIDANCE_BETA
        logits = +beta * Hmat
        logits -= logits.max(axis=1, keepdims=True)                        
        w = np.exp(logits)
        w /= np.sum(w, axis=1, keepdims=True)
        smax = np.sum(w * Hmat, axis=1, keepdims=True)                                  
        return smax

                                                         
    batch_fn = globals().get("_soft_violation_batch", None)
    row_fn   = globals().get("_soft_violation", None)
    if callable(batch_fn):
        h = np.asarray(batch_fn(X_hints), dtype=float).reshape(-1, 1)
    elif callable(row_fn):
        h = np.asarray([row_fn(r) for r in X_hints], dtype=float).reshape(-1, 1)
    else:
        h = np.zeros((X_hints.shape[0], 1), dtype=float)
    return np.clip(h, 0.0, 1.0)

def _h_and_grad_batch(X_bo: np.ndarray):
                                                                                
    X_bo = np.atleast_2d(X_bo)
    N, D = X_bo.shape
    H = _h_eval_only_batch(X_bo)
    grad = np.zeros((N, D))
    if _cont_idx.size:
        spans = (_cont_bounds[:, 1] - _cont_bounds[:, 0])
        h_fd = np.maximum(1e-6, 1e-4 * spans)
        for k, j in enumerate(_cont_idx):
            hj = float(h_fd[k]); lo, hi = _cont_bounds[k]
            Xp = X_bo.copy(); Xm = X_bo.copy()
            Xp[:, j] = np.clip(Xp[:, j] + hj, lo, hi)
            Xm[:, j] = np.clip(Xm[:, j] - hj, lo, hi)
            Hp = _h_eval_only_batch(Xp); Hm = _h_eval_only_batch(Xm)
            dH = (Hp - Hm) / (2.0 * hj)
            grad[:, j] = dH[:, 0]
    return H, grad

def _tr_weight_and_grad_batch(X_bo: np.ndarray):
                                                                               
    X_bo = np.atleast_2d(X_bo)
    N, D = X_bo.shape
    if (not TR_ENABLED) or (_TR_STATE.get("center") is None) or _cont_idx.size == 0:
        return np.ones((N, 1)), np.zeros((N, D))
    center = _TR_STATE["center"]; half = _TR_STATE["half"]
    low, high = center - half, center + half
    sigma = np.maximum(1e-12, TR_SIGMA_FRAC * half)
    M = X_bo[:, _cont_idx]         
    delta_low  = np.maximum(0.0, low  - M)
    delta_high = np.maximum(0.0, M    - high)
    delta = np.maximum(delta_low, delta_high)                       
    sign = np.where(delta_low > 0.0, -1.0, 0.0) + np.where(delta_high > 0.0, 1.0, 0.0)
    Z = (delta / sigma)**2
    w_tr = np.exp(-np.sum(Z, axis=1, keepdims=True))               
    dlog_w = -2.0 * (delta / (sigma**2)) * sign                    
    grad_tr_cont = w_tr * dlog_w                                   
    grad_tr = np.zeros((N, D))
    if _cont_idx.size:
        grad_tr[:, _cont_idx] = grad_tr_cont
    return w_tr, grad_tr


                                                                                       
                                                                                  
_ITER_SCALERS = {"iter": None, "base_den": 1.0}


def install_tempered_acquisition(bo, to_real_fn):
    

    _orig_acq_fun = bo.acquisition.acquisition_function
                                                                                         
    bo._orig_acq_fun = _orig_acq_fun

    _has_grad = hasattr(bo.acquisition, "acquisition_function_withGradients")
    if _has_grad:
        _orig_acq_fun_grad = bo.acquisition.acquisition_function_withGradients
        bo._orig_acq_fun_grad = _orig_acq_fun_grad

                                                       
    _to_real = to_real_fn

                                                             
    _dom = globals().get("_CURRENT_DOMAIN", None)
    _span = None
    if _dom is not None:
        _lo, _hi = _extract_minmax_from_domain(_dom)                               
        _span = np.where((_hi - _lo) == 0.0, 1.0, (_hi - _lo)).astype(float)        


    def _acq_tempered_no_grad(X_bo: np.ndarray) -> np.ndarray:
        base = _orig_acq_fun(X_bo).reshape(-1, 1)                                              
                                                                                               
        s = globals().get("_ITER_SCALERS", {})
        if s.get("iter") == getattr(bo, "_acq_iter", None):
            den_base = float(s.get("base_den", 1.0))
        else:
            med = float(np.median(base))
            den_base = float(np.median(np.abs(base - med)) + SCALING_EPS)
        base_scaled = base / den_base                                    

                                               
        X_real = _to_real(X_bo)
        H, _ = _h_and_grad_batch(X_real)                                              
        temper = np.exp(-GUIDANCE_LAMBDA * H)                         
        w_tr, _ = _tr_weight_and_grad_batch(X_real)           

                                                                  
        globals()["_LAST_ACQ_DIAGNOSTICS"] = {
                                                                                
            "X": np.array(X_real, copy=True),
            "H": np.array(H, copy=True),
            "H_hat": None,
            "temper": np.array(temper, copy=True),
            "w_tr": np.array(w_tr, copy=True),
            "base_hat": np.array(base_scaled, copy=True),
        }
        return base_scaled * temper * w_tr

    def _acq_tempered_with_grad(X_bo: np.ndarray):
        val_base, grad_base = _orig_acq_fun_grad(X_bo)                   
                                                                   
        s = globals().get("_ITER_SCALERS", {})
        if s.get("iter") == getattr(bo, "_acq_iter", None):
            den_base = float(s.get("base_den", 1.0))
        else:
            med = float(np.median(val_base))
            den_base = float(np.median(np.abs(val_base - med)) + SCALING_EPS)
        base_scaled      = val_base / den_base
        grad_base_scaled = grad_base / den_base

                                                                                            
        X_real = _to_real(X_bo)
        H,   gradH_real   = _h_and_grad_batch(X_real)                                 
        wtr, gradTR_real  = _tr_weight_and_grad_batch(X_real)                            
        if _span is not None:
            J = _span.reshape(1, -1)                                              
            gradH_scaled  = gradH_real  * J                                     
            gradTR_scaled = gradTR_real * J
        else:
            gradH_scaled  = gradH_real
            gradTR_scaled = gradTR_real
        temper = np.exp(-GUIDANCE_LAMBDA * H)                        
                                   
        g1 = grad_base_scaled * temper * wtr
        g2 = base_scaled * (temper * (-GUIDANCE_LAMBDA * gradH_scaled)) * wtr
        g3 = base_scaled * temper * gradTR_scaled
        grad_total = g1 + g2 + g3

                                                                                 

        globals()["_LAST_ACQ_DIAGNOSTICS"] = {
            "X": np.array(X_real, copy=True),
            "H": np.array(H, copy=True),
            "H_hat": None,
            "temper": np.array(temper, copy=True),
            "w_tr": np.array(wtr, copy=True),
            "base_hat": np.array(base_scaled, copy=True),
        }
        return (base_scaled * temper * wtr), grad_total


    bo.acquisition.acquisition_function = _acq_tempered_no_grad
    if _has_grad:
        bo.acquisition.acquisition_function_withGradients = _acq_tempered_with_grad



def _load_guidance_hints(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def _to_guidance_features_row(x: np.ndarray) -> np.ndarray:

       
    x = np.asarray(x, dtype=float).ravel()
    name_to_idx = {d["name"]: j for j, d in enumerate(_CURRENT_DOMAIN or [])}
    def g(name: str, default: float = 0.0) -> float:
        j = name_to_idx.get(name)
        return float(x[j]) if j is not None else float(default)

                         
    pattern_idx = int(round(g("pattern_idx")))
    perimeters  = int(round(g("perimeters")))
    support_mat = int(round(g("support_material")))
    seam_idx    = int(round(g("seam_idx")))
    top_solid_layers = int(round(g("top_solid_layers")))
    bottom_solid_layers = int(round(g("bottom_solid_layers")))

                       
    if "orient_pair_idx" in name_to_idx:
        orient_pair_idx = int(round(g("orient_pair_idx")))
                                                      
    else:
        print("derive orient pair idx from up idx and yaw_deg")

    feats = [
        g("layer_height"),                                     
        100.0 * g("infill_density"),                              
        float(pattern_idx),                                        
        float(perimeters),                                      
        g("filament_max_volumetric_speed"),                                  
        float(support_mat),                                       
        g("first_layer_height"),                                     
        g("first_layer_extrusion_width"),                                     
        g("elefant_foot_compensation"),                                     
        float(seam_idx),                                  
        float(orient_pair_idx),                                  
        g("brim_width"),                                  
        float(top_solid_layers),                       
        float(bottom_solid_layers),                         
    ]
                                    
                                                  
    row = np.asarray(feats, dtype=float)
    return row


def _to_hints_batch(X_bo: np.ndarray) -> np.ndarray:

    X_abs = np.asarray([_to_guidance_features_row(xi) for xi in X_bo], dtype=float)
                                    
    base = globals().get("_HINTS_BASE_VEC", None)
    hints = globals().get("_guidance_hints", None)
    if base is None or hints is None:
        return X_abs

                                                              
    name_to_idx = {nm: j for j, nm in enumerate(GUIDANCE_FEATURE_ORDER)}

                                                                          
    relative_targets = set()
    try:
        for block in (hints or []):
            for cl in block.get("clauses", []):
                rt = (cl.get("residual_type", "") or "").lower()
                if rt in _RELATIVE_RESIDUAL_TYPES:
                    for t in cl.get("targets", []):
                        if t in name_to_idx:
                            relative_targets.add(t)
    except Exception:
                                                      
        return X_abs

    if not relative_targets:
        return X_abs

    X_rel = X_abs.copy()

    def _wrap_deg(d):
        return (d + 180.0) % 360.0 - 180.0

    for t in sorted(relative_targets):
        j = name_to_idx[t]
        if t.startswith(_ANGLE_FEATURE_PREFIX):
            print("found an angle feature somewhere")
            X_rel[:, j] = _wrap_deg(X_abs[:, j] - float(base[j]))
        else:
            X_rel[:, j] = X_abs[:, j] - float(base[j])

    return X_rel


def _init_tr_from_hints_and_domain(domain: List[Dict[str, Any]]):
                                                                                
    cont_names, cont_bounds = [], []
    for d in domain:
        if d.get("type") == "continuous":
            cont_names.append(d["name"])
            lo, hi = map(float, d["domain"])
            cont_bounds.append([lo, hi])
    if not cont_names:
        _TR_STATE.update({"center": None, "half": None, "bounds": None, "names": []})
        return
    bounds = np.asarray(cont_bounds, dtype=float)
    names  = list(cont_names)

                                                                                       
    hinted_L, hinted_U = {}, {}
    try:
        for block in (_guidance_hints or []):
            for cl in block.get("clauses", []):
                if cl.get("residual_type") == "in_box":
                    targets = cl.get("targets", [])
                    if len(targets) == 1:
                        nm = targets[0]
                        params = cl.get("parameters", {})
                        if "L" in params and "U" in params:
                            L = float(params["L"]); U = float(params["U"])
                            hinted_L[nm] = min(L, hinted_L.get(nm, L))
                            hinted_U[nm] = max(U, hinted_U.get(nm, U))
    except Exception:
        pass

    center = np.zeros(len(names)); half = np.zeros(len(names))
    for i, nm in enumerate(names):
        lo, hi = bounds[i, 0], bounds[i, 1]
        if nm in hinted_L and nm in hinted_U:
            L, U = max(hinted_L[nm], lo), min(hinted_U[nm], hi)
            c = 0.5*(L+U); h = max(1e-12, 0.5*(U-L))
            if h <= 0:
                c = 0.5*(lo+hi); h = 0.2*(hi-lo)
        else:
            c = 0.5*(lo+hi); h = 0.2*(hi-lo)
        center[i] = c
        half[i]   = max(h, TR_MIN_FRAC*(hi-lo))

    _TR_STATE.update({"center": center, "half": half, "bounds": bounds, "names": names})



def _update_tr_after_step(improved: bool):
    if (not TR_ENABLED) or (_TR_STATE.get("center") is None):
        return
    bounds = _TR_STATE["bounds"]; half = _TR_STATE["half"]
    span = bounds[:,1] - bounds[:,0]
    if improved:
        _TR_STATE["fail_streak"] = 0
        half = np.minimum(half * TR_EXPAND, 0.5 * TR_MAX_FRAC * span)
    else:
        _TR_STATE["fail_streak"] += 1
        if _TR_STATE["fail_streak"] >= TR_PATIENCE:
            half = np.maximum(half * TR_SHRINK, 0.5 * TR_MIN_FRAC * span)
            _TR_STATE["fail_streak"] = 0
    _TR_STATE["half"] = half



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
    filament_max_volumetric_speed: float          
    support_material: int                      
    first_layer_height: float                 
    first_layer_extrusion_width: float        
    elefant_foot_compensation: float               
    seam_position: str                                           
    rotate_y: int                                                                
    rotate_x: int                                                                
    rotate_z: int                                                              
    brim_width: float
    top_solid_layers: int
    bottom_solid_layers: int    
    
    
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
        "--top-solid-layers", str(params.top_solid_layers),
        "--bottom-solid-layers", str(params.bottom_solid_layers),
        "--loglevel", str(2),
        "--center", str(125) + ',' + str(105),                                             

    ]
                                   
    if params.support_material:
                                                                        
        cmd += [
            "--support-material",
            "--support-material-auto",
            "--support-material-threshold", str(0),
        ]
                                            
    if config_save_path:
        cmd += ["--save", config_save_path]
                                                               

                                                                                                        
    cmdstl = cmd.copy()
    cmdstl[4] = "--export-stl"
    cmdstl[17] = mesh_export_path

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


                    
                                                                                
_MESH_PATH_CACHED: Optional[Tuple[str, Tuple[int,int], Tuple[float,float], float]] = None                                                           
_MESH_DATA_CACHED = None
_MESH_PRECOMP_CACHED = None

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


    
def _effective_param_dict_for_cache(params: SlicingParams) -> Dict[str, Any]:

    def r(x, nd=6):                              
        return round(float(x), nd)

    d: Dict[str, Any] = {
        "layer_height": r(params.layer_height),
        "infill_density": r(params.infill_density),
        "infill_pattern": params.infill_pattern,
        "perimeters": int(params.perimeters),
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
        "top_solid_layers": int(params.top_solid_layers),
        "bottom_solid_layers": int(params.bottom_solid_layers),
    }
    if params.support_material:
        d["support_material_auto"] = True
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

    def _attach_quality(metrics: Dict, gcode_path: str, config_save_path: str, mesh_export_path: str , io_dict, LOAD_BEARING,  LOAD_DIRECTION,rx_deg, ry_deg, rz_deg) -> None:

        try:
                           
            q_final, q_info = _compute_quality_final_via_new_pipeline(
                gcode_path, config_save_path, _mesh_path(mesh_export_path), io_dict, LOAD_BEARING,  LOAD_DIRECTION,rx_deg, ry_deg, rz_deg
            )
                                            
        except Exception as e:
            print(f"Failed to compute quality (new pipeline): {e}")
            q_final, q_info = float("inf"), {}

        if not np.isfinite(q_final):
            q_final = FAIL_PENALTY

        metrics["quality_J"] = float(q_final)
        metrics["quality_info"] = q_info

    key = _cache_key(params)
    rx_deg = params.rotate_x
    ry_deg = params.rotate_y
    rz_deg = params.rotate_z
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
            print(f"Slicing failed (exit {proc.returncode}):\nSTDERR:\n{proc.stderr}\n")
            return float("inf")

        if not _gcode_exists_nonempty(gcode_path):
            print(f"G-code not generated:\nSTDERR:\n{proc.stderr}\n")
            return float("inf")

        try:
            metrics = parse_gcode_metrics(gcode_path)
        except Exception:
            print("Failed to parse time/cost from G-code")
            return float("inf")

                                                          
        _attach_quality(metrics, gcode_path, config_save_path, mesh_export_path, io_dict, LOAD_BEARING,  LOAD_DIRECTION,rx_deg, ry_deg, rz_deg)

        _MEM_CACHE[key] = (gcode_path, config_save_path, mesh_export_path, metrics)

                                                                            
    if "quality_J" not in metrics:
        try:
            _attach_quality(metrics, gcode_path, config_save_path, mesh_export_path, io_dict, LOAD_BEARING,  LOAD_DIRECTION,rx_deg, ry_deg, rz_deg)
            _MEM_CACHE[key] = (gcode_path, config_save_path, mesh_export_path, metrics)
        except Exception as e:
                                                                                                    
            print(f"Failed to compute quality from cache (new pipeline): {e}")
            metrics["quality_J"] = FAIL_PENALTY

    try:
        if USE_COMBINED and len(OBJECTIVE_COMBINE) > 0:
            return objective_from_metrics_combined(metrics, OBJECTIVE_COMBINE)
        else:
            print("Objective combination not specified properly")
            return float("inf")
    except Exception:
        print("Failed to obtain objective value from parsed metrics")
        return float("inf")
                                          
INFILL_PATTERNS: List[str] = [
    "concentric",          
    "rectilinear",         
    "grid",                
    "cubic",               
    "honeycomb",           
    "gyroid",              
]

                            
SEAM_POSITIONS: List[str] = ["rear", "aligned","nearest", "random"]



def _x_to_params(x: np.ndarray) -> SlicingParams:
    x = np.asarray(x).ravel()

    layer_height                  = float(x[0])
    infill_density               = float(x[1])
    pat_idx                      = int(round(x[2]))
    perimeters                   = int(round(x[3]))
    filament_max_volumetric_speed= float(x[4])
    support_material             = int(round(x[5]))       
    first_layer_height           = float(x[6])
    first_layer_extrusion_width  = float(x[7])
    elefant_foot_compensation         = float(x[8])
    seam_idx                     = int(round(x[9]))
                                                          
    orient_pair_idx             = int(round(x[10]))
    brim_width                  = float(x[11])
    top_solid_layers            = int(round(x[12]))
    bottom_solid_layers         = int(round(x[13]))

                               
    pat_idx  = max(0, min(pat_idx,  len(INFILL_PATTERNS) - 1))
    seam_idx = max(0, min(seam_idx, len(SEAM_POSITIONS) - 1))
    perimeters = max(2, min(perimeters, 6))
    top_solid_layers = max(3, min(top_solid_layers, 10))
    bottom_solid_layers = max(3, min(bottom_solid_layers, 10))

                                                                        
    if ORIENTATION_LUT is None or ORIENT_PAIRS is None:
        raise RuntimeError("Orientation LUT not loaded or ORIENT_PAIRS missing. Load --orientation_tables earlier.")
    orient_pair_idx = max(0, min(int(orient_pair_idx), len(ORIENT_PAIRS) - 1))
    up_idx, yaw_deg = ORIENT_PAIRS[orient_pair_idx]
    rotate_y, rotate_x, rotate_z = ORIENTATION_LUT[(up_idx, yaw_deg)]



    return SlicingParams(
        layer_height=layer_height,
        infill_density=infill_density,
        infill_pattern=INFILL_PATTERNS[pat_idx],
        perimeters=perimeters,
        filament_max_volumetric_speed=filament_max_volumetric_speed,
        support_material=support_material,
        first_layer_height=first_layer_height,
        first_layer_extrusion_width=first_layer_extrusion_width,
        elefant_foot_compensation=elefant_foot_compensation,
        seam_position=SEAM_POSITIONS[seam_idx],
        rotate_y =int(rotate_y),
        rotate_x =int(rotate_x),
        rotate_z = int(rotate_z),
        brim_width = brim_width,
        top_solid_layers=top_solid_layers,
        bottom_solid_layers = bottom_solid_layers,
    )


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
    mvs0  = _get_num_from_cfg(cfg, ["filament_max_volumetric_speed"], default=11.5)

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


    mvs_min = max(3.0,  0.5 * mvs0)
    mvs_max = min(11.5, 1.0 * mvs0)                                                                                
                                                                                                            
                                                                                                                                  


    per_min, per_max = 2, max(5, min(6, per0 + 2))

    bw_min, bw_max = 0.0, 8.0

    tsl_min = min(3, int(0.7/lh0), int(0.7/lh_min))
    tsl_max = 10

    bsl_min = min(3, int(0.5/lh0), int(0.5/lh_min))
    bsl_max = 10

    

    return {
        "layer_height":                  (lh_min, lh_max),
        "infill_density":                (ild_min, ild_max),
        "perimeters":                    (per_min, per_max),
        "filament_max_volumetric_speed": (mvs_min, mvs_max),
        "first_layer_height":            (flh_min, flh_max),
        "first_layer_extrusion_width":   (flew_min, flew_max),
        "elefant_foot_compensation":     (xy_min, xy_max),
        "brim_width":                    (bw_min, bw_max),
        "top_solid_layers":              (tsl_min, tsl_max),
        "bottom_solid_layers":           (bsl_min, bsl_max),
    }


                                                     

def _extract_minmax_from_domain(domain):
                                                                                               
    lows, highs = [], []
    for d in domain:
        typ = d.get("type")
        dom = d.get("domain")
        if typ == "continuous":
            a, b = float(dom[0]), float(dom[1])
        elif typ == "discrete":
                                                              
            vals = [float(v) for v in dom]
            a, b = float(min(vals)), float(max(vals))
        else:
            raise ValueError(f"Unsupported domain type for scaling: {typ}")
        lows.append(a); highs.append(b)
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def _scale_domain_01(domain):

       
    lo, hi = _extract_minmax_from_domain(domain)
    span = np.where((hi - lo) == 0.0, 1.0, (hi - lo))

                                                                                              
    domain_scaled = []
    for d, a, b in zip(domain, lo, hi):
        if d["type"] == "continuous":
            domain_scaled.append({"name": d["name"], "type": "continuous", "domain": (0.0, 1.0)})
        elif d["type"] == "discrete":
            vals = [float(v) for v in d["domain"]]
            vals_s = [float((v - a) / (b - a) if b > a else 0.0) for v in vals]
            domain_scaled.append({"name": d["name"], "type": "discrete", "domain": tuple(vals_s)})
        else:
            raise ValueError(f"Unsupported domain type: {d['type']}")

                
    def to_real(Xs):
        Xs = np.asarray(Xs, dtype=float)
        return lo + Xs * span

    def to_scaled(Xr):
        Xr = np.asarray(Xr, dtype=float)
        return (Xr - lo) / span

    return domain_scaled, to_real, to_scaled

def _bo_objective_from_scaled_factory(to_real_fn):

       
    def _f_scaled(Xs: np.ndarray) -> np.ndarray:
        Xr = to_real_fn(Xs)                                    
        n = Xr.shape[0]
        vals = np.empty((n, 1), dtype=float)
        for i in range(n):
            params = _x_to_params(Xr[i])
            y = float(evaluate_params(params))
            if not np.isfinite(y):
                y = FAIL_PENALTY
            vals[i, 0] = y
        return vals
    return _f_scaled


def build_domain_from_profile(ini_path: str) -> List[Dict[str, Any]]:
    cfg = parse_config_ini(ini_path)
    B   = _profile_aware_bounds(cfg)

    return [
        {"name": "layer_height",      "type": "continuous", "domain": B["layer_height"]},
        {"name": "infill_density",    "type": "continuous", "domain": B["infill_density"]},
        {"name": "pattern_idx",       "type": "discrete",   "domain": tuple(range(len(INFILL_PATTERNS)))},
        {"name": "perimeters",        "type": "discrete",   "domain": tuple(range(int(B["perimeters"][0]),
                                                                                   int(B["perimeters"][1]) + 1))},
        {"name": "filament_max_volumetric_speed", "type": "continuous", "domain": B["filament_max_volumetric_speed"]},
        {"name": "support_material",  "type": "discrete",   "domain": (0, 1)},

        {"name": "first_layer_height","type": "continuous", "domain": B["first_layer_height"]},
        {"name": "first_layer_extrusion_width","type": "continuous", "domain": B["first_layer_extrusion_width"]},
        {"name": "elefant_foot_compensation","type": "continuous", "domain": B["elefant_foot_compensation"]},
        {"name": "seam_idx",          "type": "discrete",   "domain": tuple(range(len(SEAM_POSITIONS)))},
        {"name": "orient_pair_idx", "type": "discrete", "domain": tuple(range(len(ORIENT_PAIRS or [])))},
        {"name": "brim_width",        "type": "continuous", "domain": B["brim_width"]},
        {"name": "top_solid_layers",        "type": "discrete",   "domain": tuple(range(int(B["top_solid_layers"][0]),
                                                                                   int(B["top_solid_layers"][1]) + 1))},
        {"name": "bottom_solid_layers",        "type": "discrete",   "domain": tuple(range(int(B["bottom_solid_layers"][0]),
                                                                                   int(B["bottom_solid_layers"][1]) + 1))},
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



                                                                                 

def _sample_uniform_from_domain(domain: List[dict], n: int, seed: int | None = None) -> np.ndarray:

       
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(n):
        row = []
        for d in domain:
            typ = d.get("type")
            if typ == "continuous":
                a, b = d["domain"]
                row.append(rng.uniform(float(a), float(b)))
            elif typ == "discrete":
                                                              
                choices = list(d["domain"])
                row.append(float(choices[rng.integers(0, len(choices))]))
            else:
                                                                          
                dom = d.get("domain")
                if isinstance(dom, (list, tuple)) and len(dom) == 2:
                    a, b = dom
                    row.append(rng.uniform(float(a), float(b)))
                else:
                    raise ValueError(f"Unsupported domain entry: {d}")
        X.append(row)
    return np.asarray(X, dtype=float)

def _predict_mean_var_robust(bo, X: np.ndarray):
                                                                                  
    try:
        m, v = bo.model.predict(X);  return np.asarray(m).reshape(-1), np.asarray(v).reshape(-1)
    except Exception:
        pass
    try:
        m, v = bo.model.model.predict(X);  return np.asarray(m).reshape(-1), np.asarray(v).reshape(-1)
    except Exception:
        pass
                                                                
    return np.full(X.shape[0], np.nan), np.full(X.shape[0], np.nan)




def snapshot_acq_and_var(bo, domain: List[dict], out_dir: str, domain_names: list[str],
                         iter_idx: int, nsamples: int = 512, seed: int | None = None,
                         to_real_fn=None, context: dict | None = None) -> None:                      
                                                          
    gp_submodel = getattr(getattr(bo, "model", None), "model", None)
    if gp_submodel is None:
        print(f"[snapshot] iter={iter_idx}: GP model not initialized yet; skipping snapshot.")
                                                                                                   
        return
                                                            
    try:
        _orig = getattr(bo, "_orig_acq_fun", None)
        if _orig is not None:
                                                                                        
            Xs_scaled = _sample_uniform_from_domain(domain, nsamples, seed=seed)
                                                                         
            if context:
                name2col = {nm: j for j, nm in enumerate(domain_names)}
                for nm, val in context.items():
                    j = name2col.get(nm)
                    if j is not None:
                        Xs_scaled[:, j] = float(val)
            base_raw = np.asarray(_orig(Xs_scaled)).reshape(-1)
            b_med = float(np.median(base_raw))
            b_mad = float(np.median(np.abs(base_raw - b_med)))
            globals()["_ITER_SCALERS"] = {"iter": int(iter_idx), "base_den": b_mad + SCALING_EPS}
    except Exception:
        pass


    Xs_scaled = _sample_uniform_from_domain(domain, nsamples, seed=seed)
    if context:
        name2col = {nm: j for j, nm in enumerate(domain_names)}
        for nm, val in context.items():
            j = name2col.get(nm)
            if j is not None:
                Xs_scaled[:, j] = float(val)
                                                
    vals = bo.acquisition.acquisition_function(Xs_scaled)
    vals = np.asarray(vals).reshape(-1)
                        


    mean, var = _predict_mean_var_robust(bo, Xs_scaled)


                                    
    uniq = np.unique(np.round(vals, decimals=12)).size
    print(f"[acq-snapshot] iter={iter_idx} nsamples={nsamples} "
          f"min={vals.min():.6g} max={vals.max():.6g} mean={vals.mean():.6g} "
          f"std={vals.std():.6g} unique≈{uniq}")
    if np.isfinite(var).any():
        print(f"[var-snapshot]  iter={iter_idx} var_min={np.nanmin(var):.6g} "
              f"var_max={np.nanmax(var):.6g} var_mean={np.nanmean(var):.6g}")
                       
       

def dump_kernel_hypers(bo, out_dir: str, iter_idx: int):
                                                          
    try:
        m = bo.model.model             
        kern = getattr(m, 'kern', None)
        if kern is None:
            return
                                                 
        ls = getattr(kern, 'lengthscale', None)
        var = getattr(kern, 'variance', None)
        noise = getattr(getattr(m, 'Gaussian_noise', None), 'variance', None)

        def _to_list(p):
            try:
                return [float(x) for x in np.asarray(p.values).ravel()]
            except Exception:
                return None

        row = {
            "iter": iter_idx,
            "lengthscale": _to_list(ls),
            "signal_variance": float(var.values[0]) if var is not None else None,
            "noise_variance": float(noise.values[0]) if noise is not None else None,
        }
        path = os.path.join(out_dir, "kernel_hypers.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print("couldn't dump kernel hyperparams because: ", e)
        pass                                                    



def install_acquisition_probe(bo, out_dir: str, domain_names: list[str]):

    os.makedirs(out_dir, exist_ok=True)

                         
    bo._acq_iter = 0
    bo._acq_probe_buffer = []                                        

    D_expected = len(domain_names)

    def _record_batch(x, a_values):
                                                                                
        x_arr = np.asarray(x)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim > 2:
            x_arr = x_arr.reshape(x_arr.shape[0], -1)

        a_arr = np.asarray(a_values)
                                            
        if a_arr.ndim == 0:
            a_arr = np.repeat(a_arr, x_arr.shape[0])
        else:
            a_arr = a_arr.reshape(-1)
            if a_arr.size == 1 and x_arr.shape[0] > 1:
                a_arr = np.repeat(a_arr[0], x_arr.shape[0])

        for i in range(x_arr.shape[0]):
            xi = x_arr[i].ravel()
                                                                            
            if xi.size != D_expected:
                if xi.size > D_expected:
                    xi = xi[:D_expected]
                else:
                    xi = np.pad(xi, (0, D_expected - xi.size), mode="constant")
            try:
                vi = float(np.asarray(a_arr[i]).squeeze())
            except Exception:
                vi = None
            bo._acq_probe_buffer.append((xi.astype(float), vi))

    def _dump_buffer():
        it = getattr(bo, "_acq_iter", 0)
        buf = getattr(bo, "_acq_probe_buffer", None)
        if it <= 0 or not buf:
            return
        path = os.path.join(out_dir, f"acq_iter_{it}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "acq_value"] + list(domain_names))
            for j, (xj, vj) in enumerate(buf):
                w.writerow([j, vj] + [float(t) for t in np.asarray(xj).ravel()])
                                  
        bo._acq_probe_buffer = []

                                                                          
    acq = bo.acquisition

    if hasattr(acq, "acquisition_function"):
        _orig_acq = acq.acquisition_function
        def _acq_wrapped(x):
            vals = _orig_acq(x)
            _record_batch(x, vals)
            return vals
        acq.acquisition_function = _acq_wrapped

    if hasattr(acq, "acquisition_function_withGradients"):
        _orig_acqg = acq.acquisition_function_withGradients
        def _acqg_wrapped(x):
            vals, grads = _orig_acqg(x)
            _record_batch(x, vals)
            return vals, grads
        acq.acquisition_function_withGradients = _acqg_wrapped

                                                          
    acq_opt = bo.acquisition_optimizer

    def _wrap_opt(method_name: str):
        if not hasattr(acq_opt, method_name):
            return
        _orig = getattr(acq_opt, method_name)
        def _wrapped(f, *args, **kwargs):
                                                        
            bo._acq_probe_buffer = []
            res = _orig(f, *args, **kwargs)
                                                                                
            _dump_buffer()
                                                     
            return res
        setattr(acq_opt, method_name, _wrapped)

    for _name in ("optimize", "optimize_batch", "_optimize", "_optimize_batch"):
        _wrap_opt(_name)





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
        default="./logs_my_dovetail_context_Bp7_unfreeze_at4/llm_guidance/",
        help="Root directory where run-specific outputs are created.",
    )
    p.add_argument(
        "--refs_store",
        type=str,
        default="./logs_my_dovetail_context_Bp7_unfreeze_at4/refs_store.json",
        help="Path to refs_store.json produced by compute_refs_persist.py",
    )
    p.add_argument(
        "--require_refs",
        action="store_true",
        help="If set, abort if persisted refs are missing instead of computing them.",
    )
    p.add_argument(
        "--guidance_hints_dir",
        type=str,
        default=None,
        help="Directory containing guidance JSON hint files. If omitted, uses GUIDANCE_HINTS_DIR.",
    )

    p.add_argument(
        "--orientation_tables", type=str,
        default=str(Path(__file__).with_name("orientation_tables_prusa_30deg.npz")),
        help="Path to .npz with orientation tables (ups, yaws, lut_keys, lut_vals).",
    )

    p.add_argument(
        "--load_bearing",
        action="store_true",
        help="If set, abort if persisted refs are missing instead of computing them.",
    )
    
    p.add_argument(
        "--load_direction",
        type=str,
        default='z',  
        help="String label for load direction, passed into the quality computation.",
    )

    p.add_argument(
        "--use_dynamic_llm_cache",
        action="store_true",
        help="Enable dynamic cache: on signature miss, call live LLM once and cache the result keyed by signature.",
    )
    p.add_argument(
        "--dynamic_llm_cache_path",
        type=str,
        default=None,
        help=f"Path to dynamic LLM cache JSON (default: {DYNAMIC_LLM_CACHE_PATH}).",
    )
    p.add_argument(
        "--live_llm_model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model name (overrides env LIVE_LLM_MODEL).",
    )
    p.add_argument(
        "--live_llm_system_file",
        type=str,
        default=None,
        help="Path to system prompt file for live calls (overrides env LIVE_LLM_SYSTEM_FILE).",
    )
    p.add_argument(
        "--live_llm_backend",
        type=str,
        default="openai",
        choices=["openai"],
        help="Live LLM backend (overrides env LIVE_LLM_BACKEND). Only supported: openai.",
    )
    p.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (overrides env OPENAI_API_KEY). Required when --live_llm_backend=openai unless OPENAI_API_KEY is already set.",
    )

    p.add_argument(
        "--no_dynamic_llm_cache",
        action="store_true",
        help="Disable dynamic cache (overrides env USE_DYNAMIC_LLM_CACHE).",
     )

    p.add_argument(
        "--live_llm_reasoning_effort",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="OpenAI reasoning effort (overrides env LIVE_LLM_REASONING_EFFORT). "
             "If not 'none', sampling params (temperature/top_p) will be omitted.",
    )
    p.add_argument(
        "--live_llm_text_verbosity",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="OpenAI text verbosity (overrides env LIVE_LLM_TEXT_VERBOSITY).",
    )
    return p.parse_args()



def main():

    global _LOGGER, _RUN_ID, GUIDANCE_LAMBDA
    
    global MODEL_STL, PROFILE_INI, OUTPUT_DIR, CONFIG_SAVE_DIR, GCODE_OUT_DIR, MESH_OUT_DIR   

    global _BEST_QUALITY_INFO, _BEST_HINTS_BASE_VEC, _BEST_PARAMS, _BEST_OBJECTIVE_VALUE

    global _HINTS_BASE_VEC

    global  LOAD_BEARING, LOAD_DIRECTION

    
                                           
    _RUN_ID = uuid.uuid4().hex
    args = _parse_args()
    seed = args.seed


                                                             
    global USE_DYNAMIC_LLM_CACHE, DYNAMIC_LLM_CACHE_PATH
    global LIVE_LLM_MODEL, LIVE_LLM_SYSTEM_FILE, LIVE_LLM_BACKEND
    global LIVE_LLM_TEMPERATURE, LIVE_LLM_TOP_P, LIVE_LLM_MAX_NEW_TOKENS
    global LIVE_LLM_REASONING_EFFORT, LIVE_LLM_TEXT_VERBOSITY

    if getattr(args, "use_dynamic_llm_cache", False):
        USE_DYNAMIC_LLM_CACHE = True
    if getattr(args, "dynamic_llm_cache_path", None):
        DYNAMIC_LLM_CACHE_PATH = args.dynamic_llm_cache_path

    if getattr(args, "no_dynamic_llm_cache", False):
        USE_DYNAMIC_LLM_CACHE = False

                          
    if getattr(args, "openai_api_key", None):
        os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)
    if getattr(args, "live_llm_reasoning_effort", None) is not None:
        LIVE_LLM_REASONING_EFFORT = str(args.live_llm_reasoning_effort).strip().lower()
    if getattr(args, "live_llm_text_verbosity", None) is not None:
        LIVE_LLM_TEXT_VERBOSITY = str(args.live_llm_text_verbosity).strip().lower()

    if getattr(args, "live_llm_model", None):
        LIVE_LLM_MODEL = args.live_llm_model
    if getattr(args, "live_llm_system_file", None):
        LIVE_LLM_SYSTEM_FILE = args.live_llm_system_file
    if getattr(args, "live_llm_backend", None):
        LIVE_LLM_BACKEND = str(args.live_llm_backend).lower()
    if getattr(args, "live_llm_temperature", None) is not None:
        LIVE_LLM_TEMPERATURE = float(args.live_llm_temperature)
    if getattr(args, "live_llm_top_p", None) is not None:
        LIVE_LLM_TOP_P = float(args.live_llm_top_p)
    if getattr(args, "live_llm_max_new_tokens", None) is not None:
        LIVE_LLM_MAX_NEW_TOKENS = int(args.live_llm_max_new_tokens)


    MODEL_STL = args.model_stl         
    PROFILE_INI = args.profile_ini
    LOAD_BEARING = args.load_bearing
    LOAD_DIRECTION = args.load_direction
    
    if getattr(args, "guidance_hints_dir", None) is not None:
        GUIDANCE_HINTS_DIR = args.guidance_hints_dir


    OUTPUT_DIR  = os.path.join(args.output_root_dir, str(_RUN_ID)) + "/"                       
    CONFIG_SAVE_DIR = os.path.join(OUTPUT_DIR, "configs")   
    GCODE_OUT_DIR = os.path.join(OUTPUT_DIR, "gcodes")     
    MESH_OUT_DIR = os.path.join(OUTPUT_DIR, "meshes")
    
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    os.makedirs(GCODE_OUT_DIR, exist_ok=True)
    os.makedirs(MESH_OUT_DIR, exist_ok=True)    

    ACQ_LOG_DIR = os.path.join(OUTPUT_DIR, "acq_logs")
    os.makedirs(ACQ_LOG_DIR, exist_ok=True)

                                                        
    try:
        _load_orientation_tables_npz(args.orientation_tables)
    except Exception as e:
        raise RuntimeError(f"Failed to load orientation tables ({args.orientation_tables}): {e}")
    
    domain = build_domain_from_profile(PROFILE_INI)
    
                                                                              
    domain_scaled, to_real, to_scaled = _scale_domain_01(domain)
    f_scaled = _bo_objective_from_scaled_factory(to_real)
    D = len(domain_scaled)
                                                            
    kern_warm = GPy.kern.Matern52(input_dim=D, ARD=True)
                                                        
    kern_warm.variance.constrain_bounded(1e-6, 10.0)
    
    is_discrete = np.array([d.get("type") == "discrete" for d in domain], dtype=bool)
    kern_warm.lengthscale[is_discrete].constrain_bounded(0.2, 0.7)
    kern_warm.lengthscale[~is_discrete].constrain_bounded(0.03, 0.7)

    
    print(kern_warm)



                                                                                     
    global _CURRENT_DOMAIN
    _CURRENT_DOMAIN = domain
    _refresh_cont_indices()

    np.random.seed(seed)

    _LOGGER = BOLogger(
        csv_path=os.path.join(OUTPUT_DIR, f"bo_gpt_guidance_{_RUN_ID}.csv"),
        run_id=_RUN_ID,
        optimizer="gpt_guidance",
        seed=seed,
    )
    
                                                              
    if USE_DOMAIN_BASELINE_SCALING and not REFS_FROZEN:
        loaded = _try_load_persisted_refs(args.refs_store)
        if args.require_refs and not loaded:
            raise RuntimeError("Persisted refs required (--require_refs) but not found in refs_store.")

                                                                                                         

    bo_warm = GPyOpt.methods.BayesianOptimization(
        f=f_scaled,
        domain=domain_scaled,
        initial_design_numdata=INITIAL_DESIGN,
        initial_design_type=INITIAL_DESIGN_TYPE,
        acquisition_type='EI',
        exact_feval=True,
        normalize_Y=True,
        de_duplication=True,
        jitter = 0,
        kernel = kern_warm
    )

                                                                                       
    bo_warm.run_optimization(max_iter=1, eps=0)

                                                                   
                                                                                
    if USE_DOMAIN_BASELINE_SCALING and not REFS_FROZEN:
        if args.require_refs:
            raise RuntimeError("Persisted refs were not loaded and --require_refs forbids legacy compute.")
                                               

                                                                
                                                                               
    X_all_scaled = np.asarray(bo_warm.X)
    X_warm_scaled = np.asarray(X_all_scaled[:INITIAL_DESIGN], dtype=float)
    X_warm_real = to_real(X_warm_scaled)
    Y_warm = np.array(
        [float(evaluate_params(_x_to_params(xr))) for xr in X_warm_real],
        dtype=float
    )
    Y_warm[~np.isfinite(Y_warm)] = FAIL_PENALTY
    sig = hashlib.sha256(np.asarray(X_warm_scaled, float).tobytes()).hexdigest()
    print("X_warm_sha256:", sig)
    
                                                                                      
    
    for k, xw_s in enumerate(X_warm_scaled):
                                                               
        xw = to_real(xw_s[None, :])[0]
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
                                                                  
            x_vec = X_warm_real[k]
            try:
                H_now = float(_h_eval_only_batch(np.atleast_2d(x_vec))[0, 0])
            except Exception:
                H_now = None
                                                                                       
            H_hat_now = None
            try:
                diag = globals().get("_LAST_ACQ_DIAGNOSTICS", {})
                            
                Xb = diag.get("X"); Hhat_b = diag.get("H_hat")
                if Xb is not None and Hhat_b is not None:
                    j = int(np.argmin(np.sum((Xb - np.atleast_2d(x_vec))**2, axis=1)))
                    if np.allclose(Xb[j], x_vec, atol=1e-8):
                        H_hat_now = float(Hhat_b[j, 0])
            except Exception:
                pass
            if H_hat_now is None:
                H_hat_now = H_now                                                    
            try:
                w_tr_now = float(_tr_weight_and_grad_batch(np.atleast_2d(x_vec))[0][0, 0])
            except Exception:
                w_tr_now = None
                                                                                  
            temper_now = None
            try:
                diag = globals().get("_LAST_ACQ_DIAGNOSTICS", {})
                Xb = diag.get("X"); temper_b = diag.get("temper")
                if Xb is not None and temper_b is not None:
                    j = int(np.argmin(np.sum((Xb - np.atleast_2d(x_vec))**2, axis=1)))
                    if np.allclose(Xb[j], x_vec, atol=1e-8):
                        temper_now = float(temper_b[j, 0])
            except Exception:
                pass
                                                                     
            if temper_now is None and H_hat_now is not None:
                temper_now = float(np.exp(-GUIDANCE_LAMBDA * H_hat_now))

            try:
                active_hints = [s.get("name") or s.get("path") for s in (_H_SOURCES or [])]
            except Exception:
                active_hints = None
            _LOGGER.log_eval(
                iter_idx=-(len(X_warm_real) - k),                                     
                objective=y_w,
                time_s=time_s,
                cost=cost,
                quality_J=qJ,
                quality_info=qInfo,
                slicer_params = _params_to_dict(params_w),
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
                guidance_lambda=GUIDANCE_LAMBDA,
                H=H_now,
                H_hat=H_hat_now,
                temper=temper_now,
                w_tr=w_tr_now,
                active_hints=active_hints,
            )

                                                                                           
    try:
        if len(Y_warm) > 0:
                                                                                          
            idx_best_warm = int(np.nanargmin(np.asarray(Y_warm, dtype=float)))
            x_best_now = np.asarray(X_warm_real[idx_best_warm], dtype=float)
            _BEST_HINTS_BASE_VEC = _to_guidance_features_row(x_best_now)
            globals()["_HINTS_BASE_VEC"] = np.asarray(_BEST_HINTS_BASE_VEC, dtype=float)
                                                            
            best_params = _x_to_params(x_best_now)
            best_key = _cache_key(best_params)
            _, _, _, best_metrics = _MEM_CACHE.get(best_key, (None, None, None, {}))
            _BEST_QUALITY_INFO = (best_metrics or {}).get("quality_info", {}) if isinstance(best_metrics, dict) else {}
                                        
                                                                                                                          
            _BEST_PARAMS = x_best_now
            _BEST_OBJECTIVE_VALUE = Y_warm[idx_best_warm]
    except Exception as _e:
        print("Warm-up best-anchor init failed (non-fatal):", _e)
    

                                                                                         
    global _soft_violation, _soft_violation_batch, _guidance_hints
    _soft_violation = None
    _soft_violation_batch = None
    _guidance_hints = None
    try:
        if build_soft_violation is None:
            raise RuntimeError("soft_violation_from_hints.build_soft_violation not available")
        warm_last_key = _cache_key(_x_to_params(X_warm_real[-1]))
        _, _, _, warm_last_metrics = _MEM_CACHE.get(warm_last_key, (None, None, None, {}))
        warm_info = (warm_last_metrics or {}).get("quality_info", {}) if isinstance(warm_last_metrics, dict) else {}
                                                                                           
        src_info = (_BEST_QUALITY_INFO or {}) if isinstance(_BEST_QUALITY_INFO, dict) else {}
                    # Guidance generator: select next action(s) from diagnostics

        first_json = choose_hints_json_from_info(src_info or warm_info)
        print(f"[guidance] source={'BEST' if (src_info and isinstance(src_info, dict)) else 'WARM_LAST'}; keys={list((((src_info or warm_info) or {}).get('messages') or {}).keys())}")
        _reload_hints(first_json, domain)
    except Exception as e:
        print("[warn] Guidance not available; running without cEI/TR. ", e)


                                                                                     
    kern = GPy.kern.Matern52(input_dim=D, ARD=True)
                                                   
    kern.variance.constrain_bounded(1e-6, 10.0)
    kern.lengthscale[is_discrete].constrain_bounded(0.2, 0.7)
    kern.lengthscale[~is_discrete].constrain_bounded(0.03, 0.7)
    
    print(kern)

    
    bo = GPyOpt.methods.BayesianOptimization(
        f=f_scaled,
        domain=domain_scaled,
        X=X_warm_scaled, Y=Y_warm.reshape(-1, 1),
        acquisition_type='EI',
        exact_feval=True,
        normalize_Y=True,
        de_duplication=True,
        jitter = 0,
        kernel = kern
    )


                                    
                                   
                                    
    
                                                                          
    install_tempered_acquisition(bo, to_real)


                                                                        
                                                                       
    domain_names = [
        "layer_height",
        "infill_density",
        "pattern_idx",
        "perimeters",
        "filament_max_volumetric_speed",
        "support_material",
        "first_layer_height",
        "first_layer_extrusion_width",
        "elefant_foot_compensation",
        "seam_idx",
        "orient_pair_idx",
        "brim_width",
        "top_solid_layers",
        "bottom_solid_layers",
    ]

    install_acquisition_probe(bo, ACQ_LOG_DIR, domain_names)


                                      

    max_iter = MAX_ITER
    no_improvement_limit = 100
    improvement_threshold = 1e-4

                                                      
    idx_best_warm = int(np.argmin(Y_warm))
    best_so_far = float(Y_warm[idx_best_warm])
    best_values = [best_so_far]
    no_improvement_counter = 0

                                                                         
    x_best_warm = np.asarray(X_warm_real)[idx_best_warm]
    best_params = _x_to_params(x_best_warm)
    key_best_warm = _cache_key(best_params)
    best_gcode, best_config, best_stl, _ = _MEM_CACHE.get(key_best_warm, (None, None, None, {}))

                                                                           
    x_anchor_real = np.asarray(x_best_warm, dtype=float)
                                                                                     


                                                                                                           
    VETO_DIM_NAMES = {"brim_width", "support_material"}
    def _toggle_veto_defaults_real(x_real, domain):

           
        x_new = np.asarray(x_real, dtype=float).copy()
        lo, hi = _extract_minmax_from_domain(domain)
        name_to_idx = {d["name"]: j for j, d in enumerate(domain)}
                           
        j = name_to_idx.get("brim_width")
        if j is not None:
            x_new[j] = float(lo[j] + hi[j])/2
                                          
        j = name_to_idx.get("support_material")
        if j is not None:
            x_new[j] = 1.0

        return x_new

                                      
    pending_veto_toggle = False
    _last_fail_x_real = None


                           
    for i in range(max_iter):
        bo._acq_iter = i+1
                                                                                
                                                                                  

                                                                                         
                                                                   
                                                                                
        ctx_for_iter = _build_context_scaled(
            active_hints=_guidance_hints,
            domain_scaled=domain_scaled,
            to_scaled=to_scaled,
            x_anchor_real=x_anchor_real,
        )

        snapshot_acq_and_var(
            bo=bo,
            domain=domain_scaled,
            out_dir=ACQ_LOG_DIR,
            domain_names=domain_names,
            iter_idx=bo._acq_iter,
            nsamples=512,
            seed=seed + i,
            to_real_fn=to_real,
            context=ctx_for_iter,
        )



                                                                                      
        dump_kernel_hypers(bo, ACQ_LOG_DIR, bo._acq_iter)

        

                                                                                              
        if pending_veto_toggle and _last_fail_x_real is not None:
            print(f"Iteration {bo._acq_iter} is repair path")
                                                                                           
            x_last_real = _toggle_veto_defaults_real(_last_fail_x_real, domain)
            x_last_scaled = to_scaled(np.atleast_2d(x_last_real))[0]
            params_last = _x_to_params(x_last_real)
            t_suggest0 = t_suggest1 = time.time()                            
            t_eval0 = time.time()
            y_last = float(evaluate_params(params_last))
            if not np.isfinite(y_last):
                y_last = FAIL_PENALTY
            t_eval1 = time.time()
                                                                                                  
            try:
                X_new = np.atleast_2d(x_last_scaled)
                Y_new = np.atleast_2d([y_last])
                bo.X = np.vstack([np.asarray(bo.X), X_new])
                bo.Y = np.vstack([np.asarray(bo.Y), Y_new])
                bo.model.updateModel(bo.X, bo.Y, X_new, Y_new)
            except Exception as _e:
                print(f"[warn] Could not inject veto-toggled observation into model: {_e}")
                                      
            pending_veto_toggle = False
            _last_fail_x_real = None
        else:
                                            

                                                                                      
            ctx = ctx_for_iter

            t_suggest0 = time.time()
            bo.run_optimization(max_iter=1, eps=0, context=ctx)

            t_suggest1 = time.time()
            x_last_scaled = np.asarray(bo.X[-1],dtype=float)
            x_last_real = to_real(x_last_scaled[None, :])[0]
            params_last = _x_to_params(x_last_real)
            t_eval0 = time.time()
            y_last = float(evaluate_params(params_last))
            if not np.isfinite(y_last):
                y_last = FAIL_PENALTY
            t_eval1 = time.time()
                                                                                                               
            if y_last >= float(FAIL_PENALTY) - 1e-9:
                reorient_hint_active = False
                try:
                    reorient_hint_active = any(
                        str(src.get("name", "")).lower().startswith("reorient")
                        for src in (_H_SOURCES or [])
                    )
                except Exception:
                    reorient_hint_active = False

                if reorient_hint_active:
                    pending_veto_toggle = True
                    _last_fail_x_real = np.asarray(x_last_real, dtype=float)


                                        
        if _LOGGER is not None:
            key = _cache_key(params_last)
            gcode_path, cfg_path, mesh_path, metrics = _MEM_CACHE.get(key, (None, None, None, {}))
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
        
                                                                    
            x_vec = x_last_real
            try:
                H_now = float(_h_eval_only_batch(np.atleast_2d(x_vec))[0, 0])
            except Exception:
                H_now = None
                                                                                       
            H_hat_now = None
            try:
                diag = globals().get("_LAST_ACQ_DIAGNOSTICS", {})
                            
                Xb = diag.get("X"); Hhat_b = diag.get("H_hat")
                if Xb is not None and Hhat_b is not None:
                    j = int(np.argmin(np.sum((Xb - np.atleast_2d(x_vec))**2, axis=1)))
                    if np.allclose(Xb[j], x_vec, atol=1e-8):
                        H_hat_now = float(Hhat_b[j, 0])
            except Exception:
                pass
            if H_hat_now is None:
                H_hat_now = H_now                                                    
            try:
                w_tr_now = float(_tr_weight_and_grad_batch(np.atleast_2d(x_vec))[0][0, 0])
            except Exception:
                w_tr_now = None
                                                                                  
            temper_now = None
            try:
                diag = globals().get("_LAST_ACQ_DIAGNOSTICS", {})
                Xb = diag.get("X"); temper_b = diag.get("temper")
                if Xb is not None and temper_b is not None:
                    j = int(np.argmin(np.sum((Xb - np.atleast_2d(x_vec))**2, axis=1)))
                    if np.allclose(Xb[j], x_vec, atol=1e-8):
                        temper_now = float(temper_b[j, 0])
            except Exception:
                pass
                                                                     
            if temper_now is None and H_hat_now is not None:
                temper_now = float(np.exp(-GUIDANCE_LAMBDA * H_hat_now))

            try:
                active_hints = [s.get("name") or s.get("path") for s in (_H_SOURCES or [])]
            except Exception:
                active_hints = None
            _LOGGER.log_eval(
                iter_idx=i + 1,                                                
                objective=y_last,                                                                                           
                time_s=time_s,
                cost=cost,
                quality_J=qJ,
                quality_info=qInfo,
                slicer_params = _params_to_dict(params_last),
                gcode_path=gcode_path,
                config_path=cfg_path,
                mesh_path=mesh_path,
                suggest_secs=(t_suggest1 - t_suggest0),
                eval_secs=(t_eval1 - t_eval0),
                total_secs=(t_suggest1 - t_suggest0) + (t_eval1 - t_eval0),
                use_domain_baseline_scaling=USE_DOMAIN_BASELINE_SCALING,
                time_ref=TIME_REF,
                cost_ref=COST_REF,
                quality_ref=QUALITY_REF,
                use_combined=bool(USE_COMBINED and len(OBJECTIVE_COMBINE) > 0),
                objective_kind=("combined" if USE_COMBINED else "not specified"),
                objective_combine_pairs=(OBJECTIVE_COMBINE if USE_COMBINED else None),
                                                          
                guidance_lambda=GUIDANCE_LAMBDA,
                H=H_now,
                H_hat=H_hat_now,
                temper=temper_now,
                w_tr=w_tr_now,
                active_hints=active_hints,
            )



        t_eval1 = time.time()
                                                        
        improved_true = (best_so_far - y_last) > improvement_threshold

                                   
        if improved_true:
            best_so_far = y_last
            no_improvement_counter = 0
            key = _cache_key(params_last)
            gcode_path, cfg_path, mesh_path, metrics = _MEM_CACHE.get(key, (None, None, None, {}))
            best_gcode = gcode_path
            best_config = cfg_path
            best_stl = mesh_path
                                                                               
            try:
                x_best_scaled_now = np.asarray(bo.x_opt, dtype=float)                   
                x_best_now = to_real(x_best_scaled_now[None,:])[0]
                globals()["_BEST_HINTS_BASE_VEC"] = _to_guidance_features_row(x_best_now)
            except Exception as _e:
                print("Fail safe (best baseline map):", _e)
            try:
                best_info = (metrics or {}).get("quality_info", {}) if isinstance(metrics, dict) else {}
                globals()["_BEST_QUALITY_INFO"] = best_info
                globals()["_BEST_PARAMS"] = x_best_now
                globals()["_BEST_OBJECTIVE_VALUE"] = best_so_far
            except Exception as _e:
                print("Fail safe (stash best info):", _e)
        else:
            no_improvement_counter+= 1

                                                                   
                                                                                     
        try:
                                                                                                            
            x_last_vec = _to_guidance_features_row(np.asarray(x_last_real, dtype=float))
            x_best_vec = globals().get("_BEST_HINTS_BASE_VEC", None)
            if H_BASELINE_MODE == "best" and x_best_vec is not None:
                base_vec = x_best_vec
            elif H_BASELINE_MODE == "blend" and x_best_vec is not None:
                a = float(H_BASELINE_BLEND_ALPHA)
                base_vec = a * x_best_vec + (1.0 - a) * x_last_vec
            else:
                base_vec = x_last_vec
            globals()["_HINTS_BASE_VEC"] = np.asarray(base_vec, dtype=float)
        except Exception as _e:
            print("Fail safe (set _HINTS_BASE_VEC):", _e)

        try:
            use_best = (H_BASELINE_MODE in ("best", "blend")) and (globals().get("_BEST_QUALITY_INFO") is not None)
            if use_best:
                                                                
                src_info = globals().get("_BEST_QUALITY_INFO", {}) or {}
            else:
                                                             
                last_key = _cache_key(params_last)
                _, _, _, last_metrics = _MEM_CACHE.get(last_key, (None, None, None, {}))
                src_info = (last_metrics or {}).get("quality_info", {}) if isinstance(last_metrics, dict) else {}
                        # Guidance generator: select next action(s) from diagnostics

            next_json = choose_hints_json_from_info(src_info)
            print(f"[guidance] source={'BEST' if use_best else 'LAST'}; keys={list(((src_info or {}).get('messages') or {}).keys())}")
            _reload_hints(next_json, domain)
                                                                          
                                                                    
                                                                      
        except Exception as e:
            print(f"[warn] Could not reload next-iteration guidance: {e}")

                                             
        _update_tr_after_step(improved=improved_true)

        best_values.append(best_so_far)
        x_best_scaled = bo.x_opt
        x_best = to_real(x_best_scaled[None, :])[0]
                                                                                   
        x_anchor_real = np.asarray(x_best, dtype=float)

                                                                          
        if no_improvement_counter == 0:
                                           
            GUIDANCE_LAMBDA = min(2.0, GUIDANCE_LAMBDA + 0.1)
        elif no_improvement_counter >= 5 and (no_improvement_counter % 5) == 0:
                                                                   
            GUIDANCE_LAMBDA = max(0.2, GUIDANCE_LAMBDA - 0.2)
        
        print(f"Iter {i+1}: last f(x) = {y_last:.6f}, x_last = {_x_to_params(x_last_real)}")
        print(f"Iter {i+1}: best_so_far = {best_so_far:.6f}, x_best = {_x_to_params(x_best)}")

        if no_improvement_counter >= no_improvement_limit:
            print(f"Stopped early after {i+1} iterations (no improvement in {no_improvement_limit})")
            break



    x_best = bo.x_opt
    x_best_scaled = np.asarray(bo.x_opt, dtype=float)                         
    x_best = to_real(x_best_scaled[None, :])[0]                             
    f_best = float(bo.fx_opt)
    best_params = _x_to_params(x_best)

    if USE_COMBINED:
        result = {
            "best_params": _effective_param_dict_for_cache(best_params),
            "best_objective": f_best,
            "objective": {
                "use_combined": USE_COMBINED,
                "objective_combine": OBJECTIVE_COMBINE,
                "scaling": {
                    "use_domain_baseline": USE_DOMAIN_BASELINE_SCALING,
                    "time_ref_seconds": TIME_REF,
                    "cost_ref_dollars": COST_REF,
                    "quality_ref": QUALITY_REF
                },
                "best_file_paths": {
                    "best_gcode": best_gcode,
                    "best_config": best_config,
                    "best_stl": best_stl
                }
            },
        }

    else:
        result = {}
    print(json.dumps(result, indent=2))

        
if __name__ == "__main__":
    main()
