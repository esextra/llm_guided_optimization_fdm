from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


import print_quality.pipeline.metrics.printability_penalty as printability
import print_quality.pipeline.metrics.structural_penalty as structural
import print_quality.pipeline.metrics.geometric_penalty as geometric

# --------------------------- meaning utility -------------------

from typing import Dict, Any, Tuple

# ---- severity bucketing used for printability/geometric/functional penalties ----
def _severity_label(x: float) -> str:
    x = float(x)
    if x < 0.10: return "negligible"
    if x < 0.25: return "mild"
    if x < 0.40: return "moderate"
    if x < 0.60: return "high"
    if x < 0.80: return "very high"
    return "extreme"

# ---- crisp base texts for veto affirmations/negations ----
_VETO_OK_TEXT = {
    "bed_adhesion":      "OK - bed adhesion sufficient",
    "warping_lever":     "OK - warping risk acceptable",
    "overhang_exposure": "OK - overhangs adequately backed/supported",
    "bridge_exposure":   "OK - bridge spans within capability",
    "island_starts":     "OK - no mid-air island starts",
    "slender_towers":    "OK - no slender-tower risk",
}
_VETO_BAD_TEXT = {
    "bed_adhesion":      "TRIGGERED - part won’t stick to the bed",
    "warping_lever":     "TRIGGERED - corners likely to lift/warp",
    "overhang_exposure": "TRIGGERED - unsupported faces exist",
    "bridge_exposure":   "TRIGGERED - bridges exceed safe span",
    "island_starts":     "TRIGGERED - islands start in air",
    "slender_towers":    "TRIGGERED - towers too slender/wobbly",
}

# exact keys as defined by your aggregator
_GEOMETRIC_KEYS  = ("surface_distance", "stair_stepping", "thin_wall", "seam_visibility")
_FUNCTIONAL_KEYS = ("strength_reserve", "z_bonding_proxy", "perim_infill_contact",
                    "xy_dimensional_risk", "thermal_creep_risk")
_VETO_KEYS = ("bed_adhesion", "warping_lever", "overhang_exposure",
              "bridge_exposure", "island_starts", "slender_towers")
_PRINTABILITY_KEYS = ("stringing_exposure","support_removal")

def crisp_summary(metrics_out: Dict[str, Any],
                  return_only_bad: bool = True,
                  bad_levels: Tuple[str, ...] = ("very high", "extreme", "moderate", "mild", 'high','negligible')) -> Dict[str, str]:
    """
    Output: flat dict containing ONLY
      - veto keys (affirm/negate text with numeric value & threshold),
      - 2 printability keys (severity + value),
      - 4 geometric keys (severity + value),
      - 5 functional keys (severity + value).

    Filtering:
      - If return_only_bad=True (default), keep:
          * ONLY vetos that TRIGGER (value > threshold), and
          * ONLY non-veto penalties whose severity ∈ bad_levels.
      - If return_only_bad=False, include all requested keys that are PRESENT (no fabrication).
    """
    penalties = metrics_out.get("penalties", {}) or {}
    P = penalties.get("printability", {}) or {}
    G = penalties.get("geometric", {}) or {}
    S = penalties.get("structural", {}) or {}

    veto = metrics_out.get("veto", {}) or {}
    thr  = veto.get("thresholds", {}) or {}

    out: Dict[str, str] = {}

    # --- vetos: only consider keys that have BOTH a value and a threshold ---
    for k in _VETO_KEYS:
        if k not in P or k not in thr:
            continue  # do not fabricate entries
        val = float(P[k])
        t   = float(thr[k])
        triggered = val > t
        if return_only_bad and not triggered:
            continue
        prefix = _VETO_BAD_TEXT[k] if triggered else _VETO_OK_TEXT[k]
        comparator = ">" if triggered else "≤"
        out[k] = f"{prefix} (value={val:.3f} {comparator} threshold={t:.3f})"

    # helper that only adds keys that exist; includes severity + value
    def _maybe_add(group: Dict[str, Any], key: str):
        if key not in group:
            return
        val = float(group[key])
        sev = _severity_label(val)
        if return_only_bad and (sev not in bad_levels):
            return
        out[key] = f"{sev} (value={val:.3f})"

    # --- printability(only present keys) ---
    for k in _PRINTABILITY_KEYS:
        _maybe_add(P, k)

    # --- geometric (only present keys) ---
    for k in _GEOMETRIC_KEYS:
        _maybe_add(G, k)

    # --- functional (only present keys) ---
    for k in _FUNCTIONAL_KEYS:
        _maybe_add(S, k)

    return out



# -------------------------- utilities --------------------------


def build_veto_thresholds(cfg: dict,
                          printability_keys: List[str],
                          default: float = 0.40, epsilon: float = 0.01) -> Dict[str, float]:
    """
    Build a per-metric veto threshold dict. 
    """
    out = {}
    out["bed_adhesion"]       = 0.0 + epsilon
    out["warping_lever"]      = 0.1 + epsilon
    out["overhang_exposure"]  = 0.0 + epsilon
    out["bridge_exposure"]    = 0.1 + epsilon
    out["island_starts"]      = 0.1 + epsilon
    out["slender_towers"]     = 0.1 + epsilon

    return out


def veto_check(printability_metrics: Dict[str, float],
               thresholds: Dict[str, float],
               exclude: Optional[List[str]] = None) -> Tuple[bool, Dict[str, float], float]:
    """
    Returns (failed, offenders_dict, max_veto_risk) where offenders exceed their thresholds.
    You can exclude certain keys (e.g., 'stringing_exposure', 'support removal') from veto consideration.
    """
    exclude = set(exclude or [])
    offenders = {}
    max_risk = 0.0
    failed = False
    for k, thr in thresholds.items():
        if k in exclude:
            continue
        val = float(printability_metrics.get(k, 0.0))
        max_risk = max(max_risk, val)
        if val > thr:
            offenders[k] = val
            failed = True
    return failed, offenders, max_risk


def vetos_from_slicer_warnings(slicer_warnings, printability_metrics):
    if "Low bed adhesion" in slicer_warnings: #Describes that the print has low bed adhesion and may became loose.
        printability_metrics['bed_adhesion'] = 1
        
    if "Loose extrusions" in slicer_warnings: #Describes extrusions that are not supported enough and come out curled or loose.
        printability_metrics["overhang_exposure"] = 1  
        
    if "Collapsing overhang" in slicer_warnings: #Describes that the print has large overhang area which will print badly or not print at all.
        printability_metrics["overhang_exposure"] = 1
    
    if "Floating object part" in slicer_warnings: #Describes that the object has part that is not connected to the bed and will not print at all without supports.
        printability_metrics["island_starts"] = 1

    if "Long bridging extrusions" in slicer_warnings: #Describes that the model has long bridging extrusions which may print badly 
        printability_metrics["bridge_exposure"] = 1
    
    if "Floating bridge anchors" in slicer_warnings: #Describes bridge anchors/turns in the air, which will definitely print badly
        printability_metrics["bridge_exposure"] = 1
        
    if "Consider enabling supports" in slicer_warnings:
        printability_metrics["overhang_exposure"] = 1  ## safeguard
        
    if "Also consider enabling brim" in slicer_warnings:
        printability_metrics["bed_adhesion"] = 1 ##safe guard
    
    return printability_metrics


# ---------------------- main aggregation API ----------------------

def compute_metrics_add_vetos_use_slicer_io(mesh_pc, print_job, layer_rasters, pj_dict, cfg, io_dict, load_bearing,load_direction, rx_deg, ry_deg, rz_deg):
    #print(load_bearing)
    slicer_warnings = io_dict["slice_stdout"]
    #slicer_errors = io_dict["slice_stderr"]

    # Correct geometric function name per your module
    geometric_metrics     = geometric.compute_geometric_penalties(mesh_pc, print_job, cfg)
    printability_metrics_to_process  = printability.compute_printability_penalties(mesh_pc, print_job, layer_rasters, pj_dict, cfg)
    structural_metrics    = structural.compute_structural_penalties(mesh_pc, pj_dict, layer_rasters, cfg, load_bearing,load_direction, rx_deg, ry_deg, rz_deg)

    printability_metrics = vetos_from_slicer_warnings(slicer_warnings, printability_metrics_to_process)
        
    keys_to_skip = ['stringing_exposure', 'support_removal']
    

    
    
    # Printability veto keys: everything in printability except stringing and support removal
    veto_keys = [k for k in printability_metrics.keys() if k not in keys_to_skip]


    # Veto thresholds and status
    thresholds = build_veto_thresholds(cfg, veto_keys, default=float(cfg.get('default_veto_threshold', 0.40)))
    failed, offenders, pmax = veto_check(printability_metrics, thresholds, exclude=[])

    output = {
        'penalties': {
            'printability': dict(printability_metrics),
            'structural':   dict(structural_metrics),
            'geometric':    dict(geometric_metrics),
        },
        'veto': {
            'failed':     bool(failed),
            'offenders':  dict(offenders),
            'thresholds': dict(thresholds),
        },
    }
    return output

def compute_metrics_add_vetos_no_slicer_io(mesh_pc, print_job, layer_rasters, pj_dict, cfg, load_bearing, load_direction, rx_deg, ry_deg, rz_deg):
    #print(load_bearing)
    # Correct geometric function name per your module
    geometric_metrics     = geometric.compute_geometric_penalties(mesh_pc, print_job, cfg)
    printability_metrics  = printability.compute_printability_penalties(mesh_pc, print_job, layer_rasters, pj_dict, cfg)
    structural_metrics    = structural.compute_structural_penalties(mesh_pc, pj_dict, layer_rasters, cfg, load_bearing, load_direction, rx_deg, ry_deg, rz_deg)

    keys_to_skip = ['stringing_exposure', 'support_removal']
    
    # Printability veto keys: everything in printability except stringing and support removal
    veto_keys = [k for k in printability_metrics.keys() if k not in keys_to_skip]


    # Veto thresholds and status
    thresholds = build_veto_thresholds(cfg, veto_keys, default=float(cfg.get('default_veto_threshold', 0.40)))
    failed, offenders, pmax = veto_check(printability_metrics, thresholds, exclude=[])

    output = {
        'penalties': {
            'printability': dict(printability_metrics),
            'structural':   dict(structural_metrics),
            'geometric':    dict(geometric_metrics),
        },
        'veto': {
            'failed':     bool(failed),
            'offenders':  dict(offenders),
            'thresholds': dict(thresholds),
        },
    }
    return output



# --- Bounded group softmax (inputs in [0,1] -> output in [0,1]) ---
# def _softmax_group_risk_bounded(values: List[float], beta: float = 10.0) -> float:
#     if not values:
#         return 0.0
#     K = len(values)
#     s = sum(math.exp(beta * float(v)) for v in values) / K
#     return float((1.0 / beta) * math.log(s))

def _softmax_group_risk_bounded(values: List[float], beta: float = 10.0) -> float:
    # Replace NaN values with 0.0 before applying softmax
    values = [0.0 if math.isnan(v) else v for v in values]
    if not values:
        return 0.0
    K = len(values)
    s = sum(math.exp(beta * float(v)) for v in values) / K
    return float((1.0 / beta) * math.log(s))


# --- Build grouped objectives directly from agg_out['penalties'] ---
def _grouped_objectives_from_metrics(agg_out: Dict[str, dict], beta: Optional[float]) -> Dict[str, float]:
    P = agg_out['penalties']['printability']
    S = agg_out['penalties']['structural']
    G = agg_out['penalties']['geometric']
    b = float(beta) if (beta is not None) else 10.0

    functional_keys = [k for k in ('strength_reserve', 'z_bonding_proxy',
                                   'perim_infill_contact', 'xy_dimensional_risk',
                                   'thermal_creep_risk') if k and k in S]
    geometric_keys  = [k for k in ('surface_distance', 'stair_stepping',
                                   'thin_wall', 'seam_visibility') if k in G]
    
    printability_keys = [k for k in ('stringing_exposure', 'support_removal') if k in P]

    F   = _softmax_group_risk_bounded([float(S[k]) for k in functional_keys], beta=b) if functional_keys else 0.0
    Geo = _softmax_group_risk_bounded([float(G[k]) for k in geometric_keys],  beta=b) if geometric_keys else 0.0
    Pr = _softmax_group_risk_bounded([float(P[k]) for k in printability_keys],  beta=b) if printability_keys else 0.0
    return {'functional': F, 'geometric': Geo, 'printability': Pr}

# --- Goals-only exceedance (stay in [0,1]) ---
def _exceed_goal_unit(p: float, g: float, eps: float = 1e-6) -> float:
    p = float(min(max(p, 0.0), 1.0))
    g = float(min(max(g, 0.0), 1.0 - eps))
    return float(min(max((p - g) / (1.0 - g + eps), 0.0), 1.0))

def _exceed_vector_unit(objectives: Dict[str, float], goals: Dict[str, float]) -> Dict[str, float]:
    return {k: _exceed_goal_unit(objectives[k], goals.get(k, 0.0)) for k in objectives.keys()}

# --- Headroom-OR Chebyshev (bounded in [0,1]) ---
def _J_optionA_goal_cheb(objectives: Dict[str, float], goals: Dict[str, float], lam: float = 0.15) -> float:
    e = _exceed_vector_unit(objectives, goals)
    if not e:
        return 0.0
    a = max(e.values())                       # worst exceedance
    b = float(sum(e.values())) / len(e)       # mean exceedance
    return float(a + lam * (1.0 - a) * b)

def _J_optionC_goal_cheb(objectives: Dict[str, float], goals: Dict[str, float],
                         rng: Optional[np.random.Generator] = None, lam: float = 0.15
                         ) -> Tuple[float, Dict[str, float]]:
    rng = rng or np.random.default_rng()
    e = _exceed_vector_unit(objectives, goals)
    if not e:
        return 0.0, {}
    keys = list(e.keys())
    w = rng.dirichlet(np.ones(len(keys)))
    a = max(e.values())
    b = float(np.dot(w, np.array([e[k] for k in keys], dtype=float)))
    J = float(a + lam * (1.0 - a) * b)
    return J, {k: float(w[i]) for i, k in enumerate(keys)}

# --- Compute goals from warm-up samples (no budgets) ---


def compute_goals_from_samples(samples: List[Dict[str, float]], q: float = 0.25) -> Dict[str, float]:
    if not samples:
        return {}
    keys = list(samples[0].keys())
    arr = {k: np.array([float(s[k]) for s in samples if k in s], dtype=float) for k in keys}
    goals = {k: float(np.quantile(arr[k], q)) for k in keys}
    eps = 1e-6
    return {k: float(min(max(v, 0.0), 1.0 - eps)) for k, v in goals.items()}
    
    
# --- Public entry point: penalties-only scalar from an aggregate(...) output ---
def aggregate(mesh_pc, print_job, layer_rasters, pj_dict, cfg, load_bearing, load_direction, rx_deg, ry_deg, rz_deg,
                                  mode: str = "A",
                                  lam: float = 0.15,
                                  samples: Optional[List[Dict[str, float]]]=None,
                                  beta: Optional[float] = None,
                                  rng_for_option_c: Optional[np.random.Generator] = None
                                  ) -> Tuple[float, Dict[str, object]]:
    """
    Returns (J, info) with J in [0,1].
    Veto is a hard constraint: if agg_out['veto']['failed'] is truthy -> J=1.0 and info['infeasible']=True.
    """
    #print(load_bearing)
    metrics_out = compute_metrics_add_vetos_no_slicer_io(mesh_pc, print_job, layer_rasters, pj_dict, cfg, load_bearing,load_direction, rx_deg, ry_deg, rz_deg)
    info = crisp_summary(metrics_out, cfg)
    veto = metrics_out.get('veto', {})
    failed = veto.get('failed', False)
    #print(metrics_out)
    if failed:
        return float("inf"), metrics_out, {'infeasible': True, 'offenders': veto.get('offenders', {})}, info
    
    if samples:
        goals = compute_goals_from_samples(samples,q=0.25,default_value=0.0)
    else:
        goals = {'functional': 0.0, 'geometric': 0.2, 'printability': 0.0}
    
    objectives = _grouped_objectives_from_metrics(metrics_out, beta=beta)

    if mode.upper() == "A":
        J = _J_optionA_goal_cheb(objectives, goals, lam=lam)
        return J, metrics_out, {'infeasible': False, 'objectives': objectives}, info
    elif mode.upper() == "C":
        J, weights = _J_optionC_goal_cheb(objectives, goals, rng=rng_for_option_c, lam=lam)
        return J, metrics_out, {'infeasible': False, 'objectives': objectives, 'weights': weights}, info
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'A' or 'C'.")

def aggregate_for_bo(mesh_pc, print_job, layer_rasters, pj_dict, cfg, io_dict,load_bearing,load_direction, rx_deg, ry_deg, rz_deg,
                                  mode: str = "A",
                                  lam: float = 0.15,
                                  samples: Optional[List[Dict[str, float]]]=None,
                                  beta: Optional[float] = None,
                                  rng_for_option_c: Optional[np.random.Generator] = None
                                  ) -> Tuple[float, Dict[str, object]]:
    """
    Returns (J, info) with J in [0,1].
    Veto is a hard constraint: if agg_out['veto']['failed'] is truthy -> J=1.0 and info['infeasible']=True.
    """
    #print(load_bearing)
    metrics_out = compute_metrics_add_vetos_use_slicer_io(mesh_pc, print_job, layer_rasters, pj_dict, cfg, io_dict,load_bearing,load_direction, rx_deg, ry_deg, rz_deg)
    info = crisp_summary(metrics_out, cfg)
    veto = metrics_out.get('veto', {})
    failed = veto.get('failed', False)
    #print(metrics_out)
    if failed:
        return float("inf"), metrics_out, {'infeasible': True, 'offenders': veto.get('offenders', {})}, info
    
    if samples:
        goals = compute_goals_from_samples(samples,q=0.25,default_value=0.0)
    else:
        goals = {'functional': 0.0, 'geometric': 0.2, 'printability': 0.0}
    
    objectives = _grouped_objectives_from_metrics(metrics_out, beta=beta)

    if mode.upper() == "A":
        J = _J_optionA_goal_cheb(objectives, goals, lam=lam)
        return J, metrics_out, {'infeasible': False, 'objectives': objectives}, info
    elif mode.upper() == "C":
        J, weights = _J_optionC_goal_cheb(objectives, goals, rng=rng_for_option_c, lam=lam)
        return J, metrics_out, {'infeasible': False, 'objectives': objectives, 'weights': weights}, info
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'A' or 'C'.")
