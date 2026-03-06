#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, Union, List, Tuple

Number = Union[int, float]

                                            
_HEADER: List[str] = [
    "timestamp_iso",
    "run_id",
    "optimizer",
    "seed",
    "row_kind",
                            
    "iter",
    "eval_index_in_iter",
    "suggest_secs",
    "eval_secs",
    "total_secs",
                          
    "objective",
    "objective_kind",
    "use_combined",
    "objective_combine_json",
                     
    "time_s",
    "cost",
    "quality_J",
    "quality_info_json",
                   
    "gcode_path",
    "config_path",
    "mesh_path",
                                
    "use_domain_baseline_scaling",
    "time_ref",
    "cost_ref",
    "quality_ref",
                                                          
    "slicer_params_json",
                                                      
    "guidance_lambda",
    "H",
    "H_hat",
    "temper",
    "w_tr",
    "active_hints_json",
]

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _b01(x: Optional[bool]) -> Optional[int]:
    if x is None:
        return None
    return 1 if x else 0

def _to_json(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
                                      
        return str(obj)

@dataclass
class BOEvalRow:
              
    csv_path: str
    run_id: str
    optimizer: str
    seed: Optional[int]

             
    iter: Optional[int] = None
    eval_index_in_iter: int = 0
    suggest_secs: Optional[Number] = None
    eval_secs: Optional[Number] = None
    total_secs: Optional[Number] = None

    objective: Optional[Number] = None
    objective_kind: str = "combined"                    
    use_combined: Optional[bool] = None
    objective_combine_json: Optional[str] = None

    time_s: Optional[Number] = None
    cost: Optional[Number] = None
    quality_J: Optional[Number] = None
    quality_info_json: Optional[str] = None

    gcode_path: Optional[str] = None
    config_path: Optional[str] = None
    mesh_path: Optional[str] = None

    use_domain_baseline_scaling: Optional[bool] = None
    time_ref: Optional[Number] = None
    cost_ref: Optional[Number] = None
    quality_ref: Optional[Number] = None

    slicer_params_json: Optional[str] = None       

                                              
    guidance_lambda: Optional[Number] = None
    H: Optional[Number] = None
    H_hat: Optional[Number] = None
    temper: Optional[Number] = None
    w_tr: Optional[Number] = None
    active_hints_json: Optional[str] = None

    def write(self) -> None:
        _ensure_dir(self.csv_path)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_HEADER, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({
                "timestamp_iso": _now_iso(),
                "run_id": self.run_id,
                "optimizer": self.optimizer,
                "seed": self.seed,
                "row_kind": "eval",
                "iter": self.iter,
                "eval_index_in_iter": self.eval_index_in_iter,
                "suggest_secs": self.suggest_secs,
                "eval_secs": self.eval_secs,
                "total_secs": self.total_secs,
                "objective": self.objective,
                "objective_kind": self.objective_kind,
                "use_combined": _b01(self.use_combined),
                "objective_combine_json": self.objective_combine_json,
                "time_s": self.time_s,
                "cost": self.cost,
                "quality_J": self.quality_J,
                "quality_info_json": self.quality_info_json,
                "gcode_path": self.gcode_path,
                "config_path": self.config_path,
                "mesh_path": self.mesh_path,
                "use_domain_baseline_scaling": _b01(self.use_domain_baseline_scaling),
                "time_ref": self.time_ref,
                "cost_ref": self.cost_ref,
                "quality_ref": self.quality_ref,
                "slicer_params_json": self.slicer_params_json,
                "guidance_lambda": self.guidance_lambda,
                "H": self.H,
                "H_hat": self.H_hat,
                "temper": self.temper,
                "w_tr": self.w_tr,
                "active_hints_json": self.active_hints_json,
            })

class BOLogger:
                                                                       

    def __init__(self, csv_path: str, run_id: str, optimizer: str, seed: Optional[int] = None):
        self.csv_path = csv_path
        self.run_id = run_id
        self.optimizer = optimizer
        self.seed = seed

    def log_eval(
        self,
        *,
        iter_idx: int,
        objective: Number,
                    
        time_s: Optional[Number],
        cost: Optional[Number],
        quality_J: Optional[Number],
        quality_info: Optional[Dict[str, Any]],
                             
        slicer_params: Optional[Dict[str, Any]],
                    
        gcode_path: Optional[str],
        config_path: Optional[str],
        mesh_path: Optional[str],
                
        suggest_secs: Optional[Number],
        eval_secs: Optional[Number],
        total_secs: Optional[Number],
                 
        use_domain_baseline_scaling: Optional[bool],
        time_ref: Optional[Number],
        cost_ref: Optional[Number],
        quality_ref: Optional[Number],
                        
        use_combined: bool,
        objective_kind: str,
        objective_combine_pairs: Optional[List[Tuple[str, float]]] = None,
                                                  
        guidance_lambda: Optional[Number] = None,
        H: Optional[Number] = None,
        H_hat: Optional[Number] = None,
        temper: Optional[Number] = None,
        w_tr: Optional[Number] = None,
        active_hints: Optional[List[str]] = None,
    ) -> None:
        row = BOEvalRow(
            csv_path=self.csv_path,
            run_id=self.run_id,
            optimizer=self.optimizer,
            seed=self.seed,
            iter=iter_idx,
            eval_index_in_iter=0,
            suggest_secs=suggest_secs,
            eval_secs=eval_secs,
            total_secs=total_secs,
            objective=objective,
            objective_kind=objective_kind,
            use_combined=use_combined,
            objective_combine_json=_to_json(objective_combine_pairs) if use_combined else None,
            time_s=time_s,
            cost=cost,
            quality_J=quality_J,
            quality_info_json=_to_json(quality_info) if isinstance(quality_info, dict) else None,
            gcode_path=gcode_path,
            config_path=config_path,
            mesh_path=mesh_path,
            use_domain_baseline_scaling=use_domain_baseline_scaling,
            time_ref=time_ref,
            cost_ref=cost_ref,
            quality_ref=quality_ref,
            slicer_params_json=_to_json(slicer_params),       
            guidance_lambda=guidance_lambda,
            H=H,
            H_hat=H_hat,
            temper=temper,
            w_tr=w_tr,
            active_hints_json=_to_json(active_hints) if isinstance(active_hints, list) else None,

        )
        row.write()
