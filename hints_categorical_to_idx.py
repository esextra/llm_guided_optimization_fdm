                             
from typing import List, Dict, Any

def normalize_hints_categoricals_to_indices(
    hints: List[Dict[str, Any]],
    infill_patterns: List[str],
    seam_positions: List[str],
) -> List[Dict[str, Any]]:
       
    def _map_val(val, choices):
        if isinstance(val, str):
            try:
                return float(choices.index(val))
            except ValueError:
                return val                            
        if isinstance(val, list):
            return [_map_val(v, choices) for v in val]
        return val

    out: List[Dict[str, Any]] = []
    for block in hints or []:
        nb = dict(block)
        new_clauses = []
        for cl in block.get("clauses", []):
            cl2 = dict(cl)
            targets = list(cl2.get("targets", []))

                                               
            norm_targets = []
            seen_cat = {"infill": False, "seam": False}
            for t in targets:
                if t == "infill_pattern":
                    norm_targets.append("infill_pattern_id"); seen_cat["infill"] = True
                elif t == "seam_position":
                    norm_targets.append("seam_idx"); seen_cat["seam"] = True
                else:
                    norm_targets.append(t)
            cl2["targets"] = norm_targets

                                                                             
            params = dict(cl2.get("parameters", {}) or {})
            if seen_cat["infill"] or ("infill_pattern_id" in norm_targets):
                for key in ("v","y","L","U"):
                    if key in params:
                        params[key] = _map_val(params[key], infill_patterns)
            if seen_cat["seam"] or ("seam_idx" in norm_targets):
                for key in ("v","y","L","U"):
                    if key in params:
                        params[key] = _map_val(params[key], seam_positions)

            cl2["parameters"] = params
            new_clauses.append(cl2)
        nb["clauses"] = new_clauses
        out.append(nb)
    return out
