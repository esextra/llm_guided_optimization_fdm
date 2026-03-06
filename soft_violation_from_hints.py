import numpy as np
from typing import List, Dict, Any, Tuple, Callable

EPS = 1e-12                                    

                            
                         
                            

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _atomic_residual_attached(
    rt: str,
    cols: List[np.ndarray],
    params: Dict[str, Any],
    N: int,
    eps_div: float = 1e-8,
) -> np.ndarray:
       
    rt = rt.lower()

    if rt == "ratio_eq":
        r_tgt = float(params.get("r"))
        x, y = cols
        denom = y + eps_div
        d = x / denom - r_tgt
        return d * d

    elif rt == "eq_const":
        v = float(params.get("v"))
        x = cols[0]
        d = x - v
        return d * d

    elif rt == "eq_var":
        x, y = cols
        d = x - y
        return d * d

    elif rt == "in_box":
        L = float(params.get("L"))
        U = float(params.get("U"))
        x = cols[0]
        return _relu(L - x) + _relu(x - U)

    elif rt == "inc":
        k = float(params.get("k"))
        x = cols[0]
        return _relu(k - x)                       

    elif rt == "dec":
        k = float(params.get("k"))
        x = cols[0]
        return _relu(x - k)                       

    elif rt == "diff_ge":
                             
        delta = float(params.get("δ", params.get("delta", 0.0)))
        x, y = cols
        return _relu((y + delta) - x)

    elif rt == "sum_le":
                             
        k = float(params.get("k"))
        total = np.zeros(N)
        for a in cols:
            total += a
        return _relu(total - k)

    elif rt == "monotone":
                                                                            
        xstack = np.stack(cols, axis=1)          
        if xstack.shape[1] < 2:
            return np.zeros(N)
        diffs = xstack[:, :-1] - xstack[:, 1:]                                     
        return _relu(diffs).sum(axis=1)

    else:
        raise ValueError(f"Unknown residual_type: {rt}")


                            
                         
                            

def _block_penalty_attached(
    X: np.ndarray,
    feature_order: List[str],
    block: Dict[str, Any],
) -> np.ndarray:
       
    N, d = X.shape
    name_to_col = {name: j for j, name in enumerate(feature_order)}

    residuals: List[np.ndarray] = []
    weights: List[float] = []

    for clause in block.get("clauses", []):
        rt = clause["residual_type"]
        params = clause.get("parameters", {})
        targets = clause.get("targets", [])

        try:
            cols = [X[:, name_to_col[t]] for t in targets]
        except KeyError as e:
            raise ValueError(f"Target '{e.args[0]}' not found in feature_order.") from None

        r_raw = _atomic_residual_attached(rt, cols, params, N=N)
        r_unit = r_raw / (1.0 + r_raw)                                      
        r_unit = np.minimum(r_unit, 1.0 - EPS)                                              

        residuals.append(r_unit)
        w = float(clause.get("importance", 1.0)) * float(clause.get("confidence", 1.0)) * 1.0
        weights.append(w)

    if not residuals:
        return np.zeros(N)

    agg = (block.get("aggregator", "and") or "and").lower()

    if agg == "and":
        prod = np.ones(N)
        for r, w in zip(residuals, weights):
            prod *= (1.0 - r) ** w
        pen = 1.0 - prod

    elif agg in ("or", "null", "none"):
        prod = np.ones(N)
        for r, w in zip(residuals, weights):
            prod *= r ** w
        pen = prod

    else:
        raise ValueError(f"Unsupported aggregator: '{agg}'. Use 'and' or 'or'.")

    return np.minimum(1.0, np.maximum(0.0, pen))


                            
                    
                            

def build_soft_violation(
    heuristic_hints: List[Dict[str, Any]],
    feature_order: List[str],
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
       
    hints = list(heuristic_hints)
    feats = list(feature_order)

    def soft_violation_batch(X: np.ndarray) -> np.ndarray:
           
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
                           
                     
            raise ValueError("X must have shape (N, d).")
        N, d = X.shape
        if d != len(feats):
            raise ValueError(f"X.shape[1] ({d}) must match len(feature_order) ({len(feats)}).")

                                                   
        one_minus = np.ones(N)
        for block in hints:
            pen_b = _block_penalty_attached(X, feats, block)        
            one_minus *= (1.0 - pen_b)
        total = 1.0 - one_minus
        return np.minimum(1.0, np.maximum(0.0, total))

    def soft_violation(x: np.ndarray) -> float:
           
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != len(feats):
            raise ValueError(f"x must have length {len(feats)} matching feature_order.")
        return float(soft_violation_batch(x[None, :])[0])

    return soft_violation, soft_violation_batch
