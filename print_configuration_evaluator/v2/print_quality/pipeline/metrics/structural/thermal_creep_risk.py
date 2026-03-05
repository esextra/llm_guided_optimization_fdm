#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional


def thermal_creep_risk(
    cfg: Any,
    env_temp_c: Optional[float] = None,
    use_hdt_first: bool = True,
    safety_margin_c: float = 25.0,
) -> Dict[str, Any]:
    # not implemented in current version
    return {
        "penalty": 0.0,
        "material": "",
        "env_temp_c": float(env_temp_c) if env_temp_c is not None else float("nan"),
        "limit_basis": "",
        "limit_c": float("nan"),
        "margin_c": float("nan"),
        "safety_margin_c": float(safety_margin_c),
        "tg_c": float("nan"),
        "hdt45_c": float("nan"),
        "use_hdt_first": bool(use_hdt_first),
    }


def compute_thermal_creep_risk_penalty(cfg: Any) -> float:
    # not implemented in current version
    return 0.0