from __future__ import annotations

from typing import Any, Dict, Optional


def bed_adhesion(layer_rasters: Any, cfg: Any, segments: Optional[Any] = None) -> Dict[str, float]:
    # not implemented because slicer warnings override any compute that would be written here
    return {"score": 0.0}


def compute_bed_adhesion_penalty(layer_rasters: Any, cfg: Any, segments: Any) -> float:
    # not implemented because slicer warnings override any compute that would be written here
    return 0.0
