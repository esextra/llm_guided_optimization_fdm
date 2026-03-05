from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

class Channel(Enum):
    P_OUTER = auto()
    P_INNER = auto()
    P_OVERHANG = auto()
    SKIN_TOP = auto()
    SKIN_SOLID = auto()
    INFILL = auto()
    BRIDGE = auto()
    SUPPORT = auto()
    SUP_IFC = auto()
    ADHESION = auto()
    TRAVEL = auto()
    UNKNOWN = auto()

PRUSA_TYPES_TO_CHANNEL = {
    "External perimeter": Channel.P_OUTER,
    "Perimeter": Channel.P_INNER,
    "Overhang perimeter": Channel.P_OVERHANG,
    "Top solid infill": Channel.SKIN_TOP,
    "Solid infill": Channel.SKIN_SOLID,
    "Internal infill": Channel.INFILL,
    "Bridge infill": Channel.BRIDGE,
    "Support material": Channel.SUPPORT,
    "Support material interface": Channel.SUP_IFC,
    "Skirt/Brim": Channel.ADHESION,
    "Gap fill": Channel.SKIN_SOLID,
    "Ironing": Channel.SKIN_TOP,
}

@dataclass
class Config:
    # Geometry / process
    nozzle_diameter: Optional[float] = None
    layer_height: Optional[float] = None
    first_layer_height: Optional[float] = None
    seam_position: Optional[str] = None

    # Shells / infill (densities stay as strings like "5%"; metrics convert)
    fill_density: Optional[str] = None
    infill_density: Optional[str] = None        # JSON synonym
    fill_pattern: Optional[str] = None
    perimeters: Optional[float] = None

    # Widths (mm; parser will turn "%" into mm using nozzle_diameter)
    extrusion_width: Optional[float] = None
    perimeter_extrusion_width: Optional[float] = None
    external_perimeter_extrusion_width: Optional[float] = None
    infill_extrusion_width: Optional[float] = None
    solid_infill_extrusion_width: Optional[float] = None
    top_infill_extrusion_width: Optional[float] = None
    first_layer_extrusion_width: Optional[float] = None

    # Top skins
    top_solid_layers: Optional[float] = None

    # Temps / material
    bed_temperature: Optional[float] = None
    filament_type: Optional[str] = None
    filament_settings_id: Optional[str] = None

    # Speeds / timing / flow
    max_print_speed: Optional[float] = None
    travel_speed: Optional[float] = None
    first_layer_speed: Optional[float] = None
    perimeter_speed: Optional[float] = None
    solid_infill_speed: Optional[float] = None
    top_solid_infill_speed: Optional[float] = None
    infill_speed: Optional[float] = None
    support_material_speed: Optional[float] = None
    support_material_interface_speed: Optional[float] = None  # may be "80%" in INI
    min_layer_time: Optional[float] = None
    bridge_speed: Optional[float] = None
    bridge_flow_ratio: Optional[float] = None
    filament_max_volumetric_speed: Optional[float] = None

    # Supports
    support_material: Optional[bool] = None
    support_material_threshold: Optional[float] = None

    # Dimensional XY
    elefant_foot_compensation: Optional[float] = None

    # Unmapped keys retained here (metrics will also look here as fallback)
    extras: Optional[Dict[str, Any]] = None

@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: Optional[np.ndarray] = None
    vertex_normals: Optional[np.ndarray] = None
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None

@dataclass
class GCodeState:
    absolute_coordinates: bool = True
    absolute_extrusion: bool = True
    current_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_e: float = 0.0
    current_f: Optional[float] = None
    current_tool: Optional[int] = None
    fan_speed: Optional[int] = None
    nozzle_temps: Dict[int, float] = field(default_factory=dict)
    bed_temp: Optional[float] = None

@dataclass
class Segment:
    id: int
    layer_index: int
    z: float
    channel: Channel
    feature_type: str
    polyline: np.ndarray
    feedrates_mm_per_min: List[float]
    e_deltas: List[float]
    is_extruding: bool
    width_mm: Optional[float]
    height_mm: Optional[float]
    fan_speed: Optional[int]
    tool: Optional[int]
    start_time_s: Optional[float] = None
    duration_s: Optional[float] = None
    retractions_before: int = 0
    retractions_after: int = 0
    comments: List[str] = field(default_factory=list)

@dataclass
class LayerInfo:
    index: int
    z: float
    segments: List[Segment] = field(default_factory=list)
    time_s: Optional[float] = None

@dataclass
class PrintJob:
    segments: List[Segment]
    layers: List[LayerInfo]
    config: Config
    first_layer_footprint_hint: Optional[object] = None
