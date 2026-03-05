from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import re
import numpy as np

from ..data.types import (
    Channel, PRUSA_TYPES_TO_CHANNEL,
    GCodeState, Segment, LayerInfo, PrintJob, Config
)
from ..utils.logging_utils import get_logger
from ..utils.config_resolution import channel_default_width_mm, channel_default_speed_mms

log = get_logger(__name__)

WORD_RE = re.compile(r'([A-Za-z])([-+0-9.]+)')
TYPE_RE = re.compile(r';\s*TYPE\s*:\s*(.+)\s*$')
LAYER_CHANGE_RE = re.compile(r';\s*LAYER_CHANGE', re.IGNORECASE)
LAYER_NUM_RE = re.compile(r';\s*LAYER\s*:\s*(-?\d+)', re.IGNORECASE)
HEIGHT_RE = re.compile(r';\s*HEIGHT\s*:\s*([-+0-9.]+)', re.IGNORECASE)
WIDTH_RE = re.compile(r';\s*WIDTH\s*:\s*([-+0-9.]+)', re.IGNORECASE)

def _parse_word_codes(line: str) -> Dict[str, float]:
    return {k: float(v) for k, v in WORD_RE.findall(line)}

@dataclass
class _SegBuilder:
    feature_type: str
    channel: Channel
    layer_index: int
    z: float
    points: List[Tuple[float, float, float]]
    feedrates: List[float]
    e_deltas: List[float]
    is_extruding: bool
    width_mm: Optional[float]
    height_mm: Optional[float]
    fan_speed: Optional[int]
    tool: Optional[int]
    comments: List[str]
    retractions_before: int = 0
    retractions_after: int = 0

    def to_segment(self, seg_id: int) -> Segment:
        poly = np.array(self.points, dtype=np.float64)
        return Segment(
            id=seg_id,
            layer_index=self.layer_index,
            z=self.z,
            channel=self.channel,
            feature_type=self.feature_type,
            polyline=poly,
            feedrates_mm_per_min=list(self.feedrates),
            e_deltas=list(self.e_deltas),
            is_extruding=self.is_extruding,
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            fan_speed=self.fan_speed,
            tool=self.tool,
            comments=list(self.comments),
            retractions_before=self.retractions_before,
            retractions_after=self.retractions_after
        )

def _maybe_close_segment(segments: List[Segment], builder: Optional[_SegBuilder], seg_id: int) -> Tuple[Optional[_SegBuilder], int]:
    if builder is None or len(builder.points) < 2:
        return None, seg_id
    seg = builder.to_segment(seg_id)
    segments.append(seg)
    return None, seg_id + 1

def parse_gcode(path: str, cfg: Optional[Config] = None, *, skip_until_first_layer: bool = True) -> PrintJob:
    """
    Improvements added:
      (#1) Track retractions that occur BEFORE an extruding segment starts, storing them in Segment.retractions_before.
      (#5) Feedrate fallback: if a line lacks F and state.current_f is None, use the last non-zero modal F we saw;
           if none seen yet, fall back to channel_default_speed_mms(cfg) * 60; else 0.
    """
    state = GCodeState()
    current_feature = "Unknown"
    current_channel = Channel.UNKNOWN
    current_layer_index = 0 if not skip_until_first_layer else -1
    current_z = 0.0
    current_width: Optional[float] = None
    current_height: Optional[float] = None
    current_fan: Optional[int] = None
    current_tool: Optional[int] = None

    segments: List[Segment] = []
    seg_builder: Optional[_SegBuilder] = None
    seg_id = 0
    seen_first_layer = not skip_until_first_layer

    # (#1) count retractions occurring while not extruding; they will be attached to the next extruding segment
    pending_retractions_before = 0
    # (#5) remember the last effective modal feedrate (mm/min) once we see a non-zero F
    last_effective_f: Optional[float] = None

    # per-layer time accumulation (coarse: dist / feedrate)
    layer_time_accum: Dict[int, float] = {}

    
    def _flush_builder():
        nonlocal seg_builder, seg_id, segments
        if seg_builder is None:
            return
        if len(seg_builder.points) >= 2:
            segments.append(seg_builder.to_segment(seg_id))
            seg_id += 1
        # Always reset, even if it had <2 points (drop it)
        seg_builder = None

    def begin_new_segment(*, is_extruding: bool, retractions_before_seed: int = 0):
        nonlocal seg_builder
        # Close any existing segment if it has >= 2 points
        nonlocal seg_id
        if seg_builder is not None and len(seg_builder.points) >= 2:
            seg, seg_id_new = seg_builder.to_segment(seg_id), seg_id + 1
            segments.append(seg)
            seg_id = seg_id_new

        # Build a fresh segment
        # Non-extruding segments are TRAVEL regardless of the last ;TYPE: value.
        feat = current_feature if is_extruding else "Travel"
        chan = current_channel if is_extruding else Channel.TRAVEL
        seg_builder = _SegBuilder(
            feature_type=feat,
            channel=chan,
            layer_index=current_layer_index,
            z=current_z,
            points=[],
            feedrates=[],
            e_deltas=[],
            is_extruding=is_extruding,
            width_mm=current_width,
            height_mm=current_height,
            fan_speed=current_fan,
            tool=current_tool,
            comments=[],
            retractions_before=retractions_before_seed,
            retractions_after=0
        )

    # --- parsing ---

    # Threshold to classify retraction from E delta (mm of filament)
    retraction_threshold = 0.01

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            # Comments / Prusa metadata
            if line.startswith(";"):
                if LAYER_CHANGE_RE.match(line):
                    # Close ongoing segment at layer change
                    _flush_builder()
                    if not seen_first_layer:
                        seen_first_layer = True
                        current_layer_index += 1  # -1 -> 0
                    else:
                        current_layer_index += 1
                    continue

                m = LAYER_NUM_RE.match(line)
                if m:
                    if seg_builder is not None and len(seg_builder.points) >= 2:
                        seg, seg_id = seg_builder.to_segment(seg_id), seg_id + 1
                        segments.append(seg)
                        seg_builder = None
                    _flush_builder()
                    current_layer_index = int(m.group(1))
                    if not seen_first_layer:
                        seen_first_layer = True
                    continue

                m = HEIGHT_RE.match(line)
                if m:
                    current_height = float(m.group(1))
                    continue

                m = WIDTH_RE.match(line)
                if m:
                    current_width = float(m.group(1))
                    continue

                m = TYPE_RE.match(line)
                if m:
                    # New feature/channel → close current segment and start fresh on next moves
                    _flush_builder()
                    current_feature = m.group(1)
                    current_channel = PRUSA_TYPES_TO_CHANNEL.get(current_feature, Channel.UNKNOWN)
                    continue

                # Collect comments only after we hit first layer
                if seen_first_layer and seg_builder is not None:
                    seg_builder.comments.append(line[1:].strip())
                continue  # end of handling for ';' lines

            # Fan control
            if line.startswith("M106"):
                codes = _parse_word_codes(line)
                # S is 0..255; we store int for consistency with your types
                current_fan = int(codes.get("S", current_fan if current_fan is not None else 0))
                continue
            if line.startswith("M107"):
                current_fan = 0
                continue

            # Absolute/relative modes
            if line.startswith("G90"):
                state.absolute_coordinates = True
                continue
            if line.startswith("G91"):
                state.absolute_coordinates = False
                continue
            if line.startswith("M82"):
                state.absolute_extrusion = True
                continue
            if line.startswith("M83"):
                state.absolute_extrusion = False
                continue

            # G92 (set position / E)
            if line.startswith("G92"):
                codes = _parse_word_codes(line)
                # Close segment on coordinate discontinuity
                _flush_builder()

                x, y, z = state.current_xyz
                if "X" in codes:
                    x = codes["X"] if state.absolute_coordinates else (x + codes["X"])
                if "Y" in codes:
                    y = codes["Y"] if state.absolute_coordinates else (y + codes["Y"])
                if "Z" in codes:
                    z = codes["Z"] if state.absolute_coordinates else (z + codes["Z"])
                state.current_xyz = (x, y, z)

                if "E" in codes:
                    state.current_e = codes["E"] if state.absolute_extrusion else (state.current_e + codes["E"])
                continue

            # Motion
            if line.startswith("G0") or line.startswith("G1"):
                codes = _parse_word_codes(line)

                prev_x, prev_y, prev_z = state.current_xyz
                x, y, z = state.current_xyz
                e = state.current_e
                f_modal = state.current_f  # may be None

                # Position updates (modal)
                if "X" in codes:
                    x = codes["X"] if state.absolute_coordinates else (x + codes["X"])
                if "Y" in codes:
                    y = codes["Y"] if state.absolute_coordinates else (y + codes["Y"])
                if "Z" in codes:
                    z = codes["Z"] if state.absolute_coordinates else (z + codes["Z"])

                # E update and delta
                e_new = e
                if "E" in codes:
                    if state.absolute_extrusion:
                        e_new = codes["E"]
                        e_delta = e_new - e
                    else:
                        e_delta = codes["E"]
                        e_new = e + e_delta
                else:
                    e_delta = 0.0

                # F (feedrate) update (modal, mm/min)
                f_on_line = codes.get("F", None)
                if f_on_line is not None:
                    f_modal = f_on_line
                    state.current_f = f_on_line
                    if f_on_line > 0:
                        last_effective_f = f_on_line

                # Is this move extruding?
                is_extruding = e_delta > 1e-9
                started_new_segment = False

                # Geometry flags
                has_xy = (x != prev_x) or (y != prev_y)
                has_z  = (z != prev_z)
                if has_z:
                    # track current Z but don't assume geometry
                    current_z = z

                # --- Skip recording segments until the first ;LAYER_CHANGE if requested ---
                # Purpose: drop bed-edge priming/intro moves but keep skirts/brims/rafts (which occur at layer 0+).
                if skip_until_first_layer and not seen_first_layer:
                    # Update modal state only; do NOT begin/append segments yet.
                    state.current_xyz = (x, y, z)
                    state.current_e = e_new
                    if f_on_line is not None:
                        state.current_f = f_on_line
                        if f_on_line > 0:
                            last_effective_f = f_on_line
                    # Do not let pre-layer retractions leak into layer 0 segments.
                    pending_retractions_before = 0
                    continue


                # (#1) Handle retractions with BEFORE/AFTER semantics
                # (applies even if there's no XY geometry on this line)
                if e_delta < -retraction_threshold:
                    # Retraction happened on this move
                    if seg_builder is not None and seg_builder.is_extruding:
                        # We were in an extruding segment → this retraction is AFTER it
                        seg_builder.retractions_after += 1
                        # Close that segment because the material flow stopped
                        _flush_builder()
                    else:
                        # We are not in an extruding segment → count for the NEXT extruding segment
                        pending_retractions_before += 1


                # If there's NO XY movement (pure E change or Z-only hop),
                # do not create/extend a segment. Just commit modal state.
                if not has_xy:
                    if has_z:
                        _flush_builder()  # keep layers clean across Z-hops
                    state.current_xyz = (x, y, z)
                    state.current_e = e_new
                    # keep feedrate modal updated
                    if f_on_line is not None:
                        state.current_f = f_on_line
                        if f_on_line > 0:
                            last_effective_f = f_on_line
                    continue

                # Start a new segment if needed (mode switch or none active)
                if seg_builder is None or seg_builder.is_extruding != is_extruding:
                    # When starting an EXTRUDING segment, consume pending BEFORE retractions
                    seed = pending_retractions_before if is_extruding else 0
                    begin_new_segment(is_extruding=is_extruding, retractions_before_seed=seed)
                    started_new_segment = True
                    if is_extruding:
                        pending_retractions_before = 0  # consumed

                # Effective feedrate for this point (#5)
                if state.current_f is not None and state.current_f > 0:
                    effective_f = state.current_f
                elif last_effective_f is not None and last_effective_f > 0:
                    effective_f = last_effective_f
                else:
                    fallback_speed_mms = channel_default_speed_mms(cfg, current_channel) if cfg else None
                    effective_f = (fallback_speed_mms or 0.0) * 60.0  # mm/min

                # If we just opened a new segment, seed the anchor point with the previous position
                if started_new_segment:
                    seg_builder.points.append((prev_x, prev_y, prev_z))
                    seg_builder.feedrates.append(effective_f)
                    seg_builder.e_deltas.append(0.0)
                # Append geometry and attributes
                seg_builder.points.append((x, y, z))
                seg_builder.feedrates.append(effective_f)
                seg_builder.e_deltas.append(max(e_delta, 0.0))  # store only positive extrusion

                # Time accumulation (distance / speed)
                if len(seg_builder.points) >= 2:
                    p0x, p0y, p0z = seg_builder.points[-2]
                    p1x, p1y, p1z = seg_builder.points[-1]
                    dx = p1x - p0x
                    dy = p1y - p0y
                    dz = p1z - p0z
                    dist = float((dx * dx + dy * dy + dz * dz) ** 0.5)
                    v_mms = (effective_f or 0.0) / 60.0
                    if v_mms > 1e-9 and dist > 0.0:
                        dt = dist / v_mms
                        layer_time_accum[current_layer_index] = layer_time_accum.get(current_layer_index, 0.0) + dt

                # Commit modal positions
                state.current_xyz = (x, y, z)
                state.current_e = e_new
                continue

            # Tool changes
            if line.startswith("T"):
                try:
                    current_tool = int(line[1:].strip())
                except ValueError:
                    pass
                # Close current segment on tool change for cleanliness
                _flush_builder()
                continue

            # For any other codes we do not affect geometry/segments

    # Flush the last open segment
    if seg_builder is not None and len(seg_builder.points) >= 2:
        seg, seg_id = seg_builder.to_segment(seg_id), seg_id + 1
        segments.append(seg)

    # Fill per-segment width/height defaults when missing
    if cfg is not None:
        for s in segments:
            if s.width_mm is None:
                s.width_mm = channel_default_width_mm(cfg, s.channel)
            if s.height_mm is None:
                if s.layer_index == 0 and cfg.first_layer_height is not None:
                    s.height_mm = cfg.first_layer_height
                elif cfg.layer_height is not None:
                    s.height_mm = cfg.layer_height

    # Build LayerInfo objects
    layers: Dict[int, LayerInfo] = {}

    for s in segments:
        # Drop any leftover pre-first-layer segments (layer_index < 0) just in case.
        if s.layer_index < 0:
            continue
        if s.layer_index not in layers:
            layers[s.layer_index] = LayerInfo(index=s.layer_index, z=s.z)
        layers[s.layer_index].segments.append(s)
        
    for li in layers.values():
        li.time_s = layer_time_accum.get(li.index, None)

    layers_list = [layers[k] for k in sorted(layers.keys())]
    return PrintJob(segments=segments, layers=layers_list, config=(cfg or Config()))
