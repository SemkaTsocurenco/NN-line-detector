from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import List, Optional, Tuple

import numpy as np


class ManeuverFlags(IntFlag):
    NONE = 0
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 4
    UTURN = 8


class LaneType(IntEnum):
    UNKNOWN = 0
    SOLID = 1
    DASHED = 2
    DOUBLE = 3
    BOTTS = 4


@dataclass
class DetectionRaw:
    class_id: int
    confidence: float
    mask: Optional[np.ndarray] = None  # Expected shape (H, W) uint8
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2 in pixels
    polygon: Optional[np.ndarray] = None  # Nx2 array of points in image coords


@dataclass
class MarkingObject:
    class_id: int
    x_dm: int
    y_dm: int
    length_dm: int
    width_dm: int
    yaw_decideg: int
    confidence_byte: int
    flags: int = 0


@dataclass
class LaneSummary:
    left_offset_dm: int = 0
    right_offset_dm: int = 0
    left_type: LaneType = LaneType.UNKNOWN
    right_type: LaneType = LaneType.UNKNOWN
    allowed_maneuvers: ManeuverFlags = ManeuverFlags.NONE
    quality: int = 0  # 0..255


@dataclass
class FrameDetections:
    raw: List[DetectionRaw] = field(default_factory=list)
    objects: List[MarkingObject] = field(default_factory=list)
    summary: LaneSummary = field(default_factory=LaneSummary)
