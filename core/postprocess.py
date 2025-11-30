from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from core.detections import DetectionRaw, LaneSummary, LaneType, ManeuverFlags, MarkingObject

logger = logging.getLogger(__name__)


@dataclass
class PostprocessParams:
    confidence_threshold: float
    min_area: float
    merge_iou_threshold: float


class DetectionPostprocessor:
    """Filters raw detections and merges overlapping ones of the same class."""

    def __init__(self, params: PostprocessParams) -> None:
        self.params = params

    def update_params(self, params: PostprocessParams) -> None:
        self.params = params

    def process(self, detections: Iterable[DetectionRaw], frame_shape: Tuple[int, int, int]) -> List[DetectionRaw]:
        filtered: List[DetectionRaw] = []
        for det in detections:
            if det.confidence < self.params.confidence_threshold:
                continue
            area = self._area(det)
            if area is not None and area < self.params.min_area:
                continue
            filtered.append(det)

        merged = self._merge_by_class(filtered)
        return merged

    def build_lane_summary(self, detections: Iterable[DetectionRaw], frame_shape: Tuple[int, int, int]) -> LaneSummary:
        h, w, _ = frame_shape
        detections = list(detections)
        if not detections:
            return LaneSummary()

        centers = [self._bbox_center(det) for det in detections if det.bbox is not None]
        if not centers:
            return LaneSummary()
        xs = [c[0] for c in centers]
        left_offset_px = min(xs) - w / 2.0
        right_offset_px = max(xs) - w / 2.0

        scale_dm = 100.0 / max(w, 1)  # crude scale: full width ~10m (100 dm)
        left_offset_dm = int(round(left_offset_px * scale_dm))
        right_offset_dm = int(round(right_offset_px * scale_dm))

        maneuvers = ManeuverFlags.STRAIGHT
        if left_offset_px < 0:
            maneuvers |= ManeuverFlags.LEFT
        if right_offset_px > 0:
            maneuvers |= ManeuverFlags.RIGHT

        quality = min(255, len(detections) * 20)

        return LaneSummary(
            left_offset_dm=left_offset_dm,
            right_offset_dm=right_offset_dm,
            left_type=LaneType.UNKNOWN,
            right_type=LaneType.UNKNOWN,
            allowed_maneuvers=maneuvers,
            quality=quality,
        )

    @staticmethod
    def _area(det: DetectionRaw) -> Optional[float]:
        if det.mask is not None:
            return float((det.mask > 0).sum())
        if det.bbox is not None:
            x1, y1, x2, y2 = det.bbox
            return float(max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1))
        return None

    @staticmethod
    def _bbox_center(det: DetectionRaw) -> Tuple[float, float]:
        if det.bbox:
            x1, y1, x2, y2 = det.bbox
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        return (0.0, 0.0)

    def _merge_by_class(self, detections: List[DetectionRaw]) -> List[DetectionRaw]:
        merged: List[DetectionRaw] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            placed = False
            for idx, m in enumerate(merged):
                if det.class_id != m.class_id or det.bbox is None or m.bbox is None:
                    continue
                if self._iou(det.bbox, m.bbox) >= self.params.merge_iou_threshold:
                    merged[idx] = self._merge_detections(m, det)
                    placed = True
                    break
            if not placed:
                merged.append(det)
        return merged

    @staticmethod
    def _merge_detections(a: DetectionRaw, b: DetectionRaw) -> DetectionRaw:
        bbox = None
        if a.bbox and b.bbox:
            x1 = min(a.bbox[0], b.bbox[0])
            y1 = min(a.bbox[1], b.bbox[1])
            x2 = max(a.bbox[2], b.bbox[2])
            y2 = max(a.bbox[3], b.bbox[3])
            bbox = (x1, y1, x2, y2)

        mask = None
        if a.mask is not None and b.mask is not None:
            mask = np.maximum(a.mask, b.mask)
        elif a.mask is not None:
            mask = a.mask
        elif b.mask is not None:
            mask = b.mask

        confidence = max(a.confidence, b.confidence)
        return DetectionRaw(class_id=a.class_id, confidence=confidence, mask=mask, bbox=bbox)

    @staticmethod
    def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter_w = max(0, xB - xA + 1)
        inter_h = max(0, yB - yA + 1)
        inter = inter_w * inter_h
        area1 = max(0, b1[2] - b1[0] + 1) * max(0, b1[3] - b1[1] + 1)
        area2 = max(0, b2[2] - b2[0] + 1) * max(0, b2[3] - b2[1] + 1)
        union = area1 + area2 - inter
        if union == 0:
            return 0.0
        return inter / union


class GeometryMapper:
    """Converts image detections to protocol-ready MarkingObjects."""

    def __init__(self, calibration: Optional[dict] = None) -> None:
        self.calibration = calibration or {}

    def to_marking_objects(self, detections: Iterable[DetectionRaw], frame_shape: Tuple[int, int, int]) -> List[MarkingObject]:
        h, w, _ = frame_shape
        result: List[MarkingObject] = []
        for det in detections:
            if det.bbox is None:
                continue
            x1, y1, x2, y2 = det.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            # Crude mapping: normalize image coords to +-50 dm window.
            x_dm = int(round((cx - w / 2.0) / max(w, 1) * 100))
            y_dm = int(round((h - cy) / max(h, 1) * 100))
            length_dm = int(round(bh / max(h, 1) * 100))
            width_dm = int(round(bw / max(w, 1) * 100))

            confidence_byte = int(max(0, min(255, det.confidence * 255)))
            obj = MarkingObject(
                class_id=det.class_id,
                x_dm=x_dm,
                y_dm=y_dm,
                length_dm=length_dm,
                width_dm=width_dm,
                yaw_decideg=0,
                confidence_byte=confidence_byte,
                flags=0,
            )
            result.append(obj)
        return result
