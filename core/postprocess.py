from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from core.detections import (
    DetectionRaw,
    LaneBoundaryPoint,
    LaneSummary,
    LaneType,
    LineColor,
    ManeuverFlags,
    MarkingObject,
)
from core.line_fitting import LineFitter

logger = logging.getLogger(__name__)


@dataclass
class PostprocessParams:
    confidence_threshold: float
    min_area: float
    merge_iou_threshold: float


class DetectionPostprocessor:
    """Filters raw detections and merges overlapping ones of the same class."""

    def __init__(
        self,
        params: PostprocessParams,
        geometry_mapper: Optional[GeometryMapper] = None,
        line_fitting_params: Optional[dict] = None,
    ) -> None:
        self.params = params
        self.geometry_mapper = geometry_mapper or GeometryMapper()

        # Line fitting parameters
        lf_params = line_fitting_params or {}
        self.line_fitting_enabled = lf_params.get("enabled", True)
        self.split_by_center = lf_params.get("split_by_center", True)

        self.line_fitter = LineFitter(
            poly_degree=lf_params.get("poly_degree", 2),
            ransac_iterations=lf_params.get("ransac_iterations", 100),
            inlier_threshold=lf_params.get("inlier_threshold", 5.0),
            min_inlier_ratio=lf_params.get("min_inlier_ratio", 0.3),
            min_points=lf_params.get("min_points", 20),
            split_margin=lf_params.get("split_margin", 20),
        )

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

    def fit_lines(self, detections: Iterable[DetectionRaw]) -> List:
        """
        Fit polynomial lines to lane marking detections.

        Args:
            detections: List of DetectionRaw objects

        Returns:
            List of FittedLine objects
        """
        if not self.line_fitting_enabled:
            return []

        return self.line_fitter.fit_lines_from_detections(
            detections,
            split_by_center=self.split_by_center,
        )

    def build_lane_summary(self, detections: Iterable[DetectionRaw], frame_shape: Tuple[int, int, int]) -> LaneSummary:
        h, w, _ = frame_shape
        detections = list(detections)
        if not detections:
            return LaneSummary()

        left_det, right_det = self._pick_left_right(detections, frame_shape)

        def offsets(det: Optional[DetectionRaw]) -> int:
            if det and det.bbox:
                cx = 0.5 * (det.bbox[0] + det.bbox[2])
                return int(round((cx - w / 2.0) / max(w, 1) * 100))
            return 0

        left_offset_dm = offsets(left_det)
        right_offset_dm = offsets(right_det)

        maneuvers = ManeuverFlags.STRAIGHT
        if left_offset_dm < 0:
            maneuvers |= ManeuverFlags.LEFT
        if right_offset_dm > 0:
            maneuvers |= ManeuverFlags.RIGHT

        quality = min(255, len(detections) * 20)

        left_quality = int(min(255, (left_det.confidence * 255) if left_det else 0))
        right_quality = int(min(255, (right_det.confidence * 255) if right_det else 0))

        left_width_dm = self._width_dm(left_det, w)
        right_width_dm = self._width_dm(right_det, w)

        left_type = GeometryMapper._line_style_from_class(left_det.class_id) if left_det else LaneType.UNKNOWN
        right_type = GeometryMapper._line_style_from_class(right_det.class_id) if right_det else LaneType.UNKNOWN
        left_color = GeometryMapper._line_color_from_class(left_det.class_id) if left_det else LineColor.UNKNOWN
        right_color = GeometryMapper._line_color_from_class(right_det.class_id) if right_det else LineColor.UNKNOWN

        left_boundary = (
            self.geometry_mapper.boundary_points_from_bbox(left_det, frame_shape)
            if left_det
            else [LaneBoundaryPoint(0, 0), LaneBoundaryPoint(0, 0), LaneBoundaryPoint(0, 0)]
        )
        right_boundary = (
            self.geometry_mapper.boundary_points_from_bbox(right_det, frame_shape)
            if right_det
            else [LaneBoundaryPoint(0, 0), LaneBoundaryPoint(0, 0), LaneBoundaryPoint(0, 0)]
        )

        return LaneSummary(
            left_offset_dm=left_offset_dm,
            right_offset_dm=right_offset_dm,
            left_type=left_type,
            right_type=right_type,
            left_color=left_color,
            right_color=right_color,
            allowed_maneuvers=maneuvers,
            quality=quality,
            left_quality=left_quality,
            right_quality=right_quality,
            left_width_dm=left_width_dm,
            right_width_dm=right_width_dm,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
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

    def _width_dm(self, det: Optional[DetectionRaw], frame_w: int) -> int:
        if det and det.bbox:
            x1, _, x2, _ = det.bbox
            return int(round(max(1.0, x2 - x1) / max(frame_w, 1) * 100))
        return 0

    def _pick_left_right(self, detections: List[DetectionRaw], frame_shape: Tuple[int, int, int]) -> Tuple[Optional[DetectionRaw], Optional[DetectionRaw]]:
        w = frame_shape[1]
        left: Optional[DetectionRaw] = None
        right: Optional[DetectionRaw] = None
        for det in detections:
            if not det.bbox:
                continue
            cx = 0.5 * (det.bbox[0] + det.bbox[2])
            if cx < w / 2.0:
                if (left is None) or (det.confidence > left.confidence):
                    left = det
            else:
                if (right is None) or (det.confidence > right.confidence):
                    right = det
        return left, right

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
                line_color=self._line_color_from_class(det.class_id),
                line_style=self._line_style_from_class(det.class_id),
            )
            result.append(obj)
        return result

    def boundary_points_from_bbox(self, det: DetectionRaw, frame_shape: Tuple[int, int, int]) -> List[LaneBoundaryPoint]:
        if det.bbox is None:
            return []
        h, w, _ = frame_shape
        x1, y1, x2, y2 = det.bbox
        xs = [x1, 0.5 * (x1 + x2), x2]
        ys = [y1, 0.5 * (y1 + y2), y2]
        points: List[LaneBoundaryPoint] = []
        for x_px, y_px in zip(xs, ys):
            x_dm = int(round((x_px - w / 2.0) / max(w, 1) * 100))
            y_dm = int(round((h - y_px) / max(h, 1) * 100))
            points.append(LaneBoundaryPoint(x_dm=x_dm, y_dm=y_dm))
        return points

    @staticmethod
    def _line_color_from_class(class_id: int) -> LineColor:
        if class_id in {5, 8, 10, 12, 14, 15}:  # yellow-ish classes
            return LineColor.YELLOW
        if class_id in {6}:  # red
            return LineColor.RED
        return LineColor.WHITE if class_id != 0 else LineColor.UNKNOWN

    @staticmethod
    def _line_style_from_class(class_id: int) -> LaneType:
        if class_id in {4, 5, 6}:  # solid single
            return LaneType.SOLID
        if class_id in {7, 8}:  # double
            return LaneType.DOUBLE
        if class_id in {9, 10}:  # dashed
            return LaneType.DASHED
        return LaneType.UNKNOWN
