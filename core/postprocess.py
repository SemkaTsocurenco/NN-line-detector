from __future__ import annotations

import logging
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

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


def load_postprocess_config(config_path: str = "config/postprocess.yaml") -> Dict[str, Any]:
    """Load postprocessing configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Postprocess config not found at %s, using defaults", config_path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.params = params
        self.geometry_mapper = geometry_mapper or GeometryMapper(config=config)

        # Load configuration
        if config is None:
            config = {}

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
            config=lf_params,
        )

        # Load config values
        self.config = config
        geom_cfg = config.get("geometry", {})
        self.normalization_scale = geom_cfg.get("normalization_scale", 100)
        self.quality_multiplier = geom_cfg.get("quality_multiplier", 20)
        self.max_quality = geom_cfg.get("max_quality", 255)
        self.confidence_to_byte_scale = geom_cfg.get("confidence_to_byte_scale", 255)
        self.bbox_area_offset = geom_cfg.get("bbox_area_offset", 1)

        # Arrow clustering parameters
        arrow_cfg = config.get("arrow_clustering", {})
        self.arrow_clustering_enabled = arrow_cfg.get("enabled", True)
        self.arrow_classes = set(arrow_cfg.get("arrow_classes", [11, 12, 13, 14, 15]))
        self.arrow_absorb_classes = set(arrow_cfg.get("absorb_classes", [4, 5, 6, 7, 8, 9, 10]))
        self.arrow_iou_threshold = float(arrow_cfg.get("iou_threshold", 0.05))
        self.arrow_overlap_ratio = float(arrow_cfg.get("overlap_ratio_threshold", 0.25))
        self.arrow_absorb_overlap_ratio = float(arrow_cfg.get("absorb_overlap_ratio", 0.6))
        self.arrow_require_center_inside = bool(arrow_cfg.get("require_center_inside", True))
        self.arrow_max_absorb_area_ratio = float(arrow_cfg.get("max_absorb_area_ratio", 1.2))
        self.arrow_forward_factor = float(arrow_cfg.get("forward_factor", 3.0))
        self.arrow_lateral_factor = float(arrow_cfg.get("lateral_factor", 1.0))
        self.arrow_min_forward_px = float(arrow_cfg.get("min_forward_px", 20.0))
        self.arrow_min_lateral_px = float(arrow_cfg.get("min_lateral_px", 10.0))
        self.arrow_tail_overlap_min = float(arrow_cfg.get("tail_overlap_min", 0.1))
        self.arrow_tail_area_ratio = float(arrow_cfg.get("tail_area_ratio", 3.0))

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
        if self.arrow_clustering_enabled:
            merged = self._cluster_arrows(merged)
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
                return int(round((cx - w / 2.0) / max(w, 1) * self.normalization_scale))
            return 0

        left_offset_dm = offsets(left_det)
        right_offset_dm = offsets(right_det)

        maneuvers = ManeuverFlags.STRAIGHT
        if left_offset_dm < 0:
            maneuvers |= ManeuverFlags.LEFT
        if right_offset_dm > 0:
            maneuvers |= ManeuverFlags.RIGHT

        quality = min(self.max_quality, len(detections) * self.quality_multiplier)

        left_quality = int(min(self.max_quality, (left_det.confidence * self.confidence_to_byte_scale) if left_det else 0))
        right_quality = int(min(self.max_quality, (right_det.confidence * self.confidence_to_byte_scale) if right_det else 0))

        left_width_dm = self._width_dm(left_det, w)
        right_width_dm = self._width_dm(right_det, w)

        left_type = self.geometry_mapper._line_style_from_class(left_det.class_id) if left_det else LaneType.UNKNOWN
        right_type = self.geometry_mapper._line_style_from_class(right_det.class_id) if right_det else LaneType.UNKNOWN
        left_color = self.geometry_mapper._line_color_from_class(left_det.class_id) if left_det else LineColor.UNKNOWN
        right_color = self.geometry_mapper._line_color_from_class(right_det.class_id) if right_det else LineColor.UNKNOWN

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

    def _area(self, det: DetectionRaw) -> Optional[float]:
        if det.mask is not None:
            return float((det.mask > 0).sum())
        if det.bbox is not None:
            x1, y1, x2, y2 = det.bbox
            offset = self.bbox_area_offset
            return float(max(0, x2 - x1 + offset) * max(0, y2 - y1 + offset))
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
            return int(round(max(1.0, x2 - x1) / max(frame_w, 1) * self.normalization_scale))
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

    def _cluster_arrows(self, detections: List[DetectionRaw]) -> List[DetectionRaw]:
        """
        Merge arrow detections with nearby line-like masks so arrows render as a single object.

        Optimized for small N: uses bbox prefilter then mask overlap on the intersecting window.
        """
        if not detections:
            return []

        result: List[DetectionRaw] = []
        consumed = set()

        # First, merge arrows with nearby masks
        for idx, det in enumerate(detections):
            if det.class_id not in self.arrow_classes or det.mask is None or idx in consumed:
                continue

            consumed.add(idx)
            seed_mask = det.mask
            seed_bbox = det.bbox or self._bbox_from_mask(seed_mask)
            arrow_info = self._arrow_info(seed_mask)
            if arrow_info is None:
                result.append(det)
                continue

            combined_mask = seed_mask.copy()
            combined_bbox = seed_bbox
            max_conf = det.confidence

            for j, other in enumerate(detections):
                if j in consumed or other.mask is None:
                    continue
                if other.class_id not in self.arrow_classes and other.class_id not in self.arrow_absorb_classes:
                    continue
                if not self._should_absorb_into_arrow(seed_mask, seed_bbox, arrow_info, other):
                    continue

                combined_mask = np.maximum(combined_mask, other.mask)
                combined_bbox = self._bbox_from_mask(combined_mask)
                max_conf = max(max_conf, other.confidence)
                consumed.add(j)

            result.append(
                DetectionRaw(
                    class_id=det.class_id,
                    confidence=max_conf,
                    mask=combined_mask,
                    bbox=combined_bbox,
                    polygon=det.polygon,
                )
            )

        # Then add everything that wasn't consumed
        for idx, det in enumerate(detections):
            if idx in consumed:
                continue
            result.append(det)

        return result

    def _should_absorb_into_arrow(
        self,
        arrow_mask: np.ndarray,
        arrow_bbox: Optional[Tuple[int, int, int, int]],
        arrow_info: dict,
        other: DetectionRaw,
    ) -> bool:
        """
        Decide if other detection should be merged into arrow cluster based on overlap and direction.
        - Arrow fragments: directional gate + modest overlap.
        - Line-like objects: stricter, require directional gate and high coverage.
        Uses only the seed arrow geometry (arrow_info) to avoid runaway growth.
        """
        other_bbox = other.bbox or self._bbox_from_mask(other.mask)
        if other_bbox is None or arrow_bbox is None or other.mask is None:
            return False

        # Avoid runaway growth: only absorb if the other mask is not disproportionately larger than the arrow seed
        arrow_area = (arrow_mask > 0).sum()
        other_area = (other.mask > 0).sum()
        if arrow_area == 0 or other_area == 0:
            return False
        if other_area / float(arrow_area) > self.arrow_max_absorb_area_ratio:
            return False

        # Directional gate: only consider masks along the arrow axis within a corridor
        if not self._directional_gate(arrow_info, other_bbox):
            return False

        overlap_ratio = self._mask_overlap_ratio(arrow_mask, arrow_bbox, other.mask, other_bbox)
        bbox_iou = self._iou(arrow_bbox, other_bbox)

        # Arrow-to-arrow merge: allow modest overlap
        if other.class_id in self.arrow_classes:
            return (bbox_iou >= self.arrow_iou_threshold) or (overlap_ratio >= self.arrow_overlap_ratio)

        # Absorbing line-like objects:
        # Primary path: strong coverage
        if overlap_ratio >= self.arrow_absorb_overlap_ratio:
            if self.arrow_require_center_inside and not self._is_center_inside(arrow_mask, arrow_bbox, other_bbox):
                return False
            return bbox_iou >= self.arrow_iou_threshold

        # Secondary path: allow thin tail pieces along the arrow direction with modest overlap
        if overlap_ratio >= self.arrow_tail_overlap_min:
            area_ratio = other_area / float(arrow_area)
            if area_ratio <= self.arrow_tail_area_ratio:
                if self.arrow_require_center_inside:
                    return self._is_center_inside(arrow_mask, arrow_bbox, other_bbox)
                return True

        return False

    def _directional_gate(self, arrow_info: dict, other_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if other bbox lies within a corridor along the arrow's principal direction.
        """
        cx_o = 0.5 * (other_bbox[0] + other_bbox[2])
        cy_o = 0.5 * (other_bbox[1] + other_bbox[3])
        cx_a, cy_a = arrow_info["center"]
        dx, dy = arrow_info["direction"]
        rel_x = cx_o - cx_a
        rel_y = cy_o - cy_a

        # Projection onto arrow direction and perpendicular distance
        proj = abs(rel_x * dx + rel_y * dy)
        lateral = abs(-rel_x * dy + rel_y * dx)

        max_forward = max(self.arrow_forward_factor * arrow_info["length_scale"], self.arrow_min_forward_px)
        lateral_limit = max(self.arrow_lateral_factor * arrow_info["width_scale"], self.arrow_min_lateral_px)

        return proj <= max_forward and lateral <= lateral_limit

    @staticmethod
    def _is_center_inside(
        arrow_mask: np.ndarray,
        arrow_bbox: Optional[Tuple[int, int, int, int]],
        other_bbox: Tuple[int, int, int, int],
    ) -> bool:
        if arrow_bbox is None or arrow_mask.size == 0:
            return False
        cx = int(round(0.5 * (other_bbox[0] + other_bbox[2])))
        cy = int(round(0.5 * (other_bbox[1] + other_bbox[3])))
        h, w = arrow_mask.shape
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        return arrow_mask[cy, cx] > 0

    @staticmethod
    def _arrow_info(mask: np.ndarray) -> Optional[dict]:
        """Compute principal direction and scales for an arrow mask."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        cx = float(xs.mean())
        cy = float(ys.mean())
        coords = np.column_stack([xs - cx, ys - cy])
        if len(coords) < 2:
            return None
        cov = np.cov(coords, rowvar=False)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        idx = int(np.argmax(eigvals))
        direction = eigvecs[:, idx]
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return None
        direction = direction / norm
        length_scale = float(np.sqrt(max(eigvals[idx], 1e-6)) * 2.0)
        width_scale = float(np.sqrt(max(eigvals[1 - idx], 1e-6)) * 2.0)
        return {
            "center": (cx, cy),
            "direction": (float(direction[0]), float(direction[1])),
            "length_scale": length_scale,
            "width_scale": width_scale,
        }

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _mask_overlap_ratio(
        self,
        mask_a: np.ndarray,
        bbox_a: Optional[Tuple[int, int, int, int]],
        mask_b: np.ndarray,
        bbox_b: Optional[Tuple[int, int, int, int]],
    ) -> float:
        """
        Compute overlap area(mask_a ∩ mask_b) / area(mask_b) using bbox window to limit work.
        """
        if mask_a is None or mask_b is None:
            return 0.0

        if bbox_a is None:
            bbox_a = self._bbox_from_mask(mask_a)
        if bbox_b is None:
            bbox_b = self._bbox_from_mask(mask_b)
        if bbox_a is None or bbox_b is None:
            return 0.0

        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        # Slice only the intersecting window to keep it cheap
        window_a = mask_a[y1 : y2 + 1, x1 : x2 + 1]
        window_b = mask_b[y1 : y2 + 1, x1 : x2 + 1]
        intersection = np.logical_and(window_a > 0, window_b > 0).sum()
        if intersection == 0:
            return 0.0

        area_b = (mask_b > 0).sum()
        if area_b == 0:
            return 0.0
        return float(intersection) / float(area_b)

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

    def _iou(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        offset = self.bbox_area_offset
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter_w = max(0, xB - xA + offset)
        inter_h = max(0, yB - yA + offset)
        inter = inter_w * inter_h
        area1 = max(0, b1[2] - b1[0] + offset) * max(0, b1[3] - b1[1] + offset)
        area2 = max(0, b2[2] - b2[0] + offset) * max(0, b2[3] - b2[1] + offset)
        union = area1 + area2 - inter
        if union == 0:
            return 0.0
        return inter / union


class GeometryMapper:
    """Converts image detections to protocol-ready MarkingObjects."""

    def __init__(self, calibration: Optional[dict] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.calibration = calibration or {}

        # Load config
        if config is None:
            config = {}

        geom_cfg = config.get("geometry", {})
        self.normalization_scale = geom_cfg.get("normalization_scale", 100)
        self.confidence_to_byte_scale = geom_cfg.get("confidence_to_byte_scale", 255)

        # Load line color and style mappings from config
        line_colors_cfg = config.get("line_colors", {})
        self.yellow_classes = set(line_colors_cfg.get("yellow_classes", [5, 8, 10, 12, 14, 15]))
        self.red_classes = set(line_colors_cfg.get("red_classes", [6]))

        line_styles_cfg = config.get("line_styles", {})
        self.solid_classes = set(line_styles_cfg.get("solid_classes", [4, 5, 6]))
        self.double_classes = set(line_styles_cfg.get("double_classes", [7, 8]))
        self.dashed_classes = set(line_styles_cfg.get("dashed_classes", [9, 10]))
        arrow_cfg = config.get("arrow_clustering", {})
        self.arrow_classes = set(arrow_cfg.get("arrow_classes", [11, 12, 13, 14, 15]))

        # Line classes that should NOT be sent as road_objects (they go as fitted_lines)
        self.line_classes = self.solid_classes | self.double_classes | self.dashed_classes

    def to_marking_objects(self, detections: Iterable[DetectionRaw], frame_shape: Tuple[int, int, int]) -> List[MarkingObject]:
        """
        Convert detections to MarkingObjects for road_objects protocol message.

        Note: Line classes (4-10) are excluded - they are sent via fitted_lines message.
        """
        h, w, _ = frame_shape
        result: List[MarkingObject] = []
        for det in detections:
            # Skip line classes - they go as fitted_lines, not road_objects
            if det.class_id in self.line_classes:
                continue

            geometry = self._extract_object_geometry(det)
            if geometry is None:
                continue
            cx, cy, length_px, width_px, yaw_deg = geometry

            # Crude mapping: normalize image coords to +-50 dm window.
            x_dm = int(round((cx - w / 2.0) / max(w, 1) * self.normalization_scale))
            y_dm = int(round((h - cy) / max(h, 1) * self.normalization_scale))
            length_dm = int(round(length_px / max(h, 1) * self.normalization_scale))
            width_dm = int(round(width_px / max(w, 1) * self.normalization_scale))
            yaw_decideg = int(round(yaw_deg * 10.0))

            confidence_byte = int(max(0, min(self.confidence_to_byte_scale, det.confidence * self.confidence_to_byte_scale)))

            # Convert yaw to radians for v2 protocol
            yaw_rad = math.radians(yaw_deg)

            obj = MarkingObject(
                class_id=det.class_id,
                x_dm=x_dm,
                y_dm=y_dm,
                length_dm=length_dm,
                width_dm=width_dm,
                yaw_decideg=yaw_decideg,
                confidence_byte=confidence_byte,
                flags=0,
                line_color=self._line_color_from_class(det.class_id),
                line_style=self._line_style_from_class(det.class_id),
                # Store original pixel coordinates for accurate transformation in v2 protocol
                bbox_px=det.bbox,
                center_px=(cx, cy),
                yaw_rad=yaw_rad,
            )
            result.append(obj)
        return result

    def _extract_object_geometry(self, det: DetectionRaw) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Extract center, length, width and yaw for a detection.

        For arrows we prefer contour-based geometry (same as renderer) to keep TCP data aligned
        with what is drawn. For other classes fall back to the axis-aligned bbox.
        """
        if det.mask is not None and (det.class_id in self.arrow_classes or det.bbox is None):
            geom = self._geometry_from_mask(det.mask)
            if geom is not None:
                return geom

        if det.bbox is None:
            return None

        x1, y1, x2, y2 = det.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        return (cx, cy, bh, bw, 0.0)

    @staticmethod
    def _geometry_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Derive geometry from a binary mask using min-area rectangle and PCA direction.

        Returns (cx, cy, length_px, width_px, yaw_deg).
        """
        if mask is None or mask.size == 0:
            return None

        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return None

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        (cx, cy), (w_box, h_box), _ = cv2.minAreaRect(contour)
        length_px = float(max(w_box, h_box))
        width_px = float(max(1.0, min(w_box, h_box)))

        # Estimate orientation via PCA on mask coordinates for stability
        yaw_deg = 0.0
        ys, xs = np.nonzero(binary)
        if len(xs) >= 2:
            coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
            try:
                eigvals, eigvecs = np.linalg.eigh(np.cov(coords, rowvar=False))
                idx = int(np.argmax(eigvals))
                direction = eigvecs[:, idx]
                norm = float(np.linalg.norm(direction))
                if norm > 0:
                    dx, dy = direction / norm
                    # Prefer forward (-y) orientation to keep yaw stable across frames
                    if dy > 0:
                        dx, dy = -dx, -dy
                    yaw_deg = math.degrees(math.atan2(dx, -dy))
            except np.linalg.LinAlgError:
                yaw_deg = 0.0

        return (float(cx), float(cy), max(length_px, 1.0), max(width_px, 1.0), yaw_deg)

    def boundary_points_from_bbox(self, det: DetectionRaw, frame_shape: Tuple[int, int, int]) -> List[LaneBoundaryPoint]:
        if det.bbox is None:
            return []
        h, w, _ = frame_shape
        x1, y1, x2, y2 = det.bbox
        xs = [x1, 0.5 * (x1 + x2), x2]
        ys = [y1, 0.5 * (y1 + y2), y2]
        points: List[LaneBoundaryPoint] = []
        for x_px, y_px in zip(xs, ys):
            x_dm = int(round((x_px - w / 2.0) / max(w, 1) * self.normalization_scale))
            y_dm = int(round((h - y_px) / max(h, 1) * self.normalization_scale))
            points.append(LaneBoundaryPoint(x_dm=x_dm, y_dm=y_dm))
        return points

    def _line_color_from_class(self, class_id: int) -> LineColor:
        if class_id in self.yellow_classes:
            return LineColor.YELLOW
        if class_id in self.red_classes:
            return LineColor.RED
        return LineColor.WHITE if class_id != 0 else LineColor.UNKNOWN

    def _line_style_from_class(self, class_id: int) -> LaneType:
        if class_id in self.solid_classes:
            return LaneType.SOLID
        if class_id in self.double_classes:
            return LaneType.DOUBLE
        if class_id in self.dashed_classes:
            return LaneType.DASHED
        return LaneType.UNKNOWN
