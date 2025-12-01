from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from core.detections import DetectionRaw, LaneSummary, LineColor, LaneType

logger = logging.getLogger(__name__)


def load_renderer_config(config_path: str = "config/renderer.yaml") -> Dict[str, Any]:
    """Load renderer configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Renderer config not found at %s, using defaults", config_path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Renderer:
    def __init__(self, class_names: Optional[Dict[int, str]] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.class_names = class_names or {}
        self._color_cache: Dict[int, tuple[int, int, int]] = {}

        # Load configuration
        if config is None:
            config = load_renderer_config()

        # Load all config sections
        mask_cfg = config.get("mask", {})
        overlap_cfg = config.get("overlap", {})
        line_cfg = config.get("line", {})
        arrow_cfg = config.get("arrow", {})
        crosswalk_cfg = config.get("crosswalk", {})
        stop_line_cfg = config.get("stop_line", {})
        box_junction_cfg = config.get("box_junction", {})
        channelizing_cfg = config.get("channelizing", {})
        icon_cfg = config.get("icon", {})
        priority_cfg = config.get("priority", {})
        class_groups_cfg = config.get("class_groups", {})
        line_colors_cfg = config.get("line_colors", {})

        # Mask parameters
        self.mask_binarization_threshold = mask_cfg.get("binarization_threshold", 127)
        self.mask_overlay_alpha = mask_cfg.get("overlay_alpha", 0.4)
        self.mask_contour_outline_thickness = mask_cfg.get("contour_outline_thickness", 3)
        self.mask_contour_color_thickness = mask_cfg.get("contour_color_thickness", 2)

        # Overlap parameters
        self.overlap_threshold = overlap_cfg.get("threshold", 0.7)
        self.line_detection_threshold = overlap_cfg.get("line_detection_threshold", 0.4)

        # Line parameters
        self.line_sample_points_min = line_cfg.get("sample_points_min", 50)
        self.line_thickness = line_cfg.get("thickness", 6)
        self.line_outline_offset = line_cfg.get("outline_offset", 2)
        self.dash_length = line_cfg.get("dash_length", 20)
        self.gap_length = line_cfg.get("gap_length", 10)
        self.double_offset = line_cfg.get("double_offset", 6)

        # Arrow parameters
        self.arrow_min_mask_area = arrow_cfg.get("min_mask_area", 25)
        self.arrow_min_contour_area = arrow_cfg.get("min_contour_area", 20)
        self.arrow_min_box_width = arrow_cfg.get("min_box_width", 3)
        self.arrow_min_box_height = arrow_cfg.get("min_box_height", 3)
        self.arrow_template_size = arrow_cfg.get("template_size", 200)
        self.arrow_thickness = arrow_cfg.get("thickness", 22)
        self.arrow_outline_thickness = arrow_cfg.get("outline_thickness", 28)
        self.arrow_tip_size = arrow_cfg.get("tip_size", 45)
        self.arrow_combined_thickness = arrow_cfg.get("combined_thickness", 15)
        self.arrow_combined_outline = arrow_cfg.get("combined_outline", 21)
        self.arrow_combined_tip_size = arrow_cfg.get("combined_tip_size", 35)
        self.arrow_alpha = arrow_cfg.get("alpha", 0.9)

        # Crosswalk parameters
        self.crosswalk_template_size = crosswalk_cfg.get("template_size", 200)
        self.crosswalk_num_stripes = crosswalk_cfg.get("num_stripes", 8)
        self.crosswalk_alpha = crosswalk_cfg.get("alpha", 0.9)

        # Stop line parameters
        self.stop_line_outline_thickness = stop_line_cfg.get("outline_thickness", 12)
        self.stop_line_line_thickness = stop_line_cfg.get("line_thickness", 8)

        # Box junction parameters
        self.box_junction_template_size = box_junction_cfg.get("template_size", 200)
        self.box_junction_line_thickness = box_junction_cfg.get("line_thickness", 7)
        self.box_junction_line_spacing = box_junction_cfg.get("line_spacing", 25)
        self.box_junction_alpha = box_junction_cfg.get("alpha", 0.85)

        # Channelizing parameters
        self.channelizing_alpha = channelizing_cfg.get("alpha", 0.85)

        # Icon parameters
        self.icon_outline_offset = icon_cfg.get("outline_offset", 2)
        self.icon_base_thickness = icon_cfg.get("base_thickness", 3)

        # Priority levels
        self.priority_lines = priority_cfg.get("lines", 1)
        self.priority_surface_marks = priority_cfg.get("surface_marks", 2)
        self.priority_stop_cross = priority_cfg.get("stop_cross", 3)
        self.priority_arrows_icons = priority_cfg.get("arrows_icons", 4)

        # Class groups
        self.class_group_lines = set(class_groups_cfg.get("lines", [4, 5, 6, 7, 8, 9, 10]))
        self.class_group_surface_marks = set(class_groups_cfg.get("surface_marks", [1, 16]))
        self.class_group_stop_cross = set(class_groups_cfg.get("stop_cross", [2, 3]))
        self.class_group_arrows_icons = set(class_groups_cfg.get("arrows_icons", [11, 12, 13, 14, 15, 22, 23]))

        # Define colors for lane markings (BGR format for OpenCV)
        # Using bright, saturated colors for visibility
        self.line_colors = {
            LineColor.WHITE: tuple(line_colors_cfg.get("white", [255, 255, 255])),
            LineColor.YELLOW: tuple(line_colors_cfg.get("yellow", [0, 255, 255])),
            LineColor.RED: tuple(line_colors_cfg.get("red", [0, 0, 255])),
            LineColor.UNKNOWN: tuple(line_colors_cfg.get("unknown", [200, 200, 200])),
        }

        # Arrow and icon colors
        self.arrow_color = tuple(config.get("arrow_color", [0, 255, 255]))
        self.icon_color = tuple(config.get("icon_color", [0, 255, 255]))
        self.outline_color = tuple(config.get("outline_color", [0, 0, 0]))

    def render(
        self,
        frame: np.ndarray,
        detections: Iterable[DetectionRaw],
        summary: Optional[LaneSummary] = None,
        fitted_lines: Optional[List] = None,
        show_masks: bool = False,
    ) -> np.ndarray:
        img = frame.copy()

        # Draw masks from neural network if requested
        if show_masks:
            for det in detections:
                if det.mask is not None:
                    self._draw_mask(img, det)

        # Convert detections to list and filter by priority
        det_list = list(detections)

        # Remove overlapping detections - keep higher priority ones
        filtered_dets = self._filter_overlapping_detections(det_list)

        # Separate detections by priority (render order: low to high)
        # Priority groups:
        # 1. Lines (lowest priority - draw first, can be covered)
        # 2. Road surface markings (channelizing, box junction)
        # 3. Stop lines and crosswalks
        # 4. Arrows and icons (highest priority - draw last)

        lines = []
        surface_marks = []
        stop_cross = []
        arrows_icons = []

        for det in filtered_dets:
            if det.class_id in self.class_group_lines:  # Lines
                lines.append(det)
            elif det.class_id in self.class_group_surface_marks:  # Box junction, channelizing
                surface_marks.append(det)
            elif det.class_id in self.class_group_stop_cross:  # Crosswalk, stop line
                stop_cross.append(det)
            elif det.class_id in self.class_group_arrows_icons:  # Arrows, icons
                arrows_icons.append(det)

        # Draw in priority order (lowest to highest)

        # 1. Draw fitted lines (lowest priority)
        if fitted_lines:
            # Filter out lines that are overlapped by higher priority objects
            filtered_lines = self._filter_lines_by_detections(fitted_lines, surface_marks + stop_cross + arrows_icons)
            for fitted_line in filtered_lines:
                self._draw_fitted_line(img, fitted_line)

        # 2. Draw surface markings
        for det in surface_marks:
            if det.class_id == 1:
                self._draw_box_junction(img, det)
            elif det.class_id == 16:
                self._draw_channelizing_line(img, det)

        # 3. Draw stop lines and crosswalks
        for det in stop_cross:
            if det.class_id == 2:
                self._draw_crosswalk(img, det)
            elif det.class_id == 3:
                self._draw_stop_line(img, det)

        # 4. Draw arrows and icons (highest priority - always visible)
        logger.debug(f"Drawing {len(arrows_icons)} arrows/icons")
        for det in arrows_icons:
            if det.class_id in {11, 12, 13, 14, 15}:
                logger.debug(f"Attempting to draw arrow: class_id={det.class_id}, confidence={det.confidence:.2f}")
                self._draw_arrow(img, det)
            elif det.class_id in {22, 23}:
                self._draw_icon(img, det)

        if summary:
            text = (
                f"Left: {summary.left_offset_dm}dm  Right: {summary.right_offset_dm}dm  "
                f"Quality: {summary.quality}"
            )
            cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def _color_for_class(self, class_id: int) -> tuple[int, int, int]:
        """Get a consistent color for each class ID."""
        if class_id not in self._color_cache:
            rng = np.random.default_rng(class_id)
            self._color_cache[class_id] = tuple(int(x) for x in rng.integers(0, 255, size=3))
        return self._color_cache[class_id]

    def _draw_mask(self, img: np.ndarray, det: DetectionRaw) -> None:
        """
        Draw detection mask with semi-transparent color overlay.

        Args:
            img: Image to draw on (modified in-place)
            det: Detection with mask to draw
        """
        if det.mask is None:
            return

        # Get color for this class
        color = self._color_for_class(det.class_id)

        # Convert mask to binary
        mask = (det.mask > self.mask_binarization_threshold).astype(np.uint8)

        # Create colored overlay
        overlay = img.copy()
        overlay[mask > 0] = color

        # Blend with original image (semi-transparent)
        cv2.addWeighted(overlay, self.mask_overlay_alpha, img, 1 - self.mask_overlay_alpha, 0, img)

        # Draw mask contour for better visibility
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Draw black outline first
            cv2.drawContours(img, contours, -1, (0, 0, 0), self.mask_contour_outline_thickness)
            # Draw colored contour on top
            cv2.drawContours(img, contours, -1, color, self.mask_contour_color_thickness)

    def _get_detection_priority(self, class_id: int) -> int:
        """
        Get priority for class ID. Higher number = higher priority (drawn last).

        Priority levels are loaded from config.
        """
        if class_id in self.class_group_lines:  # Lines
            return self.priority_lines
        elif class_id in self.class_group_surface_marks:  # Box junction, channelizing
            return self.priority_surface_marks
        elif class_id in self.class_group_stop_cross:  # Crosswalk, stop line
            return self.priority_stop_cross
        elif class_id in self.class_group_arrows_icons:  # Arrows, icons
            return self.priority_arrows_icons
        else:
            return 0  # Unknown

    def _masks_overlap(self, det1: DetectionRaw, det2: DetectionRaw, threshold: Optional[float] = None) -> bool:
        """
        Check if two detection masks overlap significantly.

        Args:
            det1, det2: Detections to check
            threshold: Minimum overlap ratio to consider as overlapping

        Returns:
            True if masks overlap more than threshold
        """
        if threshold is None:
            threshold = self.overlap_threshold

        if det1.mask is None or det2.mask is None:
            return False

        # Convert masks to binary
        mask1 = (det1.mask > self.mask_binarization_threshold).astype(np.uint8)
        mask2 = (det2.mask > self.mask_binarization_threshold).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.sum(mask1 & mask2)
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)

        if area1 == 0 or area2 == 0:
            return False

        # Calculate overlap ratio (intersection / smaller area)
        overlap_ratio = intersection / min(area1, area2)

        return overlap_ratio > threshold

    def _filter_overlapping_detections(self, detections: List[DetectionRaw]) -> List[DetectionRaw]:
        """
        Filter overlapping detections, keeping only the highest priority ones.

        Args:
            detections: List of all detections

        Returns:
            Filtered list with overlapping lower-priority detections removed
        """
        if not detections:
            return []

        # Sort by priority (highest first) and confidence
        sorted_dets = sorted(
            detections,
            key=lambda d: (self._get_detection_priority(d.class_id), d.confidence),
            reverse=True
        )

        filtered = []

        for det in sorted_dets:
            # Check if this detection overlaps with any already accepted higher-priority detection
            is_overlapped = False
            for accepted in filtered:
                # Only filter if overlapped by HIGHER priority
                if self._get_detection_priority(accepted.class_id) > self._get_detection_priority(det.class_id):
                    if self._masks_overlap(det, accepted, threshold=self.overlap_threshold):
                        is_overlapped = True
                        logger.debug(f"Filtering class {det.class_id} overlapped by class {accepted.class_id}")
                        break

            if not is_overlapped:
                filtered.append(det)
            else:
                logger.debug(f"Skipped detection: class_id={det.class_id}, confidence={det.confidence:.2f}")

        return filtered

    def _line_overlaps_detection(self, fitted_line, detection: DetectionRaw, threshold: Optional[float] = None) -> bool:
        """
        Check if a fitted line overlaps with a detection mask.

        Args:
            fitted_line: FittedLine object
            detection: Detection to check against
            threshold: Overlap threshold

        Returns:
            True if line passes through the detection mask
        """
        if threshold is None:
            threshold = self.line_detection_threshold

        if detection.mask is None:
            return False

        # Get line points
        x_start, y_start = fitted_line.start_point
        x_end, y_end = fitted_line.end_point

        if y_start == y_end:
            return False

        # Sample points along the line
        num_points = max(abs(y_end - y_start), self.line_sample_points_min)
        y_values = np.linspace(y_start, y_end, num_points)
        x_values = np.polyval(fitted_line.poly_coeffs, y_values)

        # Check how many line points fall inside the detection mask
        mask = (detection.mask > self.mask_binarization_threshold).astype(np.uint8)
        h, w = mask.shape

        points_in_mask = 0
        valid_points = 0

        for x, y in zip(x_values, y_values):
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                valid_points += 1
                if mask[yi, xi] > 0:
                    points_in_mask += 1

        if valid_points == 0:
            return False

        overlap_ratio = points_in_mask / valid_points
        return overlap_ratio > threshold

    def _filter_lines_by_detections(self, fitted_lines: List, detections: List[DetectionRaw]) -> List:
        """
        Filter out fitted lines that are overlapped by higher-priority detections.

        Args:
            fitted_lines: List of FittedLine objects
            detections: List of higher-priority detections

        Returns:
            Filtered list of lines
        """
        if not fitted_lines or not detections:
            return fitted_lines

        filtered = []
        for line in fitted_lines:
            # Check if this line is overlapped by any detection
            is_overlapped = False
            for det in detections:
                if self._line_overlaps_detection(line, det):
                    is_overlapped = True
                    break

            if not is_overlapped:
                filtered.append(line)

        return filtered

    def _draw_fitted_line(self, img: np.ndarray, fitted_line) -> None:
        """
        Draw a fitted polynomial line on the image.

        Args:
            img: Image to draw on (modified in-place)
            fitted_line: FittedLine object containing polynomial and style info
        """
        # Get line color based on class ID
        line_color = self._get_line_color_from_class(fitted_line.class_id)
        bgr_color = self.line_colors.get(line_color, (255, 255, 255))

        # Get line style (solid, dashed, double)
        line_style = self._get_line_style_from_class(fitted_line.class_id)

        # Generate points along the polynomial
        # Note: poly_coeffs are for x = f(y), not y = f(x)
        x_start, y_start = fitted_line.start_point
        x_end, y_end = fitted_line.end_point

        if y_start == y_end:
            return  # Degenerate line

        # Sample points along y-axis (since x = f(y))
        num_points = max(abs(y_end - y_start), self.line_sample_points_min)
        y_values = np.linspace(y_start, y_end, num_points)
        x_values = np.polyval(fitted_line.poly_coeffs, y_values)

        # Convert to integer pixel coordinates
        points = np.column_stack([x_values, y_values]).astype(np.int32)

        # Clip to image bounds
        h, w = img.shape[:2]
        valid_mask = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        points = points[valid_mask]

        if len(points) < 2:
            return

        # Draw based on line style with black outline for visibility
        if line_style == LaneType.SOLID:
            self._draw_solid_line(img, points, bgr_color, thickness=self.line_thickness)
        elif line_style == LaneType.DASHED:
            self._draw_dashed_line(img, points, bgr_color, thickness=self.line_thickness)
        elif line_style == LaneType.DOUBLE:
            self._draw_double_line(img, points, bgr_color, thickness=self.line_thickness)
        else:
            # Default: solid line
            self._draw_solid_line(img, points, bgr_color, thickness=self.line_thickness)

    def _draw_solid_line(self, img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
        """Draw a solid line through points with black outline."""
        # Draw black outline first (thicker)
        cv2.polylines(img, [points], isClosed=False, color=(0, 0, 0), thickness=thickness + self.line_outline_offset, lineType=cv2.LINE_AA)
        # Draw colored line on top
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: Optional[int] = None,
        gap_length: Optional[int] = None,
    ) -> None:
        """Draw a dashed line through points."""
        if dash_length is None:
            dash_length = self.dash_length
        if gap_length is None:
            gap_length = self.gap_length

        if len(points) < 2:
            return

        total_length = 0.0
        segment_lengths = []

        # Calculate cumulative lengths
        for i in range(len(points) - 1):
            segment_len = np.linalg.norm(points[i + 1] - points[i])
            segment_lengths.append(segment_len)
            total_length += segment_len

        # Draw dashes
        current_length = 0.0
        is_dash = True
        segment_idx = 0
        segment_progress = 0.0

        while current_length < total_length and segment_idx < len(segment_lengths):
            if is_dash:
                target_length = current_length + dash_length
            else:
                target_length = current_length + gap_length

            # Find start point
            start_pt = self._interpolate_point(points, segment_lengths, current_length)

            # Find end point
            end_length = min(target_length, total_length)
            end_pt = self._interpolate_point(points, segment_lengths, end_length)

            if is_dash and start_pt is not None and end_pt is not None:
                # Draw black outline first
                cv2.line(img, tuple(start_pt), tuple(end_pt), (0, 0, 0), thickness + self.line_outline_offset, lineType=cv2.LINE_AA)
                # Draw colored dash on top
                cv2.line(img, tuple(start_pt), tuple(end_pt), color, thickness, lineType=cv2.LINE_AA)

            current_length = end_length
            is_dash = not is_dash

    def _draw_double_line(
        self,
        img: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
        offset: Optional[int] = None,
    ) -> None:
        """Draw a double solid line through points."""
        if offset is None:
            offset = self.double_offset

        if len(points) < 2:
            return

        # Calculate perpendicular offsets
        left_points = []
        right_points = []

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            # Direction vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)

            if length < 1e-6:
                continue

            # Perpendicular vector (normalized)
            perp_x = -dy / length
            perp_y = dx / length

            # Offset points
            left_pt = (int(p1[0] + perp_x * offset), int(p1[1] + perp_y * offset))
            right_pt = (int(p1[0] - perp_x * offset), int(p1[1] - perp_y * offset))

            left_points.append(left_pt)
            right_points.append(right_pt)

        # Add last point
        if len(points) > 0:
            p_last = points[-1]
            if len(left_points) > 0:
                # Use same offset as previous
                perp_x = (left_points[-1][0] - points[-2][0]) / offset if len(points) > 1 else 0
                perp_y = (left_points[-1][1] - points[-2][1]) / offset if len(points) > 1 else 0
                left_points.append((int(p_last[0] + perp_x * offset), int(p_last[1] + perp_y * offset)))
                right_points.append((int(p_last[0] - perp_x * offset), int(p_last[1] - perp_y * offset)))

        # Draw both lines with black outline
        if len(left_points) > 1:
            left_arr = np.array(left_points, dtype=np.int32)
            right_arr = np.array(right_points, dtype=np.int32)
            line_thickness = thickness // 2

            # Draw black outlines first
            cv2.polylines(img, [left_arr], isClosed=False, color=(0, 0, 0), thickness=line_thickness + self.line_outline_offset, lineType=cv2.LINE_AA)
            cv2.polylines(img, [right_arr], isClosed=False, color=(0, 0, 0), thickness=line_thickness + self.line_outline_offset, lineType=cv2.LINE_AA)

            # Draw colored lines on top
            cv2.polylines(img, [left_arr], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
            cv2.polylines(img, [right_arr], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _interpolate_point(points: np.ndarray, segment_lengths: List[float], target_length: float) -> Optional[np.ndarray]:
        """Interpolate a point at a given distance along the polyline."""
        if target_length <= 0:
            return points[0]

        cumulative_length = 0.0
        for i, seg_len in enumerate(segment_lengths):
            if cumulative_length + seg_len >= target_length:
                # Interpolate within this segment
                t = (target_length - cumulative_length) / seg_len if seg_len > 0 else 0
                pt = points[i] + t * (points[i + 1] - points[i])
                return pt.astype(np.int32)
            cumulative_length += seg_len

        return points[-1] if len(points) > 0 else None

    @staticmethod
    def _get_line_color_from_class(class_id: int) -> LineColor:
        """Map class ID to line color."""
        if class_id in {5, 8, 10}:  # yellow lines
            return LineColor.YELLOW
        if class_id == 6:  # red line
            return LineColor.RED
        return LineColor.WHITE

    @staticmethod
    def _get_line_style_from_class(class_id: int) -> LaneType:
        """Map class ID to line style."""
        if class_id in {4, 5, 6}:  # solid single
            return LaneType.SOLID
        if class_id in {7, 8}:  # double
            return LaneType.DOUBLE
        if class_id in {9, 10}:  # dashed
            return LaneType.DASHED
        return LaneType.UNKNOWN

    def _draw_arrow(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw arrow based on detection with perspective transformation."""
        if det.mask is None:
            return

        # Get arrow direction from class_id
        # 11: left, 12: straight, 13: right, 14: left_straight, 15: right_straight
        arrow_type = self._get_arrow_type(det.class_id)

        # Extract mask contour for perspective
        mask = (det.mask > self.mask_binarization_threshold).astype(np.uint8) * 255

        # Check if mask has enough pixels
        mask_area = np.sum(mask > 0)
        if mask_area < self.arrow_min_mask_area:
            logger.debug(f"Arrow mask too small: {mask_area} pixels, class_id={det.class_id}")
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.debug(f"No contours found for arrow, class_id={det.class_id}")
            return

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)

        if contour_area < self.arrow_min_contour_area:
            logger.debug(f"Arrow contour too small: {contour_area}, class_id={det.class_id}")
            return

        # Get rotated bounding box for perspective
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Check if box is degenerate
        box_width = np.linalg.norm(box[0] - box[1])
        box_height = np.linalg.norm(box[1] - box[2])

        if box_width < self.arrow_min_box_width or box_height < self.arrow_min_box_height:
            logger.debug(f"Arrow box too small: {box_width:.1f}x{box_height:.1f}, class_id={det.class_id}")
            return

        # Sort points: top-left, top-right, bottom-right, bottom-left
        try:
            box_sorted = self._sort_box_points(box)
        except Exception as e:
            logger.debug(f"Failed to sort arrow box: {e}, class_id={det.class_id}")
            return

        # Create arrow template
        arrow_img = self._create_arrow_template(arrow_type)

        # Warp arrow to fit detection with perspective
        try:
            self._warp_and_draw(img, arrow_img, box_sorted)
        except Exception as e:
            logger.debug(f"Failed to warp arrow: {e}, class_id={det.class_id}")

    @staticmethod
    def _get_arrow_type(class_id: int) -> str:
        """Get arrow type from class ID."""
        arrow_types = {
            11: "left",
            12: "straight",
            13: "right",
            14: "left_straight",
            15: "right_straight",
        }
        return arrow_types.get(class_id, "straight")

    def _create_arrow_template(self, arrow_type: str, size: Optional[int] = None) -> np.ndarray:
        """Create arrow template image with bright color and black outline."""
        if size is None:
            size = self.arrow_template_size

        img = np.zeros((size, size, 4), dtype=np.uint8)  # RGBA

        # Use bright cyan/green color for better visibility
        color = tuple(list(self.arrow_color) + [255])  # Add alpha
        outline_color = tuple(list(self.outline_color) + [255])  # Add alpha
        thickness = self.arrow_thickness
        outline_thickness = self.arrow_outline_thickness
        tip_size = self.arrow_tip_size

        center_x, center_y = size // 2, size // 2

        if arrow_type == "straight":
            # Vertical arrow up - draw outline first
            cv2.line(img, (center_x, size - 40), (center_x, 40), outline_color, outline_thickness)
            pts_outline = np.array([
                [center_x, 20],
                [center_x - tip_size - 5, 20 + tip_size + 5],
                [center_x + tip_size + 5, 20 + tip_size + 5]
            ], np.int32)
            cv2.fillPoly(img, [pts_outline], outline_color)

            # Draw colored arrow on top
            cv2.line(img, (center_x, size - 40), (center_x, 40), color, thickness)
            pts = np.array([
                [center_x, 20],
                [center_x - tip_size, 20 + tip_size],
                [center_x + tip_size, 20 + tip_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif arrow_type == "left":
            # Arrow pointing left - outline first
            cv2.line(img, (size - 40, center_y), (40, center_y), outline_color, outline_thickness)
            pts_outline = np.array([
                [20, center_y],
                [20 + tip_size + 5, center_y - tip_size - 5],
                [20 + tip_size + 5, center_y + tip_size + 5]
            ], np.int32)
            cv2.fillPoly(img, [pts_outline], outline_color)

            # Colored arrow on top
            cv2.line(img, (size - 40, center_y), (40, center_y), color, thickness)
            pts = np.array([
                [20, center_y],
                [20 + tip_size, center_y - tip_size],
                [20 + tip_size, center_y + tip_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif arrow_type == "right":
            # Arrow pointing right - outline first
            cv2.line(img, (40, center_y), (size - 40, center_y), outline_color, outline_thickness)
            pts_outline = np.array([
                [size - 20, center_y],
                [size - 20 - tip_size - 5, center_y - tip_size - 5],
                [size - 20 - tip_size - 5, center_y + tip_size + 5]
            ], np.int32)
            cv2.fillPoly(img, [pts_outline], outline_color)

            # Colored arrow on top
            cv2.line(img, (40, center_y), (size - 40, center_y), color, thickness)
            pts = np.array([
                [size - 20, center_y],
                [size - 20 - tip_size, center_y - tip_size],
                [size - 20 - tip_size, center_y + tip_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif arrow_type == "left_straight":
            # Combined left and straight with outlines
            comb_thick = self.arrow_combined_thickness
            comb_outline = self.arrow_combined_outline
            tip_s = self.arrow_combined_tip_size

            # Straight arrow - outline
            cv2.line(img, (center_x + 30, size - 40), (center_x + 30, 40), outline_color, comb_outline)
            pts_s_outline = np.array([
                [center_x + 30, 20],
                [center_x + 30 - tip_s - 3, 20 + tip_s + 3],
                [center_x + 30 + tip_s + 3, 20 + tip_s + 3]
            ], np.int32)
            cv2.fillPoly(img, [pts_s_outline], outline_color)

            # Left arrow - outline
            cv2.line(img, (size - 40, center_y - 30), (40, center_y - 30), outline_color, comb_outline)
            pts_l_outline = np.array([
                [20, center_y - 30],
                [20 + tip_s + 3, center_y - 30 - tip_s - 3],
                [20 + tip_s + 3, center_y - 30 + tip_s + 3]
            ], np.int32)
            cv2.fillPoly(img, [pts_l_outline], outline_color)

            # Colored arrows on top
            cv2.line(img, (center_x + 30, size - 40), (center_x + 30, 40), color, comb_thick)
            pts_straight = np.array([
                [center_x + 30, 20],
                [center_x + 30 - tip_s, 20 + tip_s],
                [center_x + 30 + tip_s, 20 + tip_s]
            ], np.int32)
            cv2.fillPoly(img, [pts_straight], color)

            cv2.line(img, (size - 40, center_y - 30), (40, center_y - 30), color, comb_thick)
            pts_left = np.array([
                [20, center_y - 30],
                [20 + tip_s, center_y - 30 - tip_s],
                [20 + tip_s, center_y - 30 + tip_s]
            ], np.int32)
            cv2.fillPoly(img, [pts_left], color)

        elif arrow_type == "right_straight":
            # Combined right and straight with outlines
            comb_thick = self.arrow_combined_thickness
            comb_outline = self.arrow_combined_outline
            tip_s = self.arrow_combined_tip_size

            # Straight arrow - outline
            cv2.line(img, (center_x - 30, size - 40), (center_x - 30, 40), outline_color, comb_outline)
            pts_s_outline = np.array([
                [center_x - 30, 20],
                [center_x - 30 - tip_s - 3, 20 + tip_s + 3],
                [center_x - 30 + tip_s + 3, 20 + tip_s + 3]
            ], np.int32)
            cv2.fillPoly(img, [pts_s_outline], outline_color)

            # Right arrow - outline
            cv2.line(img, (40, center_y + 30), (size - 40, center_y + 30), outline_color, comb_outline)
            pts_r_outline = np.array([
                [size - 20, center_y + 30],
                [size - 20 - tip_s - 3, center_y + 30 - tip_s - 3],
                [size - 20 - tip_s - 3, center_y + 30 + tip_s + 3]
            ], np.int32)
            cv2.fillPoly(img, [pts_r_outline], outline_color)

            # Colored arrows on top
            cv2.line(img, (center_x - 30, size - 40), (center_x - 30, 40), color, comb_thick)
            pts_straight = np.array([
                [center_x - 30, 20],
                [center_x - 30 - tip_s, 20 + tip_s],
                [center_x - 30 + tip_s, 20 + tip_s]
            ], np.int32)
            cv2.fillPoly(img, [pts_straight], color)

            cv2.line(img, (40, center_y + 30), (size - 40, center_y + 30), color, comb_thick)
            pts_right = np.array([
                [size - 20, center_y + 30],
                [size - 20 - tip_s, center_y + 30 - tip_s],
                [size - 20 - tip_s, center_y + 30 + tip_s]
            ], np.int32)
            cv2.fillPoly(img, [pts_right], color)

        return img

    @staticmethod
    def _sort_box_points(box: np.ndarray) -> np.ndarray:
        """Sort box points: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y coordinate
        sorted_by_y = box[np.argsort(box[:, 1])]

        # Top two points
        top_points = sorted_by_y[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]  # Sort by x

        # Bottom two points
        bottom_points = sorted_by_y[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Sort by x

        return np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.float32)

    def _warp_and_draw(self, img: np.ndarray, overlay: np.ndarray, dst_points: np.ndarray, alpha: Optional[float] = None) -> None:
        """Warp overlay image to fit dst_points and blend with img."""
        if alpha is None:
            alpha = self.arrow_alpha

        h, w = overlay.shape[:2]

        # Source points (corners of overlay)
        src_points = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        # Get perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Warp overlay
        warped = cv2.warpPerspective(overlay, matrix, (img.shape[1], img.shape[0]))

        # Extract alpha channel if exists
        if warped.shape[2] == 4:
            alpha_channel = warped[:, :, 3] / 255.0 * alpha
            overlay_rgb = warped[:, :, :3]
        else:
            # Create alpha from non-black pixels
            alpha_channel = np.any(warped > 0, axis=2).astype(np.float32) * alpha
            overlay_rgb = warped

        # Blend with increased opacity for better visibility
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - alpha_channel) + overlay_rgb[:, :, c] * alpha_channel

    def _draw_crosswalk(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw crosswalk with perspective-aware stripes."""
        if det.mask is None:
            return

        # Get mask contour
        mask = (det.mask > self.mask_binarization_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box_sorted = self._sort_box_points(box)

        # Create crosswalk template
        crosswalk_img = self._create_crosswalk_template()

        # Warp and draw with higher opacity
        self._warp_and_draw(img, crosswalk_img, box_sorted, alpha=self.crosswalk_alpha)

    def _create_crosswalk_template(self, size: Optional[int] = None) -> np.ndarray:
        """Create crosswalk template with stripes."""
        if size is None:
            size = self.crosswalk_template_size

        img = np.zeros((size, size, 4), dtype=np.uint8)
        color = (255, 255, 255, 255)  # White

        # Draw vertical stripes
        num_stripes = self.crosswalk_num_stripes
        stripe_width = size // (num_stripes * 2)

        for i in range(num_stripes):
            x = i * stripe_width * 2 + stripe_width // 2
            cv2.rectangle(img, (x, 0), (x + stripe_width, size), color, -1)

        return img

    def _draw_stop_line(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw stop line with outline for visibility."""
        if det.mask is None:
            return

        # Get mask contour
        mask = (det.mask > self.mask_binarization_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Draw black outline first
        cv2.drawContours(img, [box], 0, (0, 0, 0), self.stop_line_outline_thickness)
        # Draw bright white line on top
        cv2.drawContours(img, [box], 0, (255, 255, 255), self.stop_line_line_thickness)

    def _draw_box_junction(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw box junction (grid pattern)."""
        if det.mask is None:
            return

        mask = (det.mask > self.mask_binarization_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_sorted = self._sort_box_points(box)

        # Create box junction template (diagonal lines)
        junction_img = self._create_box_junction_template()
        self._warp_and_draw(img, junction_img, box_sorted, alpha=self.box_junction_alpha)

    def _create_box_junction_template(self, size: Optional[int] = None) -> np.ndarray:
        """Create box junction template with diagonal grid and bright colors."""
        if size is None:
            size = self.box_junction_template_size

        img = np.zeros((size, size, 4), dtype=np.uint8)
        color = (0, 255, 255, 255)  # Bright yellow (BGR)

        thickness = self.box_junction_line_thickness
        spacing = self.box_junction_line_spacing

        # Draw diagonal lines (both directions)
        for offset in range(-size, size * 2, spacing):
            # Top-left to bottom-right
            cv2.line(img, (offset, 0), (offset + size, size), color, thickness)
            # Top-right to bottom-left
            cv2.line(img, (size - offset, 0), (-offset, size), color, thickness)

        return img

    def _draw_channelizing_line(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw channelizing line with bright yellow color."""
        if det.mask is None:
            return

        mask = (det.mask > self.mask_binarization_threshold).astype(np.uint8) * 255
        # Draw mask with bright yellow color
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[mask > 0] = [0, 255, 255]  # Bright yellow (BGR)

        # Blend with higher opacity
        mask_f = (mask.astype(np.float32) / 255.0) * self.channelizing_alpha
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask_f) + colored_mask[:, :, c] * mask_f

    def _draw_icon(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw icons (motor, bike) as bright symbols with outline."""
        if det.bbox is None:
            return

        x1, y1, x2, y2 = det.bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = min(x2 - x1, y2 - y1)

        # Use config colors
        icon_color = self.icon_color
        outline_color = self.outline_color
        thickness = self.icon_base_thickness

        if det.class_id == 22:  # motor_icon
            # Draw simple car shape with outline
            outline_thick = thickness + self.icon_outline_offset
            # Outline
            cv2.rectangle(img, (center_x - size // 3, center_y - size // 4),
                         (center_x + size // 3, center_y + size // 4), outline_color, outline_thick)
            cv2.circle(img, (center_x - size // 4, center_y + size // 4), size // 10, outline_color, outline_thick)
            cv2.circle(img, (center_x + size // 4, center_y + size // 4), size // 10, outline_color, outline_thick)
            # Icon
            cv2.rectangle(img, (center_x - size // 3, center_y - size // 4),
                         (center_x + size // 3, center_y + size // 4), icon_color, thickness)
            cv2.circle(img, (center_x - size // 4, center_y + size // 4), size // 10, icon_color, -1)
            cv2.circle(img, (center_x + size // 4, center_y + size // 4), size // 10, icon_color, -1)

        elif det.class_id == 23:  # bike_icon
            # Draw simple bike shape with outline
            outline_thick = thickness + self.icon_outline_offset
            # Outline
            cv2.circle(img, (center_x - size // 4, center_y), size // 8, outline_color, outline_thick)
            cv2.circle(img, (center_x + size // 4, center_y), size // 8, outline_color, outline_thick)
            cv2.line(img, (center_x - size // 4, center_y), (center_x + size // 4, center_y), outline_color, outline_thick)
            # Icon
            cv2.circle(img, (center_x - size // 4, center_y), size // 8, icon_color, thickness)
            cv2.circle(img, (center_x + size // 4, center_y), size // 8, icon_color, thickness)
            cv2.line(img, (center_x - size // 4, center_y), (center_x + size // 4, center_y), icon_color, thickness)
