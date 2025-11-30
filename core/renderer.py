from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from core.detections import DetectionRaw, LaneSummary, LineColor, LaneType


class Renderer:
    def __init__(self, class_names: Optional[Dict[int, str]] = None) -> None:
        self.class_names = class_names or {}
        self._color_cache: Dict[int, tuple[int, int, int]] = {}

        # Define colors for lane markings (BGR format for OpenCV)
        self.line_colors = {
            LineColor.WHITE: (255, 255, 255),
            LineColor.YELLOW: (0, 255, 255),
            LineColor.RED: (0, 0, 255),
            LineColor.UNKNOWN: (128, 128, 128),
        }

    def render(
        self,
        frame: np.ndarray,
        detections: Iterable[DetectionRaw],
        summary: Optional[LaneSummary] = None,
        fitted_lines: Optional[List] = None,
    ) -> np.ndarray:
        img = frame.copy()

        # Draw fitted lines (classes 4-10)
        if fitted_lines:
            for fitted_line in fitted_lines:
                self._draw_fitted_line(img, fitted_line)

        # Draw other objects (arrows, crosswalks, etc.) with perspective
        for det in detections:
            # Skip line classes (they are drawn via fitted_lines)
            if det.class_id in {4, 5, 6, 7, 8, 9, 10}:
                continue

            # Draw objects based on class
            if det.class_id in {11, 12, 13, 14, 15}:  # Arrows
                self._draw_arrow(img, det)
            elif det.class_id == 2:  # Crosswalk
                self._draw_crosswalk(img, det)
            elif det.class_id == 3:  # Stop line
                self._draw_stop_line(img, det)
            elif det.class_id == 1:  # Box junction
                self._draw_box_junction(img, det)
            elif det.class_id in {16}:  # Channelizing line
                self._draw_channelizing_line(img, det)
            elif det.class_id in {22, 23}:  # Icons (motor, bike)
                self._draw_icon(img, det)
            # Add more object types as needed

        if summary:
            text = (
                f"Left: {summary.left_offset_dm}dm  Right: {summary.right_offset_dm}dm  "
                f"Quality: {summary.quality}"
            )
            cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def _color_for_class(self, class_id: int) -> tuple[int, int, int]:
        if class_id not in self._color_cache:
            rng = np.random.default_rng(class_id)
            self._color_cache[class_id] = tuple(int(x) for x in rng.integers(0, 255, size=3))
        return self._color_cache[class_id]

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
        x_start, y_start = fitted_line.start_point
        x_end, y_end = fitted_line.end_point

        if x_start == x_end:
            return  # Degenerate line

        # Sample points along x-axis
        num_points = max(abs(x_end - x_start), 50)
        x_values = np.linspace(x_start, x_end, num_points)
        y_values = np.polyval(fitted_line.poly_coeffs, x_values)

        # Convert to integer pixel coordinates
        points = np.column_stack([x_values, y_values]).astype(np.int32)

        # Clip to image bounds
        h, w = img.shape[:2]
        valid_mask = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        points = points[valid_mask]

        if len(points) < 2:
            return

        # Draw based on line style
        if line_style == LaneType.SOLID:
            self._draw_solid_line(img, points, bgr_color, thickness=4)
        elif line_style == LaneType.DASHED:
            self._draw_dashed_line(img, points, bgr_color, thickness=4)
        elif line_style == LaneType.DOUBLE:
            self._draw_double_line(img, points, bgr_color, thickness=4)
        else:
            # Default: solid line
            self._draw_solid_line(img, points, bgr_color, thickness=4)

    def _draw_solid_line(self, img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int = 4) -> None:
        """Draw a solid line through points."""
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 4,
        dash_length: int = 20,
        gap_length: int = 10,
    ) -> None:
        """Draw a dashed line through points."""
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
                cv2.line(img, tuple(start_pt), tuple(end_pt), color, thickness, lineType=cv2.LINE_AA)

            current_length = end_length
            is_dash = not is_dash

    def _draw_double_line(
        self,
        img: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 4,
        offset: int = 6,
    ) -> None:
        """Draw a double solid line through points."""
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

        # Draw both lines
        if len(left_points) > 1:
            cv2.polylines(img, [np.array(left_points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness // 2, lineType=cv2.LINE_AA)
            cv2.polylines(img, [np.array(right_points, dtype=np.int32)], isClosed=False, color=color, thickness=thickness // 2, lineType=cv2.LINE_AA)

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
        if det.mask is None or det.bbox is None:
            return

        # Get arrow direction from class_id
        # 11: left, 12: straight, 13: right, 14: left_straight, 15: right_straight
        arrow_type = self._get_arrow_type(det.class_id)

        # Extract mask contour for perspective
        mask = (det.mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Get rotated bounding box for perspective
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Sort points: top-left, top-right, bottom-right, bottom-left
        box_sorted = self._sort_box_points(box)

        # Create arrow template
        arrow_img = self._create_arrow_template(arrow_type)

        # Warp arrow to fit detection with perspective
        self._warp_and_draw(img, arrow_img, box_sorted)

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

    @staticmethod
    def _create_arrow_template(arrow_type: str, size: int = 200) -> np.ndarray:
        """Create arrow template image."""
        img = np.zeros((size, size, 4), dtype=np.uint8)  # RGBA
        color = (255, 255, 255, 255)  # White with alpha
        thickness = 20
        tip_size = 40

        center_x, center_y = size // 2, size // 2

        if arrow_type == "straight":
            # Vertical arrow up
            cv2.line(img, (center_x, size - 40), (center_x, 40), color, thickness)
            # Arrow head
            pts = np.array([
                [center_x, 20],
                [center_x - tip_size, 20 + tip_size],
                [center_x + tip_size, 20 + tip_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif arrow_type == "left":
            # Arrow pointing left
            cv2.line(img, (size - 40, center_y), (40, center_y), color, thickness)
            # Arrow head
            pts = np.array([
                [20, center_y],
                [20 + tip_size, center_y - tip_size],
                [20 + tip_size, center_y + tip_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif arrow_type == "right":
            # Arrow pointing right
            cv2.line(img, (40, center_y), (size - 40, center_y), color, thickness)
            # Arrow head
            pts = np.array([
                [size - 20, center_y],
                [size - 20 - tip_size, center_y - tip_size],
                [size - 20 - tip_size, center_y + tip_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

        elif arrow_type == "left_straight":
            # Combined left and straight
            # Straight arrow
            cv2.line(img, (center_x + 30, size - 40), (center_x + 30, 40), color, thickness - 5)
            pts_straight = np.array([
                [center_x + 30, 20],
                [center_x + 30 - 30, 20 + 30],
                [center_x + 30 + 30, 20 + 30]
            ], np.int32)
            cv2.fillPoly(img, [pts_straight], color)
            # Left arrow
            cv2.line(img, (size - 40, center_y - 30), (40, center_y - 30), color, thickness - 5)
            pts_left = np.array([
                [20, center_y - 30],
                [20 + 30, center_y - 30 - 30],
                [20 + 30, center_y - 30 + 30]
            ], np.int32)
            cv2.fillPoly(img, [pts_left], color)

        elif arrow_type == "right_straight":
            # Combined right and straight
            # Straight arrow
            cv2.line(img, (center_x - 30, size - 40), (center_x - 30, 40), color, thickness - 5)
            pts_straight = np.array([
                [center_x - 30, 20],
                [center_x - 30 - 30, 20 + 30],
                [center_x - 30 + 30, 20 + 30]
            ], np.int32)
            cv2.fillPoly(img, [pts_straight], color)
            # Right arrow
            cv2.line(img, (40, center_y + 30), (size - 40, center_y + 30), color, thickness - 5)
            pts_right = np.array([
                [size - 20, center_y + 30],
                [size - 20 - 30, center_y + 30 - 30],
                [size - 20 - 30, center_y + 30 + 30]
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

    @staticmethod
    def _warp_and_draw(img: np.ndarray, overlay: np.ndarray, dst_points: np.ndarray, alpha: float = 0.8) -> None:
        """Warp overlay image to fit dst_points and blend with img."""
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

        # Blend
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - alpha_channel) + overlay_rgb[:, :, c] * alpha_channel

    def _draw_crosswalk(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw crosswalk with perspective-aware stripes."""
        if det.mask is None:
            return

        # Get mask contour
        mask = (det.mask > 127).astype(np.uint8) * 255
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

        # Warp and draw
        self._warp_and_draw(img, crosswalk_img, box_sorted, alpha=0.7)

    @staticmethod
    def _create_crosswalk_template(size: int = 200) -> np.ndarray:
        """Create crosswalk template with stripes."""
        img = np.zeros((size, size, 4), dtype=np.uint8)
        color = (255, 255, 255, 255)  # White

        # Draw vertical stripes
        num_stripes = 8
        stripe_width = size // (num_stripes * 2)

        for i in range(num_stripes):
            x = i * stripe_width * 2 + stripe_width // 2
            cv2.rectangle(img, (x, 0), (x + stripe_width, size), color, -1)

        return img

    def _draw_stop_line(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw stop line."""
        if det.mask is None:
            return

        # Get mask contour
        mask = (det.mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Draw thick white line along the contour
        cv2.drawContours(img, [box], 0, (255, 255, 255), 8)

    def _draw_box_junction(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw box junction (grid pattern)."""
        if det.mask is None:
            return

        mask = (det.mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_sorted = self._sort_box_points(box)

        # Create box junction template (diagonal lines)
        junction_img = self._create_box_junction_template()
        self._warp_and_draw(img, junction_img, box_sorted, alpha=0.6)

    @staticmethod
    def _create_box_junction_template(size: int = 200) -> np.ndarray:
        """Create box junction template with diagonal grid."""
        img = np.zeros((size, size, 4), dtype=np.uint8)
        color = (0, 255, 255, 255)  # Yellow

        thickness = 5
        spacing = 30

        # Draw diagonal lines (both directions)
        for offset in range(-size, size * 2, spacing):
            # Top-left to bottom-right
            cv2.line(img, (offset, 0), (offset + size, size), color, thickness)
            # Top-right to bottom-left
            cv2.line(img, (size - offset, 0), (-offset, size), color, thickness)

        return img

    def _draw_channelizing_line(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw channelizing line (similar to solid line but thicker)."""
        if det.mask is None:
            return

        mask = (det.mask > 127).astype(np.uint8) * 255
        # Draw mask with yellow color
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[mask > 0] = [0, 255, 255]  # Yellow

        # Blend
        alpha = 0.6
        mask_f = (mask.astype(np.float32) / 255.0) * alpha
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask_f) + colored_mask[:, :, c] * mask_f

    def _draw_icon(self, img: np.ndarray, det: DetectionRaw) -> None:
        """Draw icons (motor, bike) as simple symbols."""
        if det.bbox is None:
            return

        x1, y1, x2, y2 = det.bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = min(x2 - x1, y2 - y1)

        if det.class_id == 22:  # motor_icon
            # Draw simple car shape
            cv2.rectangle(img, (center_x - size // 3, center_y - size // 4),
                         (center_x + size // 3, center_y + size // 4), (255, 255, 255), 2)
            cv2.circle(img, (center_x - size // 4, center_y + size // 4), size // 10, (255, 255, 255), -1)
            cv2.circle(img, (center_x + size // 4, center_y + size // 4), size // 10, (255, 255, 255), -1)

        elif det.class_id == 23:  # bike_icon
            # Draw simple bike shape
            cv2.circle(img, (center_x - size // 4, center_y), size // 8, (255, 255, 255), 2)
            cv2.circle(img, (center_x + size // 4, center_y), size // 8, (255, 255, 255), 2)
            cv2.line(img, (center_x - size // 4, center_y), (center_x + size // 4, center_y), (255, 255, 255), 2)
