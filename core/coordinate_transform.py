"""
Coordinate transformation utilities for converting pixel coordinates to real-world meters.

Uses perspective transformation matrix to convert image pixel coordinates
to real-world coordinates on the road surface.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Transform pixel coordinates to real-world road surface coordinates in meters.

    Coordinate system:
    - Origin: At camera position
    - X axis: Points right (perpendicular to direction of motion)
    - Y axis: Points forward (direction of motion, splits image vertically in half)
    - Z axis: Points up (not used for road markings)

    The perspective transform matrix maps image pixels to meters on the road surface.
    """

    def __init__(
        self,
        perspective_matrix_path: str = "config/PerspectiveTransform.yaml",
        calibration_path: str = "config/calibration.json"
    ):
        """
        Initialize coordinate transformer.

        Args:
            perspective_matrix_path: Path to perspective transformation matrix YAML
            calibration_path: Path to camera calibration JSON
        """
        self.perspective_matrix = self._load_perspective_matrix(perspective_matrix_path)
        self.calibration = self._load_calibration(calibration_path)

        # Get image center from calibration
        if self.calibration:
            self.image_width = self.calibration.get("image_width", 1920)
            self.image_height = self.calibration.get("image_height", 1080)
            self.center_x = self.image_width / 2.0
        else:
            self.image_width = 1920
            self.image_height = 1080
            self.center_x = 960.0

    def _load_perspective_matrix(self, config_path: str) -> Optional[np.ndarray]:
        """Load perspective transformation matrix from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Perspective transform config not found at %s", config_path)
            return None

        try:
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            matrix_node = fs.getNode("perspectiveTransformMatrix")
            if matrix_node.empty():
                logger.error("perspectiveTransformMatrix not found in %s", config_path)
                return None

            matrix = matrix_node.mat()
            fs.release()

            if matrix is None or matrix.shape != (3, 3):
                logger.error("Invalid perspective matrix shape: %s", matrix.shape if matrix is not None else None)
                return None

            logger.info("Loaded perspective transform matrix from %s", config_path)
            return matrix
        except Exception as e:
            logger.error("Failed to load perspective matrix: %s", e)
            return None

    def _load_calibration(self, config_path: str) -> Optional[Dict]:
        """Load camera calibration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Calibration config not found at %s", config_path)
            return None

        try:
            with path.open("r") as f:
                calibration = json.load(f)
            logger.info("Loaded camera calibration from %s", config_path)
            return calibration
        except Exception as e:
            logger.error("Failed to load calibration: %s", e)
            return None

    def pixel_to_meters(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to real-world meters on road surface.

        Args:
            x_px: X coordinate in pixels
            y_px: Y coordinate in pixels

        Returns:
            Tuple of (x_meters, y_meters) where:
            - x_meters: lateral position in meters (positive = right, negative = left)
            - y_meters: forward distance in meters (always positive)
        """
        if self.perspective_matrix is None:
            # Fallback: use simple scaling if no perspective matrix available
            logger.warning("No perspective matrix, using fallback pixel scaling")
            x_m = (x_px - self.center_x) * 0.01  # rough estimate
            y_m = (self.image_height - y_px) * 0.01
            return (x_m, y_m)

        # Apply perspective transformation
        # Input point in homogeneous coordinates
        point_px = np.array([x_px, y_px, 1.0])

        # Transform: [x', y', w'] = M * [x, y, 1]
        point_transformed = self.perspective_matrix @ point_px

        # Normalize by w' to get actual coordinates
        w = point_transformed[2]
        if abs(w) < 1e-6:
            logger.warning("Perspective transform resulted in near-zero w: %f", w)
            return (0.0, 0.0)

        x_m = point_transformed[0] / w
        y_m = point_transformed[1] / w

        return (x_m, y_m)

    def pixels_to_meters_batch(self, points_px: np.ndarray) -> np.ndarray:
        """
        Convert multiple pixel coordinates to meters.

        Args:
            points_px: Nx2 array of (x, y) pixel coordinates

        Returns:
            Nx2 array of (x, y) coordinates in meters
        """
        if self.perspective_matrix is None:
            logger.warning("No perspective matrix, using fallback pixel scaling")
            points_m = points_px.copy().astype(np.float32)
            points_m[:, 0] = (points_m[:, 0] - self.center_x) * 0.01
            points_m[:, 1] = (self.image_height - points_m[:, 1]) * 0.01
            return points_m

        # Convert points to homogeneous coordinates
        n_points = len(points_px)
        points_homog = np.hstack([points_px, np.ones((n_points, 1))]).T  # 3xN

        # Apply transformation
        points_transformed = self.perspective_matrix @ points_homog  # 3xN

        # Normalize by w coordinate
        w = points_transformed[2, :]
        w = np.where(np.abs(w) < 1e-6, 1e-6, w)  # Avoid division by zero

        points_m = np.zeros((n_points, 2), dtype=np.float32)
        points_m[:, 0] = points_transformed[0, :] / w
        points_m[:, 1] = points_transformed[1, :] / w

        return points_m

    def get_line_points_in_meters(
        self,
        poly_coeffs: np.ndarray,
        y_start_px: int,
        y_end_px: int,
        num_samples: int = 3
    ) -> List[Tuple[float, float]]:
        """
        Sample points along a polynomial line and convert to meters.

        Args:
            poly_coeffs: Polynomial coefficients [a, b, c] for x = ay^2 + by + c
            y_start_px: Starting Y coordinate in pixels
            y_end_px: Ending Y coordinate in pixels
            num_samples: Number of points to sample (default: 3 for top, middle, bottom)

        Returns:
            List of (x_meters, y_meters) tuples
        """
        # Sample Y coordinates evenly along the line
        y_values = np.linspace(y_start_px, y_end_px, num_samples)

        # Calculate X coordinates using polynomial
        x_values = np.polyval(poly_coeffs, y_values)

        # Stack into Nx2 array
        points_px = np.column_stack([x_values, y_values])

        # Convert to meters
        points_m = self.pixels_to_meters_batch(points_px)

        # Return as list of tuples
        return [(float(x), float(y)) for x, y in points_m]

    def get_bbox_center_in_meters(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """
        Convert bounding box center to meters.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in pixels

        Returns:
            Tuple of (x_meters, y_meters) for bbox center
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.pixel_to_meters(cx, cy)

    def get_bbox_dimensions_in_meters(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """
        Convert bounding box dimensions to meters (approximate).

        Note: This is an approximation since perspective distortion affects
        different parts of the box differently. Uses center point scaling.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in pixels

        Returns:
            Tuple of (width_meters, height_meters)
        """
        x1, y1, x2, y2 = bbox

        # Get corners in meters
        tl_m = self.pixel_to_meters(x1, y1)
        tr_m = self.pixel_to_meters(x2, y1)
        bl_m = self.pixel_to_meters(x1, y2)
        br_m = self.pixel_to_meters(x2, y2)

        # Calculate average dimensions
        width_top = abs(tr_m[0] - tl_m[0])
        width_bottom = abs(br_m[0] - bl_m[0])
        width_m = (width_top + width_bottom) / 2.0

        height_left = abs(bl_m[1] - tl_m[1])
        height_right = abs(br_m[1] - tr_m[1])
        height_m = (height_left + height_right) / 2.0

        return (width_m, height_m)
