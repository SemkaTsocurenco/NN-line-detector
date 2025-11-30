"""
RANSAC-based polynomial line fitting for lane marking detection.

This module extracts polynomial curves from segmentation masks using RANSAC
to robustly fit lines even with noisy detections.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FittedLine:
    """Represents a fitted polynomial line."""
    poly_coeffs: np.ndarray  # Polynomial coefficients [a, b, c] for y = ax^2 + bx + c
    start_point: Tuple[int, int]  # (x, y) starting point
    end_point: Tuple[int, int]  # (x, y) ending point
    class_id: int  # Lane marking class ID
    confidence: float  # Detection confidence
    inlier_ratio: float  # Ratio of inlier points (quality metric)
    side: str = "unknown"  # "left", "right", or "center"
    curvature: float = 0.0  # Curvature metric (0 = straight)
    is_valid: bool = True  # Passed validation checks


class PolynomialRANSAC:
    """
    RANSAC-based polynomial fitting for line detection.

    Fits a polynomial of specified degree to a set of points,
    robustly handling outliers.
    """

    def __init__(
        self,
        degree: int = 2,
        max_iterations: int = 100,
        inlier_threshold: float = 5.0,
        min_inlier_ratio: float = 0.3,
        min_points: int = 20,
    ):
        """
        Initialize RANSAC fitter.

        Args:
            degree: Polynomial degree (1=line, 2=parabola, etc.)
            max_iterations: Maximum RANSAC iterations
            inlier_threshold: Distance threshold for inliers (pixels)
            min_inlier_ratio: Minimum ratio of inliers to accept fit
            min_points: Minimum number of points required for fitting
        """
        self.degree = degree
        self.max_iterations = max_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_points = min_points

    def fit(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fit polynomial to points using RANSAC.

        Args:
            points: Nx2 array of (x, y) coordinates

        Returns:
            Tuple of (coefficients, inlier_mask) or None if fitting fails
            coefficients: polynomial coefficients [a, b, c, ...] for y = ax^n + bx^(n-1) + ...
            inlier_mask: boolean mask of inlier points
        """
        if len(points) < self.min_points:
            logger.debug(f"Not enough points for fitting: {len(points)} < {self.min_points}")
            return None

        n_samples = self.degree + 1  # Minimum samples needed
        best_inliers = None
        best_coeffs = None
        best_inlier_count = 0

        for iteration in range(self.max_iterations):
            # Randomly sample minimum points needed
            if len(points) < n_samples:
                break

            sample_indices = np.random.choice(len(points), n_samples, replace=False)
            sample_points = points[sample_indices]

            # Fit polynomial to sample
            try:
                x_sample = sample_points[:, 0]
                y_sample = sample_points[:, 1]
                coeffs = np.polyfit(x_sample, y_sample, self.degree)
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.debug(f"Polyfit failed at iteration {iteration}: {e}")
                continue

            # Evaluate model on all points
            x_all = points[:, 0]
            y_all = points[:, 1]
            y_pred = np.polyval(coeffs, x_all)

            # Find inliers
            distances = np.abs(y_all - y_pred)
            inlier_mask = distances < self.inlier_threshold
            inlier_count = np.sum(inlier_mask)

            # Update best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_coeffs = coeffs
                best_inliers = inlier_mask

        # Check if we have enough inliers
        if best_inliers is None:
            logger.debug("RANSAC failed to find any valid model")
            return None

        inlier_ratio = best_inlier_count / len(points)
        if inlier_ratio < self.min_inlier_ratio:
            logger.debug(f"Inlier ratio too low: {inlier_ratio:.2f} < {self.min_inlier_ratio}")
            return None

        # Refit on all inliers for better accuracy
        inlier_points = points[best_inliers]
        try:
            final_coeffs = np.polyfit(inlier_points[:, 0], inlier_points[:, 1], self.degree)
        except (np.linalg.LinAlgError, ValueError):
            final_coeffs = best_coeffs

        return final_coeffs, best_inliers


class LineValidator:
    """
    Validates fitted lines for realism.
    """

    def __init__(
        self,
        max_curvature: float = 0.002,  # Maximum allowed curvature (a coefficient)
        max_horizontal_change: float = 100.0,  # Max pixels X can change over line
        min_length: int = 30,  # Minimum line length in pixels
        max_length: int = 400,  # Maximum line length in pixels
        max_angle_deviation: float = 45.0,  # Max deviation from vertical (degrees)
    ):
        """
        Initialize line validator.

        Args:
            max_curvature: Maximum absolute value of quadratic coefficient
            max_horizontal_change: Maximum horizontal drift allowed
            min_length: Minimum line length
            max_length: Maximum line length
            max_angle_deviation: Maximum angle from vertical
        """
        self.max_curvature = max_curvature
        self.max_horizontal_change = max_horizontal_change
        self.min_length = min_length
        self.max_length = max_length
        self.max_angle_deviation = max_angle_deviation

    def validate(self, fitted_line: FittedLine) -> bool:
        """
        Validate if a fitted line is realistic.

        Args:
            fitted_line: FittedLine to validate

        Returns:
            True if line passes all validation checks
        """
        # Check 1: Curvature
        if not self._check_curvature(fitted_line):
            logger.debug(f"Line failed curvature check: {fitted_line.curvature:.4f}")
            return False

        # Check 2: Length
        if not self._check_length(fitted_line):
            logger.debug("Line failed length check")
            return False

        # Check 3: Horizontal drift
        if not self._check_horizontal_drift(fitted_line):
            logger.debug("Line failed horizontal drift check")
            return False

        # Check 4: General direction (should be mostly vertical)
        if not self._check_direction(fitted_line):
            logger.debug("Line failed direction check")
            return False

        return True

    def _check_curvature(self, line: FittedLine) -> bool:
        """Check if curvature is within realistic bounds."""
        # For polynomial y = ax^2 + bx + c, 'a' represents curvature
        if len(line.poly_coeffs) >= 3:
            curvature = abs(line.poly_coeffs[0])
            return curvature <= self.max_curvature
        return True

    def _check_length(self, line: FittedLine) -> bool:
        """Check if line length is reasonable."""
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return self.min_length <= length <= self.max_length

    def _check_horizontal_drift(self, line: FittedLine) -> bool:
        """Check if line drifts too much horizontally."""
        x1, _ = line.start_point
        x2, _ = line.end_point
        drift = abs(x2 - x1)
        return drift <= self.max_horizontal_change

    def _check_direction(self, line: FittedLine) -> bool:
        """Check if line is roughly vertical (within angle tolerance)."""
        x1, y1 = line.start_point
        x2, y2 = line.end_point

        # Calculate angle from vertical
        dx = x2 - x1
        dy = y2 - y1

        if dy == 0:  # Horizontal line
            return False

        # Angle from vertical in degrees
        angle_from_vertical = abs(np.degrees(np.arctan2(dx, abs(dy))))

        return angle_from_vertical <= self.max_angle_deviation


class LineFitter:
    """
    Extracts and fits polynomial lines from detection masks.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        ransac_iterations: int = 100,
        inlier_threshold: float = 5.0,
        min_inlier_ratio: float = 0.3,
        min_points: int = 20,
        split_margin: int = 20,
        validate_lines: bool = False,
        max_curvature: float = 0.2,
        max_horizontal_change: float = 100.0,
    ):
        """
        Initialize line fitter.

        Args:
            poly_degree: Degree of polynomial to fit
            ransac_iterations: Number of RANSAC iterations
            inlier_threshold: RANSAC inlier distance threshold (pixels)
            min_inlier_ratio: Minimum inlier ratio to accept fit
            min_points: Minimum points required for fitting
            split_margin: Margin around center when splitting masks (pixels)
            validate_lines: Enable line validation
            max_curvature: Maximum allowed curvature
            max_horizontal_change: Maximum horizontal drift
        """
        self.ransac = PolynomialRANSAC(
            degree=poly_degree,
            max_iterations=ransac_iterations,
            inlier_threshold=inlier_threshold,
            min_inlier_ratio=min_inlier_ratio,
            min_points=min_points,
        )
        self.split_margin = split_margin
        self.validate_lines = validate_lines
        self.validator = LineValidator(
            max_curvature=max_curvature,
            max_horizontal_change=max_horizontal_change,
        ) if validate_lines else None

    @staticmethod
    def _split_mask_by_center(mask: np.ndarray, margin: int = 20) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Split a mask into left and right halves based on image center.

        Args:
            mask: Binary mask (H, W)
            margin: Margin around center to exclude (pixels)

        Returns:
            Tuple of (left_mask, right_mask), each can be None if no pixels
        """
        if mask.max() == 0:
            return None, None

        h, w = mask.shape
        center_x = w // 2

        # Create left and right masks
        left_mask = mask.copy()
        right_mask = mask.copy()

        # Zero out right half for left mask (with margin)
        left_mask[:, center_x + margin:] = 0

        # Zero out left half for right mask (with margin)
        right_mask[:, :center_x - margin] = 0

        # Check if masks have content
        left_mask = left_mask if left_mask.max() > 0 else None
        right_mask = right_mask if right_mask.max() > 0 else None

        return left_mask, right_mask

    def extract_line_from_mask(
        self,
        mask: np.ndarray,
        class_id: int,
        confidence: float,
        side: str = "unknown",
    ) -> Optional[FittedLine]:
        """
        Extract fitted line from a binary mask.

        Args:
            mask: Binary mask (H, W) with 255 for line pixels
            class_id: Class ID of the line
            confidence: Detection confidence

        Returns:
            FittedLine object or None if fitting fails
        """
        # Extract points from mask
        points = self._extract_points_from_mask(mask)

        if points is None or len(points) < self.ransac.min_points:
            logger.debug(f"Not enough points in mask for class {class_id}")
            return None

        # Fit polynomial using RANSAC
        result = self.ransac.fit(points)

        if result is None:
            logger.debug(f"RANSAC fitting failed for class {class_id}")
            return None

        coeffs, inlier_mask = result
        inlier_ratio = np.sum(inlier_mask) / len(points)

        # Determine start and end points
        inlier_points = points[inlier_mask]
        x_min, x_max = int(inlier_points[:, 0].min()), int(inlier_points[:, 0].max())

        # Evaluate polynomial at endpoints
        y_start = int(np.polyval(coeffs, x_min))
        y_end = int(np.polyval(coeffs, x_max))

        # Calculate curvature (absolute value of quadratic coefficient)
        curvature = abs(coeffs[0]) if len(coeffs) >= 3 else 0.0

        fitted_line = FittedLine(
            poly_coeffs=coeffs,
            start_point=(x_min, y_start),
            end_point=(x_max, y_end),
            class_id=class_id,
            confidence=confidence,
            inlier_ratio=inlier_ratio,
            side=side,
            curvature=curvature,
            is_valid=True,  # Will be set by validator
        )

        # Validate if enabled
        if self.validate_lines and self.validator is not None:
            fitted_line.is_valid = self.validator.validate(fitted_line)

        return fitted_line

    @staticmethod
    def _extract_points_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract (x, y) points from binary mask.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Nx2 array of (x, y) coordinates or None
        """
        if mask.max() == 0:
            return None

        # Convert to binary if needed
        binary_mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask

        # Get coordinates of non-zero pixels
        ys, xs = np.nonzero(binary_mask)

        if len(xs) == 0:
            return None

        # Return as (x, y) pairs
        points = np.column_stack([xs, ys])

        return points

    def fit_lines_from_detections(
        self,
        detections: List,  # List[DetectionRaw]
        split_by_center: bool = True,
    ) -> List[FittedLine]:
        """
        Fit lines from all detections.

        Args:
            detections: List of DetectionRaw objects
            split_by_center: If True, split masks into left/right halves

        Returns:
            List of FittedLine objects
        """
        fitted_lines = []

        # Line class IDs (4-10: various lane markings)
        LINE_CLASS_IDS = {4, 5, 6, 7, 8, 9, 10}

        for det in detections:
            # Only process line classes
            if det.class_id not in LINE_CLASS_IDS:
                continue

            if det.mask is None:
                logger.debug(f"No mask for detection class {det.class_id}")
                continue

            # Split mask into left and right if requested
            if split_by_center:
                left_mask, right_mask = self._split_mask_by_center(det.mask, margin=self.split_margin)

                # Fit left line
                if left_mask is not None:
                    left_line = self.extract_line_from_mask(
                        mask=left_mask,
                        class_id=det.class_id,
                        confidence=det.confidence,
                        side="left",
                    )
                    if left_line is not None and left_line.is_valid:
                        fitted_lines.append(left_line)
                        logger.debug(
                            f"Fitted LEFT line for class {det.class_id}: "
                            f"inlier_ratio={left_line.inlier_ratio:.2f}, "
                            f"curvature={left_line.curvature:.4f}, "
                            f"start={left_line.start_point}, end={left_line.end_point}"
                        )
                    elif left_line is not None and not left_line.is_valid:
                        logger.debug(f"LEFT line for class {det.class_id} failed validation")

                # Fit right line
                if right_mask is not None:
                    right_line = self.extract_line_from_mask(
                        mask=right_mask,
                        class_id=det.class_id,
                        confidence=det.confidence,
                        side="right",
                    )
                    if right_line is not None and right_line.is_valid:
                        fitted_lines.append(right_line)
                        logger.debug(
                            f"Fitted RIGHT line for class {det.class_id}: "
                            f"inlier_ratio={right_line.inlier_ratio:.2f}, "
                            f"curvature={right_line.curvature:.4f}, "
                            f"start={right_line.start_point}, end={right_line.end_point}"
                        )
                    elif right_line is not None and not right_line.is_valid:
                        logger.debug(f"RIGHT line for class {det.class_id} failed validation")
            else:
                # Fit whole mask without splitting
                fitted_line = self.extract_line_from_mask(
                    mask=det.mask,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    side="unknown",
                )

                if fitted_line is not None and fitted_line.is_valid:
                    fitted_lines.append(fitted_line)
                    logger.debug(
                        f"Fitted line for class {det.class_id}: "
                        f"inlier_ratio={fitted_line.inlier_ratio:.2f}, "
                        f"curvature={fitted_line.curvature:.4f}, "
                        f"start={fitted_line.start_point}, end={fitted_line.end_point}"
                    )
                elif fitted_line is not None and not fitted_line.is_valid:
                    logger.debug(f"Line for class {det.class_id} failed validation")

        return fitted_lines
