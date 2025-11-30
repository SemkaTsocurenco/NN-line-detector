#!/usr/bin/env python3
"""
Test script for line fitting functionality.
"""
import numpy as np
import cv2
from core.line_fitting import LineFitter, FittedLine
from core.detections import DetectionRaw

def create_synthetic_line_mask(shape=(384, 384), poly_coeffs=None):
    """Create a synthetic line mask for testing."""
    if poly_coeffs is None:
        poly_coeffs = [0.0005, -0.3, 250]  # Parabola: y = 0.0005*x^2 - 0.3*x + 250

    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Generate points along the polynomial
    x_vals = np.arange(50, w - 50, 2)
    y_vals = np.polyval(poly_coeffs, x_vals)

    # Add some noise
    noise = np.random.randn(len(y_vals)) * 3
    y_vals = y_vals + noise

    # Draw on mask
    for x, y in zip(x_vals, y_vals):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), 3, 255, -1)

    return mask

def test_line_fitting():
    """Test the line fitting pipeline."""
    print("Testing line fitting...")

    # Create synthetic data
    mask = create_synthetic_line_mask()

    # Create a DetectionRaw object
    detection = DetectionRaw(
        class_id=4,  # solid_single_white
        confidence=0.95,
        mask=mask,
        bbox=(50, 200, 330, 300)
    )

    # Initialize fitter
    fitter = LineFitter(
        poly_degree=2,
        ransac_iterations=100,
        inlier_threshold=5.0,
        min_inlier_ratio=0.3,
        min_points=20,
    )

    # Fit line
    fitted_line = fitter.extract_line_from_mask(
        mask=detection.mask,
        class_id=detection.class_id,
        confidence=detection.confidence,
    )

    if fitted_line is None:
        print("ERROR: Line fitting failed!")
        return False

    print(f"✓ Line fitted successfully!")
    print(f"  Class ID: {fitted_line.class_id}")
    print(f"  Confidence: {fitted_line.confidence:.3f}")
    print(f"  Inlier ratio: {fitted_line.inlier_ratio:.3f}")
    print(f"  Polynomial coeffs: {fitted_line.poly_coeffs}")
    print(f"  Start point: {fitted_line.start_point}")
    print(f"  End point: {fitted_line.end_point}")

    # Visualize result
    vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw fitted line
    x_start, y_start = fitted_line.start_point
    x_end, y_end = fitted_line.end_point

    if x_start != x_end:
        num_points = max(abs(x_end - x_start), 50)
        x_values = np.linspace(x_start, x_end, num_points)
        y_values = np.polyval(fitted_line.poly_coeffs, x_values)
        points = np.column_stack([x_values, y_values]).astype(np.int32)

        cv2.polylines(vis_img, [points], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.circle(vis_img, fitted_line.start_point, 5, (0, 0, 255), -1)
        cv2.circle(vis_img, fitted_line.end_point, 5, (255, 0, 0), -1)

    # Save visualization
    cv2.imwrite('/tmp/line_fitting_test.png', vis_img)
    print(f"✓ Visualization saved to /tmp/line_fitting_test.png")

    return True

def test_multiple_lines():
    """Test fitting multiple lines."""
    print("\nTesting multiple line fitting...")

    detections = []

    # Create multiple synthetic line detections
    for i, class_id in enumerate([4, 5, 9, 10]):  # Different line types
        # Different polynomials for each line
        poly_coeffs = [0.0003 * (i + 1), -0.2 * (i + 1), 150 + 50 * i]
        mask = create_synthetic_line_mask(poly_coeffs=poly_coeffs)

        det = DetectionRaw(
            class_id=class_id,
            confidence=0.85 + i * 0.03,
            mask=mask,
            bbox=(50, 100, 330, 300)
        )
        detections.append(det)

    # Initialize fitter
    fitter = LineFitter()

    # Fit all lines
    fitted_lines = fitter.fit_lines_from_detections(detections)

    print(f"✓ Fitted {len(fitted_lines)} out of {len(detections)} detections")

    for i, fitted_line in enumerate(fitted_lines):
        print(f"  Line {i+1}: class={fitted_line.class_id}, inliers={fitted_line.inlier_ratio:.2f}")

    return len(fitted_lines) > 0

if __name__ == "__main__":
    print("=" * 60)
    print("Line Fitting Test Suite")
    print("=" * 60)

    success = True

    try:
        success &= test_line_fitting()
        success &= test_multiple_lines()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
