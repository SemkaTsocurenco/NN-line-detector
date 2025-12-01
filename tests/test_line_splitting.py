#!/usr/bin/env python3
"""
Test script for line splitting functionality.
"""
import numpy as np
import cv2
from core.line_fitting import LineFitter
from core.detections import DetectionRaw

def create_two_lane_mask(shape=(384, 384)):
    """Create a synthetic mask with two lane lines (left and right).

    Note: Now generates vertical lines using x = f(y) formulation.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Left line (goes through left half) - x = f(y)
    left_poly = [0.0003, -0.15, 100]  # x = 0.0003*y^2 - 0.15*y + 100
    y_vals_left = np.arange(30, h - 30, 2)
    x_vals_left = np.polyval(left_poly, y_vals_left)
    x_vals_left += np.random.randn(len(x_vals_left)) * 2  # Add noise

    for x, y in zip(x_vals_left, y_vals_left):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), 3, 255, -1)

    # Right line (goes through right half) - x = f(y)
    right_poly = [0.0004, -0.2, 280]  # x = 0.0004*y^2 - 0.2*y + 280
    y_vals_right = np.arange(30, h - 30, 2)
    x_vals_right = np.polyval(right_poly, y_vals_right)
    x_vals_right += np.random.randn(len(x_vals_right)) * 2  # Add noise

    for x, y in zip(x_vals_right, y_vals_right):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), 3, 255, -1)

    return mask

def test_line_splitting():
    """Test splitting of lines into left and right."""
    print("Testing line splitting...")

    # Create mask with two lines
    mask = create_two_lane_mask()

    # Create detection
    detection = DetectionRaw(
        class_id=4,  # solid_single_white
        confidence=0.95,
        mask=mask,
        bbox=(30, 100, 354, 250)
    )

    # Initialize fitter
    fitter = LineFitter(split_margin=20)

    # Test WITHOUT splitting
    print("\n1. Testing WITHOUT splitting (should get one merged line):")
    lines_no_split = fitter.fit_lines_from_detections([detection], split_by_center=False)
    print(f"   Found {len(lines_no_split)} line(s)")
    for i, line in enumerate(lines_no_split):
        print(f"   Line {i+1}: side={line.side}, start={line.start_point}, end={line.end_point}")

    # Test WITH splitting
    print("\n2. Testing WITH splitting (should get two separate lines):")
    lines_with_split = fitter.fit_lines_from_detections([detection], split_by_center=True)
    print(f"   Found {len(lines_with_split)} line(s)")
    for i, line in enumerate(lines_with_split):
        print(f"   Line {i+1}: side={line.side}, start={line.start_point}, end={line.end_point}")

    # Visualize results
    vis_img = np.zeros((384, 384 * 3, 3), dtype=np.uint8)

    # Original mask
    vis_img[:, :384] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis_img, "Original Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Without splitting (poly_coeffs are now x = f(y))
    no_split_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    for line in lines_no_split:
        x_start, y_start = line.start_point
        x_end, y_end = line.end_point
        if y_start != y_end:
            num_points = max(abs(y_end - y_start), 50)
            y_values = np.linspace(y_start, y_end, num_points)
            x_values = np.polyval(line.poly_coeffs, y_values)
            points = np.column_stack([x_values, y_values]).astype(np.int32)
            cv2.polylines(no_split_img, [points], False, (0, 255, 0), 3)
    vis_img[:, 384:768] = no_split_img
    cv2.putText(vis_img, "No Split (1 line)", (394, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # With splitting (poly_coeffs are now x = f(y))
    split_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 0, 255)]  # Blue for left, Red for right
    for i, line in enumerate(lines_with_split):
        x_start, y_start = line.start_point
        x_end, y_end = line.end_point
        if y_start != y_end:
            num_points = max(abs(y_end - y_start), 50)
            y_values = np.linspace(y_start, y_end, num_points)
            x_values = np.polyval(line.poly_coeffs, y_values)
            points = np.column_stack([x_values, y_values]).astype(np.int32)
            color = colors[i % len(colors)]
            cv2.polylines(split_img, [points], False, color, 3)
            # Label
            mid_y = (y_start + y_end) // 2
            mid_x = int(np.polyval(line.poly_coeffs, mid_y))
            cv2.putText(split_img, line.side.upper(), (mid_x - 20, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw center line
    cv2.line(split_img, (192, 0), (192, 384), (255, 255, 0), 2)
    vis_img[:, 768:] = split_img
    cv2.putText(vis_img, "With Split (2 lines)", (778, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Save visualization
    cv2.imwrite('/tmp/line_splitting_test.png', vis_img)
    print(f"\n✓ Visualization saved to /tmp/line_splitting_test.png")

    # Verify results
    success = True
    if len(lines_no_split) != 1:
        print(f"✗ Expected 1 line without splitting, got {len(lines_no_split)}")
        success = False
    else:
        print(f"✓ Correct: 1 line without splitting")

    if len(lines_with_split) != 2:
        print(f"✗ Expected 2 lines with splitting, got {len(lines_with_split)}")
        success = False
    else:
        print(f"✓ Correct: 2 lines with splitting")

        # Check sides
        sides = {line.side for line in lines_with_split}
        if sides != {"left", "right"}:
            print(f"✗ Expected sides {{left, right}}, got {sides}")
            success = False
        else:
            print(f"✓ Correct: Lines labeled as 'left' and 'right'")

    return success

if __name__ == "__main__":
    print("=" * 60)
    print("Line Splitting Test")
    print("=" * 60)

    try:
        success = test_line_splitting()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ Test passed!")
    else:
        print("✗ Test failed")
    print("=" * 60)
