#!/usr/bin/env python3
"""
Test script for object rendering with perspective.
"""
import numpy as np
import cv2
from core.renderer import Renderer
from core.detections import DetectionRaw

def create_test_mask_arrow(shape=(384, 384), rotation_angle=0):
    """Create a test mask for arrow."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create arrow-shaped region
    center_x, center_y = w // 2, h // 2 + 50

    # Define arrow points (pointing up)
    pts = np.array([
        [center_x, center_y - 60],  # tip
        [center_x - 40, center_y - 20],
        [center_x - 20, center_y - 20],
        [center_x - 20, center_y + 40],
        [center_x + 20, center_y + 40],
        [center_x + 20, center_y - 20],
        [center_x + 40, center_y - 20],
    ], np.int32)

    # Rotate if needed
    if rotation_angle != 0:
        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.int32)

    cv2.fillPoly(mask, [pts], 255)

    # Get bbox
    x, y, w_box, h_box = cv2.boundingRect(pts)
    bbox = (x, y, x + w_box, y + h_box)

    return mask, bbox

def create_test_mask_crosswalk(shape=(384, 384)):
    """Create a test mask for crosswalk with perspective."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Trapezoid shape (perspective view)
    pts = np.array([
        [100, 200],  # top-left
        [280, 200],  # top-right
        [320, 300],  # bottom-right
        [60, 300],   # bottom-left
    ], np.int32)

    cv2.fillPoly(mask, [pts], 255)

    x, y, w_box, h_box = cv2.boundingRect(pts)
    bbox = (x, y, x + w_box, y + h_box)

    return mask, bbox

def create_test_mask_stop_line(shape=(384, 384)):
    """Create a test mask for stop line."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Horizontal line with slight perspective
    pts = np.array([
        [50, 250],
        [330, 260],
        [330, 280],
        [50, 270],
    ], np.int32)

    cv2.fillPoly(mask, [pts], 255)

    x, y, w_box, h_box = cv2.boundingRect(pts)
    bbox = (x, y, x + w_box, y + h_box)

    return mask, bbox

def test_arrow_rendering():
    """Test arrow rendering for all directions."""
    print("Testing arrow rendering...")

    frame = np.zeros((384, 384, 3), dtype=np.uint8)
    renderer = Renderer()

    detections = []

    # Test all arrow types
    arrow_classes = {
        11: "left",
        12: "straight",
        13: "right",
        14: "left_straight",
        15: "right_straight",
    }

    y_offset = 0
    for class_id, name in arrow_classes.items():
        mask, bbox = create_test_mask_arrow()
        # Shift mask down for each arrow
        mask_shifted = np.roll(mask, y_offset, axis=0)
        bbox_shifted = (bbox[0], bbox[1] + y_offset, bbox[2], bbox[3] + y_offset)

        det = DetectionRaw(
            class_id=class_id,
            confidence=0.9,
            mask=mask_shifted,
            bbox=bbox_shifted
        )
        detections.append(det)
        y_offset += 80

    # Render
    result = renderer.render(frame, detections)

    cv2.imwrite('/tmp/test_arrows.png', result)
    print("✓ Arrow rendering test saved to /tmp/test_arrows.png")

    return True

def test_crosswalk_rendering():
    """Test crosswalk rendering with perspective."""
    print("Testing crosswalk rendering...")

    frame = np.zeros((384, 384, 3), dtype=np.uint8)
    renderer = Renderer()

    mask, bbox = create_test_mask_crosswalk()

    det = DetectionRaw(
        class_id=2,  # crosswalk
        confidence=0.9,
        mask=mask,
        bbox=bbox
    )

    result = renderer.render(frame, [det])

    cv2.imwrite('/tmp/test_crosswalk.png', result)
    print("✓ Crosswalk rendering test saved to /tmp/test_crosswalk.png")

    return True

def test_stop_line_rendering():
    """Test stop line rendering."""
    print("Testing stop line rendering...")

    frame = np.zeros((384, 384, 3), dtype=np.uint8)
    renderer = Renderer()

    mask, bbox = create_test_mask_stop_line()

    det = DetectionRaw(
        class_id=3,  # stop_line
        confidence=0.9,
        mask=mask,
        bbox=bbox
    )

    result = renderer.render(frame, [det])

    cv2.imwrite('/tmp/test_stop_line.png', result)
    print("✓ Stop line rendering test saved to /tmp/test_stop_line.png")

    return True

def test_combined_rendering():
    """Test rendering multiple objects together."""
    print("Testing combined rendering...")

    frame = np.zeros((384, 384, 3), dtype=np.uint8)
    renderer = Renderer()

    detections = []

    # Add straight arrow
    mask_arrow, bbox_arrow = create_test_mask_arrow()
    mask_arrow = np.roll(mask_arrow, -80, axis=0)  # Move up
    detections.append(DetectionRaw(
        class_id=12,
        confidence=0.9,
        mask=mask_arrow,
        bbox=(bbox_arrow[0], bbox_arrow[1] - 80, bbox_arrow[2], bbox_arrow[3] - 80)
    ))

    # Add crosswalk
    mask_crosswalk, bbox_crosswalk = create_test_mask_crosswalk()
    detections.append(DetectionRaw(
        class_id=2,
        confidence=0.9,
        mask=mask_crosswalk,
        bbox=bbox_crosswalk
    ))

    # Add stop line
    mask_stop, bbox_stop = create_test_mask_stop_line()
    detections.append(DetectionRaw(
        class_id=3,
        confidence=0.9,
        mask=mask_stop,
        bbox=bbox_stop
    ))

    result = renderer.render(frame, detections)

    cv2.imwrite('/tmp/test_combined.png', result)
    print("✓ Combined rendering test saved to /tmp/test_combined.png")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Object Rendering Test Suite")
    print("=" * 60)

    success = True

    try:
        success &= test_arrow_rendering()
        success &= test_crosswalk_rendering()
        success &= test_stop_line_rendering()
        success &= test_combined_rendering()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All rendering tests passed!")
        print("\nGenerated test images:")
        print("  - /tmp/test_arrows.png")
        print("  - /tmp/test_crosswalk.png")
        print("  - /tmp/test_stop_line.png")
        print("  - /tmp/test_combined.png")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
