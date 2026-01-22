#!/usr/bin/env python3
"""
Full pipeline diagnostic - tests the complete flow including split_by_center.
"""
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.line_fitting import LineFitter
from core.detections import DetectionRaw


def create_left_side_line(shape=(384, 384), line_width=10):
    """Create a line on the left side of the image."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Line on the left side (x ~ 80)
    for y in range(30, h - 30):
        x_center = int(80 + 0.05 * y)  # Slight slant
        for dx in range(-line_width // 2, line_width // 2 + 1):
            x = x_center + dx
            if 0 <= x < w:
                mask[y, x] = 255
    return mask


def create_right_side_line(shape=(384, 384), line_width=10):
    """Create a line on the right side of the image."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Line on the right side (x ~ 300)
    for y in range(30, h - 30):
        x_center = int(300 - 0.05 * y)  # Slight slant
        for dx in range(-line_width // 2, line_width // 2 + 1):
            x = x_center + dx
            if 0 <= x < w:
                mask[y, x] = 255
    return mask


def create_center_line(shape=(384, 384), line_width=10):
    """Create a line near the center of the image."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    center_x = w // 2  # 192

    # Line near center
    for y in range(30, h - 30):
        x_center = center_x + int(0.02 * y)
        for dx in range(-line_width // 2, line_width // 2 + 1):
            x = x_center + dx
            if 0 <= x < w:
                mask[y, x] = 255
    return mask


def create_two_lane_mask(shape=(384, 384), line_width=10):
    """Create a mask with two parallel lane markings."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Left lane line
    for y in range(30, h - 30):
        x_left = int(100 + 0.03 * y)
        for dx in range(-line_width // 2, line_width // 2 + 1):
            x = x_left + dx
            if 0 <= x < w:
                mask[y, x] = 255

    # Right lane line
    for y in range(30, h - 30):
        x_right = int(280 - 0.03 * y)
        for dx in range(-line_width // 2, line_width // 2 + 1):
            x = x_right + dx
            if 0 <= x < w:
                mask[y, x] = 255

    return mask


def diagnose_split_behavior():
    """Test how split_by_center affects line detection."""
    print("\n" + "="*70)
    print("TESTING SPLIT_BY_CENTER BEHAVIOR")
    print("="*70)

    shape = (384, 384)
    split_margin = 20
    center_x = shape[1] // 2  # 192

    print(f"\nImage dimensions: {shape[1]}x{shape[0]}")
    print(f"Center X: {center_x}")
    print(f"Split margin: {split_margin}")
    print(f"LEFT  keeps X in [0, {center_x + split_margin})")
    print(f"RIGHT keeps X in [{center_x - split_margin}, {shape[1]})")
    print(f"DEAD ZONE: X in [{center_x - split_margin}, {center_x + split_margin}] is excluded from BOTH")

    # Test different line positions
    test_cases = [
        ("Left side line (x~80)", create_left_side_line(shape)),
        ("Right side line (x~300)", create_right_side_line(shape)),
        ("Center line (x~192)", create_center_line(shape)),
        ("Two lane mask", create_two_lane_mask(shape)),
    ]

    fitter = LineFitter(
        poly_degree=2,
        ransac_iterations=100,
        inlier_threshold=5.0,
        min_inlier_ratio=0.3,
        min_points=20,
        split_margin=split_margin,
    )

    for name, mask in test_cases:
        print(f"\n--- {name} ---")

        # Analyze mask position
        ys, xs = np.nonzero(mask)
        if len(xs) > 0:
            print(f"  Mask X range: [{xs.min()}, {xs.max()}]")
            print(f"  Mask center X: {xs.mean():.1f}")

        # Test WITHOUT split
        det = DetectionRaw(class_id=4, confidence=0.9, mask=mask, bbox=None)
        lines_no_split = fitter.fit_lines_from_detections([det], split_by_center=False)
        print(f"  Without split: {len(lines_no_split)} lines detected")

        # Test WITH split
        lines_with_split = fitter.fit_lines_from_detections([det], split_by_center=True)
        print(f"  With split: {len(lines_with_split)} lines detected")
        for line in lines_with_split:
            print(f"    -> {line.side} line, inlier_ratio={line.inlier_ratio:.3f}")

        # Manual split analysis
        left_mask, right_mask = fitter._split_mask_by_center(mask, margin=split_margin)

        left_pixels = np.sum(left_mask > 0) if left_mask is not None else 0
        right_pixels = np.sum(right_mask > 0) if right_mask is not None else 0

        print(f"  After split: left={left_pixels} px, right={right_pixels} px")


def diagnose_config_loading():
    """Check if config is loaded correctly."""
    print("\n" + "="*70)
    print("TESTING CONFIG LOADING")
    print("="*70)

    from core.line_fitting import load_line_fitting_config

    # Load from line_fitting.yaml
    config = load_line_fitting_config("config/line_fitting.yaml")
    print("\nconfig/line_fitting.yaml:")
    print(f"  ransac.max_iterations: {config.get('ransac', {}).get('max_iterations', 'NOT SET')}")
    print(f"  ransac.inlier_threshold: {config.get('ransac', {}).get('inlier_threshold', 'NOT SET')}")
    print(f"  ransac.min_inlier_ratio: {config.get('ransac', {}).get('min_inlier_ratio', 'NOT SET')}")
    print(f"  ransac.min_points: {config.get('ransac', {}).get('min_points', 'NOT SET')}")

    # Compare with config.yaml
    import yaml
    with open("config/config.yaml") as f:
        main_config = yaml.safe_load(f)

    lf = main_config.get("line_fitting", {})
    print("\nconfig/config.yaml -> line_fitting:")
    print(f"  ransac_iterations: {lf.get('ransac_iterations', 'NOT SET')}")
    print(f"  inlier_threshold: {lf.get('inlier_threshold', 'NOT SET')}")
    print(f"  min_inlier_ratio: {lf.get('min_inlier_ratio', 'NOT SET')}")
    print(f"  min_points: {lf.get('min_points', 'NOT SET')}")

    # Note the difference
    print("\n!!! IMPORTANT !!!")
    print("config/config.yaml uses FLAT structure (ransac_iterations)")
    print("config/line_fitting.yaml uses NESTED structure (ransac.max_iterations)")
    print("The code uses config/config.yaml, but may not read ransac: section!")


def diagnose_with_nn_like_mask():
    """Test with masks that simulate NN output characteristics."""
    print("\n" + "="*70)
    print("TESTING WITH NN-LIKE MASKS")
    print("="*70)

    shape = (384, 384)
    h, w = shape

    # NN often outputs masks with:
    # 1. Soft edges (gradient values)
    # 2. Discontinuous regions (gaps)
    # 3. Multiple fragments

    # Test 1: Soft edges
    print("\n--- Test 1: Soft edge mask ---")
    mask_soft = np.zeros((h, w), dtype=np.uint8)
    for y in range(50, h - 50):
        x_center = int(100 + 0.05 * y)
        for dx in range(-15, 16):
            x = x_center + dx
            if 0 <= x < w:
                # Gaussian-like falloff
                intensity = int(255 * np.exp(-dx*dx / 50))
                mask_soft[y, x] = max(mask_soft[y, x], intensity)

    print(f"  Value distribution:")
    print(f"    0-50: {np.sum((mask_soft > 0) & (mask_soft <= 50))} px")
    print(f"    51-100: {np.sum((mask_soft > 50) & (mask_soft <= 100))} px")
    print(f"    101-127: {np.sum((mask_soft > 100) & (mask_soft <= 127))} px")
    print(f"    128-255: {np.sum(mask_soft > 127)} px")

    fitter = LineFitter(
        poly_degree=2,
        ransac_iterations=100,
        inlier_threshold=5.0,
        min_inlier_ratio=0.3,
        min_points=20,
    )

    det = DetectionRaw(class_id=4, confidence=0.9, mask=mask_soft, bbox=None)
    lines = fitter.fit_lines_from_detections([det], split_by_center=False)
    print(f"  Result: {len(lines)} lines (with threshold=127)")

    # Test with lower threshold
    fitter_low = LineFitter(
        poly_degree=2,
        ransac_iterations=100,
        inlier_threshold=5.0,
        min_inlier_ratio=0.3,
        min_points=20,
        config={"mask": {"binarization_threshold": 50}}
    )

    lines_low = fitter_low.fit_lines_from_detections([det], split_by_center=False)
    print(f"  Result: {len(lines_low)} lines (with threshold=50)")

    # Test 2: Fragmented mask (gaps)
    print("\n--- Test 2: Fragmented mask (with gaps) ---")
    mask_frag = np.zeros((h, w), dtype=np.uint8)
    for y in range(50, h - 50):
        # Skip every 30 pixels to create gaps
        if (y // 30) % 2 == 0:
            continue
        x_center = int(100 + 0.05 * y)
        for dx in range(-5, 6):
            x = x_center + dx
            if 0 <= x < w:
                mask_frag[y, x] = 255

    print(f"  Total pixels: {np.sum(mask_frag > 0)}")

    det_frag = DetectionRaw(class_id=4, confidence=0.9, mask=mask_frag, bbox=None)
    lines_frag = fitter.fit_lines_from_detections([det_frag], split_by_center=False)
    print(f"  Result: {len(lines_frag)} lines")

    # Test 3: Very thin line
    print("\n--- Test 3: Very thin line (1-2px) ---")
    mask_thin = np.zeros((h, w), dtype=np.uint8)
    for y in range(50, h - 50):
        x = int(100 + 0.05 * y)
        if 0 <= x < w:
            mask_thin[y, x] = 255
            if x + 1 < w:
                mask_thin[y, x + 1] = 255

    print(f"  Total pixels: {np.sum(mask_thin > 0)}")

    det_thin = DetectionRaw(class_id=4, confidence=0.9, mask=mask_thin, bbox=None)
    lines_thin = fitter.fit_lines_from_detections([det_thin], split_by_center=False)
    print(f"  Result: {len(lines_thin)} lines")


def save_diagnostic_images():
    """Save diagnostic images to /tmp for visual inspection."""
    print("\n" + "="*70)
    print("SAVING DIAGNOSTIC IMAGES")
    print("="*70)

    shape = (384, 384)

    # Create and save test images
    test_cases = [
        ("left_line", create_left_side_line(shape)),
        ("right_line", create_right_side_line(shape)),
        ("center_line", create_center_line(shape)),
        ("two_lanes", create_two_lane_mask(shape)),
    ]

    for name, mask in test_cases:
        # Create visualization with split zones
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        center_x = shape[1] // 2
        margin = 20

        # Draw split zones
        cv2.line(vis, (center_x, 0), (center_x, shape[0]), (0, 255, 255), 1)  # Center
        cv2.line(vis, (center_x - margin, 0), (center_x - margin, shape[0]), (0, 128, 255), 1)  # Left boundary
        cv2.line(vis, (center_x + margin, 0), (center_x + margin, shape[0]), (255, 128, 0), 1)  # Right boundary

        # Add labels
        cv2.putText(vis, "L", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, "DEAD", (center_x - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis, "R", (shape[1] - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        path = f"/tmp/diag_{name}.png"
        cv2.imwrite(path, vis)
        print(f"  Saved: {path}")


def main():
    print("="*70)
    print("FULL PIPELINE DIAGNOSTICS")
    print("="*70)

    diagnose_config_loading()
    diagnose_split_behavior()
    diagnose_with_nn_like_mask()
    save_diagnostic_images()

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
1. CONFIG MISMATCH: config/config.yaml uses flat structure, but
   PolynomialRANSAC looks for nested 'ransac:' key.

2. SPLIT_BY_CENTER: Lines near the center (within 20px) are excluded
   from BOTH left and right halves.

3. SOFT EDGES: NN masks with gradient values below 127 are lost
   during binarization.

4. FRAGMENTED MASKS: Gaps in masks reduce the number of points,
   potentially below min_points threshold.

RECOMMENDATIONS:
- Unify config structure (use nested 'ransac:' in config.yaml)
- Lower binarization_threshold for soft masks
- Consider disabling split_by_center for certain use cases
- Increase inlier_threshold for thick lines
""")


if __name__ == "__main__":
    main()
