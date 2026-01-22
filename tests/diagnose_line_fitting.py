#!/usr/bin/env python3
"""
Diagnostic script to understand why good masks don't produce fitted lines.

Run this to see where lines are being lost in the pipeline.
"""
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.line_fitting import LineFitter, PolynomialRANSAC
from core.detections import DetectionRaw


def create_realistic_line_mask(shape=(384, 384), line_width=15, noise_level=0.1):
    """
    Create a realistic thick line mask similar to what NN produces.

    Args:
        shape: Image shape (H, W)
        line_width: Width of the line in pixels
        noise_level: Fraction of noise points to add
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create a slightly curved vertical line (x = f(y))
    # Small curvature to simulate real road markings
    poly_coeffs = [0.0002, -0.15, 100]  # x = 0.0002*y^2 - 0.15*y + 100

    # Generate the line with thickness
    for y in range(50, h - 30):
        x_center = int(np.polyval(poly_coeffs, y))

        # Draw thick line
        for dx in range(-line_width // 2, line_width // 2 + 1):
            x = x_center + dx
            if 0 <= x < w:
                # Add some vertical noise too
                for dy in range(-1, 2):
                    yy = y + dy
                    if 0 <= yy < h:
                        mask[yy, x] = 255

    # Add some noise points
    n_noise = int(noise_level * np.sum(mask > 0))
    noise_y = np.random.randint(0, h, n_noise)
    noise_x = np.random.randint(0, w, n_noise)
    for ny, nx in zip(noise_y, noise_x):
        mask[ny, nx] = 255

    return mask, poly_coeffs


def diagnose_single_mask(mask, name="test_mask"):
    """
    Run detailed diagnostics on a single mask to understand where fitting fails.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS FOR: {name}")
    print(f"{'='*60}")

    # Step 1: Analyze mask properties
    print(f"\n[1] MASK PROPERTIES:")
    print(f"    Shape: {mask.shape}")
    print(f"    dtype: {mask.dtype}")
    print(f"    min: {mask.min()}, max: {mask.max()}")
    print(f"    Non-zero pixels: {np.sum(mask > 0)}")
    print(f"    Pixels > 127: {np.sum(mask > 127)}")

    # Step 2: Test point extraction with different thresholds
    print(f"\n[2] POINT EXTRACTION:")
    for threshold in [0, 50, 100, 127, 200]:
        binary = (mask > threshold).astype(np.uint8)
        ys, xs = np.nonzero(binary)
        print(f"    Threshold {threshold}: {len(xs)} points")

    # Step 3: Test RANSAC with current (strict) config
    print(f"\n[3] RANSAC WITH STRICT CONFIG (from config.yaml):")
    strict_ransac = PolynomialRANSAC(
        degree=2,
        max_iterations=100,
        inlier_threshold=5.0,
        min_inlier_ratio=0.3,
        min_points=20,
    )
    test_ransac(mask, strict_ransac, "strict")

    # Step 4: Test RANSAC with relaxed config
    print(f"\n[4] RANSAC WITH RELAXED CONFIG:")
    relaxed_ransac = PolynomialRANSAC(
        degree=2,
        max_iterations=400,
        inlier_threshold=15.0,  # Increased for thick lines
        min_inlier_ratio=0.05,  # Much lower
        min_points=10,
    )
    test_ransac(mask, relaxed_ransac, "relaxed")

    # Step 5: Test with very relaxed config
    print(f"\n[5] RANSAC WITH VERY RELAXED CONFIG:")
    very_relaxed_ransac = PolynomialRANSAC(
        degree=2,
        max_iterations=500,
        inlier_threshold=25.0,  # For very thick lines
        min_inlier_ratio=0.02,
        min_points=5,
    )
    test_ransac(mask, very_relaxed_ransac, "very_relaxed")

    return mask


def test_ransac(mask, ransac, config_name):
    """Test RANSAC fitting and report detailed results."""
    # Extract points (using threshold 127 as in current code)
    binary_mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask
    ys, xs = np.nonzero(binary_mask)

    if len(xs) == 0:
        print(f"    [{config_name}] FAILED: No points after binarization (threshold=127)")
        return None

    points = np.column_stack([xs, ys])
    print(f"    [{config_name}] Points extracted: {len(points)}")

    if len(points) < ransac.min_points:
        print(f"    [{config_name}] FAILED: Not enough points ({len(points)} < {ransac.min_points})")
        return None

    # Try fitting
    result = ransac.fit(points)

    if result is None:
        print(f"    [{config_name}] FAILED: RANSAC returned None")
        # Try to understand why
        rotated_points = np.column_stack([points[:, 1], points[:, 0]])

        # Manually check inlier ratios
        best_ratio = 0
        for _ in range(min(100, ransac.max_iterations)):
            if len(rotated_points) < ransac.degree + 1:
                break
            indices = np.random.choice(len(rotated_points), ransac.degree + 1, replace=False)
            sample = rotated_points[indices]
            try:
                coeffs = np.polyfit(sample[:, 0], sample[:, 1], ransac.degree)
                y_all = rotated_points[:, 0]
                x_all = rotated_points[:, 1]
                x_pred = np.polyval(coeffs, y_all)
                distances = np.abs(x_all - x_pred)
                inlier_count = np.sum(distances < ransac.inlier_threshold)
                ratio = inlier_count / len(rotated_points)
                if ratio > best_ratio:
                    best_ratio = ratio
            except:
                continue

        print(f"    [{config_name}] Best inlier ratio found: {best_ratio:.3f} (required: {ransac.min_inlier_ratio})")
        print(f"    [{config_name}] Distance stats: checking with threshold={ransac.inlier_threshold}")
        return None

    coeffs, inlier_mask = result
    inlier_ratio = np.sum(inlier_mask) / len(points)
    print(f"    [{config_name}] SUCCESS!")
    print(f"    [{config_name}] Coefficients: {coeffs}")
    print(f"    [{config_name}] Inlier ratio: {inlier_ratio:.3f}")
    print(f"    [{config_name}] Inlier count: {np.sum(inlier_mask)}/{len(points)}")

    return coeffs, inlier_mask


def diagnose_with_different_widths():
    """Test how line width affects fitting success."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT LINE WIDTHS")
    print("="*60)

    for width in [3, 5, 10, 15, 20, 30]:
        mask, _ = create_realistic_line_mask(line_width=width, noise_level=0.05)

        # Test with strict config
        strict_ransac = PolynomialRANSAC(
            degree=2,
            max_iterations=100,
            inlier_threshold=5.0,
            min_inlier_ratio=0.3,
            min_points=20,
        )

        binary_mask = (mask > 127).astype(np.uint8)
        ys, xs = np.nonzero(binary_mask)
        points = np.column_stack([xs, ys])

        result = strict_ransac.fit(points) if len(points) >= 20 else None
        status = "SUCCESS" if result else "FAILED"

        print(f"Width={width:2d}px: {len(points):5d} points -> {status}")


def diagnose_threshold_sensitivity():
    """Test how binarization threshold affects results."""
    print("\n" + "="*60)
    print("TESTING BINARIZATION THRESHOLD SENSITIVITY")
    print("="*60)

    # Create mask with gradient values (simulating soft NN output)
    h, w = 384, 384
    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(50, h - 30):
        x_center = int(100 + 0.1 * y)  # Simple line
        for dx in range(-10, 11):
            x = x_center + dx
            if 0 <= x < w:
                # Gradient: center is brighter, edges are dimmer
                intensity = int(255 * (1 - abs(dx) / 15))
                mask[y, x] = max(mask[y, x], intensity)

    print(f"Mask value distribution:")
    for low, high in [(0, 50), (50, 100), (100, 150), (150, 200), (200, 256)]:
        count = np.sum((mask >= low) & (mask < high))
        print(f"  [{low:3d}-{high:3d}): {count:6d} pixels")

    print(f"\nFitting results with different thresholds:")
    for threshold in [0, 50, 100, 127, 150, 200]:
        binary = (mask > threshold).astype(np.uint8)
        ys, xs = np.nonzero(binary)

        if len(xs) < 20:
            print(f"  Threshold {threshold:3d}: {len(xs):5d} points -> TOO FEW")
            continue

        points = np.column_stack([xs, ys])
        ransac = PolynomialRANSAC(
            degree=2,
            max_iterations=100,
            inlier_threshold=5.0,
            min_inlier_ratio=0.3,
            min_points=20,
        )

        result = ransac.fit(points)
        if result:
            _, inlier_mask = result
            ratio = np.sum(inlier_mask) / len(points)
            print(f"  Threshold {threshold:3d}: {len(xs):5d} points -> SUCCESS (inlier_ratio={ratio:.3f})")
        else:
            print(f"  Threshold {threshold:3d}: {len(xs):5d} points -> FAILED")


def main():
    print("="*60)
    print("LINE FITTING DIAGNOSTIC TOOL")
    print("="*60)

    # Test 1: Realistic thick line
    print("\n>>> TEST 1: Realistic thick line (15px width)")
    mask, true_coeffs = create_realistic_line_mask(line_width=15)
    print(f"True polynomial: x = {true_coeffs[0]:.6f}*y^2 + {true_coeffs[1]:.4f}*y + {true_coeffs[2]:.2f}")
    diagnose_single_mask(mask, "thick_line_15px")

    # Test 2: Thin line
    print("\n>>> TEST 2: Thin line (5px width)")
    mask_thin, _ = create_realistic_line_mask(line_width=5)
    diagnose_single_mask(mask_thin, "thin_line_5px")

    # Test 3: Very thick line
    print("\n>>> TEST 3: Very thick line (25px width)")
    mask_thick, _ = create_realistic_line_mask(line_width=25)
    diagnose_single_mask(mask_thick, "thick_line_25px")

    # Test 4: Width sensitivity
    diagnose_with_different_widths()

    # Test 5: Threshold sensitivity
    diagnose_threshold_sensitivity()

    print("\n" + "="*60)
    print("SUMMARY OF RECOMMENDATIONS")
    print("="*60)
    print("""
Based on diagnostics, the main issues are:

1. INLIER_THRESHOLD too small (5px) for thick lines (10-20px)
   -> Increase to 15-20px or make it proportional to line width

2. MIN_INLIER_RATIO too strict (0.3 = 30%)
   -> Reduce to 0.05-0.1 (5-10%)

3. BINARIZATION_THRESHOLD may be too high (127) for soft masks
   -> Consider using Otsu's method or lower threshold (50-100)

4. MAX_ITERATIONS may be too low (100)
   -> Increase to 300-500 for better convergence

Update config/config.yaml with these values:
  line_fitting:
    ransac_iterations: 400
    inlier_threshold: 15.0
    min_inlier_ratio: 0.1
    min_points: 10
""")


if __name__ == "__main__":
    main()
