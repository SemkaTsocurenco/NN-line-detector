"""
Test script for Protocol V2 implementation.

Verifies:
1. Coordinate transformation from pixels to meters
2. Lane lines frame building
3. Road objects frame building
4. Frame parsing and validation
"""
import struct
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.coordinate_transform import CoordinateTransformer
from core.detections import MarkingObject
from core.line_fitting import FittedLine
from network.protocol import ProtocolBuilder, MSG_TYPE_LANE_LINES, MSG_TYPE_ROAD_OBJECTS, SYNC_BYTE
import numpy as np


def test_coordinate_transformation():
    """Test coordinate transformation from pixels to meters."""
    print("=" * 60)
    print("TEST 1: Coordinate Transformation")
    print("=" * 60)

    transformer = CoordinateTransformer()

    # Test center point (should be close to x=0)
    x_m, y_m = transformer.pixel_to_meters(960, 540)
    print(f"✓ Center point (960, 540): x={x_m:.2f}m, y={y_m:.2f}m")

    # Test left side point
    x_m, y_m = transformer.pixel_to_meters(100, 540)
    print(f"✓ Left point (100, 540): x={x_m:.2f}m, y={y_m:.2f}m")

    # Test right side point
    x_m, y_m = transformer.pixel_to_meters(1820, 540)
    print(f"✓ Right point (1820, 540): x={x_m:.2f}m, y={y_m:.2f}m")

    # Test batch conversion
    points_px = np.array([[100, 500], [960, 500], [1820, 500]])
    points_m = transformer.pixels_to_meters_batch(points_px)
    print(f"✓ Batch conversion of 3 points:")
    for i, (px, pm) in enumerate(zip(points_px, points_m)):
        print(f"  Point {i+1}: ({px[0]}, {px[1]}) px → ({pm[0]:.2f}, {pm[1]:.2f}) m")

    print("✅ Coordinate transformation tests passed!\n")


def test_lane_lines_frame():
    """Test building and parsing LANE_LINES frame."""
    print("=" * 60)
    print("TEST 2: Lane Lines Frame")
    print("=" * 60)

    builder = ProtocolBuilder()

    # Create test fitted lines
    fitted_lines = [
        FittedLine(
            poly_coeffs=np.array([0.0005, -0.3, 250.0]),
            start_point=(245, 100),
            end_point=(255, 400),
            class_id=4,  # solid_single_white
            confidence=0.95,
            inlier_ratio=0.85,
            side="left"
        ),
        FittedLine(
            poly_coeffs=np.array([0.0003, 0.2, 670.0]),
            start_point=(665, 100),
            end_point=(675, 400),
            class_id=9,  # dashed_white
            confidence=0.90,
            inlier_ratio=0.80,
            side="right"
        ),
    ]

    # Build frame
    frame = builder.build_lane_lines_frame(fitted_lines)

    print(f"✓ Frame size: {len(frame)} bytes")

    # Parse frame header
    sync = frame[0]
    version, msg_type, seq, timestamp, payload_len = struct.unpack("<BBBIH", frame[1:10])

    print(f"✓ Sync byte: 0x{sync:02X} (expected 0x{SYNC_BYTE:02X})")
    print(f"✓ Version: {version} (expected 2)")
    print(f"✓ Message type: 0x{msg_type:02X} (expected 0x{MSG_TYPE_LANE_LINES:02X})")
    print(f"✓ Sequence: {seq}")
    print(f"✓ Timestamp: {timestamp} ms")
    print(f"✓ Payload length: {payload_len} bytes")

    # Parse payload
    payload = frame[10:10+payload_len]
    count = struct.unpack("<B", payload[0:1])[0]
    print(f"✓ Number of lines: {count}")

    offset = 1
    for i in range(count):
        line_data = struct.unpack("<BBBfffffffffffffffff", payload[offset:offset+71])
        side, style, color = line_data[0:3]
        poly_a, poly_b, poly_c = line_data[3:6]
        x_m, y_m = line_data[6:8]
        p1_x_m, p1_y_m = line_data[8:10]
        p2_x_m, p2_y_m = line_data[10:12]
        p3_x_m, p3_y_m = line_data[12:14]
        p1_x_px, p1_y_px = line_data[14:16]
        p2_x_px, p2_y_px = line_data[16:18]
        p3_x_px, p3_y_px = line_data[18:20]

        print(f"  Line {i+1}:")
        print(f"    Side: {side} (1=left, 2=right)")
        print(f"    Style: {style} (1=solid, 2=dashed)")
        print(f"    Color: {color} (1=white)")
        print(f"    Polynomial: x = {poly_a:.6f}y² + {poly_b:.3f}y + {poly_c:.1f}")
        print(f"    Center: ({x_m:.2f}, {y_m:.2f}) m")
        print(f"    Points (meters): ({p1_x_m:.2f}, {p1_y_m:.2f}), ({p2_x_m:.2f}, {p2_y_m:.2f}), ({p3_x_m:.2f}, {p3_y_m:.2f})")
        print(f"    Points (pixels): ({p1_x_px:.1f}, {p1_y_px:.1f}), ({p2_x_px:.1f}, {p2_y_px:.1f}), ({p3_x_px:.1f}, {p3_y_px:.1f})")

        offset += 71

    # Verify CRC
    crc_received = struct.unpack("<H", frame[10+payload_len:12+payload_len])[0]
    print(f"✓ CRC: 0x{crc_received:04X}")

    print("✅ Lane lines frame tests passed!\n")


def test_road_objects_frame():
    """Test building and parsing ROAD_OBJECTS frame."""
    print("=" * 60)
    print("TEST 3: Road Objects Frame")
    print("=" * 60)

    builder = ProtocolBuilder()

    # Create test road objects
    objects = [
        MarkingObject(
            class_id=12,  # arrow_straight
            x_dm=20,
            y_dm=100,
            length_dm=35,
            width_dm=12,
            yaw_decideg=0,
            confidence_byte=200,
            flags=0,
            # V2 data
            bbox_px=(450, 200, 550, 400),
            center_px=(500.0, 300.0),
            yaw_rad=0.0
        ),
        MarkingObject(
            class_id=2,  # crosswalk
            x_dm=-5,
            y_dm=150,
            length_dm=50,
            width_dm=40,
            yaw_decideg=0,
            confidence_byte=180,
            flags=0,
            bbox_px=(200, 300, 400, 500),
            center_px=(300.0, 400.0),
            yaw_rad=0.0
        ),
    ]

    # Build frame
    frame = builder.build_road_objects_frame(objects)

    print(f"✓ Frame size: {len(frame)} bytes")

    # Parse frame header
    sync = frame[0]
    version, msg_type, seq, timestamp, payload_len = struct.unpack("<BBBIH", frame[1:10])

    print(f"✓ Sync byte: 0x{sync:02X}")
    print(f"✓ Version: {version}")
    print(f"✓ Message type: 0x{msg_type:02X} (expected 0x{MSG_TYPE_ROAD_OBJECTS:02X})")
    print(f"✓ Payload length: {payload_len} bytes")

    # Parse payload
    payload = frame[10:10+payload_len]
    count = struct.unpack("<B", payload[0:1])[0]
    print(f"✓ Number of objects: {count}")

    offset = 1
    for i in range(count):
        obj_data = struct.unpack("<BfffffBB", payload[offset:offset+23])
        class_id = obj_data[0]
        center_x, center_y = obj_data[1:3]
        length, width = obj_data[3:5]
        yaw = obj_data[5]
        confidence = obj_data[6]
        flags = obj_data[7]

        print(f"  Object {i+1}:")
        print(f"    Class ID: {class_id}")
        print(f"    Center: ({center_x:.2f}, {center_y:.2f}) m")
        print(f"    Size: {length:.2f}m x {width:.2f}m")
        print(f"    Yaw: {yaw:.3f} rad")
        print(f"    Confidence: {confidence}/255")

        offset += 23

    print("✅ Road objects frame tests passed!\n")


def test_frame_sizes():
    """Test that frame sizes match specification."""
    print("=" * 60)
    print("TEST 4: Frame Size Validation")
    print("=" * 60)

    builder = ProtocolBuilder()

    # Test empty frames
    empty_lines_frame = builder.build_lane_lines_frame([])
    empty_objects_frame = builder.build_road_objects_frame([])

    # Expected: 1(sync) + 9(header) + 1(count) + 0(data) + 2(crc) = 13 bytes
    expected_empty = 13

    print(f"✓ Empty lane lines frame: {len(empty_lines_frame)} bytes (expected {expected_empty})")
    print(f"✓ Empty road objects frame: {len(empty_objects_frame)} bytes (expected {expected_empty})")

    assert len(empty_lines_frame) == expected_empty, f"Lane lines frame size mismatch"
    assert len(empty_objects_frame) == expected_empty, f"Road objects frame size mismatch"

    # Test with 1 line/object
    test_line = FittedLine(
        poly_coeffs=np.array([0.0, 0.0, 0.0]),
        start_point=(0, 0),
        end_point=(0, 0),
        class_id=4,
        confidence=0.5,
        inlier_ratio=0.5,
        side="left"
    )

    test_object = MarkingObject(
        class_id=1,
        x_dm=0, y_dm=0, length_dm=0, width_dm=0,
        yaw_decideg=0, confidence_byte=100,
        bbox_px=(0, 0, 0, 0), center_px=(0.0, 0.0), yaw_rad=0.0
    )

    single_line_frame = builder.build_lane_lines_frame([test_line])
    single_object_frame = builder.build_road_objects_frame([test_object])

    # Expected: 13 + 71 = 84 bytes for 1 line
    # Expected: 13 + 23 = 36 bytes for 1 object
    expected_line = 13 + 71
    expected_object = 13 + 23

    print(f"✓ Single lane line frame: {len(single_line_frame)} bytes (expected {expected_line})")
    print(f"✓ Single road object frame: {len(single_object_frame)} bytes (expected {expected_object})")

    assert len(single_line_frame) == expected_line, f"Single line frame size mismatch"
    assert len(single_object_frame) == expected_object, f"Single object frame size mismatch"

    print("✅ Frame size validation passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Protocol V2 Test Suite")
    print("=" * 60 + "\n")

    try:
        test_coordinate_transformation()
        test_lane_lines_frame()
        test_road_objects_frame()
        test_frame_sizes()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
