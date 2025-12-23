# Migration Guide: Protocol V1 → V2

## Overview

Protocol V2 simplifies the TCP communication structure from **5 message types to 2**:

### V1 (Legacy - 5 messages)
1. `LANE_SUMMARY` (0x01) - Basic lane info
2. `MARKING_OBJECTS` (0x02) - Objects with basic data
3. `LANE_DETAILS` (0x03) - Detailed lane boundaries
4. `MARKING_OBJECTS_EX` (0x04) - Objects with extended data
5. `FITTED_LINES` (0x05) - Polynomial lines

### V2 (Current - 2 messages)
1. **`LANE_LINES` (0x01)** - Lane marking lines with polynomials + 3 points in meters
2. **`ROAD_OBJECTS` (0x02)** - Road objects with coordinates in meters

---

## Key Changes

### Coordinate System

**V1:**
- Mixed units: deci-meters (dm), pixels, decidegrees
- Normalized coordinates relative to image size
- No real-world measurements

**V2:**
- All coordinates in **meters** (real-world distances)
- Uses perspective transformation matrix
- Origin at camera position
- X-axis: right (positive) / left (negative)
- Y-axis: forward (always positive)

### Message Structure

**V1 sent 5 messages per frame:**
```python
tcp_client.send(lane_summary_frame)      # 18 bytes
tcp_client.send(marking_objects_frame)   # 12 + N*13 bytes
tcp_client.send(lane_details_frame)      # 44 bytes
tcp_client.send(marking_objects_ex_frame) # 12 + N*15 bytes
tcp_client.send(fitted_lines_frame)      # 12 + N*23 bytes
```

**V2 sends 2 messages per frame:**
```python
tcp_client.send(lane_lines_frame)        # 12 + N*38 bytes
tcp_client.send(road_objects_frame)      # 12 + N*25 bytes
```

---

## Data Transformation

### Lane Lines (V1 → V2)

**V1 (FITTED_LINES):**
```python
{
    'class_id': 4,
    'poly_a': 0.0005,
    'poly_b': -0.3,
    'poly_c': 250.0,
    'y_start': 50,    # pixels
    'y_end': 350,     # pixels
    'confidence': 0.95,
    'inlier_ratio': 0.8
}
```

**V2 (LANE_LINES):**
```python
{
    'side': 1,        # 1=left, 2=right, 3=center
    'style': 1,       # 1=solid, 2=dashed, 3=double
    'color': 1,       # 1=white, 2=yellow, 3=red
    'poly_a': 0.0005,
    'poly_b': -0.3,
    'poly_c': 250.0,
    'points': [       # 3 points in METERS
        (1.2, 5.0),   # top (x, y) in meters
        (0.8, 15.0),  # middle
        (0.5, 25.0)   # bottom
    ]
}
```

### Road Objects (V1 → V2)

**V1 (MARKING_OBJECTS_EX):**
```python
{
    'class_id': 12,
    'x_dm': 20,           # deci-meters
    'y_dm': 100,          # deci-meters
    'length_dm': 35,      # deci-meters
    'width_dm': 12,       # deci-meters
    'yaw_decideg': 0,     # deci-degrees
    'confidence': 200,
    'line_color': 1,
    'line_style': 1
}
```

**V2 (ROAD_OBJECTS):**
```python
{
    'class_id': 12,
    'center_x': 0.2,      # meters (right of center)
    'center_y': 10.0,     # meters (forward)
    'length': 3.5,        # meters
    'width': 1.2,         # meters
    'yaw': 0.0,           # radians
    'confidence': 200
}
```

---

## Implementation Changes

### Protocol Builder

**File:** `network/protocol.py`

**V1 methods (deprecated):**
- `build_lane_summary_frame()`
- `build_marking_objects_frame()`
- `build_lane_details_frame()`
- `build_marking_objects_ex_frame()`
- `build_fitted_lines_frame()`

**V2 methods (use these):**
- `build_lane_lines_frame(fitted_lines)`
- `build_road_objects_frame(objects)`

### Main Window

**File:** `ui/main_window.py`

**Before:**
```python
def _on_detection_data(self, summary, objects, fitted_lines):
    lane_frame = self._protocol_builder.build_lane_summary_frame(summary)
    objects_frame = self._protocol_builder.build_marking_objects_frame(objects)
    lane_details = self._protocol_builder.build_lane_details_frame(summary)
    objects_ex = self._protocol_builder.build_marking_objects_ex_frame(objects)
    fitted_lines_frame = self._protocol_builder.build_fitted_lines_frame(fitted_lines)

    self._tcp_client.send(lane_frame)
    self._tcp_client.send(objects_frame)
    self._tcp_client.send(lane_details)
    self._tcp_client.send(objects_ex)
    self._tcp_client.send(fitted_lines_frame)
```

**After:**
```python
def _on_detection_data(self, summary, objects, fitted_lines):
    # V2 Protocol: Only 2 messages
    lane_lines_frame = self._protocol_builder.build_lane_lines_frame(fitted_lines)
    road_objects_frame = self._protocol_builder.build_road_objects_frame(objects)

    self._tcp_client.send(lane_lines_frame)
    self._tcp_client.send(road_objects_frame)
```

---

## Coordinate Transformation

### New Component

**File:** `core/coordinate_transform.py`

Provides perspective transformation from pixels to meters:

```python
from core.coordinate_transform import CoordinateTransformer

transformer = CoordinateTransformer()

# Convert single point
x_m, y_m = transformer.pixel_to_meters(x_px, y_px)

# Convert multiple points
points_m = transformer.pixels_to_meters_batch(points_px)

# Convert line points
points = transformer.get_line_points_in_meters(
    poly_coeffs, y_start, y_end, num_samples=3
)

# Convert bbox
center_m = transformer.get_bbox_center_in_meters(bbox)
dims_m = transformer.get_bbox_dimensions_in_meters(bbox)
```

### Configuration Files

**Required files:**
- `config/PerspectiveTransform.yaml` - Perspective transformation matrix
- `config/calibration.json` - Camera calibration parameters

---

## Benefits of V2

| Aspect | V1 | V2 |
|--------|----|----|
| **Messages per frame** | 5 | 2 |
| **Coordinate units** | Mixed (dm, pixels) | Meters |
| **Real-world accuracy** | Approximate | Precise (with calibration) |
| **Protocol complexity** | High | Low |
| **Bandwidth** | Higher | Lower |
| **3D rendering** | Requires conversion | Direct use |

---

## Backward Compatibility

V1 protocol methods are kept for backward compatibility but marked as **legacy**:

```python
# Legacy message types (moved to 0x11-0x15 range)
MSG_TYPE_LANE_SUMMARY_LEGACY = 0x11
MSG_TYPE_MARKING_OBJECTS_LEGACY = 0x12
MSG_TYPE_LANE_DETAILS_LEGACY = 0x13
MSG_TYPE_MARKING_OBJECTS_EX_LEGACY = 0x14
MSG_TYPE_FITTED_LINES_LEGACY = 0x15
```

**Recommendation:** Update dashboard receivers to use V2 protocol.

---

## Testing

### Verify Coordinate Transformation

```python
import numpy as np
from core.coordinate_transform import CoordinateTransformer

transformer = CoordinateTransformer()

# Test point in center of image (should be close to x=0)
x_m, y_m = transformer.pixel_to_meters(960, 540)
print(f"Center point: x={x_m:.2f}m, y={y_m:.2f}m")

# Test batch conversion
points_px = np.array([[100, 500], [960, 500], [1820, 500]])
points_m = transformer.pixels_to_meters_batch(points_px)
print(f"Converted points:\n{points_m}")
```

### Verify Protocol Messages

```python
from network.protocol import ProtocolBuilder

builder = ProtocolBuilder()

# Build V2 messages
lane_lines_frame = builder.build_lane_lines_frame(fitted_lines)
road_objects_frame = builder.build_road_objects_frame(objects)

print(f"Lane lines frame size: {len(lane_lines_frame)} bytes")
print(f"Road objects frame size: {len(road_objects_frame)} bytes")
```

---

## Documentation

- **V2 Protocol Specification:** [TCP_PROTOCOL_V2.md](./TCP_PROTOCOL_V2.md)
- **V1 Protocol (Legacy):** [TCP_PROTOCOL.md](./TCP_PROTOCOL.md)
- **Code Reference:**
  - [network/protocol.py](../network/protocol.py) - Protocol builders
  - [core/coordinate_transform.py](../core/coordinate_transform.py) - Coordinate transformation
  - [ui/main_window.py](../ui/main_window.py) - Message sending
