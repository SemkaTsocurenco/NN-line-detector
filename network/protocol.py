from __future__ import annotations

import logging
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from core.detections import LaneBoundaryPoint, LaneSummary, MarkingObject
from core.coordinate_transform import CoordinateTransformer

logger = logging.getLogger(__name__)


def load_protocol_config(config_path: str = "config/protocol.yaml") -> Dict[str, Any]:
    """Load protocol configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Protocol config not found at %s, using defaults", config_path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Load config values or use defaults
_protocol_config = load_protocol_config()
_prot_cfg = _protocol_config.get("protocol", {})
_crc_cfg = _protocol_config.get("crc", {})
_seq_cfg = _protocol_config.get("sequence", {})
_msg_sizes = _protocol_config.get("message_sizes", {})

SYNC_BYTE = _prot_cfg.get("sync_byte", 0xAA)
VERSION = _prot_cfg.get("version", 0x02)  # Updated to v2 for new protocol

# New message types (v2 protocol)
MSG_TYPE_LANE_LINES = 0x01  # Lane marking lines with polynomial curves
MSG_TYPE_ROAD_OBJECTS = 0x02  # Road objects (arrows, crosswalks, etc.)

# Legacy message types (deprecated in v2)
MSG_TYPE_LANE_SUMMARY_LEGACY = 0x11
MSG_TYPE_MARKING_OBJECTS_LEGACY = 0x12
MSG_TYPE_LANE_DETAILS_LEGACY = 0x13
MSG_TYPE_MARKING_OBJECTS_EX_LEGACY = 0x14
MSG_TYPE_FITTED_LINES_LEGACY = 0x15

MAX_PAYLOAD_SIZE = _prot_cfg.get("max_payload_size", 1024)  # per spec: payload length constraint
CRC_INITIAL = _crc_cfg.get("initial_value", 0xFFFF)
CRC_POLYNOMIAL = _crc_cfg.get("polynomial", 0xA001)
SEQ_MAX_VALUE = _seq_cfg.get("max_value", 256)


def crc16_ibm(data: bytes) -> int:
    """CRC16-IBM (reflected), configurable initial value and polynomial."""
    crc = CRC_INITIAL
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ CRC_POLYNOMIAL
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc


class SeqCounter:
    """Thread-safe sequence counter 0..255 with wrap."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            val = self._value
            self._value = (self._value + 1) % SEQ_MAX_VALUE
            return val


class TimeProvider:
    """Provides timestamp in milliseconds (uint32) using monotonic time."""

    def __call__(self) -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF


class ProtocolBuilder:
    def __init__(self, seq_counter: SeqCounter | None = None, time_provider: TimeProvider | None = None, config: Dict[str, Any] | None = None) -> None:
        self.seq_counter = seq_counter or SeqCounter()
        self.time_provider = time_provider or TimeProvider()

        # Initialize coordinate transformer
        self.coordinate_transformer = CoordinateTransformer()

        # Load config
        if config is None:
            config = _protocol_config

        msg_sizes = config.get("message_sizes", {})
        self.marking_object_size = msg_sizes.get("marking_object", 13)
        self.marking_object_ex_size = msg_sizes.get("marking_object_ex", 15)
        self.fitted_line_size = msg_sizes.get("fitted_line", 23)
        self.lane_line_size = msg_sizes.get("lane_line", 71)  # V2: 3 bytes (BBB) + 68 bytes (17 floats) = 71 bytes per line
        self.road_object_size = msg_sizes.get("road_object", 23)  # V2: 1+20(5 floats)+1+1 = 23 bytes per object

        conv_cfg = config.get("conversion", {})
        self.confidence_scale = conv_cfg.get("confidence_scale", 255)
        self.inlier_ratio_scale = conv_cfg.get("inlier_ratio_scale", 255)

        line_class_cfg = config.get("line_classification", {})
        self.yellow_classes = set(line_class_cfg.get("yellow_classes", [5, 8, 10]))
        self.red_classes = set(line_class_cfg.get("red_classes", [6]))
        self.solid_classes = set(line_class_cfg.get("solid_classes", [4, 5, 6]))
        self.double_classes = set(line_class_cfg.get("double_classes", [7, 8]))
        self.dashed_classes = set(line_class_cfg.get("dashed_classes", [9, 10]))

        side_map_cfg = config.get("side_mapping", {})
        self.side_mapping = side_map_cfg if side_map_cfg else {"unknown": 0, "left": 1, "right": 2, "center": 3}

    def build_lane_summary_frame(self, summary: LaneSummary) -> bytes:
        payload = struct.pack(
            "<hhBBBB",
            summary.left_offset_dm,
            summary.right_offset_dm,
            summary.left_type,
            summary.right_type,
            int(summary.allowed_maneuvers),
            summary.quality,
        )
        return self._build_frame(MSG_TYPE_LANE_SUMMARY, payload)

    def build_marking_objects_frame(self, objects: Iterable[MarkingObject]) -> bytes:
        objs_list: List[MarkingObject] = list(objects)
        max_objects = (MAX_PAYLOAD_SIZE - 1) // self.marking_object_size  # 1 byte for count
        if len(objs_list) > max_objects:
            logger.warning("Trimming objects from %d to %d to satisfy frame size", len(objs_list), max_objects)
            objs_list = objs_list[:max_objects]
        count = len(objs_list)
        payload_parts = [struct.pack("<B", count)]
        for obj in objs_list:
            payload_parts.append(
                struct.pack(
                    "<BhhHHhBB",
                    obj.class_id & 0xFF,
                    obj.x_dm,
                    obj.y_dm,
                    obj.length_dm & 0xFFFF,
                    obj.width_dm & 0xFFFF,
                    obj.yaw_decideg,
                    obj.confidence_byte & 0xFF,
                    obj.flags & 0xFF,
                )
            )
        payload = b"".join(payload_parts)
        return self._build_frame(MSG_TYPE_MARKING_OBJECTS, payload)

    def build_lane_details_frame(self, summary: LaneSummary) -> bytes:
        def pad_points(points: List[LaneBoundaryPoint], count: int = 3) -> List[LaneBoundaryPoint]:
            pts = list(points)[:count]
            while len(pts) < count:
                pts.append(LaneBoundaryPoint(0, 0))
            return pts

        left_pts = pad_points(summary.left_boundary)
        right_pts = pad_points(summary.right_boundary)
        payload_parts = [
            struct.pack(
                "<BBBBBBHH",
                summary.left_type & 0xFF,
                summary.right_type & 0xFF,
                summary.left_color & 0xFF,
                summary.right_color & 0xFF,
                summary.left_quality & 0xFF,
                summary.right_quality & 0xFF,
                summary.left_width_dm & 0xFFFF,
                summary.right_width_dm & 0xFFFF,
            )
        ]
        for pt in left_pts + right_pts:
            payload_parts.append(struct.pack("<hh", pt.x_dm, pt.y_dm))
        payload = b"".join(payload_parts)
        return self._build_frame(MSG_TYPE_LANE_DETAILS, payload)

    def build_marking_objects_ex_frame(self, objects: Iterable[MarkingObject]) -> bytes:
        objs_list: List[MarkingObject] = list(objects)
        max_objects = (MAX_PAYLOAD_SIZE - 1) // self.marking_object_ex_size  # 1 byte for count
        if len(objs_list) > max_objects:
            logger.warning("Trimming objects from %d to %d to satisfy extended frame size", len(objs_list), max_objects)
            objs_list = objs_list[:max_objects]
        count = len(objs_list)
        payload_parts = [struct.pack("<B", count)]
        for obj in objs_list:
            payload_parts.append(
                struct.pack(
                    "<BhhHHhBBBB",
                    obj.class_id & 0xFF,
                    obj.x_dm,
                    obj.y_dm,
                    obj.length_dm & 0xFFFF,
                    obj.width_dm & 0xFFFF,
                    obj.yaw_decideg,
                    obj.confidence_byte & 0xFF,
                    obj.flags & 0xFF,
                    obj.line_color & 0xFF,
                    obj.line_style & 0xFF,
                )
            )
        payload = b"".join(payload_parts)
        return self._build_frame(MSG_TYPE_MARKING_OBJECTS_EX, payload)

    def build_fitted_lines_frame(self, fitted_lines: Iterable) -> bytes:
        """
        Build frame with polynomial fitted lines for 3D reconstruction.

        Each line contains:
        - class_id (1 byte): line marking type (4-10)
        - side (1 byte): 0=unknown, 1=left, 2=right, 3=center
        - color (1 byte): line color enum
        - style (1 byte): line style enum (solid/dashed/double)
        - poly_a (float, 4 bytes): coefficient 'a' for x = ay^2 + by + c
        - poly_b (float, 4 bytes): coefficient 'b'
        - poly_c (float, 4 bytes): coefficient 'c'
        - y_start (int16, 2 bytes): start Y coordinate (pixels)
        - y_end (int16, 2 bytes): end Y coordinate (pixels)
        - confidence (uint8, 1 byte): detection confidence 0-255
        - quality (uint8, 1 byte): inlier ratio 0-255
        Total: 23 bytes per line
        """
        lines_list = list(fitted_lines)
        max_lines = (MAX_PAYLOAD_SIZE - 1) // self.fitted_line_size  # 1 byte for count
        if len(lines_list) > max_lines:
            logger.warning("Trimming fitted lines from %d to %d to satisfy frame size", len(lines_list), max_lines)
            lines_list = lines_list[:max_lines]

        count = len(lines_list)
        payload_parts = [struct.pack("<B", count)]

        for line in lines_list:
            # Map side string to byte
            side_byte = self.side_mapping.get(line.side, 0)

            # Get color and style from line attributes
            # These should be added to FittedLine or derived from class_id
            line_color = self._get_line_color_byte(line.class_id)
            line_style = self._get_line_style_byte(line.class_id)

            # Extract polynomial coefficients [a, b, c] for x = ay^2 + by + c
            if len(line.poly_coeffs) >= 3:
                poly_a = float(line.poly_coeffs[0])
                poly_b = float(line.poly_coeffs[1])
                poly_c = float(line.poly_coeffs[2])
            else:
                poly_a = poly_b = poly_c = 0.0

            # Get start and end Y coordinates
            _, y_start = line.start_point
            _, y_end = line.end_point

            # Convert confidence and inlier_ratio to bytes
            confidence_byte = int(line.confidence * self.confidence_scale) & 0xFF
            quality_byte = int(line.inlier_ratio * self.inlier_ratio_scale) & 0xFF

            payload_parts.append(
                struct.pack(
                    "<BBBBfffhhBB",
                    line.class_id & 0xFF,
                    side_byte & 0xFF,
                    line_color & 0xFF,
                    line_style & 0xFF,
                    poly_a,
                    poly_b,
                    poly_c,
                    y_start & 0xFFFF,
                    y_end & 0xFFFF,
                    confidence_byte,
                    quality_byte,
                )
            )

        payload = b"".join(payload_parts)
        return self._build_frame(MSG_TYPE_FITTED_LINES, payload)

    def _get_line_color_byte(self, class_id: int) -> int:
        """Map class_id to line color byte."""
        if class_id in self.yellow_classes:
            return 1  # YELLOW
        if class_id in self.red_classes:
            return 2  # RED
        return 0  # WHITE

    def _get_line_style_byte(self, class_id: int) -> int:
        """Map class_id to line style byte."""
        if class_id in self.solid_classes:
            return 1  # SOLID
        if class_id in self.double_classes:
            return 2  # DOUBLE
        if class_id in self.dashed_classes:
            return 3  # DASHED
        return 0  # UNKNOWN

    def _build_frame(self, msg_type: int, payload: bytes) -> bytes:
        seq = self.seq_counter.next()
        timestamp_ms = self.time_provider()
        header = struct.pack("<BBBIH", VERSION, msg_type & 0xFF, seq & 0xFF, timestamp_ms, len(payload))
        frame_wo_crc = bytes([SYNC_BYTE]) + header + payload
        crc = crc16_ibm(frame_wo_crc[1:])  # without sync byte
        frame = frame_wo_crc + struct.pack("<H", crc)
        return frame

    # ========== NEW V2 PROTOCOL METHODS ==========

    def build_lane_lines_frame(self, fitted_lines: Iterable) -> bytes:
        """
        Build frame with lane marking lines (v2 protocol - message type 0x01).

        Each line contains:
        - side (1 byte): 0=unknown, 1=left, 2=right, 3=center
        - style (1 byte): 0=unknown, 1=solid, 2=dashed, 3=double
        - color (1 byte): 0=unknown, 1=white, 2=yellow, 3=red
        - poly_a (float, 4 bytes): coefficient 'a' for x = ay^2 + by + c
        - poly_b (float, 4 bytes): coefficient 'b'
        - poly_c (float, 4 bytes): coefficient 'c'
        - x (float, 4 bytes): X coordinate in meters
        - y (float, 4 bytes): Y coordinate in meters
        - point1_x_m (float, 4 bytes): top point X in meters
        - point1_y_m (float, 4 bytes): top point Y in meters
        - point2_x_m (float, 4 bytes): middle point X in meters
        - point2_y_m (float, 4 bytes): middle point Y in meters
        - point3_x_m (float, 4 bytes): bottom point X in meters
        - point3_y_m (float, 4 bytes): bottom point Y in meters
        - point1_x_px (float, 4 bytes): top point X in pixels (OpenCV coords)
        - point1_y_px (float, 4 bytes): top point Y in pixels (OpenCV coords)
        - point2_x_px (float, 4 bytes): middle point X in pixels (OpenCV coords)
        - point2_y_px (float, 4 bytes): middle point Y in pixels (OpenCV coords)
        - point3_x_px (float, 4 bytes): bottom point X in pixels (OpenCV coords)
        - point3_y_px (float, 4 bytes): bottom point Y in pixels (OpenCV coords)

        Total: 59 bytes per line
        Frame size: 1(sync) + 9(header) + 1(count) + N*59 + 2(crc) bytes
        """
        lines_list = list(fitted_lines)
        max_lines = (MAX_PAYLOAD_SIZE - 1) // self.lane_line_size
        if len(lines_list) > max_lines:
            logger.warning("Trimming lane lines from %d to %d to satisfy frame size", len(lines_list), max_lines)
            lines_list = lines_list[:max_lines]

        count = len(lines_list)
        payload_parts = [struct.pack("<B", count)]

        for line in lines_list:
            # Map side string to byte
            side_byte = self.side_mapping.get(line.side, 0)

            # Get color and style from class_id
            color_byte = self._get_line_color_byte(line.class_id)
            style_byte = self._get_line_style_byte(line.class_id)

            # Extract polynomial coefficients [a, b, c] for x = ay^2 + by + c
            if len(line.poly_coeffs) >= 3:
                poly_a = float(line.poly_coeffs[0])
                poly_b = float(line.poly_coeffs[1])
                poly_c = float(line.poly_coeffs[2])
            else:
                poly_a = poly_b = poly_c = 0.0

            # Get start and end Y coordinates in pixels
            x_start_px, y_start_px = line.start_point
            x_end_px, y_end_px = line.end_point

            # Calculate 3 sample points in pixels (top, middle, bottom)
            import numpy as np
            y_samples = np.linspace(y_start_px, y_end_px, 3)
            x_samples = np.polyval(line.poly_coeffs, y_samples)

            # Points in pixels (OpenCV format: origin top-left, X right, Y down)
            p1_x_px, p1_y_px = float(x_samples[0]), float(y_samples[0])  # Top
            p2_x_px, p2_y_px = float(x_samples[1]), float(y_samples[1])  # Middle
            p3_x_px, p3_y_px = float(x_samples[2]), float(y_samples[2])  # Bottom

            # Convert 3 points along the line to meters using coordinate transformer
            # Points: top (y_start), middle, bottom (y_end)
            points_m = self.coordinate_transformer.get_line_points_in_meters(
                line.poly_coeffs,
                y_start_px,
                y_end_px,
                num_samples=3
            )

            # Unpack the 3 points in meters
            if len(points_m) >= 3:
                p1_x_m, p1_y_m = points_m[0]  # Top point in meters
                p2_x_m, p2_y_m = points_m[1]  # Middle point in meters
                p3_x_m, p3_y_m = points_m[2]  # Bottom point in meters
            else:
                # Fallback if conversion fails
                p1_x_m = p1_y_m = p2_x_m = p2_y_m = p3_x_m = p3_y_m = 0.0

            # Calculate center point in meters (use middle point)
            x_m = p2_x_m
            y_m = p2_y_m

            payload_parts.append(
                struct.pack(
                    "<BBBfffffffffffffffff",  # 3 bytes + 17 floats = 71 bytes
                    # poly(3) + xy(2) + points_m(6) + points_px(6) = 17 floats
                    side_byte & 0xFF,
                    style_byte & 0xFF,
                    color_byte & 0xFF,
                    poly_a,
                    poly_b,
                    poly_c,
                    x_m,
                    y_m,
                    p1_x_m,
                    p1_y_m,
                    p2_x_m,
                    p2_y_m,
                    p3_x_m,
                    p3_y_m,
                    p1_x_px,
                    p1_y_px,
                    p2_x_px,
                    p2_y_px,
                    p3_x_px,
                    p3_y_px,
                )
            )

        payload = b"".join(payload_parts)
        return self._build_frame(MSG_TYPE_LANE_LINES, payload)

    def build_road_objects_frame(self, objects: Iterable[MarkingObject]) -> bytes:
        """
        Build frame with road objects (v2 protocol - message type 0x02).

        Each object contains:
        - class_id (1 byte): object class (arrows, crosswalks, etc.)
        - center_x (float, 4 bytes): center X position in meters
        - center_y (float, 4 bytes): center Y position in meters
        - length (float, 4 bytes): object length in meters
        - width (float, 4 bytes): object width in meters
        - yaw (float, 4 bytes): orientation angle in radians
        - confidence (1 byte): detection confidence 0-255
        - flags (1 byte): status flags
        - reserved (2 bytes): for future use

        Total: 25 bytes per object
        Frame size: 1(sync) + 9(header) + 1(count) + N*25 + 2(crc) bytes
        """
        objs_list: List[MarkingObject] = list(objects)
        max_objects = (MAX_PAYLOAD_SIZE - 1) // self.road_object_size
        if len(objs_list) > max_objects:
            logger.warning("Trimming road objects from %d to %d to satisfy frame size", len(objs_list), max_objects)
            objs_list = objs_list[:max_objects]

        count = len(objs_list)
        payload_parts = [struct.pack("<B", count)]

        for obj in objs_list:
            # Use pixel coordinates if available, otherwise fall back to dm values
            if obj.center_px and obj.bbox_px:
                # Convert center from pixels to meters using perspective transform
                center_x_m, center_y_m = self.coordinate_transformer.pixel_to_meters(
                    obj.center_px[0],
                    obj.center_px[1]
                )

                # Convert bbox dimensions from pixels to meters
                length_m, width_m = self.coordinate_transformer.get_bbox_dimensions_in_meters(
                    obj.bbox_px
                )

                # Use stored yaw in radians
                yaw_rad = obj.yaw_rad
            else:
                # Fallback: convert existing dm values to meters
                logger.warning("Object missing pixel coordinates, using dm fallback for class_id=%d", obj.class_id)
                center_x_m = float(obj.x_dm) / 10.0
                center_y_m = float(obj.y_dm) / 10.0
                length_m = float(obj.length_dm) / 10.0
                width_m = float(obj.width_dm) / 10.0
                yaw_rad = float(obj.yaw_decideg) / 10.0 * (3.14159265359 / 180.0)

            # Confidence byte
            confidence_byte = obj.confidence_byte & 0xFF

            # Flags
            flags = obj.flags & 0xFF

            payload_parts.append(
                struct.pack(
                    "<BfffffBB",  # 1 byte + 5 floats + 2 bytes = 1+20+2 = 23 bytes total
                    obj.class_id & 0xFF,
                    center_x_m,
                    center_y_m,
                    length_m,
                    width_m,
                    yaw_rad,
                    confidence_byte,
                    flags,
                )
            )

        payload = b"".join(payload_parts)
        return self._build_frame(MSG_TYPE_ROAD_OBJECTS, payload)


if __name__ == "__main__":  # simple manual self-check
    logging.basicConfig(level=logging.INFO)
    builder = ProtocolBuilder()
    ls = LaneSummary(left_offset_dm=-5, right_offset_dm=5, quality=123)
    mo = [
        MarkingObject(class_id=1, x_dm=10, y_dm=20, length_dm=30, width_dm=5, yaw_decideg=0, confidence_byte=200),
        MarkingObject(class_id=2, x_dm=-15, y_dm=25, length_dm=40, width_dm=6, yaw_decideg=10, confidence_byte=180),
    ]
    lane_frame = builder.build_lane_summary_frame(ls)
    obj_frame = builder.build_marking_objects_frame(mo)
    lane_details = builder.build_lane_details_frame(ls)
    obj_frame_ex = builder.build_marking_objects_ex_frame(mo)
    print(f"LaneSummary frame length={len(lane_frame)} (expected 1+9+8+2=20)")
    print(f"LaneDetails frame length={len(lane_details)} (expected 1+9+46+2=58)")
    print(f"MarkingObjects frame length={len(obj_frame)} (expected 1+9+1+N*13+2)")
    print(f"MarkingObjectsEx frame length={len(obj_frame_ex)} (expected 1+9+1+N*15+2)")
