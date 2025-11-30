from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Iterable, List

from core.detections import LaneSummary, MarkingObject

logger = logging.getLogger(__name__)

SYNC_BYTE = 0xAA
VERSION = 0x01
MSG_TYPE_LANE_SUMMARY = 0x01
MSG_TYPE_MARKING_OBJECTS = 0x02
MAX_PAYLOAD_SIZE = 1024  # per spec: payload length constraint


def crc16_ibm(data: bytes) -> int:
    """CRC16-IBM (reflected), initial 0xFFFF, polynomial 0xA001."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
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
            self._value = (self._value + 1) % 256
            return val


class TimeProvider:
    """Provides timestamp in milliseconds (uint32) using monotonic time."""

    def __call__(self) -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF


class ProtocolBuilder:
    def __init__(self, seq_counter: SeqCounter | None = None, time_provider: TimeProvider | None = None) -> None:
        self.seq_counter = seq_counter or SeqCounter()
        self.time_provider = time_provider or TimeProvider()

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
        max_objects = (MAX_PAYLOAD_SIZE - 1) // 13  # 1 byte for count
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

    def _build_frame(self, msg_type: int, payload: bytes) -> bytes:
        seq = self.seq_counter.next()
        timestamp_ms = self.time_provider()
        header = struct.pack("<BBBIH", VERSION, msg_type & 0xFF, seq & 0xFF, timestamp_ms, len(payload))
        frame_wo_crc = bytes([SYNC_BYTE]) + header + payload
        crc = crc16_ibm(frame_wo_crc[1:])  # without sync byte
        frame = frame_wo_crc + struct.pack("<H", crc)
        return frame


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
    print(f"LaneSummary frame length={len(lane_frame)} (expected 1+9+8+2=20)")
    print(f"MarkingObjects frame length={len(obj_frame)} (expected 1+9+1+2*13+2=39)")
