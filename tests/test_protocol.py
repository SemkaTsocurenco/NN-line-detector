from network.protocol import ProtocolBuilder
from core.detections import LaneSummary, MarkingObject


def test_lane_summary_frame_length():
    builder = ProtocolBuilder()
    frame = builder.build_lane_summary_frame(LaneSummary(left_offset_dm=-1, right_offset_dm=2, quality=10))
    # Frame: 1(sync) + 9(header) + 8(payload) + 2(crc)
    assert len(frame) == 20
    assert frame[0] == 0xAA


def test_marking_objects_frame_length():
    builder = ProtocolBuilder()
    objs = [
        MarkingObject(class_id=1, x_dm=10, y_dm=20, length_dm=30, width_dm=5, yaw_decideg=0, confidence_byte=200),
        MarkingObject(class_id=2, x_dm=-15, y_dm=25, length_dm=40, width_dm=6, yaw_decideg=10, confidence_byte=180),
    ]
    frame = builder.build_marking_objects_frame(objs)
    # Frame: 1(sync) + 9(header) + (1 + N*13) + 2(crc)
    assert len(frame) == 1 + 9 + (1 + len(objs) * 13) + 2
    assert frame[0] == 0xAA
