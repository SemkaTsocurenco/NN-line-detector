from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import ipaddress
from urllib.parse import urlparse
from PyQt5 import QtCore

logger = logging.getLogger(__name__)

try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import GObject, Gst

    Gst.init(None)
    GST_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    logger.error("GStreamer bindings are not available: %s", exc)
    GST_AVAILABLE = False
    GObject = None  # type: ignore
    Gst = None  # type: ignore


class VideoCaptureService(QtCore.QObject):
    """GStreamer-based RTSP capture with a small frame queue."""

    frame_ready = QtCore.pyqtSignal(np.ndarray)
    state_changed = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        rtsp_uri: str,
        latency_ms: int = 100,
        drop_old_frames: bool = True,
        gst_pipeline: str = "",
    ) -> None:
        super().__init__()
        self.rtsp_uri = rtsp_uri
        self.latency_ms = latency_ms
        self.drop_old_frames = drop_old_frames
        self.gst_pipeline = gst_pipeline

        self._pipeline: Optional[Gst.Element] = None
        self._appsink: Optional[Gst.Element] = None
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._loop: Optional[GObject.MainLoop] = None
        self._bus_watch = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if not GST_AVAILABLE:
            self.error.emit("GStreamer is not available in this environment.")
            return
        if self._running:
            return
        pipeline_str = self.gst_pipeline or self._build_pipeline()
        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except Exception as exc:  # pragma: no cover - Gst.parse_launch side effects
            logger.exception("Failed to create pipeline: %s", exc)
            self.error.emit(f"Pipeline error: {exc}")
            return

        self._appsink = self._pipeline.get_by_name("appsink0")
        if not self._appsink:
            self.error.emit("appsink0 not found in pipeline")
            return
        self._appsink.set_property("emit-signals", True)
        self._appsink.connect("new-sample", self._on_new_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._loop = GObject.MainLoop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._running = True
        self._thread.start()

        self._pipeline.set_state(Gst.State.PLAYING)
        self.state_changed.emit("starting")
        logger.info("VideoCaptureService started with pipeline: %s", pipeline_str)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop:
            self._loop.quit()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._frame_queue = queue.Queue(maxsize=2)
        self.state_changed.emit("stopped")
        logger.info("VideoCaptureService stopped")

    def _run_loop(self) -> None:
        if not self._loop:
            return
        try:
            self._loop.run()
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.exception("GObject main loop error: %s", exc)
            self.error.emit(str(exc))
        finally:
            if self._pipeline:
                self._pipeline.set_state(Gst.State.NULL)

    def _build_pipeline(self) -> str:
        parsed = urlparse(self.rtsp_uri)
        host = parsed.hostname or ""
        scheme = (parsed.scheme or "").lower()
        port = parsed.port

        if scheme in ("udp", "rtp"):
            return self._udp_pipeline(host, port)

        if host and self._is_multicast(host):
            return self._udp_pipeline(host, port)

        return self._rtsp_pipeline()

    def _rtsp_pipeline(self) -> str:
        return (
            f"rtspsrc location={self.rtsp_uri} latency={self.latency_ms} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=appsink0 emit-signals=true "
            "sync=false max-buffers=2 drop=true"
        )

    def _udp_pipeline(self, host: str, port: Optional[int]) -> str:
        if not host or not port:
            logger.warning("UDP pipeline requested but host/port not resolved, falling back to RTSP")
            return self._rtsp_pipeline()

        return (
            f"udpsrc multicast-group={host} auto-multicast=true port={port} "
            'caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
            f"rtpjitterbuffer latency={self.latency_ms} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=appsink0 emit-signals=true "
            "sync=false max-buffers=2 drop=true"
        )

    @staticmethod
    def _is_multicast(host: str) -> bool:
        try:
            ip = ipaddress.ip_address(host)
            return ip.is_multicast
        except ValueError:
            return False

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        if not caps:
            return Gst.FlowReturn.ERROR
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
            frame_copy = frame.copy()  # detach from Gst buffer
        finally:
            buf.unmap(map_info)

        if self.drop_old_frames and self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame_copy)
        except queue.Full:
            pass

        self.frame_ready.emit(frame_copy)
        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> None:  # pragma: no cover
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            text = f"{err.message} ({debug})"
            logger.error("GStreamer error: %s", text)
            self.error.emit(text)
            self.stop()
        elif msg_type == Gst.MessageType.EOS:
            logger.info("GStreamer EOS received")
            self.stop()
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                old, new, _ = message.parse_state_changed()
                self.state_changed.emit(f"{old.value_nick}->{new.value_nick}")

    def fetch_latest_frame(self) -> Optional[np.ndarray]:
        """Return the freshest frame from queue without blocking."""
        latest = None
        try:
            while True:
                latest = self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        return latest
