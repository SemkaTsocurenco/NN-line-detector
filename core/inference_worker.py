from __future__ import annotations

import logging
import queue
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui

from core.nn_engine import NNEngine
from core.postprocess import DetectionPostprocessor, GeometryMapper, PostprocessParams
from core.renderer import Renderer

logger = logging.getLogger(__name__)


class InferenceWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    detection_data = QtCore.pyqtSignal(object, list)  # LaneSummary, list[MarkingObject]
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        nn_engine: NNEngine,
        postprocessor: DetectionPostprocessor,
        geometry_mapper: GeometryMapper,
        renderer: Renderer,
    ) -> None:
        super().__init__()
        self.nn_engine = nn_engine
        self.postprocessor = postprocessor
        self.geometry_mapper = geometry_mapper
        self.renderer = renderer

        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._thread: Optional[QtCore.QThread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = QtCore.QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()
        logger.info("InferenceWorker started")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._frame_queue.put_nowait(np.zeros((1, 1, 3), dtype=np.uint8))
        except queue.Full:
            pass
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        logger.info("InferenceWorker stopped")

    def submit_frame(self, frame: np.ndarray) -> None:
        if not self._running:
            return
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def update_postprocess_params(self, params: PostprocessParams) -> None:
        self.postprocessor.update_params(params)

    def _run(self) -> None:
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None or frame.size == 0:
                continue
            try:
                raw = self.nn_engine.infer(frame)
                processed = self.postprocessor.process(raw, frame.shape)
                summary = self.postprocessor.build_lane_summary(processed, frame.shape)
                objects = self.geometry_mapper.to_marking_objects(processed, frame.shape)
                rendered = self.renderer.render(frame, processed, summary)
                qimage = self._to_qimage(rendered)
                self.frame_ready.emit(qimage)
                self.detection_data.emit(summary, objects)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Inference error: %s", exc)
                self.error.emit(str(exc))

    @staticmethod
    def _to_qimage(image: np.ndarray) -> QtGui.QImage:
        h, w, _ = image.shape
        return QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
