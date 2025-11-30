from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from core.inference_worker import InferenceWorker
from core.nn_engine import NNEngine
from core.postprocess import DetectionPostprocessor, GeometryMapper, PostprocessParams
from core.renderer import Renderer
from core.video_capture import GST_AVAILABLE, VideoCaptureService
from ui.video_widget import VideoWidget
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_manager: ConfigManager, parent=None) -> None:
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("NN Line Detector")
        self.resize(1200, 800)

        self.video_widget = VideoWidget(self)
        self.start_button = QtWidgets.QPushButton("Start RTSP")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.rtsp_edit = QtWidgets.QLineEdit(self.config_manager.get_value("rtsp.uri", ""))
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(float(self.config_manager.get_value("postprocess.confidence_threshold", 0.5)))

        self.area_spin = QtWidgets.QDoubleSpinBox()
        self.area_spin.setRange(0.0, 100000.0)
        self.area_spin.setDecimals(1)
        self.area_spin.setSingleStep(10.0)
        self.area_spin.setValue(float(self.config_manager.get_value("postprocess.min_area", 100.0)))

        self.tcp_host_edit = QtWidgets.QLineEdit(self.config_manager.get_value("tcp.host", "127.0.0.1"))
        self.tcp_port_spin = QtWidgets.QSpinBox()
        self.tcp_port_spin.setRange(1, 65535)
        self.tcp_port_spin.setValue(int(self.config_manager.get_value("tcp.port", 9000)))

        self.rtsp_status = QtWidgets.QLabel("RTSP: stopped")
        self.nn_status = QtWidgets.QLabel("NN: idle")
        self.tcp_status = QtWidgets.QLabel("TCP: idle")
        self.detections_label = QtWidgets.QLabel("Detections: 0")

        self._video_service: Optional[VideoCaptureService] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._nn_engine: Optional[NNEngine] = None

        self._build_layout()
        self._connect_signals()
        if not GST_AVAILABLE:
            self.start_button.setEnabled(False)
            self.rtsp_status.setText("RTSP: GStreamer bindings not available")

    def _build_layout(self) -> None:
        control_layout = QtWidgets.QFormLayout()
        control_layout.addRow("RTSP URI:", self.rtsp_edit)
        control_layout.addRow("Confidence:", self.conf_spin)
        control_layout.addRow("Min area:", self.area_spin)
        control_layout.addRow("TCP host:", self.tcp_host_edit)
        control_layout.addRow("TCP port:", self.tcp_port_spin)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addStretch()

        status_layout = QtWidgets.QHBoxLayout()
        status_layout.addWidget(self.rtsp_status)
        status_layout.addWidget(self.nn_status)
        status_layout.addWidget(self.tcp_status)
        status_layout.addWidget(self.detections_label)
        status_layout.addStretch()

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addLayout(control_layout)
        right_layout.addLayout(buttons_layout)
        right_layout.addLayout(status_layout)
        right_layout.addStretch()

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.video_widget, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        self.area_spin.valueChanged.connect(self._on_area_changed)
        self.rtsp_edit.editingFinished.connect(self._on_rtsp_changed)
        self.tcp_host_edit.editingFinished.connect(self._on_tcp_changed)
        self.tcp_port_spin.valueChanged.connect(self._on_tcp_changed)

    def _on_start_clicked(self) -> None:
        try:
            self._start_services()
        except Exception as exc:
            logger.exception("Failed to start: %s", exc)
            self.nn_status.setText(f"NN error: {exc}")
            self._teardown_inference()
            if self._video_service:
                self._video_service.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _start_services(self) -> None:
        self._teardown_inference()
        self._start_inference()

        if self._video_service:
            self._video_service.stop()
        self._video_service = VideoCaptureService(
            rtsp_uri=self.rtsp_edit.text().strip(),
            latency_ms=int(self.config_manager.get_value("rtsp.latency_ms", 100)),
            drop_old_frames=bool(self.config_manager.get_value("rtsp.drop_old_frames", True)),
            gst_pipeline=self.config_manager.get_value("rtsp.gst_pipeline", ""),
        )
        self._video_service.frame_ready.connect(self._on_frame)
        self._video_service.state_changed.connect(self._on_rtsp_state)
        self._video_service.error.connect(self._on_rtsp_error)
        self._video_service.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.rtsp_status.setText("RTSP: connecting")

    def _start_inference(self) -> None:
        model_path = self.config_manager.get_value("nn.model_path", "NN/model_traced.pt")
        device = self.config_manager.get_value("nn.device", "cpu")
        input_size_cfg = self.config_manager.get_value("nn.input_size", None)
        input_size = None
        if isinstance(input_size_cfg, (list, tuple)) and len(input_size_cfg) == 2:
            input_size = (int(input_size_cfg[0]), int(input_size_cfg[1]))
        class_names_cfg = self.config_manager.get_value("nn.class_names", {})
        class_names: Dict[int, str] = {int(k): v for k, v in (class_names_cfg or {}).items()}

        self._nn_engine = NNEngine(model_path=model_path, device=device, input_size=input_size)
        if self._nn_engine.model is None:
            raise RuntimeError("Model not loaded")

        params = self._current_postprocess_params()
        postprocessor = DetectionPostprocessor(params)
        geometry = GeometryMapper(self.config_manager.get_value("camera.calibration", {}))
        renderer = Renderer(class_names)

        self._inference_worker = InferenceWorker(
            nn_engine=self._nn_engine,
            postprocessor=postprocessor,
            geometry_mapper=geometry,
            renderer=renderer,
        )
        self._inference_worker.frame_ready.connect(self._on_inference_frame)
        self._inference_worker.detections_updated.connect(self._on_detections_updated)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.start()
        self.nn_status.setText("NN: running")

    def _on_stop_clicked(self) -> None:
        if self._video_service:
            self._video_service.stop()
        self._teardown_inference()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.rtsp_status.setText("RTSP: stopped")

    def _on_conf_changed(self, value: float) -> None:
        self.config_manager.set_value("postprocess.confidence_threshold", float(value))
        self.config_manager.save()
        self._update_postprocess_params()

    def _on_area_changed(self, value: float) -> None:
        self.config_manager.set_value("postprocess.min_area", float(value))
        self.config_manager.save()
        self._update_postprocess_params()

    def _update_postprocess_params(self) -> None:
        if self._inference_worker:
            self._inference_worker.update_postprocess_params(self._current_postprocess_params())

    def _current_postprocess_params(self) -> PostprocessParams:
        return PostprocessParams(
            confidence_threshold=float(self.config_manager.get_value("postprocess.confidence_threshold", 0.5)),
            min_area=float(self.config_manager.get_value("postprocess.min_area", 100.0)),
            merge_iou_threshold=float(self.config_manager.get_value("postprocess.merge_iou_threshold", 0.5)),
        )

    def _on_rtsp_changed(self) -> None:
        uri = self.rtsp_edit.text().strip()
        self.config_manager.set_value("rtsp.uri", uri)
        self.config_manager.save()

    def _on_tcp_changed(self) -> None:
        self.config_manager.set_value("tcp.host", self.tcp_host_edit.text().strip())
        self.config_manager.set_value("tcp.port", int(self.tcp_port_spin.value()))
        self.config_manager.save()

    def _on_frame(self, frame: np.ndarray) -> None:
        if self._inference_worker:
            self._inference_worker.submit_frame(frame)
        else:
            self._display_raw_frame(frame)

    def _display_raw_frame(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.video_widget.set_image(image.copy())

    def _on_inference_frame(self, image: QtGui.QImage) -> None:
        self.video_widget.set_image(image)

    def _on_detections_updated(self, summary: object, count: int) -> None:
        self.detections_label.setText(f"Detections: {count}")

    def _on_inference_error(self, msg: str) -> None:
        logger.error("Inference error: %s", msg)
        self.nn_status.setText(f"NN error: {msg}")

    def _on_rtsp_state(self, state: str) -> None:
        self.rtsp_status.setText(f"RTSP: {state}")

    def _on_rtsp_error(self, msg: str) -> None:
        self.rtsp_status.setText(f"RTSP error: {msg}")
        logger.error("RTSP error: %s", msg)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _teardown_inference(self) -> None:
        if self._inference_worker:
            self._inference_worker.stop()
            self._inference_worker = None
        self.nn_status.setText("NN: idle")
        self.detections_label.setText("Detections: 0")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._video_service:
            self._video_service.stop()
        self._teardown_inference()
        super().closeEvent(event)
