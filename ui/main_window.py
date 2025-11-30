from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

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

        self._video_service: Optional[VideoCaptureService] = None

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

    def _on_stop_clicked(self) -> None:
        if self._video_service:
            self._video_service.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.rtsp_status.setText("RTSP: stopped")

    def _on_conf_changed(self, value: float) -> None:
        self.config_manager.set_value("postprocess.confidence_threshold", float(value))
        self.config_manager.save()

    def _on_area_changed(self, value: float) -> None:
        self.config_manager.set_value("postprocess.min_area", float(value))
        self.config_manager.save()

    def _on_rtsp_changed(self) -> None:
        uri = self.rtsp_edit.text().strip()
        self.config_manager.set_value("rtsp.uri", uri)
        self.config_manager.save()

    def _on_tcp_changed(self) -> None:
        self.config_manager.set_value("tcp.host", self.tcp_host_edit.text().strip())
        self.config_manager.set_value("tcp.port", int(self.tcp_port_spin.value()))
        self.config_manager.save()

    def _on_frame(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.video_widget.set_image(image.copy())

    def _on_rtsp_state(self, state: str) -> None:
        self.rtsp_status.setText(f"RTSP: {state}")

    def _on_rtsp_error(self, msg: str) -> None:
        self.rtsp_status.setText(f"RTSP error: {msg}")
        logger.error("RTSP error: %s", msg)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._video_service:
            self._video_service.stop()
        super().closeEvent(event)
