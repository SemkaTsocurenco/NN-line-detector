from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets


class VideoWidget(QtWidgets.QWidget):
    """Simple widget to display a QImage/Pixmap scaled to the widget size."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAutoFillBackground(True)

    def set_image(self, image: QtGui.QImage) -> None:
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        painter.end()
