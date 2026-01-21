import logging
import os
import sys

from utils.config_manager import ConfigManager
from utils.logging_setup import setup_logging


def configure_qt_environment() -> None:
    """
    Ensure Qt uses the correct plugin path (avoid OpenCV's bundled plugins).
    Must be called before importing modules that might load cv2/Qt.
    """
    try:
        os.environ.pop("QT_PLUGIN_PATH", None)  # remove paths injected by other libs (e.g., cv2)
        from PyQt5.QtCore import QLibraryInfo

        plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
        os.environ["QT_PLUGIN_PATH"] = plugin_path
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Could not adjust Qt plugin path: %s", exc)


def main() -> int:
    config = ConfigManager()
    setup_logging(level=config.get_value("logging.level", "INFO"), log_file=config.get_value("logging.log_file", ""))

    configure_qt_environment()

    from PyQt5 import QtWidgets

    def apply_stylesheet(app: QtWidgets.QApplication) -> None:
        from PyQt5 import QtCore

        import resources_rc  # noqa: F401

        style_file = QtCore.QFile(":/styles/dark.qss")
        if not style_file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text):
            logging.warning("Could not load stylesheet from resources")
            return
        stream = QtCore.QTextStream(style_file)
        app.setStyleSheet(stream.readAll())

    from ui.main_window import MainWindow

    logging.info("Starting NN Line Detector application")
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app)
    window = MainWindow(config)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
