import logging
import sys

from PyQt5 import QtWidgets

from ui.main_window import MainWindow
from utils.config_manager import ConfigManager
from utils.logging_setup import setup_logging


def main() -> int:
    config = ConfigManager()
    setup_logging(level=config.get_value("logging.level", "INFO"), log_file=config.get_value("logging.log_file", ""))

    logging.info("Starting NN Line Detector application")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
