from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from typing import Optional

from PyQt5 import QtCore

logger = logging.getLogger(__name__)


class TcpClient(QtCore.QObject):
    status_changed = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        host: str,
        port: int,
        reconnect_delay_s: float = 2.0,
        drop_if_disconnected: bool = True,
        max_queue: int = 100,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.reconnect_delay_s = reconnect_delay_s
        self.drop_if_disconnected = drop_if_disconnected
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TcpClient thread started")

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        with self._queue.mutex:
            self._queue.queue.clear()
        logger.info("TcpClient stopped")

    def send(self, data: bytes) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(data)
            except queue.Full:
                logger.warning("TcpClient queue full, dropping message")

    def _run(self) -> None:
        while self._running:
            if not self._sock:
                self._connect()
            if not self._sock:
                time.sleep(self.reconnect_delay_s)
                continue

            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._sock.sendall(msg)
            except Exception as exc:
                logger.error("TCP send failed: %s", exc)
                self.error.emit(str(exc))
                self._close_socket()
                if self.drop_if_disconnected:
                    with self._queue.mutex:
                        self._queue.queue.clear()
                time.sleep(self.reconnect_delay_s)

    def _connect(self) -> None:
        try:
            self.status_changed.emit("connecting")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.host, self.port))
            sock.settimeout(None)
            self._sock = sock
            self.status_changed.emit("connected")
            logger.info("TcpClient connected to %s:%d", self.host, self.port)
        except Exception as exc:
            logger.error("TCP connect failed: %s", exc)
            self.error.emit(str(exc))
            self.status_changed.emit("disconnected")
            self._close_socket()

    def _close_socket(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
