from __future__ import annotations

import socket
import threading
from collections import deque
from dataclasses import dataclass

from textop_protocol.motion import MotionBlock
from textop_protocol.motion_stream import (
    recv_textop_block,
    textop_block_from_wire,
    textop_block_to_wire,
)

__all__ = [
    "SocketOnlineSource",
    "SocketSourceCfg",
    "TextOpLiveDiagnostics",
    "textop_block_from_wire",
    "textop_block_to_wire",
]


@dataclass(frozen=True)
class SocketSourceCfg:
    host: str = "127.0.0.1"
    port: int = 8765
    max_queue_blocks: int = 32


@dataclass
class TextOpLiveDiagnostics:
    queue_depth: int = 0
    blocks_received: int = 0
    blocks_polled: int = 0
    bad_messages: int = 0
    last_error: str | None = None
    connected: bool = False


class SocketOnlineSource:
    def __init__(self, cfg: SocketSourceCfg | None = None) -> None:
        cfg = cfg or SocketSourceCfg()
        if cfg.max_queue_blocks <= 0:
            raise ValueError(
                f"max_queue_blocks must be positive, got {cfg.max_queue_blocks}"
            )
        self.cfg = cfg
        self.diagnostics = TextOpLiveDiagnostics()
        self._queue: deque[MotionBlock] = deque()
        self._queue_condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._queue_condition:
            self._queue_condition.notify_all()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._sock = None

    def poll(self) -> MotionBlock | None:
        with self._queue_condition:
            if not self._queue:
                self._sync_queue_depth_locked()
                return None
            block = self._queue.popleft()
            self.diagnostics.blocks_polled += 1
            self._sync_queue_depth_locked()
            self._queue_condition.notify()
            return block

    def append_wire_record(self, record: bytes) -> None:
        self._append_block(textop_block_from_wire(record))

    def _reader_loop(self) -> None:
        try:
            with socket.create_connection(
                (self.cfg.host, self.cfg.port), timeout=1.0
            ) as sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock
                self.diagnostics.connected = True
                while not self._stop.is_set():
                    self._handle_next_record(sock)
        except (OSError, ValueError) as exc:
            self.diagnostics.last_error = str(exc)
        finally:
            self.diagnostics.connected = False
            self._sock = None

    def _handle_next_record(self, sock: socket.socket) -> None:
        try:
            block = recv_textop_block(sock)
            if block is None:
                self._stop.set()
                return
            self._append_block(block)
        except (ValueError, UnicodeDecodeError) as exc:
            with self._queue_condition:
                self.diagnostics.bad_messages += 1
                self.diagnostics.last_error = str(exc)
            self._stop.set()

    def _append_block(self, block: MotionBlock) -> None:
        with self._queue_condition:
            while (
                len(self._queue) >= self.cfg.max_queue_blocks
                and not self._stop.is_set()
            ):
                self._queue_condition.wait()
            if self._stop.is_set():
                return
            self._queue.append(block)
            self.diagnostics.blocks_received += 1
            self._sync_queue_depth_locked()

    def _sync_queue_depth_locked(self) -> None:
        self.diagnostics.queue_depth = len(self._queue)
