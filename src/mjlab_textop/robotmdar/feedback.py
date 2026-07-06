from __future__ import annotations

import json
import threading
from base64 import b64decode
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class FeedbackObservation:
    image_bytes: bytes | None = None
    image_mime_type: str | None = None


class HttpObservationReceiver:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        path: str = "/observation",
    ) -> None:
        if port <= 0:
            raise ValueError(f"Observation port must be positive, got {port}")
        if not path.startswith("/"):
            raise ValueError(f"Observation path must start with '/', got {path!r}")
        self.host = host
        self.port = port
        self.path = path
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None
        self._latest: FeedbackObservation | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._server = self._make_server()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self) -> FeedbackObservation | None:
        with self._lock:
            return self._latest

    def handle_post(self, body: bytes) -> None:
        try:
            observation = parse_feedback_observation(body)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            raise
        with self._lock:
            self._latest = observation

    def _make_server(self) -> ThreadingHTTPServer:
        receiver = self

        class ObservationHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != receiver.path:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    receiver.handle_post(body)
                except (TypeError, ValueError, json.JSONDecodeError):
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        return ThreadingHTTPServer((self.host, self.port), ObservationHandler)


def parse_feedback_observation(
    message: bytes | str | dict[str, Any],
) -> FeedbackObservation:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if isinstance(message, str):
        message = json.loads(message)
    if not isinstance(message, dict):
        raise ValueError("Feedback observation must be a JSON object")

    image = _parse_image(message.get("image"))

    return FeedbackObservation(
        image_bytes=None if image is None else image["data"],
        image_mime_type=None if image is None else image["mime_type"],
    )


def _parse_image(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Feedback observation image must be a JSON object")
    mime_type = str(value["mime_type"])
    data = b64decode(str(value["data"]), validate=True)
    return {"mime_type": mime_type, "data": data}
