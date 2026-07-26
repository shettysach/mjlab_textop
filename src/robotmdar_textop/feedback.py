from __future__ import annotations

import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from textop_protocol.observation import (
    ObservationMessage,
    parse_observation_json,
)


@dataclass(frozen=True)
class FeedbackObservation:
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    image_revision: int = 0
    collision_stop: bool = False
    recovery_epoch: int = 0
    observation_request_id: int | None = None
    source_frame: int | None = None


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
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None
        self._latest: FeedbackObservation | None = None
        self._requested_observation: FeedbackObservation | None = None
        self._closed = False
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._closed = False
        self._server = self._make_server()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest(self) -> FeedbackObservation | None:
        with self._condition:
            return self._latest

    def wait_for_observation(self, observation_request_id: int) -> FeedbackObservation:
        with self._condition:
            while True:
                if self.last_error is not None:
                    raise RuntimeError(self.last_error)
                if self._latest is not None and self._latest.collision_stop:
                    self._requested_observation = None
                    return self._latest
                observation = self._requested_observation
                if observation is not None:
                    self._requested_observation = None
                if (
                    observation is not None
                    and observation.observation_request_id == observation_request_id
                ):
                    return observation
                if self._closed:
                    raise RuntimeError("Observation receiver closed while waiting")
                self._condition.wait()

    def handle_post(self, body: bytes) -> None:
        try:
            message = parse_observation_json(body)
        except (TypeError, ValueError) as exc:
            with self._condition:
                self.last_error = str(exc)
                self._condition.notify_all()
            raise
        with self._condition:
            self._latest = merge_feedback_message(self._latest, message)
            if message.observation_request_id is not None:
                self._requested_observation = self._latest
            self._condition.notify_all()

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
                except (TypeError, ValueError):
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
    parsed = parse_observation_json(message)
    return merge_feedback_message(None, parsed)


def merge_feedback_message(
    previous: FeedbackObservation | None,
    message: ObservationMessage,
) -> FeedbackObservation:
    previous = previous or FeedbackObservation()
    image_bytes = previous.image_bytes
    image_mime_type = previous.image_mime_type
    image_revision = previous.image_revision
    observation_request_id = previous.observation_request_id
    source_frame = previous.source_frame
    if message.image is not None:
        image_bytes = message.image.data
        image_mime_type = message.image.mime_type
        image_revision += 1
        observation_request_id = message.observation_request_id
        source_frame = message.source_frame

    return FeedbackObservation(
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        image_revision=image_revision,
        collision_stop=(
            previous.collision_stop
            if message.collision_stop is None
            else message.collision_stop
        ),
        recovery_epoch=(
            previous.recovery_epoch
            if message.recovery_epoch is None
            else message.recovery_epoch
        ),
        observation_request_id=observation_request_id,
        source_frame=source_frame,
    )
