from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

import imageio.v3 as iio

OBSERVATION_JPEG_QUALITY = 95


@dataclass(frozen=True)
class ObservationImage:
    data: bytes
    mime_type: str
    frame: int


class TextOpObservationPublisher(Protocol):
    def publish(
        self,
        state: dict[str, Any],
        *,
        image: ObservationImage | None = None,
    ) -> None:
        """Publish one MJLab observation payload."""


@dataclass(frozen=True)
class HttpObservationPublisherCfg:
    url: str = "http://127.0.0.1:8766/observation"
    timeout_sec: float = 1.0


@dataclass(frozen=True, kw_only=True)
class OnlineTextOpObservationCfg:
    publisher: TextOpObservationPublisher | None = None
    publish_interval: int = 1
    publish_images: bool = True
    image_publish_interval: int = 5

    def __post_init__(self) -> None:
        if self.publish_interval <= 0:
            raise ValueError(
                f"publish_interval must be positive, got {self.publish_interval}"
            )
        if self.image_publish_interval <= 0:
            raise ValueError(
                "image_publish_interval must be positive, "
                f"got {self.image_publish_interval}"
            )


class HttpObservationPublisher:
    def __init__(self, cfg: HttpObservationPublisherCfg) -> None:
        if not cfg.url:
            raise ValueError("Observation publisher URL must be non-empty")
        if cfg.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {cfg.timeout_sec}")
        self.cfg = cfg

    def publish(
        self,
        state: dict[str, Any],
        *,
        image: ObservationImage | None = None,
    ) -> None:
        request = urllib.request.Request(
            self.cfg.url,
            data=json.dumps(
                make_http_observation_payload(state=state, image=image),
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.cfg.timeout_sec) as response:
            response.read()


def make_online_textop_observation(
    *,
    frame: int,
    started: bool,
    current_frame: int,
    latest_frame: int | None,
    lag_frames: int,
    buffer_frames: int,
    stale_steps: int,
    consecutive_stale_steps: int,
    robot_anchor_pos_w: Any,
    robot_anchor_quat_w: Any,
    image_frame: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "mjlab_textop.online_observation.v1",
        "frame": int(frame),
        "started": bool(started),
        "current_frame": int(current_frame),
        "latest_frame": None if latest_frame is None else int(latest_frame),
        "lag_frames": int(lag_frames),
        "buffer_frames": int(buffer_frames),
        "stale_steps": int(stale_steps),
        "consecutive_stale_steps": int(consecutive_stale_steps),
        "robot_anchor_pos_w": [
            float(item)
            for item in robot_anchor_pos_w.detach().cpu().reshape(-1).tolist()
        ],
        "robot_anchor_quat_w": [
            float(item)
            for item in robot_anchor_quat_w.detach().cpu().reshape(-1).tolist()
        ],
    }
    if image_frame is not None:
        payload["image_frame"] = image_frame
    return payload


def make_http_observation_payload(
    *,
    state: dict[str, Any],
    image: ObservationImage | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"state": state}
    if image is not None:
        payload["image"] = {
            "mime_type": image.mime_type,
            "frame": image.frame,
            "data": b64encode(image.data).decode("ascii"),
        }
    return payload


def encode_render_image_jpeg(image: Any) -> bytes:
    buffer = BytesIO()
    iio.imwrite(buffer, image, extension=".jpg", quality=OBSERVATION_JPEG_QUALITY)
    return buffer.getvalue()
