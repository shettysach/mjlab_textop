from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Protocol

import imageio.v3 as iio
import torch
from mjlab.viewer import ViewerConfig

OBSERVATION_JPEG_QUALITY = 95


@dataclass(frozen=True)
class ObservationImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class OnlineObservationState:
    frame: int
    started: bool
    latest_index: int | None
    lag_frames: int
    buffer_frames: int
    stale_steps: int
    consecutive_stale_steps: int
    robot_anchor_pos_w: torch.Tensor
    robot_anchor_quat_w: torch.Tensor


class TextOpObservationPublisher(Protocol):
    def publish(
        self,
        state: dict[str, Any],
        *,
        image: ObservationImage,
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
    camera: ViewerConfig = field(default_factory=lambda: make_torso_observation_camera())

    def __post_init__(self) -> None:
        if self.publish_interval <= 0:
            raise ValueError(
                f"publish_interval must be positive, got {self.publish_interval}"
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
        image: ObservationImage,
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


def make_online_textop_observation(state: OnlineObservationState) -> dict[str, Any]:
    return {
        "schema": "mjlab_textop.online_observation.v1",
        "frame": int(state.frame),
        "started": bool(state.started),
        "latest_frame": None if state.latest_index is None else int(state.latest_index),
        "lag_frames": int(state.lag_frames),
        "buffer_frames": int(state.buffer_frames),
        "stale_steps": int(state.stale_steps),
        "consecutive_stale_steps": int(state.consecutive_stale_steps),
        "robot_anchor_pos_w": [
            float(item)
            for item in state.robot_anchor_pos_w.detach().cpu().reshape(-1).tolist()
        ],
        "robot_anchor_quat_w": [
            float(item)
            for item in state.robot_anchor_quat_w.detach().cpu().reshape(-1).tolist()
        ],
    }


def make_torso_observation_camera(
    *,
    width: int = 320,
    height: int = 240,
    distance: float = 2.0,
    azimuth: float = 0.0,
    elevation: float = -15.0,
) -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
        width=width,
        height=height,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
    )


def make_http_observation_payload(
    *,
    state: dict[str, Any],
    image: ObservationImage,
) -> dict[str, Any]:
    return {
        "state": state,
        "image": {
            "mime_type": image.mime_type,
            "data": b64encode(image.data).decode("ascii"),
        }
    }


def encode_render_image_jpeg(image: Any) -> bytes:
    buffer = BytesIO()
    iio.imwrite(buffer, image, extension=".jpg", quality=OBSERVATION_JPEG_QUALITY)
    return buffer.getvalue()
