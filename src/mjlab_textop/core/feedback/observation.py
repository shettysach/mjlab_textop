from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Protocol

import imageio.v3 as iio
from mjlab.viewer import ViewerConfig

from textop_live_protocol.observation import (
    ObservationImage,
    ObservationMessage,
    observation_to_json,
)

OBSERVATION_JPEG_QUALITY = 95


@dataclass(frozen=True)
class OnlineObservationState:
    frame: int
    started: bool


class ObservationPublisher(Protocol):
    def publish(
        self,
        *,
        image: ObservationImage | None,
        collision_stop: bool | None = None,
        recovery_epoch: int | None = None,
    ) -> None:
        """Publish one MJLab observation payload."""


@dataclass(frozen=True, kw_only=True)
class OnlineObservationCfg:
    publisher: ObservationPublisher | None = None
    publish_interval: int = 1
    camera: ViewerConfig = field(
        default_factory=lambda: make_torso_observation_camera()
    )

    def __post_init__(self) -> None:
        if self.publish_interval <= 0:
            raise ValueError(
                f"publish_interval must be positive, got {self.publish_interval}"
            )


class HttpObservationPublisher:
    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:8766/observation",
        timeout_sec: float = 1.0,
    ) -> None:
        if not url.strip():
            raise ValueError("URL must be non-empty")
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        self.url = url
        self.timeout_sec = timeout_sec

    def publish(
        self,
        *,
        image: ObservationImage | None,
        collision_stop: bool | None = None,
        recovery_epoch: int | None = None,
    ) -> None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(
                observation_to_json(
                    ObservationMessage(
                        image=image,
                        collision_stop=collision_stop,
                        recovery_epoch=recovery_epoch,
                    )
                ),
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            response.read()


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


def encode_render_image_jpeg(image: Any) -> bytes:
    buffer = BytesIO()
    iio.imwrite(buffer, image, extension=".jpg", quality=OBSERVATION_JPEG_QUALITY)
    return buffer.getvalue()
