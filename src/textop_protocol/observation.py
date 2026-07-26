from __future__ import annotations

import json
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Any

OBSERVATION_PROTOCOL_VERSION = 3


@dataclass(frozen=True)
class ObservationImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class ObservationMessage:
    protocol_version: int = OBSERVATION_PROTOCOL_VERSION
    image: ObservationImage | None = None
    collision_stop: bool | None = None
    recovery_epoch: int | None = None
    observation_request_id: int | None = None
    source_frame: int | None = None


def observation_to_json(message: ObservationMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"protocol_version": message.protocol_version}
    if message.collision_stop is not None:
        payload["collision_stop"] = message.collision_stop
    if message.recovery_epoch is not None:
        payload["recovery_epoch"] = message.recovery_epoch
    if message.observation_request_id is not None:
        payload["observation_request_id"] = message.observation_request_id
    if message.source_frame is not None:
        payload["source_frame"] = message.source_frame
    if message.image is not None:
        payload["image"] = {
            "mime_type": message.image.mime_type,
            "data": b64encode(message.image.data).decode("ascii"),
        }
    return payload


def parse_observation_json(
    payload: bytes | str | dict[str, Any],
) -> ObservationMessage:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)

    protocol_version = payload.get("protocol_version")
    if protocol_version != OBSERVATION_PROTOCOL_VERSION:
        raise ValueError(
            "Unsupported observation protocol version: "
            f"{protocol_version!r}; expected {OBSERVATION_PROTOCOL_VERSION}"
        )

    encoded_image = payload.get("image")
    image = None
    if encoded_image is not None:
        image = ObservationImage(
            data=b64decode(encoded_image["data"], validate=True),
            mime_type=str(encoded_image["mime_type"]),
        )

    observation_request_id = payload.get("observation_request_id")
    source_frame = payload.get("source_frame")
    for name, value in (
        ("observation_request_id", observation_request_id),
        ("source_frame", source_frame),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or null")
    if observation_request_id is not None and (image is None or source_frame is None):
        raise ValueError("Requested observations require an image and source_frame")

    return ObservationMessage(
        protocol_version=protocol_version,
        image=image,
        collision_stop=payload.get("collision_stop"),
        recovery_epoch=payload.get("recovery_epoch"),
        observation_request_id=observation_request_id,
        source_frame=source_frame,
    )
