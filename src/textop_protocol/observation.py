from __future__ import annotations

import json
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservationImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class ObservationMessage:
    image: ObservationImage | None = None
    collision_stop: bool | None = None
    recovery_epoch: int | None = None


def observation_to_json(message: ObservationMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if message.collision_stop is not None:
        payload["collision_stop"] = message.collision_stop
    if message.recovery_epoch is not None:
        payload["recovery_epoch"] = message.recovery_epoch
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

    encoded_image = payload.get("image")
    image = None
    if encoded_image is not None:
        image = ObservationImage(
            data=b64decode(encoded_image["data"], validate=True),
            mime_type=str(encoded_image["mime_type"]),
        )

    return ObservationMessage(
        image=image,
        collision_stop=payload.get("collision_stop"),
        recovery_epoch=payload.get("recovery_epoch"),
    )
