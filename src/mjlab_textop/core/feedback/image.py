from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from threading import Lock
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class EncodedObservationImage:
    mime_type: str
    data_base64: str
    frame: int | None = None
    width: int | None = None
    height: int | None = None


class ObservationImageStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: EncodedObservationImage | None = None

    def set_latest(self, image: EncodedObservationImage) -> None:
        with self._lock:
            self._latest = image

    def latest(self) -> EncodedObservationImage | None:
        with self._lock:
            return self._latest


_IMAGE_STORES: dict[str, ObservationImageStore] = {}


def register_observation_image_store(
    key: str,
    store: ObservationImageStore,
) -> None:
    _IMAGE_STORES[key] = store


def unregister_observation_image_store(key: str) -> None:
    _IMAGE_STORES.pop(key, None)


def get_observation_image_store(key: str) -> ObservationImageStore | None:
    return _IMAGE_STORES.get(key)


def encode_rgb_frame_as_jpeg_base64(
    frame: Any,
    *,
    width: int,
    height: int,
    quality: int,
    frame_index: int | None = None,
) -> EncodedObservationImage:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    if not 1 <= quality <= 95:
        raise ValueError(f"quality must be in [1, 95], got {quality}")

    image = Image.fromarray(frame)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return EncodedObservationImage(
        mime_type="image/jpeg",
        data_base64=base64.b64encode(output.getvalue()).decode("ascii"),
        frame=frame_index,
        width=width,
        height=height,
    )
