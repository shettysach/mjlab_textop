from __future__ import annotations

import json

import pytest

from mjlab_textop.core.feedback.observation import (
    OBSERVATION_JPEG_QUALITY,
    HttpObservationPublisher,
    OnlineObservationCfg,
    encode_render_image_jpeg,
    make_torso_observation_camera,
)
from textop_live_protocol.observation import (
    ObservationImage,
    ObservationMessage,
    observation_to_json,
)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return

    def read(self) -> bytes:
        return b""


def test_online_observation_cfg_defaults_to_torso_camera() -> None:
    cfg = OnlineObservationCfg()

    assert cfg.camera.origin_type == cfg.camera.OriginType.ASSET_BODY
    assert cfg.camera.entity_name == "robot"
    assert cfg.camera.body_name == "torso_link"
    assert cfg.camera.width == 320
    assert cfg.camera.height == 240
    assert cfg.camera.distance == 2.0
    assert cfg.camera.azimuth == 0.0
    assert cfg.camera.elevation == -15.0


def test_make_torso_observation_camera_uses_requested_dimensions() -> None:
    camera = make_torso_observation_camera(
        width=640,
        height=480,
        distance=1.5,
        azimuth=170.0,
        elevation=-5.0,
    )

    assert camera.origin_type == camera.OriginType.ASSET_BODY
    assert camera.entity_name == "robot"
    assert camera.body_name == "torso_link"
    assert camera.width == 640
    assert camera.height == 480
    assert camera.distance == 1.5
    assert camera.azimuth == 170.0
    assert camera.elevation == -5.0


def test_http_observation_publisher_posts_image_only(monkeypatch) -> None:
    posted = {}

    def fake_urlopen(request, timeout):
        posted["url"] = request.full_url
        posted["timeout"] = timeout
        posted["content_type"] = request.headers["Content-type"]
        posted["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(
        "mjlab_textop.core.feedback.observation.urllib.request.urlopen",
        fake_urlopen,
    )
    publisher = HttpObservationPublisher(
        url="http://127.0.0.1:9999/observation",
        timeout_sec=2.0,
    )

    publisher.publish(
        image=ObservationImage(data=b"jpeg bytes", mime_type="image/jpeg"),
    )

    assert posted["url"] == "http://127.0.0.1:9999/observation"
    assert posted["timeout"] == 2.0
    assert posted["content_type"] == "application/json"
    assert posted["payload"] == {
        "image": {
            "mime_type": "image/jpeg",
            "data": "anBlZyBieXRlcw==",
        },
    }


def test_http_observation_publisher_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="URL must be non-empty"):
        HttpObservationPublisher(url="")


def test_make_http_observation_payload_includes_image() -> None:
    assert observation_to_json(
        ObservationMessage(
            image=ObservationImage(data=b"jpeg bytes", mime_type="image/jpeg")
        )
    ) == {
        "image": {
            "mime_type": "image/jpeg",
            "data": "anBlZyBieXRlcw==",
        },
    }


def test_make_http_observation_payload_supports_collision_event() -> None:
    assert observation_to_json(
        ObservationMessage(collision_stop=True, recovery_epoch=7)
    ) == {"collision_stop": True, "recovery_epoch": 7}


def test_encode_render_image_jpeg_uses_high_quality(monkeypatch) -> None:
    calls = {}

    def fake_imwrite(buffer, image, *, extension, quality) -> None:
        del image
        calls["extension"] = extension
        calls["quality"] = quality
        buffer.write(b"jpeg")

    monkeypatch.setattr(
        "mjlab_textop.core.feedback.observation.iio.imwrite",
        fake_imwrite,
    )

    assert encode_render_image_jpeg(object()) == b"jpeg"
    assert calls == {
        "extension": ".jpg",
        "quality": OBSERVATION_JPEG_QUALITY,
    }
