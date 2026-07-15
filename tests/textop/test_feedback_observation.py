from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from mjlab_textop.core.feedback.observation import (
    OBSERVATION_JPEG_QUALITY,
    HttpObservationPublisher,
    HttpObservationPublisherCfg,
    ObservationImage,
    OnlineObservationState,
    OnlineTextOpObservationCfg,
    encode_render_image_jpeg,
    make_http_observation_payload,
    make_online_textop_observation,
    make_torso_observation_camera,
)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return

    def read(self) -> bytes:
        return b""


def test_make_online_textop_observation_payload() -> None:
    payload = make_online_textop_observation(
        OnlineObservationState(
            frame=10,
            started=True,
            latest_index=18,
            lag_frames=8,
            buffer_frames=32,
            stale_steps=0,
            consecutive_stale_steps=0,
            robot_anchor_pos_w=torch.tensor([1, 2, 3]),
            robot_anchor_quat_w=torch.tensor([1, 0, 0, 0]),
        )
    )

    assert payload == {
        "schema": "mjlab_textop.online_observation.v1",
        "frame": 10,
        "started": True,
        "latest_frame": 18,
        "lag_frames": 8,
        "buffer_frames": 32,
        "stale_steps": 0,
        "consecutive_stale_steps": 0,
        "robot_anchor_pos_w": [1.0, 2.0, 3.0],
        "robot_anchor_quat_w": [1.0, 0.0, 0.0, 0.0],
    }


def test_online_observation_cfg_defaults_to_torso_camera() -> None:
    cfg = OnlineTextOpObservationCfg()

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


def test_http_observation_publisher_posts_state_and_image(monkeypatch) -> None:
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
        HttpObservationPublisherCfg(
            url="http://127.0.0.1:9999/observation",
            timeout_sec=2.0,
        )
    )

    publisher.publish(
        {"frame": 1},
        image=ObservationImage(data=b"jpeg bytes", mime_type="image/jpeg"),
    )

    assert posted["url"] == "http://127.0.0.1:9999/observation"
    assert posted["timeout"] == 2.0
    assert posted["content_type"] == "application/json"
    assert posted["payload"] == {
        "state": {"frame": 1},
        "image": {
            "mime_type": "image/jpeg",
            "data": "anBlZyBieXRlcw==",
        },
    }


def test_http_observation_publisher_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="URL must be non-empty"):
        HttpObservationPublisher(HttpObservationPublisherCfg(url=""))


def test_http_observation_publisher_cfg_is_deepcopyable() -> None:
    cfg = HttpObservationPublisherCfg(url="http://127.0.0.1:8766/observation")

    copied = deepcopy(cfg)

    assert copied == cfg


def test_make_http_observation_payload_includes_image() -> None:
    assert make_http_observation_payload(
        state={"frame": 1},
        image=ObservationImage(data=b"jpeg bytes", mime_type="image/jpeg"),
    ) == {
        "state": {"frame": 1},
        "image": {
            "mime_type": "image/jpeg",
            "data": "anBlZyBieXRlcw==",
        },
    }


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
