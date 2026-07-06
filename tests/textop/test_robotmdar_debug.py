from __future__ import annotations

from dataclasses import dataclass

from mjlab_textop.robotmdar.debug import _query_vlm
from mjlab_textop.robotmdar.feedback import FeedbackObservation
from mjlab_textop.robotmdar.planner.vlm import VlmPromptSelection


class _Receiver:
    def __init__(self, observation: FeedbackObservation) -> None:
        self.observation = observation

    def latest(self) -> FeedbackObservation:
        return self.observation


@dataclass
class _Selector:
    selection: VlmPromptSelection

    def choose_prompt_with_debug(
        self,
        *,
        observation: FeedbackObservation | None,
    ) -> VlmPromptSelection:
        del observation
        return self.selection


def test_query_vlm_does_not_save_image_without_debug_dir(tmp_path, capsys) -> None:
    _query_vlm(
        _Receiver(_observation()),
        _Selector(VlmPromptSelection(prompt="wave", reasoning=None, response={})),
    )

    assert list(tmp_path.iterdir()) == []
    assert "vlm_debug_image" not in capsys.readouterr().err


def test_query_vlm_saves_image_when_debug_dir_is_passed(tmp_path, capsys) -> None:
    _query_vlm(
        _Receiver(_observation(image_bytes=b"jpg bytes", image_mime_type="image/jpeg")),
        _Selector(VlmPromptSelection(prompt="wave", reasoning=None, response={})),
        tmp_path,
    )

    image_path = tmp_path / "frame_42.jpg"
    assert image_path.read_bytes() == b"jpg bytes"
    assert f"vlm_debug_image {image_path}" in capsys.readouterr().err


def _observation(
    *,
    image_bytes: bytes | None = b"jpeg bytes",
    image_mime_type: str | None = "image/jpeg",
) -> FeedbackObservation:
    return FeedbackObservation(
        frame=42,
        started=True,
        latest_frame=40,
        lag_frames=2,
        buffer_frames=5,
        stale_steps=0,
        consecutive_stale_steps=0,
        robot_anchor_pos_w=(0.0, 0.0, 0.0),
        robot_anchor_quat_w=(1.0, 0.0, 0.0, 0.0),
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )
