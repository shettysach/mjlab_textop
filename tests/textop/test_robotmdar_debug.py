from __future__ import annotations

from dataclasses import dataclass

from robotmdar_textop.debug import _query_vlm
from robotmdar_textop.feedback import FeedbackObservation
from robotmdar_textop.planner.vlm import VlmPromptSelection


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


def test_query_vlm_saves_unique_images_when_debug_dir_is_passed(
    tmp_path,
    capsys,
) -> None:
    _query_vlm(
        _Receiver(_observation(image_bytes=b"jpg bytes", image_mime_type="image/jpeg")),
        _Selector(VlmPromptSelection(prompt="wave", reasoning=None, response={})),
        tmp_path,
    )
    _query_vlm(
        _Receiver(
            _observation(image_bytes=b"next jpg bytes", image_mime_type="image/jpeg")
        ),
        _Selector(VlmPromptSelection(prompt="stand", reasoning=None, response={})),
        tmp_path,
    )

    first_image_path = tmp_path / "vlm_observation_000001.jpg"
    second_image_path = tmp_path / "vlm_observation_000002.jpg"
    assert first_image_path.read_bytes() == b"jpg bytes"
    assert second_image_path.read_bytes() == b"next jpg bytes"
    stderr = capsys.readouterr().err
    assert f"vlm_debug_image {first_image_path}" in stderr
    assert f"vlm_debug_image {second_image_path}" in stderr


def _observation(
    *,
    image_bytes: bytes | None = b"jpeg bytes",
    image_mime_type: str | None = "image/jpeg",
) -> FeedbackObservation:
    return FeedbackObservation(
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )
