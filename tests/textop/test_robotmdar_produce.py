from __future__ import annotations

import pytest

from mjlab_textop.robotmdar.produce import (
    ScheduledPromptSource,
    parse_prompt_schedule,
)


def test_parse_prompt_schedule_repeats_phases() -> None:
    phases = parse_prompt_schedule(
        "walk forward:150, stand still:60, turn left:90",
        repeat=2,
    )

    assert [(phase.text, phase.frames) for phase in phases] == [
        ("walk forward", 150),
        ("stand still", 60),
        ("turn left", 90),
        ("walk forward", 150),
        ("stand still", 60),
        ("turn left", 90),
    ]


def test_parse_prompt_schedule_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="prompt:frames"):
        parse_prompt_schedule("walk forward", repeat=1)

    with pytest.raises(ValueError, match="positive"):
        parse_prompt_schedule("walk forward:0", repeat=1)

    with pytest.raises(ValueError, match="Invalid frame count"):
        parse_prompt_schedule("walk forward:soon", repeat=1)


def test_scheduled_prompt_source_advances_at_block_boundaries() -> None:
    source = ScheduledPromptSource(
        parse_prompt_schedule("walk forward:50,stand still:30,turn left:40")
    )

    assert source.text == "walk forward"
    assert source.advance(30) is False
    assert source.text == "walk forward"
    assert source.advance(20) is False
    assert source.text == "stand still"
    assert source.advance(30) is False
    assert source.text == "turn left"
    assert source.advance(40) is True
    assert source.text == "turn left"
