from __future__ import annotations

import pytest

from mjlab_textop.robotmdar.planner import (
    GeneratedBlockInfo,
    ManualPromptPlanner,
    PlannerContext,
    ScheduledPromptPlanner,
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


def test_manual_prompt_planner_uses_current_prompt_without_starting_thread() -> None:
    planner = ManualPromptPlanner("walk forward")

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "walk forward"
    )

    planner.prompt.text = "turn left"

    assert planner.choose_prompt(PlannerContext(frame_index=30, block_count=1)) == (
        "turn left"
    )
    assert planner.should_stop is False
    assert planner.input_active is False
    assert "Enter text prompt" in planner.log_suffix


def test_scheduled_prompt_planner_stops_after_final_phase_when_requested() -> None:
    planner = ScheduledPromptPlanner(
        parse_prompt_schedule("walk forward:50,stand still:30"),
        stop_on_complete=True,
    )

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "walk forward"
    )
    planner.on_block_sent(
        GeneratedBlockInfo(
            prompt="walk forward",
            start_frame=0,
            frames=50,
            block_count=1,
        )
    )

    assert planner.choose_prompt(PlannerContext(frame_index=50, block_count=1)) == (
        "stand still"
    )
    assert planner.should_stop is False

    planner.on_block_sent(
        GeneratedBlockInfo(
            prompt="stand still",
            start_frame=50,
            frames=30,
            block_count=2,
        )
    )

    assert planner.should_stop is True
    assert planner.log_suffix == ""
