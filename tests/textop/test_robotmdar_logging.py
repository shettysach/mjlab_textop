from __future__ import annotations

import pytest

from robotmdar_textop.logging import format_stream_status
from textop_protocol.timing import FPS

GREEN = "\x1b[92m"
YELLOW = "\x1b[93m"
RED = "\x1b[91m"
RESET = "\x1b[0m"


@pytest.mark.parametrize(
    ("budget_fraction", "color_code"),
    [(0.7, GREEN), (0.9, YELLOW), (1.1, RED)],
)
def test_generation_color_uses_runtime_block_size(
    budget_fraction: float,
    color_code: str,
) -> None:
    block_frames = 7
    block_budget_ms = block_frames / FPS * 1000.0

    message = format_stream_status(
        block_count=4,
        frame_index=28,
        prompt="walk",
        source="vlm",
        generation_ms=block_budget_ms * budget_fraction,
        lag_ms=0.0,
        block_frames=block_frames,
        color=True,
    )

    expected_value = f"{block_budget_ms * budget_fraction:.1f}"
    assert f"gen_ms={color_code}{expected_value}{RESET}" in message


@pytest.mark.parametrize(
    ("lag_ms", "color_code"),
    [(0.0, GREEN), (10.0, YELLOW), (21.0, RED)],
)
def test_lag_color_uses_control_frame_duration(
    lag_ms: float,
    color_code: str,
) -> None:
    message = format_stream_status(
        block_count=4,
        frame_index=28,
        prompt="walk",
        source="vlm",
        generation_ms=0.0,
        lag_ms=lag_ms,
        block_frames=7,
        color=True,
    )

    assert f"lag_ms={color_code}{lag_ms:.1f}{RESET}" in message


def test_stream_status_leaves_states_and_identifiers_uncolored() -> None:
    message = format_stream_status(
        block_count=4,
        frame_index=28,
        prompt="stand",
        source="recov",
        generation_ms=0.0,
        lag_ms=0.0,
        block_frames=7,
        suffix=" vlm=idle last_q=(block: 3, img_revision: 2)",
        color=True,
    )

    assert message.startswith(
        f"[BLOCK 4] [FRAME 28] prompt='stand' source={YELLOW}recov{RESET}"
    )
    assert "vlm=idle last_q=(block: 3, img_revision: 2)" in message
    assert message.count(GREEN) == 2


def test_vlm_source_is_bright_green() -> None:
    message = format_stream_status(
        block_count=4,
        frame_index=28,
        prompt="walk",
        source="vlm",
        generation_ms=0.0,
        lag_ms=0.0,
        block_frames=7,
        suffix=" vlm=idle",
        color=True,
    )

    assert f"source={GREEN}vlm{RESET}" in message
    assert f"vlm={GREEN}idle{RESET}" not in message


def test_next_source_is_bright_yellow() -> None:
    message = format_stream_status(
        block_count=4,
        frame_index=28,
        prompt="stand",
        source="next",
        generation_ms=0.0,
        lag_ms=0.0,
        block_frames=7,
        color=True,
    )

    assert f"source={YELLOW}next{RESET}" in message
