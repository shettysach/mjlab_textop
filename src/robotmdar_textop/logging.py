from __future__ import annotations

import os
import sys
from typing import Literal

from textop_protocol.timing import FPS

_RESET = "\x1b[0m"
_COLORS = {
    "good": "\x1b[92m",
    "warning": "\x1b[93m",
    "bad": "\x1b[91m",
}

LogHealth = Literal["good", "warning", "bad"]

_SOURCE_HEALTH: dict[str, LogHealth] = {
    "vlm": "good",
    "recov": "warning",
    "next": "warning",
    "sync": "warning",
}


def format_stream_status(
    *,
    block_count: int,
    frame_index: int,
    prompt: str,
    source: str,
    generation_ms: float,
    lag_ms: float,
    block_frames: int,
    suffix: str = "",
    color: bool | None = None,
) -> str:
    """Format one producer status line, coloring only timing values."""

    if block_frames <= 0:
        raise ValueError(f"block_frames must be positive, got {block_frames}")
    if color is None:
        color = terminal_colors_enabled()

    generation = _colored_number(
        f"{generation_ms:.1f}",
        _generation_health(generation_ms, block_frames=block_frames),
        enabled=color,
    )
    lag = _colored_number(
        f"{lag_ms:.1f}",
        _lag_health(lag_ms),
        enabled=color,
    )
    formatted_source = _colored_source(source, enabled=color)
    return (
        f"[BLOCK {block_count}] [FRAME {frame_index}] "
        f"prompt={prompt!r} source={formatted_source} "
        f"gen_ms={generation} lag_ms={lag}{suffix}"
    )


def terminal_colors_enabled() -> bool:
    return "NO_COLOR" not in os.environ and sys.stderr.isatty()


def _generation_health(generation_ms: float, *, block_frames: int) -> LogHealth:
    block_budget_ms = block_frames / FPS * 1000.0
    if generation_ms <= block_budget_ms * 0.8:
        return "good"
    if generation_ms <= block_budget_ms:
        return "warning"
    return "bad"


def _lag_health(lag_ms: float) -> LogHealth:
    if lag_ms <= 1.0:
        return "good"
    if lag_ms <= 1000.0 / FPS:
        return "warning"
    return "bad"


def _colored_number(value: str, health: LogHealth, *, enabled: bool) -> str:
    if not enabled:
        return value
    return f"{_COLORS[health]}{value}{_RESET}"


def _colored_source(source: str, *, enabled: bool) -> str:
    health = _SOURCE_HEALTH.get(source)
    if not enabled or health is None:
        return source
    return f"{_COLORS[health]}{source}{_RESET}"
