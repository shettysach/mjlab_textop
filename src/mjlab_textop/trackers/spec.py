from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg

TrackerEnvConfigurator = Callable[[ManagerBasedRlEnvCfg], None]


@dataclass(frozen=True)
class ReferenceWindowSpec:
    """Frames sampled from the 50 Hz motion stream for one policy step."""

    frame_offsets: tuple[int, ...]
    align_heading: bool = False

    def __post_init__(self) -> None:
        if not self.frame_offsets:
            raise ValueError("Reference window must contain at least one frame")
        if self.frame_offsets[0] != 0:
            raise ValueError("Reference window must start at the current frame")
        if any(offset < 0 for offset in self.frame_offsets):
            raise ValueError("Reference frame offsets must be non-negative")
        if tuple(sorted(set(self.frame_offsets))) != self.frame_offsets:
            raise ValueError(
                "Reference frame offsets must be strictly increasing and unique"
            )

    @property
    def sample_count(self) -> int:
        return len(self.frame_offsets)

    @property
    def required_span(self) -> int:
        return self.frame_offsets[-1] + 1


DEFAULT_REFERENCE_WINDOW = ReferenceWindowSpec(
    frame_offsets=(0, 1, 2, 3, 4),
)


@dataclass(frozen=True)
class TrackerSpec:
    """Runtime components owned by one low-level tracker backend."""

    name: str
    runner_cls: type[Any]
    configure_env: TrackerEnvConfigurator
    reference_window: ReferenceWindowSpec

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tracker name must be non-empty")
