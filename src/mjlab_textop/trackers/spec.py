from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg

TrackerEnvConfigurator = Callable[[ManagerBasedRlEnvCfg], None]


@dataclass(frozen=True)
class TrackerSpec:
    """Runtime components owned by one low-level tracker backend."""

    name: str
    runner_cls: type[Any]
    configure_env: TrackerEnvConfigurator

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tracker name must be non-empty")
