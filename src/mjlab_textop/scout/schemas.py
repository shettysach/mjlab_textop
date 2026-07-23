from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

ScoutView: TypeAlias = str


@dataclass(frozen=True)
class TaskInfo:
    name: str
    objective: str
    views: tuple[ScoutView, ...]


@dataclass(frozen=True)
class CapturedView:
    task: str
    view: ScoutView
    width: int
    height: int
    image: bytes
    mime_type: str = "image/jpeg"
