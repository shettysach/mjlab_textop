from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScoutView = Literal["agent", "overview", "overhead"]


@dataclass(frozen=True)
class TaskInfo:
    name: str
    objective: str


@dataclass(frozen=True)
class GeometrySummary:
    name: str
    kind: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class BodySummary:
    name: str
    position: tuple[float, float, float]
    geometries: tuple[GeometrySummary, ...]


@dataclass(frozen=True)
class SceneSummary:
    task: str
    objective: str
    robot_position: tuple[float, float, float]
    bodies: tuple[BodySummary, ...]
    available_views: tuple[ScoutView, ...]


@dataclass(frozen=True)
class CapturedView:
    task: str
    view: ScoutView
    width: int
    height: int
    image: bytes
    mime_type: str = "image/jpeg"
