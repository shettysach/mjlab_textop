from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from mujoco import MjSpec


def make_green_square_spec_fn(
    *,
    goal_pos_w: tuple[float, float, float],
    size: float,
    thickness: float = 0.01,
    rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> Callable[["MjSpec"], None]:
    def add_green_square(spec: MjSpec) -> None:
        body = spec.worldbody.add_body(name="green_square_goal")
        body.pos = (goal_pos_w[0], goal_pos_w[1], goal_pos_w[2] + thickness * 0.5)
        body.add_geom(
            name="green_square_goal_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(size * 0.5, size * 0.5, thickness * 0.5),
            rgba=rgba,
            contype=0,
            conaffinity=0,
            mass=0.0,
        )

    return add_green_square
