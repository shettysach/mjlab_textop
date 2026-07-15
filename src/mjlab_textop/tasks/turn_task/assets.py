from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from mujoco import MjSpec


def make_turn_task_spec_fn(
    *,
    goal_pos_w: tuple[float, float, float],
    goal_size: float,
    corridor_width: float = 4.0,
    corridor_start_x: float = 0.0,
    corner_x: float = 12.0,
    thickness: float = 0.01,
    goal_rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    wall_height: float = 1.5,
    wall_thickness: float = 0.2,
    wall_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
) -> Callable[["MjSpec"], None]:
    def add_turn_task(spec: MjSpec) -> None:
        body = spec.worldbody.add_body(name="turn_task_goal")
        body.pos = (
            goal_pos_w[0],
            goal_pos_w[1],
            goal_pos_w[2] + thickness * 0.5,
        )
        body.add_geom(
            name="turn_task_goal_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(goal_size * 0.5, goal_size * 0.5, thickness * 0.5),
            rgba=goal_rgba,
            contype=0,
            conaffinity=0,
            mass=0.0,
        )

        half_width = corridor_width * 0.5
        half_wall_height = wall_height * 0.5
        half_wall_thickness = wall_thickness * 0.5
        wall_z = goal_pos_w[2] + half_wall_height
        start_wall_x = corridor_start_x - half_width
        end_wall_y = goal_pos_w[1] - half_width

        _add_wall(
            spec,
            name="turn_task_start_wall",
            pos=(start_wall_x, 0.0, wall_z),
            size=(half_wall_thickness, half_width, half_wall_height),
            rgba=wall_rgba,
        )
        _add_wall(
            spec,
            name="turn_task_outer_forward_wall",
            pos=((start_wall_x + corner_x + half_width) * 0.5, half_width, wall_z),
            size=(
                (corner_x + half_width - start_wall_x) * 0.5,
                half_wall_thickness,
                half_wall_height,
            ),
            rgba=wall_rgba,
        )
        _add_wall(
            spec,
            name="turn_task_inner_forward_wall",
            pos=((start_wall_x + corner_x - half_width) * 0.5, -half_width, wall_z),
            size=(
                (corner_x - half_width - start_wall_x) * 0.5,
                half_wall_thickness,
                half_wall_height,
            ),
            rgba=wall_rgba,
        )
        _add_wall(
            spec,
            name="turn_task_outer_turn_wall",
            pos=(corner_x + half_width, (end_wall_y + half_width) * 0.5, wall_z),
            size=(
                half_wall_thickness,
                (half_width - end_wall_y) * 0.5,
                half_wall_height,
            ),
            rgba=wall_rgba,
        )
        _add_wall(
            spec,
            name="turn_task_inner_turn_wall",
            pos=(corner_x - half_width, (end_wall_y - half_width) * 0.5, wall_z),
            size=(
                half_wall_thickness,
                (-half_width - end_wall_y) * 0.5,
                half_wall_height,
            ),
            rgba=wall_rgba,
        )
        _add_wall(
            spec,
            name="turn_task_end_wall",
            pos=(corner_x, end_wall_y, wall_z),
            size=(half_width, half_wall_thickness, half_wall_height),
            rgba=wall_rgba,
        )

    return add_turn_task


def _add_wall(
    spec: "MjSpec",
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> None:
    body = spec.worldbody.add_body(name=name)
    body.pos = pos
    body.add_geom(
        name=f"{name}_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        rgba=rgba,
        contype=1,
        conaffinity=1,
    )
