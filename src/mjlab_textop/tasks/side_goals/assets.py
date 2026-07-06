from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]


def make_side_goals_spec_fn(
    *,
    blue_goal_pos_w: tuple[float, float, float],
    green_goal_pos_w: tuple[float, float, float],
    goal_size: float,
    arena_size: float,
    thickness: float = 0.01,
    blue_rgba: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    green_rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    wall_height: float = 1.5,
    wall_thickness: float = 0.2,
    wall_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    wall_friction: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wall_solref: tuple[float, float] = (0.05, 1.0),
    wall_solimp: tuple[float, float, float, float, float] = (
        0.8,
        0.95,
        0.01,
        0.5,
        2.0,
    ),
    arena_center_xy: tuple[float, float] = (0.0, 0.0),
) -> Callable[["MjSpec"], None]:
    def add_side_goals(spec: MjSpec) -> None:
        _add_goal(
            spec,
            name="side_goals_blue_goal",
            pos_w=blue_goal_pos_w,
            size=goal_size,
            thickness=thickness,
            rgba=blue_rgba,
        )
        _add_goal(
            spec,
            name="side_goals_green_goal",
            pos_w=green_goal_pos_w,
            size=goal_size,
            thickness=thickness,
            rgba=green_rgba,
        )

        half_size = arena_size * 0.5
        half_wall_height = wall_height * 0.5
        half_wall_thickness = wall_thickness * 0.5
        wall_z = green_goal_pos_w[2] + half_wall_height
        center_x, center_y = arena_center_xy
        _add_wall(
            spec,
            name="side_goals_left_wall",
            pos=(center_x, center_y + half_size, wall_z),
            size=(half_size, half_wall_thickness, half_wall_height),
            rgba=wall_rgba,
            friction=wall_friction,
            solref=wall_solref,
            solimp=wall_solimp,
        )
        _add_wall(
            spec,
            name="side_goals_right_wall",
            pos=(center_x, center_y - half_size, wall_z),
            size=(half_size, half_wall_thickness, half_wall_height),
            rgba=wall_rgba,
            friction=wall_friction,
            solref=wall_solref,
            solimp=wall_solimp,
        )
        _add_wall(
            spec,
            name="side_goals_front_wall",
            pos=(center_x + half_size, center_y, wall_z),
            size=(half_wall_thickness, half_size, half_wall_height),
            rgba=wall_rgba,
            friction=wall_friction,
            solref=wall_solref,
            solimp=wall_solimp,
        )
        _add_wall(
            spec,
            name="side_goals_back_wall",
            pos=(center_x - half_size, center_y, wall_z),
            size=(half_wall_thickness, half_size, half_wall_height),
            rgba=wall_rgba,
            friction=wall_friction,
            solref=wall_solref,
            solimp=wall_solimp,
        )

    return add_side_goals


def _add_goal(
    spec: "MjSpec",
    *,
    name: str,
    pos_w: tuple[float, float, float],
    size: float,
    thickness: float,
    rgba: tuple[float, float, float, float],
) -> None:
    body = spec.worldbody.add_body(name=name)
    body.pos = (pos_w[0], pos_w[1], pos_w[2] + thickness * 0.5)
    body.add_geom(
        name=f"{name}_visual",
        type=MJGEOM_BOX,
        size=(size * 0.5, size * 0.5, thickness * 0.5),
        rgba=rgba,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def _add_wall(
    spec: "MjSpec",
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    friction: tuple[float, float, float],
    solref: tuple[float, float],
    solimp: tuple[float, float, float, float, float],
) -> None:
    body = spec.worldbody.add_body(name=name)
    body.pos = pos
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=size,
        rgba=rgba,
        contype=1,
        conaffinity=1,
        condim=1,
        friction=friction,
        solref=solref,
        solimp=solimp,
    )
