from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]


def make_blocked_straight_spec_fn(
    *,
    goal_pos_w: tuple[float, float, float],
    size: float,
    obstacle_pos_xy: tuple[float, float],
    obstacle_size_xy: tuple[float, float],
    thickness: float = 0.01,
    rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    wall_height: float = 1.5,
    wall_thickness: float = 0.2,
    wall_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    obstacle_rgba: tuple[float, float, float, float] = (0.8, 0.1, 0.1, 1.0),
    collision_friction: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision_solref: tuple[float, float] = (0.05, 1.0),
    collision_solimp: tuple[float, float, float, float, float] = (
        0.8,
        0.95,
        0.01,
        0.5,
        2.0,
    ),
    corridor_start_x: float = 0.0,
    corridor_back_extension: float = 2.0,
) -> Callable[["MjSpec"], None]:
    def add_blocked_straight(spec: MjSpec) -> None:
        body = spec.worldbody.add_body(name="blocked_straight_goal")
        body.pos = (goal_pos_w[0], goal_pos_w[1], goal_pos_w[2] + thickness * 0.5)
        body.add_geom(
            name="blocked_straight_goal_visual",
            type=MJGEOM_BOX,
            size=(size * 0.5, size * 0.5, thickness * 0.5),
            rgba=rgba,
            contype=0,
            conaffinity=0,
            mass=0.0,
        )

        half_size = size * 0.5
        half_wall_height = wall_height * 0.5
        half_wall_thickness = wall_thickness * 0.5
        corridor_back_x = corridor_start_x - corridor_back_extension
        corridor_end_x = goal_pos_w[0] + half_size
        corridor_center_x = (corridor_back_x + corridor_end_x) * 0.5
        corridor_half_length = (corridor_end_x - corridor_back_x) * 0.5
        wall_z = goal_pos_w[2] + half_wall_height
        _add_box(
            spec,
            name="blocked_straight_left_wall",
            pos=(corridor_center_x, goal_pos_w[1] + half_size, wall_z),
            size=(corridor_half_length, half_wall_thickness, half_wall_height),
            rgba=wall_rgba,
            friction=collision_friction,
            solref=collision_solref,
            solimp=collision_solimp,
        )
        _add_box(
            spec,
            name="blocked_straight_right_wall",
            pos=(corridor_center_x, goal_pos_w[1] - half_size, wall_z),
            size=(corridor_half_length, half_wall_thickness, half_wall_height),
            rgba=wall_rgba,
            friction=collision_friction,
            solref=collision_solref,
            solimp=collision_solimp,
        )
        _add_box(
            spec,
            name="blocked_straight_end_wall",
            pos=(corridor_end_x, goal_pos_w[1], wall_z),
            size=(half_wall_thickness, half_size, half_wall_height),
            rgba=wall_rgba,
            friction=collision_friction,
            solref=collision_solref,
            solimp=collision_solimp,
        )
        _add_box(
            spec,
            name="blocked_straight_back_wall",
            pos=(corridor_back_x, goal_pos_w[1], wall_z),
            size=(half_wall_thickness, half_size, half_wall_height),
            rgba=wall_rgba,
            friction=collision_friction,
            solref=collision_solref,
            solimp=collision_solimp,
        )
        _add_box(
            spec,
            name="blocked_straight_center_block",
            pos=(obstacle_pos_xy[0], obstacle_pos_xy[1], wall_z),
            size=(obstacle_size_xy[0] * 0.5, obstacle_size_xy[1] * 0.5, half_wall_height),
            rgba=obstacle_rgba,
            friction=collision_friction,
            solref=collision_solref,
            solimp=collision_solimp,
        )

    return add_blocked_straight


def _add_box(
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
