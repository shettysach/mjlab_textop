from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

from mjlab_textop.tasks.wall_contact import _add_wall

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJTEXTURE_2D = mujoco.mjtTexture.mjTEXTURE_2D  # ty: ignore[unresolved-attribute]

_ASSETS_DIR = Path(__file__).resolve().parents[4] / "assets"


def make_portrait_corridors_spec_fn(
    *,
    corridor_length: float = 14.0,
    corridor_width: float = 3.0,
    room_size: float = 8.0,
    wall_height: float = 3.2,
    wall_thickness: float = 0.2,
    wall_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
) -> Callable[["MjSpec"], None]:
    """Create an enclosed hub with north, east, and west portrait corridors."""

    def add_portrait_corridors(spec: MjSpec) -> None:
        half_room = room_size * 0.5
        half_corridor = corridor_width * 0.5
        half_wall_height = wall_height * 0.5
        half_wall_thickness = wall_thickness * 0.5
        wall_z = half_wall_height
        corridor_end = half_room + corridor_length

        # The hub's four sides are bounded; the three corridor-facing sides use
        # two wall segments to leave a doorway into their respective corridor.
        _add_wall(
            spec,
            name="portrait_corridors_south_wall",
            pos=(0.0, -half_room, wall_z),
            size=(half_room, half_wall_thickness, half_wall_height),
            rgba=wall_rgba,
        )
        _add_hub_wall_segments(
            spec,
            side="north",
            room_half_size=half_room,
            corridor_half_width=half_corridor,
            wall_z=wall_z,
            half_wall_thickness=half_wall_thickness,
            half_wall_height=half_wall_height,
            rgba=wall_rgba,
        )
        _add_hub_wall_segments(
            spec,
            side="east",
            room_half_size=half_room,
            corridor_half_width=half_corridor,
            wall_z=wall_z,
            half_wall_thickness=half_wall_thickness,
            half_wall_height=half_wall_height,
            rgba=wall_rgba,
        )
        _add_hub_wall_segments(
            spec,
            side="west",
            room_half_size=half_room,
            corridor_half_width=half_corridor,
            wall_z=wall_z,
            half_wall_thickness=half_wall_thickness,
            half_wall_height=half_wall_height,
            rgba=wall_rgba,
        )

        _add_north_corridor(
            spec,
            room_half_size=half_room,
            corridor_end=corridor_end,
            half_corridor=half_corridor,
            half_wall_thickness=half_wall_thickness,
            half_wall_height=half_wall_height,
            wall_z=wall_z,
            wall_rgba=wall_rgba,
        )
        _add_horizontal_corridor(
            spec,
            direction="east",
            room_half_size=half_room,
            corridor_end=corridor_end,
            half_corridor=half_corridor,
            half_wall_thickness=half_wall_thickness,
            half_wall_height=half_wall_height,
            wall_z=wall_z,
            wall_rgba=wall_rgba,
        )
        _add_horizontal_corridor(
            spec,
            direction="west",
            room_half_size=half_room,
            corridor_end=corridor_end,
            half_corridor=half_corridor,
            half_wall_thickness=half_wall_thickness,
            half_wall_height=half_wall_height,
            wall_z=wall_z,
            wall_rgba=wall_rgba,
        )

        _add_portrait(spec, name="linus", pos=(0.0, corridor_end - 0.03, 1.6), axis="y")
        _add_portrait(spec, name="jensen", pos=(corridor_end - 0.03, 0.0, 1.6), axis="x")
        _add_portrait(spec, name="bugs", pos=(-corridor_end + 0.03, 0.0, 1.6), axis="x")

    return add_portrait_corridors


def _add_hub_wall_segments(
    spec: "MjSpec", *, side: str, room_half_size: float, corridor_half_width: float,
    wall_z: float, half_wall_thickness: float, half_wall_height: float,
    rgba: tuple[float, float, float, float],
) -> None:
    segment_half_length = (room_half_size - corridor_half_width) * 0.5
    segment_offset = (room_half_size + corridor_half_width) * 0.5
    for suffix, sign in (("negative", -1.0), ("positive", 1.0)):
        if side == "north":
            pos = (sign * segment_offset, room_half_size, wall_z)
            size = (segment_half_length, half_wall_thickness, half_wall_height)
        else:
            x = room_half_size if side == "east" else -room_half_size
            pos = (x, sign * segment_offset, wall_z)
            size = (half_wall_thickness, segment_half_length, half_wall_height)
        _add_wall(
            spec,
            name=f"portrait_corridors_{side}_{suffix}_wall",
            pos=pos,
            size=size,
            rgba=rgba,
        )


def _add_north_corridor(
    spec: "MjSpec", *, room_half_size: float, corridor_end: float,
    half_corridor: float,
    half_wall_thickness: float, half_wall_height: float, wall_z: float,
    wall_rgba: tuple[float, float, float, float],
) -> None:
    corridor_half_length = (corridor_end - room_half_size) * 0.5
    corridor_center = (corridor_end + room_half_size) * 0.5
    for side, x in (("west", -half_corridor), ("east", half_corridor)):
        _add_wall(spec, name=f"portrait_corridors_north_{side}_wall", pos=(x, corridor_center, wall_z), size=(half_wall_thickness, corridor_half_length, half_wall_height), rgba=wall_rgba)
    _add_wall(spec, name="portrait_corridors_north_end_wall", pos=(0.0, corridor_end, wall_z), size=(half_corridor, half_wall_thickness, half_wall_height), rgba=wall_rgba)


def _add_horizontal_corridor(
    spec: "MjSpec", *, direction: str, room_half_size: float,
    corridor_end: float, half_corridor: float,
    half_wall_thickness: float, half_wall_height: float, wall_z: float,
    wall_rgba: tuple[float, float, float, float],
) -> None:
    sign = 1.0 if direction == "east" else -1.0
    corridor_half_length = (corridor_end - room_half_size) * 0.5
    corridor_center = sign * (corridor_end + room_half_size) * 0.5
    for side, y in (("south", -half_corridor), ("north", half_corridor)):
        _add_wall(spec, name=f"portrait_corridors_{direction}_{side}_wall", pos=(corridor_center, y, wall_z), size=(corridor_half_length, half_wall_thickness, half_wall_height), rgba=wall_rgba)
    _add_wall(spec, name=f"portrait_corridors_{direction}_end_wall", pos=(sign * corridor_end, 0.0, wall_z), size=(half_wall_thickness, half_corridor, half_wall_height), rgba=wall_rgba)


def _add_portrait(spec: "MjSpec", *, name: str, pos: tuple[float, float, float], axis: str) -> None:
    texture = spec.add_texture(name=f"portrait_corridors_{name}_texture", type=MJTEXTURE_2D, file=str(_ASSETS_DIR / f"{name}.png"))
    material = spec.add_material(name=f"portrait_corridors_{name}_material")
    material.textures[0] = texture.name
    material.texrepeat = (1.0, 1.0)
    material.emission = 0.15
    width, height = (1.25, 1.55)
    size = (0.03, width, height) if axis == "x" else (width, 0.03, height)
    body = spec.worldbody.add_body(name=f"portrait_corridors_{name}_portrait")
    body.pos = pos
    body.add_geom(name=f"portrait_corridors_{name}_portrait_visual", type=MJGEOM_BOX, size=size, material=material.name, contype=0, conaffinity=0, mass=0.0)
