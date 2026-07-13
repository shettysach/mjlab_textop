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
    start_x: float = -4.0,
    corridor_length: float = 22.0,
    corridor_width: float = 3.0,
    wall_height: float = 3.2,
    wall_thickness: float = 0.2,
    wall_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
) -> Callable[["MjSpec"], None]:
    """Create three parallel, enclosed corridors with one portrait each."""

    def add_portrait_corridors(spec: MjSpec) -> None:
        end_x = start_x + corridor_length
        half_total_width = corridor_width * 1.5
        half_wall_height = wall_height * 0.5
        half_wall_thickness = wall_thickness * 0.5
        center_x = (start_x + end_x) * 0.5
        wall_z = half_wall_height

        # Four perimeter walls enclose the three lanes.
        _add_wall(
            spec,
            name="portrait_corridors_back_wall",
            pos=(start_x, 0.0, wall_z),
            size=(half_wall_thickness, half_total_width, half_wall_height),
            rgba=wall_rgba,
        )
        _add_wall(
            spec,
            name="portrait_corridors_end_wall",
            pos=(end_x, 0.0, wall_z),
            size=(half_wall_thickness, half_total_width, half_wall_height),
            rgba=wall_rgba,
        )
        for side, y in (("north", half_total_width), ("south", -half_total_width)):
            _add_wall(
                spec,
                name=f"portrait_corridors_{side}_wall",
                pos=(center_x, y, wall_z),
                size=(corridor_length * 0.5, half_wall_thickness, half_wall_height),
                rgba=wall_rgba,
            )

        # Two parallel dividers turn the enclosure into three corridors.
        for index, y in enumerate((-corridor_width * 0.5, corridor_width * 0.5), 1):
            _add_wall(
                spec,
                name=f"portrait_corridors_divider_{index}_wall",
                pos=(center_x, y, wall_z),
                size=(corridor_length * 0.5, half_wall_thickness, half_wall_height),
                rgba=wall_rgba,
            )

        portrait_x = end_x - half_wall_thickness - 0.03 - 0.01
        _add_portrait(spec, name="linus", pos=(portrait_x, corridor_width, 1.6))
        _add_portrait(spec, name="jensen", pos=(portrait_x, 0.0, 1.6))
        _add_portrait(spec, name="bugs", pos=(portrait_x, -corridor_width, 1.6))

    return add_portrait_corridors


def _add_portrait(
    spec: "MjSpec", *, name: str, pos: tuple[float, float, float]
) -> None:
    texture = spec.add_texture(
        name=f"portrait_corridors_{name}_texture",
        type=MJTEXTURE_2D,
        file=str(_ASSETS_DIR / f"{name}.png"),
    )
    material = spec.add_material(name=f"portrait_corridors_{name}_material")
    material.textures[0] = texture.name
    material.texrepeat = (1.0, 1.0)
    material.emission = 0.15

    body = spec.worldbody.add_body(name=f"portrait_corridors_{name}_portrait")
    body.pos = pos
    body.add_geom(
        name=f"portrait_corridors_{name}_portrait_visual",
        type=MJGEOM_BOX,
        size=(0.03, 1.25, 1.55),
        material=material.name,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
