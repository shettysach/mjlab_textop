from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]

WALL_SOLREF: tuple[float, float] = (0.05, 2.0)
WALL_SOLIMP: tuple[float, float, float, float, float] = (
    0.7,
    0.95,
    0.03,
    0.5,
    2.0,
)


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
        type=MJGEOM_BOX,
        size=size,
        rgba=rgba,
        contype=1,
        conaffinity=1,
        solref=WALL_SOLREF,
        solimp=WALL_SOLIMP,
    )
