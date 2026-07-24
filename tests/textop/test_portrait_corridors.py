from __future__ import annotations

import mujoco
import pytest

from tasks.portrait_corridors.env_cfg import (
    make_portrait_corridors_g1_env_cfg,
)


def test_portrait_corridors_env_has_a_single_instance_and_spec_fn() -> None:
    cfg = make_portrait_corridors_g1_env_cfg(play=True)

    assert cfg.scene.num_envs == 1
    assert cfg.scene.spec_fn is not None


def test_portrait_corridors_spec_adds_three_textured_portraits_and_walls() -> None:
    cfg = make_portrait_corridors_g1_env_cfg(play=True)
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]

    assert cfg.scene.spec_fn is not None
    cfg.scene.spec_fn(spec)

    assert {body.name for body in spec.bodies if body.name.endswith("_portrait")} == {
        "portrait_corridors_linus_portrait",
        "portrait_corridors_karpathy_portrait",
        "portrait_corridors_bugs_portrait",
    }
    assert {texture.name for texture in spec.textures} == {
        "portrait_corridors_linus_texture",
        "portrait_corridors_karpathy_texture",
        "portrait_corridors_bugs_texture",
    }
    assert len([body for body in spec.bodies if body.name.endswith("_wall")]) == 6
    cameras = {camera.name: camera for camera in spec.cameras}
    assert set(cameras) == {"corridor_left", "corridor_center", "corridor_right"}
    camera_positions = [tuple(camera.pos) for camera in cameras.values()]
    assert [position[0] for position in camera_positions] == pytest.approx(
        [0.6, 0.6, 0.6]
    )
    assert [position[1:] for position in camera_positions] == [
        (2.0, 1.25),
        (0.0, 1.25),
        (-2.0, 1.25),
    ]
    assert [camera.fovy for camera in cameras.values()] == [65.0, 65.0, 65.0]
    portrait_positions = {
        body.name: tuple(float(value) for value in body.pos)
        for body in spec.bodies
        if body.name.endswith("_portrait")
    }
    assert [position[1] for position in portrait_positions.values()] == [2.0, 0.0, -2.0]
    # The end wall's corridor-facing surface is at x=5.9; portraits must be
    # in front of it to remain visible from inside the corridors.
    assert all(position[0] < 5.9 for position in portrait_positions.values())
    model = spec.compile()
    assert model.ntex == 3
    assert model.nmesh == 3
    assert model.ncam == 3
    assert model.mat_texid[:, 1].tolist() == [0, 1, 2]
    assert model.mesh_texcoordnum.tolist() == [4, 4, 4]
