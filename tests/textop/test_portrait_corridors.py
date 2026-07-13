from __future__ import annotations

import mujoco

from mjlab_textop.tasks.portrait_corridors.env_cfg import (
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
        "portrait_corridors_jensen_portrait",
        "portrait_corridors_bugs_portrait",
    }
    assert {texture.name for texture in spec.textures} == {
        "portrait_corridors_linus_texture",
        "portrait_corridors_jensen_texture",
        "portrait_corridors_bugs_texture",
    }
    assert len([body for body in spec.bodies if body.name.endswith("_wall")]) == 16
    model = spec.compile()
    assert model.ntex == 3
