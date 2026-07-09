from __future__ import annotations

from types import SimpleNamespace

import mujoco
import torch
from mjlab.tasks.registry import load_env_cfg, load_runner_cls

from mjlab_textop.core.mdp.observations import future_joint_window_textop_order
from mjlab_textop.core.onnx_policy import OnnxPolicyRunner
from mjlab_textop.tasks.straight import mdp
from mjlab_textop.tasks.straight.env_cfg import (
    STRAIGHT_TASK_CFG,
    make_straight_g1_env_cfg,
)
from mjlab_textop.tasks.straight.registration import (
    register_straight_task,
)


def test_straight_env_cfg_has_fixed_goal_eval_terms() -> None:
    cfg = make_straight_g1_env_cfg(play=True)

    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 20.0
    assert cfg.rewards == {}
    assert cfg.scene.spec_fn is not None
    assert "straight_success" in cfg.terminations
    assert "straight_goal_distance" in cfg.metrics
    assert (
        cfg.metrics["straight_stop_trigger"].params["stop_trigger_radius"]
        == STRAIGHT_TASK_CFG.stop_trigger_radius
    )


def test_straight_play_task_uses_onnx_runner(tmp_path) -> None:
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    task_name = register_straight_task(
        runner_cls=OnnxPolicyRunner,
        source_mode="live",
        future_steps=2,
        num_envs=1,
    )

    assert load_runner_cls(task_name) is OnnxPolicyRunner
    env_cfg = load_env_cfg(task_name, play=True)
    assert env_cfg.scene.num_envs == 1
    assert "straight_success" in env_cfg.terminations
    assert (
        env_cfg.observations["actor"].terms["future_joint_window"].func
        is future_joint_window_textop_order
    )


def test_straight_spec_fn_adds_visual_non_colliding_geom() -> None:
    cfg = make_straight_g1_env_cfg(play=True)
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]

    assert cfg.scene.spec_fn is not None
    cfg.scene.spec_fn(spec)

    goal_body = next(body for body in spec.bodies if body.name == "straight_goal")
    assert goal_body is not None
    assert tuple(goal_body.pos) == (24.0, 0.0, 0.005)
    geom = next(geom for geom in goal_body.geoms if geom.name == "straight_goal_visual")
    assert geom is not None
    assert tuple(geom.size) == (9.0, 9.0, 0.005)
    assert geom.contype == 0
    assert geom.conaffinity == 0

    walls = {body.name: body for body in spec.bodies if body.name.endswith("_wall")}
    assert set(walls) == {
        "straight_left_wall",
        "straight_right_wall",
        "straight_end_wall",
        "straight_back_wall",
    }
    assert tuple(walls["straight_left_wall"].pos) == (15.5, 9.0, 0.75)
    assert tuple(walls["straight_right_wall"].pos) == (15.5, -9.0, 0.75)
    assert tuple(walls["straight_end_wall"].pos) == (33.0, 0.0, 0.75)
    assert tuple(walls["straight_back_wall"].pos) == (-2.0, 0.0, 0.75)

    wall_geom = walls["straight_left_wall"].geoms[0]
    assert tuple(wall_geom.size) == (17.5, 0.1, 0.75)
    assert tuple(wall_geom.rgba) == (0.5, 0.5, 0.5, 1.0)
    assert wall_geom.contype == 1
    assert wall_geom.conaffinity == 1


def test_straight_mdp_terms_use_true_goal_position() -> None:
    env = _fake_env(root_pos=(24.0, 0.0, 0.8), root_lin_vel=(0.03, 0.04, 0.0))

    assert torch.allclose(
        mdp.robot_goal_distance(env, STRAIGHT_TASK_CFG.goal_pos_w),
        torch.tensor([0.0]),
    )
    assert torch.allclose(mdp.robot_xy_speed(env), torch.tensor([0.05]))
    assert mdp.inside_goal_radius(
        env,
        STRAIGHT_TASK_CFG.goal_pos_w,
        radius=STRAIGHT_TASK_CFG.success_radius,
    ).tolist()
    assert mdp.below_speed_threshold(env, speed_threshold=0.1).tolist()


def test_straight_success_held_requires_hold_time() -> None:
    env = _fake_env(root_pos=(24.0, 0.0, 0.8), root_lin_vel=(0.0, 0.0, 0.0))
    cfg = SimpleNamespace(
        params={
            "goal_pos_w": STRAIGHT_TASK_CFG.goal_pos_w,
            "success_radius": STRAIGHT_TASK_CFG.success_radius,
            "speed_threshold": STRAIGHT_TASK_CFG.speed_threshold,
            "hold_time_s": 0.15,
        }
    )
    term = mdp.success_held(cfg=cfg, env=env)  # ty: ignore[invalid-argument-type]

    assert term(env, **cfg.params).tolist() == [False]
    assert term(env, **cfg.params).tolist() == [False]
    assert term(env, **cfg.params).tolist() == [True]
    term.reset()
    assert term(env, **cfg.params).tolist() == [False]


def _fake_env(
    *,
    root_pos: tuple[float, float, float],
    root_lin_vel: tuple[float, float, float],
):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([root_pos], dtype=torch.float32),
            root_link_lin_vel_w=torch.tensor([root_lin_vel], dtype=torch.float32),
        )
    )
    scene = {"robot": robot}
    return SimpleNamespace(
        device="cpu",
        num_envs=1,
        step_dt=0.05,
        scene=scene,
    )
