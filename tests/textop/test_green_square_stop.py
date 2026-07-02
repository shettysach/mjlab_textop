from __future__ import annotations

from types import SimpleNamespace

import mujoco
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.mdp.observations import future_joint_window_textop_order
from mjlab_textop.core.onnx_policy import CustomOnnxPolicyRunner
from mjlab_textop.scripts.utils import (
    ResolvedPolicy,
    register_green_square_stop_play_task,
)
from mjlab_textop.tasks import register_tasks
from mjlab_textop.tasks.green_square_stop import mdp
from mjlab_textop.tasks.green_square_stop.env_cfg import (
    GREEN_SQUARE_GOAL_POS_W,
    GREEN_SQUARE_STOP_TRIGGER_RADIUS,
)
from mjlab_textop.tasks.green_square_stop.registration import (
    GREEN_SQUARE_STOP_TASK_NAME,
)


def test_green_square_stop_task_registers() -> None:
    register_tasks()

    assert GREEN_SQUARE_STOP_TASK_NAME in list_tasks()
    assert load_runner_cls(GREEN_SQUARE_STOP_TASK_NAME) is MotionTrackingOnPolicyRunner


def test_green_square_stop_env_cfg_has_fixed_goal_eval_terms() -> None:
    register_tasks()
    cfg = load_env_cfg(GREEN_SQUARE_STOP_TASK_NAME, play=True)

    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 20.0
    assert cfg.rewards == {}
    assert cfg.scene.spec_fn is not None
    assert "green_square_success" in cfg.terminations
    assert "green_square_goal_distance" in cfg.metrics
    assert (
        cfg.metrics["green_square_stop_trigger"].params["stop_trigger_radius"]
        == GREEN_SQUARE_STOP_TRIGGER_RADIUS
    )


def test_green_square_stop_play_task_uses_onnx_runner(tmp_path) -> None:
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    task_name = register_green_square_stop_play_task(
        policy=ResolvedPolicy("onnx", onnx_file),
        source_mode="live",
        future_steps=2,
        num_envs=1,
    )

    assert load_runner_cls(task_name) is CustomOnnxPolicyRunner
    env_cfg = load_env_cfg(task_name, play=True)
    assert env_cfg.scene.num_envs == 1
    assert "green_square_success" in env_cfg.terminations
    assert (
        env_cfg.observations["actor"].terms["future_joint_window"].func
        is future_joint_window_textop_order
    )


def test_green_square_marker_spec_fn_adds_visual_non_colliding_geom() -> None:
    register_tasks()
    cfg = load_env_cfg(GREEN_SQUARE_STOP_TASK_NAME, play=True)
    spec = mujoco.MjSpec()

    assert cfg.scene.spec_fn is not None
    cfg.scene.spec_fn(spec)

    goal_body = next(body for body in spec.bodies if body.name == "green_square_goal")
    assert goal_body is not None
    assert tuple(goal_body.pos) == (24.0, 0.0, 0.005)
    geom = next(
        geom for geom in goal_body.geoms if geom.name == "green_square_goal_visual"
    )
    assert geom is not None
    assert tuple(geom.size) == (9.0, 9.0, 0.005)
    assert geom.contype == 0
    assert geom.conaffinity == 0

    walls = {body.name: body for body in spec.bodies if body.name.endswith("_wall")}
    assert set(walls) == {
        "green_square_left_wall",
        "green_square_right_wall",
        "green_square_end_wall",
    }
    assert tuple(walls["green_square_left_wall"].pos) == (24.0, 9.0, 0.75)
    assert tuple(walls["green_square_right_wall"].pos) == (24.0, -9.0, 0.75)
    assert tuple(walls["green_square_end_wall"].pos) == (33.0, 0.0, 0.75)

    wall_geom = walls["green_square_left_wall"].geoms[0]
    assert tuple(wall_geom.size) == (9.0, 0.1, 0.75)
    assert tuple(wall_geom.rgba) == (0.0, 0.0, 0.0, 0.0)
    assert wall_geom.contype == 1
    assert wall_geom.conaffinity == 1


def test_green_square_mdp_terms_use_true_goal_position() -> None:
    env = _fake_env(root_pos=(24.0, 0.0, 0.8), root_lin_vel=(0.03, 0.04, 0.0))

    assert torch.allclose(
        mdp.robot_goal_distance(env, GREEN_SQUARE_GOAL_POS_W),
        torch.tensor([0.0]),
    )
    assert torch.allclose(mdp.robot_xy_speed(env), torch.tensor([0.05]))
    assert mdp.inside_goal_radius(env, GREEN_SQUARE_GOAL_POS_W, radius=0.25).tolist()
    assert mdp.below_speed_threshold(env, speed_threshold=0.1).tolist()


def test_green_square_success_held_requires_hold_time() -> None:
    env = _fake_env(root_pos=(24.0, 0.0, 0.8), root_lin_vel=(0.0, 0.0, 0.0))
    cfg = SimpleNamespace(
        params={
            "goal_pos_w": GREEN_SQUARE_GOAL_POS_W,
            "success_radius": 0.25,
            "speed_threshold": 0.1,
            "hold_time_s": 0.15,
        }
    )
    term = mdp.success_held(cfg=cfg, env=env)

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
