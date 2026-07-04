from __future__ import annotations

import mujoco
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.mdp.observations import future_joint_window_textop_order
from mjlab_textop.core.onnx_policy import CustomOnnxPolicyRunner
from mjlab_textop.scripts.utils import (
    ResolvedPolicy,
    register_blocked_straight_play_task,
)
from mjlab_textop.tasks import register_tasks
from mjlab_textop.tasks.blocked_straight.env_cfg import (
    BLOCKED_STRAIGHT_TASK_CFG,
)
from mjlab_textop.tasks.blocked_straight.registration import (
    BLOCKED_STRAIGHT_TASK_NAME,
)


def test_blocked_straight_task_registers() -> None:
    register_tasks()

    assert BLOCKED_STRAIGHT_TASK_NAME in list_tasks()
    assert load_runner_cls(BLOCKED_STRAIGHT_TASK_NAME) is MotionTrackingOnPolicyRunner


def test_blocked_straight_env_cfg_has_fixed_goal_eval_terms() -> None:
    register_tasks()
    cfg = load_env_cfg(BLOCKED_STRAIGHT_TASK_NAME, play=True)

    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 20.0
    assert cfg.rewards == {}
    assert cfg.scene.spec_fn is not None
    assert "blocked_straight_success" in cfg.terminations
    assert "blocked_straight_goal_distance" in cfg.metrics
    assert (
        cfg.metrics["blocked_straight_stop_trigger"].params["stop_trigger_radius"]
        == BLOCKED_STRAIGHT_TASK_CFG.stop_trigger_radius
    )


def test_blocked_straight_play_task_uses_onnx_runner(tmp_path) -> None:
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    task_name = register_blocked_straight_play_task(
        policy=ResolvedPolicy("onnx", onnx_file),
        source_mode="live",
        future_steps=2,
        num_envs=1,
    )

    assert load_runner_cls(task_name) is CustomOnnxPolicyRunner
    env_cfg = load_env_cfg(task_name, play=True)
    assert env_cfg.scene.num_envs == 1
    assert "blocked_straight_success" in env_cfg.terminations
    assert (
        env_cfg.observations["actor"].terms["future_joint_window"].func
        is future_joint_window_textop_order
    )


def test_blocked_straight_spec_fn_adds_centered_wide_block() -> None:
    register_tasks()
    cfg = load_env_cfg(BLOCKED_STRAIGHT_TASK_NAME, play=True)
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]

    assert cfg.scene.spec_fn is not None
    cfg.scene.spec_fn(spec)

    goal_body = next(
        body for body in spec.bodies if body.name == "blocked_straight_goal"
    )
    assert tuple(goal_body.pos) == (24.0, 0.0, 0.005)

    obstacle = next(
        body for body in spec.bodies if body.name == "blocked_straight_center_block"
    )
    assert tuple(obstacle.pos) == (12.0, 0.0, 0.75)
    obstacle_geom = obstacle.geoms[0]
    assert tuple(obstacle_geom.size) == (0.75, 4.0, 0.75)
    assert obstacle_geom.contype == 1
    assert obstacle_geom.conaffinity == 1

    walls = {
        body.name: body
        for body in spec.bodies
        if body.name.startswith("blocked_straight_") and body.name.endswith("_wall")
    }
    assert set(walls) == {
        "blocked_straight_left_wall",
        "blocked_straight_right_wall",
        "blocked_straight_end_wall",
        "blocked_straight_back_wall",
    }
    assert tuple(walls["blocked_straight_left_wall"].pos) == (15.5, 9.0, 0.75)
    assert tuple(walls["blocked_straight_right_wall"].pos) == (15.5, -9.0, 0.75)
