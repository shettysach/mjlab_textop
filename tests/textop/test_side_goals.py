from __future__ import annotations

import mujoco
from mjlab.tasks.registry import load_env_cfg, load_runner_cls

from mjlab_textop.core.mdp.observations import future_joint_window_textop_order
from mjlab_textop.core.onnx_policy import OnnxPolicyRunner
from mjlab_textop.tasks.registration import register_task
from mjlab_textop.tasks.side_goals.env_cfg import (
    SIDE_GOALS_TASK_CFG,
    make_side_goals_g1_env_cfg,
)


def test_side_goals_env_cfg_has_fixed_green_goal_eval_terms() -> None:
    cfg = make_side_goals_g1_env_cfg(play=True)

    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 20.0
    assert cfg.rewards == {}
    assert cfg.scene.spec_fn is not None
    assert "side_goals_success" in cfg.terminations
    assert "side_goals_green_goal_distance" in cfg.metrics
    assert "side_goals_blue_goal_distance" in cfg.metrics
    assert (
        cfg.metrics["side_goals_stop_trigger"].params["stop_trigger_radius"]
        == SIDE_GOALS_TASK_CFG.stop_trigger_radius
    )
    assert (
        cfg.terminations["side_goals_success"].params["goal_pos_w"]
        == SIDE_GOALS_TASK_CFG.green_goal_pos_w
    )


def test_side_goals_play_task_uses_onnx_runner(tmp_path) -> None:
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    task_name = register_task(
        "side-goals",
        runner_cls=OnnxPolicyRunner,
        source_mode="live",
        future_steps=2,
        num_envs=1,
    )

    assert load_runner_cls(task_name) is OnnxPolicyRunner
    env_cfg = load_env_cfg(task_name, play=True)
    assert env_cfg.scene.num_envs == 1
    assert "side_goals_success" in env_cfg.terminations
    assert (
        env_cfg.observations["actor"].terms["future_joint_window"].func
        is future_joint_window_textop_order
    )


def test_side_goals_spec_fn_adds_two_goals_and_four_walls() -> None:
    cfg = make_side_goals_g1_env_cfg(play=True)
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]

    assert cfg.scene.spec_fn is not None
    cfg.scene.spec_fn(spec)

    blue_goal = next(
        body for body in spec.bodies if body.name == "side_goals_blue_goal"
    )
    green_goal = next(
        body for body in spec.bodies if body.name == "side_goals_green_goal"
    )
    assert tuple(blue_goal.pos) == (0.0, 5.0, 0.005)
    assert tuple(green_goal.pos) == (0.0, -5.0, 0.005)

    blue_geom = next(
        geom for geom in blue_goal.geoms if geom.name == "side_goals_blue_goal_visual"
    )
    green_geom = next(
        geom for geom in green_goal.geoms if geom.name == "side_goals_green_goal_visual"
    )
    assert tuple(blue_geom.size) == (4.0, 4.0, 0.005)
    assert tuple(blue_geom.rgba) == (0.0, 0.0, 1.0, 1.0)
    assert blue_geom.contype == 0
    assert blue_geom.conaffinity == 0
    assert tuple(green_geom.size) == (4.0, 4.0, 0.005)
    assert tuple(green_geom.rgba) == (0.0, 1.0, 0.0, 1.0)
    assert green_geom.contype == 0
    assert green_geom.conaffinity == 0

    walls = {
        body.name: body
        for body in spec.bodies
        if body.name.startswith("side_goals_") and body.name.endswith("_wall")
    }
    assert set(walls) == {
        "side_goals_left_wall",
        "side_goals_right_wall",
        "side_goals_front_wall",
        "side_goals_back_wall",
    }
    assert tuple(walls["side_goals_left_wall"].pos) == (0.0, 9.0, 0.75)
    assert tuple(walls["side_goals_right_wall"].pos) == (0.0, -9.0, 0.75)
    assert tuple(walls["side_goals_front_wall"].pos) == (9.0, 0.0, 0.75)
    assert tuple(walls["side_goals_back_wall"].pos) == (-9.0, 0.0, 0.75)

    wall_geom = walls["side_goals_left_wall"].geoms[0]
    assert tuple(wall_geom.size) == (9.0, 0.1, 0.75)
    assert tuple(wall_geom.rgba) == (0.5, 0.5, 0.5, 1.0)
    assert wall_geom.contype == 1
    assert wall_geom.conaffinity == 1
