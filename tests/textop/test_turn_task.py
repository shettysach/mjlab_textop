from __future__ import annotations

import mujoco
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.tasks import register_tasks
from mjlab_textop.tasks.turn.env_cfg import TURN_TASK_CFG
from mjlab_textop.tasks.turn.registration import TURN_TASK_NAME


def test_turn_task_registers() -> None:
    register_tasks()

    assert TURN_TASK_NAME in list_tasks()
    assert load_runner_cls(TURN_TASK_NAME) is MotionTrackingOnPolicyRunner


def test_turn_task_env_cfg_has_fixed_goal_eval_terms() -> None:
    register_tasks()
    cfg = load_env_cfg(TURN_TASK_NAME, play=True)

    assert cfg.scene.num_envs == 1
    assert cfg.episode_length_s == 20.0
    assert cfg.rewards == {}
    assert cfg.scene.spec_fn is not None
    assert "turn_task_success" in cfg.terminations
    assert "turn_task_goal_distance" in cfg.metrics
    assert (
        cfg.metrics["turn_task_stop_trigger"].params["stop_trigger_radius"]
        == TURN_TASK_CFG.stop_trigger_radius
    )


def test_turn_task_spec_fn_adds_narrow_l_shaped_walls() -> None:
    register_tasks()
    cfg = load_env_cfg(TURN_TASK_NAME, play=True)
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]

    assert cfg.scene.spec_fn is not None
    cfg.scene.spec_fn(spec)

    goal_body = next(body for body in spec.bodies if body.name == "turn_task_goal")
    assert tuple(goal_body.pos) == (12.0, -12.0, 0.005)
    goal_geom = next(
        geom for geom in goal_body.geoms if geom.name == "turn_task_goal_visual"
    )
    assert tuple(goal_geom.size) == (1.0, 1.0, 0.005)
    assert goal_geom.contype == 0
    assert goal_geom.conaffinity == 0

    walls = {body.name: body for body in spec.bodies if body.name.endswith("_wall")}
    assert set(walls) == {
        "turn_task_start_wall",
        "turn_task_outer_forward_wall",
        "turn_task_inner_forward_wall",
        "turn_task_outer_turn_wall",
        "turn_task_inner_turn_wall",
        "turn_task_end_wall",
    }
    assert tuple(walls["turn_task_start_wall"].pos) == (-2.0, 0.0, 0.75)
    assert tuple(walls["turn_task_outer_forward_wall"].pos) == (6.0, 2.0, 0.75)
    assert tuple(walls["turn_task_inner_forward_wall"].pos) == (4.0, -2.0, 0.75)
    assert tuple(walls["turn_task_outer_turn_wall"].pos) == (14.0, -6.0, 0.75)
    assert tuple(walls["turn_task_inner_turn_wall"].pos) == (10.0, -8.0, 0.75)
    assert tuple(walls["turn_task_end_wall"].pos) == (12.0, -14.0, 0.75)

    wall_geom = walls["turn_task_outer_forward_wall"].geoms[0]
    assert tuple(wall_geom.size) == (8.0, 0.1, 0.75)
    assert tuple(wall_geom.rgba) == (0.5, 0.5, 0.5, 1.0)
    assert wall_geom.contype == 1
    assert wall_geom.conaffinity == 1
    assert wall_geom.condim == 1
    assert tuple(wall_geom.friction) == (0.0, 0.0, 0.0)
    assert tuple(wall_geom.solref) == (0.05, 1.0)
    assert tuple(wall_geom.solimp) == (0.8, 0.95, 0.01, 0.5, 2.0)
