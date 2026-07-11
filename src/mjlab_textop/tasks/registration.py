from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.core.onnx_policy import OnnxPolicyRunner
from mjlab_textop.core.schema import FUTURE_STEPS
from mjlab_textop.tasks.blocked_straight.env_cfg import make_blocked_straight_g1_env_cfg
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_flat_tracking_env_cfg,
)
from mjlab_textop.tasks.side_goals.env_cfg import make_side_goals_g1_env_cfg
from mjlab_textop.tasks.straight.env_cfg import make_straight_g1_env_cfg
from mjlab_textop.tasks.turn.env_cfg import make_turn_task_g1_env_cfg

TextOpTask = Literal["default", "straight", "blocked-straight", "side-goals", "turn"]
PolicyRunnerCls = type[MotionTrackingOnPolicyRunner] | type[OnnxPolicyRunner]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    make_env_cfg: Callable[..., Any]


TASK_SPECS: dict[TextOpTask, TaskSpec] = {
    "default": TaskSpec(
        "Mjlab-OnlineTextOp-Flat-Unitree-G1",
        make_online_textop_g1_flat_tracking_env_cfg,
    ),
    "straight": TaskSpec("Mjlab-VLA-Straight-G1", make_straight_g1_env_cfg),
    "blocked-straight": TaskSpec(
        "Mjlab-VLA-BlockedStraight-G1", make_blocked_straight_g1_env_cfg
    ),
    "side-goals": TaskSpec("Mjlab-VLA-SideGoals-G1", make_side_goals_g1_env_cfg),
    "turn": TaskSpec("Mjlab-VLA-TurnTask-G1", make_turn_task_g1_env_cfg),
}


def register_task(
    task: TextOpTask,
    *,
    runner_cls: PolicyRunnerCls = MotionTrackingOnPolicyRunner,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode,
    future_steps: int = FUTURE_STEPS,
    num_envs: int = 1,
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
) -> str:
    spec = TASK_SPECS[task]
    policy_format: Literal["pt", "onnx"] = (
        "onnx" if runner_cls is OnnxPolicyRunner else "pt"
    )
    task_name = "-".join(
        (
            spec.name,
            policy_format,
            source_mode.capitalize(),
            uuid4().hex,
        )
    )
    env_cfg = spec.make_env_cfg(
        play=True,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
        policy_format=policy_format,
    )
    env_cfg.scene.num_envs = num_envs
    register_mjlab_task(
        task_id=task_name,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        runner_cls=runner_cls,
    )
    return task_name
