from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal
from uuid import uuid4

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.core.onnx_policy import (
    OnnxExecutionProvider,
    OnnxPolicyRunner,
)
from mjlab_textop.tasks.blocked_straight.env_cfg import make_blocked_straight_g1_env_cfg
from mjlab_textop.tasks.online_textop.env_cfg import make_online_textop_g1_env_cfg
from mjlab_textop.tasks.portrait_corridors.env_cfg import (
    make_portrait_corridors_g1_env_cfg,
)
from mjlab_textop.tasks.side_goals.env_cfg import make_side_goals_g1_env_cfg
from mjlab_textop.tasks.straight.env_cfg import make_straight_g1_env_cfg
from mjlab_textop.tasks.turn.env_cfg import make_turn_task_g1_env_cfg

TextOpTask = Literal[
    "default",
    "straight",
    "blocked-straight",
    "side-goals",
    "turn",
    "portrait-corridors",
]
PolicyRunnerCls = type[MotionTrackingOnPolicyRunner] | type[OnnxPolicyRunner]

TASK_CFGS: dict[TextOpTask, Callable[..., ManagerBasedRlEnvCfg]] = {
    "default": make_online_textop_g1_env_cfg,
    "straight": make_straight_g1_env_cfg,
    "blocked-straight": make_blocked_straight_g1_env_cfg,
    "side-goals": make_side_goals_g1_env_cfg,
    "turn": make_turn_task_g1_env_cfg,
    "portrait-corridors": make_portrait_corridors_g1_env_cfg,
}


@dataclass
class OnnxPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    onnx_execution_provider: OnnxExecutionProvider = "cpu"


def register_task(
    task: TextOpTask,
    *,
    runner_cls: PolicyRunnerCls = MotionTrackingOnPolicyRunner,
    onnx_provider: OnnxExecutionProvider = "cpu",
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode,
    num_envs: int = 1,
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
) -> str:
    make_env_cfg = TASK_CFGS[task]
    policy_format: Literal["pt", "onnx"] = (
        "onnx" if issubclass(runner_cls, OnnxPolicyRunner) else "pt"
    )
    name_parts = [task, policy_format, source_mode.capitalize(), uuid4().hex]
    task_name = "-".join(name_parts)
    env_cfg = make_env_cfg(
        play=True,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
        policy_format=policy_format,
    )
    env_cfg.scene.num_envs = num_envs
    rl_cfg = unitree_g1_tracking_ppo_runner_cfg()
    if issubclass(runner_cls, OnnxPolicyRunner):
        rl_cfg = OnnxPolicyRunnerCfg(
            **{field.name: getattr(rl_cfg, field.name) for field in fields(rl_cfg)},
            onnx_execution_provider=onnx_provider,
        )

    register_mjlab_task(
        task_id=task_name,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=runner_cls,
    )
    return task_name
