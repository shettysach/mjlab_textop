from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal
from uuid import uuid4

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
from tasks.catalog import TaskSet, make_task_env_cfg

PolicyRunnerCls = type[MotionTrackingOnPolicyRunner] | type[OnnxPolicyRunner]


@dataclass
class OnnxPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    onnx_execution_provider: OnnxExecutionProvider = "cpu"


def register_task(
    task: TaskSet | None,
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
    policy_format: Literal["pt", "onnx"] = (
        "onnx" if issubclass(runner_cls, OnnxPolicyRunner) else "pt"
    )
    name_parts = [
        task or "default",
        policy_format,
        source_mode.capitalize(),
        uuid4().hex,
    ]
    task_name = "-".join(name_parts)
    env_cfg = make_task_env_cfg(
        task,
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
