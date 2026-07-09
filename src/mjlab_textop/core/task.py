from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class DynamicOnlineTaskSpec:
    base_task_name: str
    checkpoint_env_cfg: Callable[..., Any]
    onnx_env_cfg: Callable[..., Any]
    onnx_task_name: str | None = None


def register_dynamic_online_task(
    spec: DynamicOnlineTaskSpec,
    *,
    runner_cls: type,
    source: Any = None,
    live_source_cfg: Any = None,
    source_mode: str,
    future_steps: int,
    num_envs: int = 1,
    anchor_alignment: str = "align_to_robot_start",
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: Any = None,
) -> str:
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.config.g1.rl_cfg import (
        unitree_g1_tracking_ppo_runner_cfg,
    )

    from mjlab_textop.core.onnx_policy import OnnxPolicyRunner

    is_onnx = runner_cls is OnnxPolicyRunner
    task_prefix = (
        spec.onnx_task_name if is_onnx and spec.onnx_task_name else spec.base_task_name
    )
    name_parts = [task_prefix]
    if spec.onnx_task_name is None:
        name_parts.append("Onnx" if is_onnx else "Checkpoint")
    name_parts.extend([source_mode.capitalize(), uuid4().hex])
    task_name = "-".join(name_parts)

    make_env_cfg = spec.onnx_env_cfg if is_onnx else spec.checkpoint_env_cfg
    env_cfg = make_env_cfg(
        play=True,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
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
