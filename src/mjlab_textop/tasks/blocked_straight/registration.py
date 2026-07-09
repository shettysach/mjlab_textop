from __future__ import annotations

from typing import Literal

from mjlab_textop.core.feedback.observation import OnlineTextOpObservationCfg
from mjlab_textop.core.mdp.online_commands import TextOpOnlineSourceMode
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import (
    DynamicOnlineTaskSpec,
    register_dynamic_online_task,
)
from mjlab_textop.tasks.blocked_straight.env_cfg import (
    make_blocked_straight_g1_env_cfg,
    make_blocked_straight_onnx_g1_env_cfg,
)

BLOCKED_STRAIGHT_TASK_NAME = "Mjlab-VLA-BlockedStraight-G1"

DYNAMIC_TASK_SPEC = DynamicOnlineTaskSpec(
    base_task_name=BLOCKED_STRAIGHT_TASK_NAME,
    checkpoint_env_cfg=make_blocked_straight_g1_env_cfg,
    onnx_env_cfg=make_blocked_straight_onnx_g1_env_cfg,
)


def register_blocked_straight_task(
    *,
    runner_cls: type,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineTextOpObservationCfg | None = None,
) -> str:
    return register_dynamic_online_task(
        DYNAMIC_TASK_SPEC,
        runner_cls=runner_cls,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        future_steps=future_steps,
        num_envs=num_envs,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
    )
