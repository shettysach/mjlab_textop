from __future__ import annotations

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.feedback.observation import OnlineTextOpObservationCfg
from mjlab_textop.core.mdp.online_commands import TextOpOnlineSourceMode
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.onnx_policy import OnnxPolicyRunner
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import (
    DynamicOnlineTaskSpec,
    register_dynamic_online_task,
)
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_flat_tracking_env_cfg,
    make_online_textop_onnx_g1_flat_tracking_env_cfg,
)

ONLINE_TEXTOP_TASK_NAME = "Mjlab-OnlineTextOp-Flat-Unitree-G1"

DYNAMIC_TASK_SPEC = DynamicOnlineTaskSpec(
    base_task_name=ONLINE_TEXTOP_TASK_NAME,
    checkpoint_env_cfg=make_online_textop_g1_flat_tracking_env_cfg,
    onnx_env_cfg=make_online_textop_onnx_g1_flat_tracking_env_cfg,
)


def register_online_textop_task(
    *,
    runner_cls: type = MotionTrackingOnPolicyRunner,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
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
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
    )


def register_online_textop_onnx_task(
    *,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineTextOpObservationCfg | None = None,
) -> str:
    return register_online_textop_task(
        runner_cls=OnnxPolicyRunner,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        future_steps=future_steps,
        num_envs=num_envs,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
    )
