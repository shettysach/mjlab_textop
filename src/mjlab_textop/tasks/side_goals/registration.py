from __future__ import annotations

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.core.schema import FUTURE_STEPS
from mjlab_textop.core.task import (
    DynamicOnlineTaskSpec,
    register_dynamic_online_task,
)
from mjlab_textop.tasks.side_goals.env_cfg import (
    make_side_goals_g1_env_cfg,
    make_side_goals_onnx_g1_env_cfg,
)

SIDE_GOALS_TASK_NAME = "Mjlab-VLA-SideGoals-G1"

DYNAMIC_TASK_SPEC = DynamicOnlineTaskSpec(
    base_task_name=SIDE_GOALS_TASK_NAME,
    checkpoint_env_cfg=make_side_goals_g1_env_cfg,
    onnx_env_cfg=make_side_goals_onnx_g1_env_cfg,
)


def register_side_goals_task(
    *,
    runner_cls: type,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode,
    future_steps: int = FUTURE_STEPS,
    num_envs: int = 1,
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
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
