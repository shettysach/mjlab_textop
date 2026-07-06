from __future__ import annotations

from mjlab_textop.tasks.side_goals.env_cfg import (
    SIDE_GOALS_TASK_CFG,
    SideGoalsTaskCfg,
    make_side_goals_g1_env_cfg,
    make_side_goals_onnx_g1_env_cfg,
)
from mjlab_textop.tasks.side_goals.registration import (
    SIDE_GOALS_TASK_NAME,
    register_side_goals_task,
)

__all__ = [
    "SIDE_GOALS_TASK_CFG",
    "SIDE_GOALS_TASK_NAME",
    "SideGoalsTaskCfg",
    "make_side_goals_g1_env_cfg",
    "make_side_goals_onnx_g1_env_cfg",
    "register_side_goals_task",
]
