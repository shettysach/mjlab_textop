from __future__ import annotations

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.task import StaticTaskSpec
from mjlab_textop.tasks.green_square_stop.env_cfg import (
    make_green_square_stop_g1_env_cfg,
)
from mjlab_textop.tasks.green_square_stop.ppo_cfg import (
    unitree_g1_tracking_ppo_runner_cfg,
)

GREEN_SQUARE_STOP_TASK_NAME = "Mjlab-VLA-GreenSquareStop-G1"

STATIC_TASK_SPECS = [
    StaticTaskSpec(
        task_id=GREEN_SQUARE_STOP_TASK_NAME,
        make_env_cfg=lambda: make_green_square_stop_g1_env_cfg(play=True),
        make_play_env_cfg=lambda: make_green_square_stop_g1_env_cfg(play=True),
        make_rl_cfg=unitree_g1_tracking_ppo_runner_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    ),
]
