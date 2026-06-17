"""Unitree G1 TextOp-style reference tracking task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import g1_textop_tracking_env_cfg
from .rl_cfg import g1_textop_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-TextOpTracking-Flat-Unitree-G1",
  env_cfg=g1_textop_tracking_env_cfg(),
  play_env_cfg=g1_textop_tracking_env_cfg(play=True),
  rl_cfg=g1_textop_tracking_ppo_runner_cfg(),
  runner_cls=None,
)
