"""RL configuration for the G1 TextOp-style tracking task."""

from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def g1_textop_tracking_ppo_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_textop_tracking"
  return cfg
