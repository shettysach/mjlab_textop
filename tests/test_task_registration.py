import mjlab_textop_playground.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg

from mjlab_textop_playground.tasks.g1_textop_tracking.mdp import (
  MotionReferenceCommandCfg,
)


def test_task_registers_with_mjlab():
  assert "Mjlab-TextOpTracking-Flat-Unitree-G1" in list_tasks()


def test_task_uses_motion_reference_command():
  cfg = load_env_cfg("Mjlab-TextOpTracking-Flat-Unitree-G1")
  assert isinstance(cfg.commands["motion"], MotionReferenceCommandCfg)
