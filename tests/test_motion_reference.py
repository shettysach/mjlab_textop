import pytest
import torch

from mjlab_textop_playground.reference import MotionReference


def _reference() -> MotionReference:
  return MotionReference(
    root_pos=torch.zeros(2, 4, 3),
    root_quat=torch.zeros(2, 4, 4),
    root_lin_vel=torch.zeros(2, 4, 3),
    root_ang_vel=torch.zeros(2, 4, 3),
    joint_pos=torch.zeros(2, 4, 29),
    joint_vel=torch.zeros(2, 4, 29),
    valid=torch.ones(2, 4, dtype=torch.bool),
    phase=torch.zeros(2, 4),
  )


def test_motion_reference_validate_accepts_valid_shapes():
  ref = _reference()
  ref.root_quat[..., 0] = 1.0
  ref.validate()


def test_motion_reference_validate_rejects_bad_shape():
  ref = _reference()
  ref.joint_vel = torch.zeros(2, 3, 29)
  with pytest.raises(ValueError, match="joint_vel"):
    ref.validate()
