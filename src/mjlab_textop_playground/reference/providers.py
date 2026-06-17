from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from .types import MotionReference


@dataclass
class RobotState:
  """Minimal robot state needed by reference providers."""

  root_pos: torch.Tensor
  root_quat: torch.Tensor
  root_lin_vel: torch.Tensor
  root_ang_vel: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor


class MotionReferenceProvider(Protocol):
  """Provider interface for Action-Expert-agnostic reference generation."""

  def generate(
    self,
    texts: Sequence[str],
    robot_state: RobotState,
    horizon: int,
    dt: float,
  ) -> MotionReference:
    """Generate one short-horizon reference per text command."""
