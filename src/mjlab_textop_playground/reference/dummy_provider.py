from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .providers import RobotState
from .types import MotionReference


@dataclass
class DummyTextReferenceProvider:
  """Deterministic text-to-reference provider for V1 plumbing tests."""

  forward_speed: float = 0.4
  lateral_speed: float = -0.3
  yaw_speed: float = 0.6

  def generate(
    self,
    texts: Sequence[str],
    robot_state: RobotState,
    horizon: int,
    dt: float,
  ) -> MotionReference:
    if horizon <= 0:
      raise ValueError("horizon must be positive")
    n = len(texts)
    if robot_state.joint_pos.shape[0] != n:
      raise ValueError(
        f"robot_state batch size {robot_state.joint_pos.shape[0]} does not match {n}"
      )

    device = robot_state.joint_pos.device
    dtype = robot_state.joint_pos.dtype
    steps = torch.arange(horizon, device=device, dtype=dtype)

    root_pos = robot_state.root_pos[:, None, :].repeat(1, horizon, 1)
    root_quat = robot_state.root_quat[:, None, :].repeat(1, horizon, 1)
    root_lin_vel = torch.zeros(n, horizon, 3, device=device, dtype=dtype)
    root_ang_vel = torch.zeros(n, horizon, 3, device=device, dtype=dtype)
    joint_pos = robot_state.joint_pos[:, None, :].repeat(1, horizon, 1)
    joint_vel = torch.zeros_like(joint_pos)

    for env_idx, text in enumerate(texts):
      vx, vy, wz = self._command_velocity(text)
      heading_quat = _yaw_only(robot_state.root_quat[env_idx : env_idx + 1])
      local_velocity = torch.tensor(
        [[vx, vy, 0.0]], device=device, dtype=dtype
      ).repeat(horizon, 1)
      world_velocity = _quat_apply(heading_quat.repeat(horizon, 1), local_velocity)
      root_lin_vel[env_idx] = world_velocity
      root_ang_vel[env_idx, :, 2] = wz
      root_pos[env_idx] += world_velocity * (dt * steps[:, None])
      if wz != 0.0:
        yaw = wz * dt * steps
        root_quat[env_idx] = _quat_mul(_yaw_quat(yaw), robot_state.root_quat[env_idx])

    valid = torch.ones(n, horizon, device=device, dtype=torch.bool)
    if horizon == 1:
      phase = torch.zeros(n, horizon, device=device, dtype=dtype)
    else:
      phase = steps[None, :].repeat(n, 1) / float(horizon - 1)

    reference = MotionReference(
      root_pos=root_pos,
      root_quat=root_quat,
      root_lin_vel=root_lin_vel,
      root_ang_vel=root_ang_vel,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      valid=valid,
      phase=phase,
    )
    reference.validate()
    return reference

  def _command_velocity(self, text: str) -> tuple[float, float, float]:
    normalized = " ".join(text.lower().strip().split())
    if normalized in {"walk forward", "forward", "walk"}:
      return self.forward_speed, 0.0, 0.0
    if normalized in {"turn left", "left turn"}:
      return 0.0, 0.0, self.yaw_speed
    if normalized in {"sidestep right", "side step right", "right"}:
      return 0.0, self.lateral_speed, 0.0
    return 0.0, 0.0, 0.0


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  """Return quaternions in MJLab's wxyz convention."""

  half = 0.5 * yaw
  quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
  quat[:, 0] = torch.cos(half)
  quat[:, 3] = torch.sin(half)
  return quat


def _yaw_only(quat: torch.Tensor) -> torch.Tensor:
  """Return a yaw-only quaternion extracted from a wxyz quaternion."""

  w, x, y, z = quat.unbind(dim=-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return _yaw_quat(yaw)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
  """Quaternion multiplication in wxyz convention with broadcasting."""

  w1, x1, y1, z1 = lhs.unbind(dim=-1)
  w2, x2, y2, z2 = rhs.unbind(dim=-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vectors by wxyz quaternions."""

  q_vec = quat[..., 1:]
  q_w = quat[..., :1]
  t = 2.0 * torch.cross(q_vec, vec, dim=-1)
  return vec + q_w * t + torch.cross(q_vec, t, dim=-1)
