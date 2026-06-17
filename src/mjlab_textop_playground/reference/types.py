from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MotionReference:
  """Short-horizon reference consumed by MJLab reference-tracking tasks.

  Required fields are shaped ``[num_envs, horizon, ...]``. Body fields are
  optional in V1 because the dummy provider only creates root/joint references.
  """

  root_pos: torch.Tensor
  root_quat: torch.Tensor
  root_lin_vel: torch.Tensor
  root_ang_vel: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos: torch.Tensor | None = None
  body_quat: torch.Tensor | None = None
  body_lin_vel: torch.Tensor | None = None
  body_ang_vel: torch.Tensor | None = None
  valid: torch.Tensor | None = None
  phase: torch.Tensor | None = None

  @property
  def num_envs(self) -> int:
    return int(self.root_pos.shape[0])

  @property
  def horizon(self) -> int:
    return int(self.root_pos.shape[1])

  @property
  def num_dofs(self) -> int:
    return int(self.joint_pos.shape[2])

  @property
  def device(self) -> torch.device:
    return self.root_pos.device

  def validate(self) -> None:
    """Validate shapes, device, and dtype consistency."""

    n, h = self.root_pos.shape[:2]
    device = self.root_pos.device
    dtype = self.root_pos.dtype

    required_shapes = {
      "root_pos": (n, h, 3),
      "root_quat": (n, h, 4),
      "root_lin_vel": (n, h, 3),
      "root_ang_vel": (n, h, 3),
    }
    for name, shape in required_shapes.items():
      self._check_tensor(name, getattr(self, name), shape, device, dtype)

    if self.joint_pos.ndim != 3:
      raise ValueError(f"joint_pos must be rank 3, got {self.joint_pos.shape}")
    self._check_tensor("joint_vel", self.joint_vel, self.joint_pos.shape, device, dtype)

    body_specs = {
      "body_pos": 3,
      "body_quat": 4,
      "body_lin_vel": 3,
      "body_ang_vel": 3,
    }
    for name, width in body_specs.items():
      value = getattr(self, name)
      if value is None:
        continue
      if value.ndim != 4 or value.shape[:2] != (n, h) or value.shape[-1] != width:
        raise ValueError(f"{name} must be shaped [N, H, B, {width}], got {value.shape}")
      self._check_device_dtype(name, value, device, dtype)

    if self.valid is not None:
      if self.valid.shape != (n, h):
        raise ValueError(f"valid must be shaped [N, H], got {self.valid.shape}")
      if self.valid.device != device:
        raise ValueError("valid must be on the same device as root_pos")

    if self.phase is not None:
      self._check_tensor("phase", self.phase, (n, h), device, dtype)

  @staticmethod
  def _check_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    device: torch.device,
    dtype: torch.dtype,
  ) -> None:
    if tuple(value.shape) != tuple(shape):
      raise ValueError(f"{name} must be shaped {tuple(shape)}, got {tuple(value.shape)}")
    MotionReference._check_device_dtype(name, value, device, dtype)

  @staticmethod
  def _check_device_dtype(
    name: str,
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
  ) -> None:
    if value.device != device:
      raise ValueError(f"{name} must be on {device}, got {value.device}")
    if value.dtype != dtype:
      raise ValueError(f"{name} must have dtype {dtype}, got {value.dtype}")
