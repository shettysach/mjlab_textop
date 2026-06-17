from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import MotionReferenceCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionReferenceCommand:
  return cast(MotionReferenceCommand, env.command_manager.get_term(command_name))


def reference_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  return torch.exp(-error / std**2)


def reference_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.mean(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
  return torch.exp(-error / std**2)


def reference_root_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.root_lin_vel_w - command.robot_root_lin_vel_w), dim=-1
  )
  return torch.exp(-error / std**2)


def reference_root_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.root_ang_vel_w - command.robot_root_ang_vel_w), dim=-1
  )
  return torch.exp(-error / std**2)


def upright(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
