from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionReferenceCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionReferenceCommand:
  return cast(MotionReferenceCommand, env.command_manager.get_term(command_name))


def reference_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).joint_pos


def reference_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).joint_vel


def joint_pos_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.joint_pos - command.robot_joint_pos


def joint_vel_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.joint_vel - command.robot_joint_vel


def reference_root_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return torch.cat([command.root_lin_vel_w, command.root_ang_vel_w], dim=-1)


def root_velocity_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  lin_error = command.root_lin_vel_w - command.robot_root_lin_vel_w
  ang_error = command.root_ang_vel_w - command.robot_root_ang_vel_w
  return torch.cat([lin_error, ang_error], dim=-1)


def reference_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).phase[:, None]


def future_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  future_pos_w = command.future_root_pos_w()
  future_quat_w = command.future_root_quat_w()
  robot_pos = command.robot_root_pos_w[:, None, :].expand_as(future_pos_w)
  robot_quat = command.robot_root_quat_w[:, None, :].expand_as(future_quat_w)
  pos_b, _ = subtract_frame_transforms(
    robot_pos,
    robot_quat,
    future_pos_w,
    future_quat_w,
  )
  return pos_b.reshape(env.num_envs, -1)


def future_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  future_pos_w = command.future_root_pos_w()
  future_quat_w = command.future_root_quat_w()
  robot_pos = command.robot_root_pos_w[:, None, :].expand_as(future_pos_w)
  robot_quat = command.robot_root_quat_w[:, None, :].expand_as(future_quat_w)
  _, ori_b = subtract_frame_transforms(
    robot_pos,
    robot_quat,
    future_pos_w,
    future_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(env.num_envs, -1)
