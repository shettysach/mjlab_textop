from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import MotionReferenceCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def root_height_below_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = cast(MotionReferenceCommand, env.command_manager.get_term(command_name))
  return (command.robot_root_pos_w[:, 2] - command.root_pos_w[:, 2]) < -threshold


def root_tilt_too_large(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  threshold: float,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1) > threshold


def reference_exhausted(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionReferenceCommand, env.command_manager.get_term(command_name))
  if command.cfg.auto_refresh:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return command.reference_exhausted
