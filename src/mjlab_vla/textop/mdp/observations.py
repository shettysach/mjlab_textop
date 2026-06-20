from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.observations import projected_gravity as mjlab_projected_gravity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from mjlab_vla.textop.mdp.commands import TextOpMotionCommand


def _get_textop_motion_command(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> TextOpMotionCommand:
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, TextOpMotionCommand):
        raise TypeError(
            f"Expected command {command_name!r} to be TextOpMotionCommand, "
            f"got {type(command).__name__}"
        )

    return command


def future_joint_window(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_textop_motion_command(env, command_name)

    return torch.cat(
        [
            command.future_joint_pos.reshape(env.num_envs, -1),
            command.future_joint_vel.reshape(env.num_envs, -1),
        ],
        dim=-1,
    )


def future_anchor_pos_b(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_textop_motion_command(env, command_name)

    robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand_as(
        command.future_anchor_pos_w
    )
    robot_anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand_as(
        command.future_anchor_quat_w
    )

    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )

    return pos_b.reshape(env.num_envs, -1)


def future_anchor_ori_b(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_textop_motion_command(env, command_name)

    robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand_as(
        command.future_anchor_pos_w
    )
    robot_anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand_as(
        command.future_anchor_quat_w
    )

    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    return mjlab_projected_gravity(env, asset_cfg=asset_cfg)
