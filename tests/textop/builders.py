from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mjlab_vla.textop.online import TextOpMotionBlock


def motion_block(
    index: int = 0,
    frames: int = 8,
    offset: float = 0.0,
) -> TextOpMotionBlock:
    joint_pos = (
        np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) + offset
    )
    joint_vel = joint_pos + 1000.0
    anchor_pos_w = np.stack(
        [
            np.arange(frames, dtype=np.float32) + offset,
            np.zeros(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
        ],
        axis=1,
    )
    anchor_quat_w = np.tile(
        np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
    )
    return TextOpMotionBlock(
        index=index,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        anchor_pos_w=anchor_pos_w,
        anchor_quat_w=anchor_quat_w,
    )


def fake_env(
    num_envs: int = 1,
    robot_anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    body_pos = torch.tensor([[robot_anchor_pos]], dtype=torch.float32).repeat(
        num_envs, 1, 1
    )
    robot = SimpleNamespace(
        body_names=["pelvis"],
        data=SimpleNamespace(
            body_link_pos_w=body_pos,
            body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(
                num_envs, 1, 1
            ),
        ),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": robot},
    )


def write_mjlab_motion_npz(path, frames: int = 10, bodies: int = 1):
    joint_pos = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29)
    joint_vel = joint_pos + 1000.0
    body_pos_w = np.zeros((frames, bodies, 3), dtype=np.float32)
    body_quat_w = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, bodies, 1)
    )
    np.savez(
        path,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
    )
    return joint_pos, joint_vel, body_pos_w, body_quat_w
