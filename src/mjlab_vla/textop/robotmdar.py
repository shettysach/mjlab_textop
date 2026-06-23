from __future__ import annotations

from typing import Any

import numpy as np

from mjlab_vla.textop.contract import TEXTOP_G1_JOINT_COUNT
from mjlab_vla.textop.motion import reindex_mjlab_g1_joints_to_textop
from mjlab_vla.textop.online.source import TextOpMotionBlock

ROBOTMDAR_DOF_COUNT = 23
ROBOTMDAR_G1_DOF_INDEX: tuple[int, ...] = (
    *range(19),
    22,
    23,
    24,
    25,
)


def expand_robotmdar_dof_to_mjlab_g1(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ROBOTMDAR_DOF_COUNT:
        raise ValueError(
            f"Expected [T, {ROBOTMDAR_DOF_COUNT}] RobotMDAR DoF array, "
            f"got {value.shape}"
        )

    out = np.zeros((value.shape[0], TEXTOP_G1_JOINT_COUNT), dtype=np.float32)
    out[:, ROBOTMDAR_G1_DOF_INDEX] = value
    return out


def robotmdar_motion_dict_to_block(
    motion_dict: dict[str, Any],
    *,
    index: int,
) -> TextOpMotionBlock:
    joint_pos_mjlab = expand_robotmdar_dof_to_mjlab_g1(
        _to_numpy(motion_dict["dof_pos"][0])
    )
    joint_vel_mjlab = expand_robotmdar_dof_to_mjlab_g1(
        _to_numpy(motion_dict["dof_vel"][0])
    )
    root_rot_xyzw = _to_numpy(motion_dict["root_rot"][0])

    return TextOpMotionBlock(
        index=index,
        joint_pos=reindex_mjlab_g1_joints_to_textop(joint_pos_mjlab),
        joint_vel=reindex_mjlab_g1_joints_to_textop(joint_vel_mjlab),
        anchor_pos_w=_to_numpy(motion_dict["root_trans_offset"][0]),
        anchor_quat_w=root_rot_xyzw[:, [3, 0, 1, 2]],
    )


def slice_motion_dict_tail(
    motion_dict: dict[str, Any],
    frames: int,
) -> dict[str, Any]:
    result = {}
    for key, value in motion_dict.items():
        if hasattr(value, "shape") and len(value.shape) >= 2:
            result[key] = value[:, -frames:]
        else:
            result[key] = value
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
