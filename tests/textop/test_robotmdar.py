from __future__ import annotations

import numpy as np
import pytest

from mjlab_vla.textop.motion import reindex_mjlab_g1_joints_to_textop
from mjlab_vla.textop.robotmdar import (
    ROBOTMDAR_G1_DOF_INDEX,
    expand_robotmdar_dof_to_mjlab_g1,
    robotmdar_motion_dict_to_block,
    slice_motion_dict_tail,
)


def test_expand_robotmdar_dof_to_mjlab_g1_places_known_dofs() -> None:
    robotmdar_dof = np.arange(2 * 23, dtype=np.float32).reshape(2, 23)

    mjlab_dof = expand_robotmdar_dof_to_mjlab_g1(robotmdar_dof)

    assert mjlab_dof.shape == (2, 29)
    np.testing.assert_allclose(mjlab_dof[:, ROBOTMDAR_G1_DOF_INDEX], robotmdar_dof)
    np.testing.assert_allclose(mjlab_dof[:, 19:22], 0.0)
    np.testing.assert_allclose(mjlab_dof[:, 26:29], 0.0)


def test_expand_robotmdar_dof_to_mjlab_g1_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"Expected \[T, 23\] RobotMDAR DoF array"):
        expand_robotmdar_dof_to_mjlab_g1(np.zeros((2, 22), dtype=np.float32))


def test_robotmdar_motion_dict_to_block_converts_to_textop_block() -> None:
    dof_pos = np.arange(3 * 23, dtype=np.float32).reshape(1, 3, 23)
    dof_vel = dof_pos + 1000.0
    root_rot_xyzw = np.tile(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (1, 3, 1),
    )
    root_trans_offset = np.arange(9, dtype=np.float32).reshape(1, 3, 3)

    block = robotmdar_motion_dict_to_block(
        {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_rot": root_rot_xyzw,
            "root_trans_offset": root_trans_offset,
        },
        index=11,
    )

    expected_mjlab_pos = expand_robotmdar_dof_to_mjlab_g1(dof_pos[0])
    expected_mjlab_vel = expand_robotmdar_dof_to_mjlab_g1(dof_vel[0])
    assert block.index == 11
    np.testing.assert_allclose(
        block.joint_pos, reindex_mjlab_g1_joints_to_textop(expected_mjlab_pos)
    )
    np.testing.assert_allclose(
        block.joint_vel, reindex_mjlab_g1_joints_to_textop(expected_mjlab_vel)
    )
    np.testing.assert_allclose(block.anchor_pos_w, root_trans_offset[0])
    np.testing.assert_allclose(
        block.anchor_quat_w,
        np.tile(np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float32), (3, 1)),
    )


def test_slice_motion_dict_tail_slices_batched_time_arrays() -> None:
    batched = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)
    scalar = object()

    result = slice_motion_dict_tail({"batched": batched, "scalar": scalar}, 2)

    np.testing.assert_allclose(result["batched"], batched[:, -2:])
    assert result["scalar"] is scalar
