from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_vla.textop.contract import TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
from mjlab_vla.textop.mdp.online_commands import (
    OnlineTextOpMotionCommand,
    OnlineTextOpMotionCommandCfg,
)
from mjlab_vla.textop.online import (
    QueueTextOpOnlineSource,
    TextOpMotionBlock,
    TextOpRollingMotionBuffer,
)


def _motion_block(index: int = 0, frames: int = 8, offset: float = 0.0) -> TextOpMotionBlock:
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


def test_rolling_buffer_reindexes_and_slices_first_five_frames() -> None:
    block = _motion_block(frames=8)
    buffer = TextOpRollingMotionBuffer()

    buffer.append_block(block)
    joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = (
        buffer.get_future(0, 5)
    )

    assert stale_steps == 0
    assert joint_pos.shape == (5, 29)
    expected = block.joint_pos[:5, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)
    np.testing.assert_allclose(
        joint_vel.cpu().numpy(),
        block.joint_vel[:5, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)],
    )
    np.testing.assert_allclose(anchor_pos_w.cpu().numpy(), block.anchor_pos_w[:5])
    np.testing.assert_allclose(
        anchor_quat_w.cpu().numpy(),
        np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
    )


def test_rolling_buffer_overwrites_overlapping_block_frames() -> None:
    buffer = TextOpRollingMotionBuffer()
    buffer.append_block(_motion_block(index=0, frames=8, offset=0.0))
    buffer.append_block(_motion_block(index=4, frames=3, offset=5000.0))

    joint_pos, _, _, _, stale_steps = buffer.get_future(3, 4)

    assert stale_steps == 0
    expected_source = np.concatenate(
        [
            _motion_block(index=0, frames=8, offset=0.0).joint_pos[3:4],
            _motion_block(index=4, frames=3, offset=5000.0).joint_pos[:3],
        ],
        axis=0,
    )
    expected = expected_source[:, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)


def test_rolling_buffer_requires_contiguous_start_window() -> None:
    buffer = TextOpRollingMotionBuffer()
    buffer.append_block(_motion_block(index=1, frames=4))

    assert buffer.can_start(0, 5) is False
    assert buffer.can_start(1, 4) is True


def test_rolling_buffer_repeats_latest_available_frame_on_underrun() -> None:
    buffer = TextOpRollingMotionBuffer()
    block = _motion_block(index=0, frames=5)
    buffer.append_block(block)

    joint_pos, _, _, _, stale_steps = buffer.get_future(3, 5)

    assert stale_steps == 3
    expected_source = np.stack(
        [block.joint_pos[3], block.joint_pos[4], block.joint_pos[4], block.joint_pos[4], block.joint_pos[4]],
        axis=0,
    )
    expected = expected_source[:, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)


def test_rolling_buffer_evicts_old_frames() -> None:
    buffer = TextOpRollingMotionBuffer(max_frames=5)
    buffer.append_block(_motion_block(index=0, frames=8))

    assert buffer.frame_count == 5
    assert buffer.can_start(0, 5) is False
    assert buffer.can_start(3, 5) is True


def test_rolling_buffer_rejects_wrong_joint_count() -> None:
    block = _motion_block(frames=1)
    bad = TextOpMotionBlock(
        index=0,
        joint_pos=np.zeros((1, 28), dtype=np.float32),
        joint_vel=block.joint_vel,
        anchor_pos_w=block.anchor_pos_w,
        anchor_quat_w=block.anchor_quat_w,
    )

    with pytest.raises(ValueError, match="29 joints"):
        TextOpRollingMotionBuffer().append_block(bad)


def test_online_command_polls_source_and_exposes_five_step_window() -> None:
    source = QueueTextOpOnlineSource([_motion_block(frames=8)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(source=source, future_steps=5),
        _fake_env(),
    )

    command._update_command()

    assert command.future_joint_pos.shape == (1, 5, 29)
    assert command.future_joint_vel.shape == (1, 5, 29)
    assert command.future_anchor_pos_w.shape == (1, 5, 3)
    assert command.future_anchor_quat_w.shape == (1, 5, 4)
    assert command.joint_pos.shape == (1, 29)
    assert command.joint_vel.shape == (1, 29)
    assert command.anchor_pos_w.shape == (1, 3)
    assert command.anchor_quat_w.shape == (1, 4)
    assert command.current_frame == 0

    command._update_command()
    assert command.current_frame == 1


def test_online_command_rejects_vectorized_envs() -> None:
    source = QueueTextOpOnlineSource([_motion_block(frames=8)])

    with pytest.raises(ValueError, match="one environment"):
        OnlineTextOpMotionCommand(
            OnlineTextOpMotionCommandCfg(source=source),
            _fake_env(num_envs=2),
        )


def test_online_command_rejects_too_many_consecutive_stale_windows() -> None:
    source = QueueTextOpOnlineSource([_motion_block(frames=5)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            future_steps=5,
            max_stale_steps=1,
        ),
        _fake_env(),
    )
    command._update_command()

    command._update_command()
    command._update_command()
    command._update_command()
    _ = command.future_joint_pos

    command._update_command()
    with pytest.raises(RuntimeError, match="max consecutive stale"):
        _ = command.future_joint_pos


def _fake_env(num_envs: int = 1):
    robot = SimpleNamespace(
        body_names=["pelvis"],
        data=SimpleNamespace(
            body_link_pos_w=torch.zeros(num_envs, 1, 3),
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
