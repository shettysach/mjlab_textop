from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from builders import motion_block

from mjlab_textop.core.online.buffer import RollingMotionBuffer
from mjlab_textop.core.online.window import OnlineReferenceWindow
from mjlab_textop.trackers.spec import ReferenceWindowSpec


def test_reference_window_aligns_and_caches_future_motion() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(frames=8, offset=5.0))
    windows = OnlineReferenceWindow(
        buffer,
        num_envs=1,
        device="cpu",
        future_steps=5,
    )
    robot_anchor = torch.tensor([[10.0, 20.0, 30.0]])
    windows.align(0, robot_anchor)

    first = windows.get(0)
    second = windows.get(0)

    assert second is first
    torch.testing.assert_close(first.anchor_pos_w[0], robot_anchor[0])
    assert windows.last_stale_steps == 0


def test_reference_window_tracks_stale_frames_once_per_consumer_frame() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(frames=3))
    windows = OnlineReferenceWindow(
        buffer,
        num_envs=1,
        device="cpu",
        future_steps=5,
    )
    windows.align(0, torch.zeros(1, 3))

    assert windows.get(0).stale_steps == 2
    windows.clear_cache()
    assert windows.get(0).stale_steps == 2
    assert windows.consecutive_stale_steps == 1

    windows.clear_cache()
    assert windows.get(1).stale_steps == 3
    assert windows.consecutive_stale_steps == 2


def test_reference_window_reset_clears_alignment_cache_and_diagnostics() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(frames=3))
    windows = OnlineReferenceWindow(
        buffer,
        num_envs=1,
        device="cpu",
        future_steps=5,
    )
    windows.align(0, torch.ones(1, 3))
    windows.get(0)

    windows.reset()

    assert windows.cached_for(0) is None
    assert windows.last_stale_steps == 0
    assert windows.consecutive_stale_steps == 0
    torch.testing.assert_close(windows.robot_start_anchor_pos_w, torch.zeros(1, 3))


def test_reference_window_derives_world_root_velocity_from_adjacent_poses() -> None:
    block = motion_block(frames=2)
    angle = 0.1
    block = replace(
        block,
        motion=replace(
            block.motion,
            anchor_quat_w=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)],
                ],
                dtype=np.float32,
            ),
        ),
    )
    buffer = RollingMotionBuffer()
    buffer.append_block(block)
    windows = OnlineReferenceWindow(
        buffer,
        num_envs=1,
        device="cpu",
        future_steps=5,
    )

    expected = torch.tensor([50.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    torch.testing.assert_close(
        windows.reference_root_velocity(0, dt=0.02),
        expected,
    )
    torch.testing.assert_close(
        windows.reference_root_velocity(1, dt=0.02),
        expected,
    )


def test_reference_window_uses_zero_velocity_without_an_adjacent_pose() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(frames=1))
    windows = OnlineReferenceWindow(
        buffer,
        num_envs=1,
        device="cpu",
        future_steps=5,
    )

    torch.testing.assert_close(
        windows.reference_root_velocity(0, dt=0.02),
        torch.zeros(6),
    )


def test_reference_window_can_align_reference_heading_to_robot() -> None:
    half_sqrt = np.sqrt(0.5)
    block = motion_block(frames=2)
    block = replace(
        block,
        motion=replace(
            block.motion,
            anchor_quat_w=np.tile(
                np.array([half_sqrt, 0.0, 0.0, half_sqrt], dtype=np.float32),
                (2, 1),
            ),
        ),
    )
    buffer = RollingMotionBuffer()
    buffer.append_block(block)
    windows = OnlineReferenceWindow(
        buffer,
        num_envs=1,
        device="cpu",
        spec=ReferenceWindowSpec(
            frame_offsets=(0, 1),
            align_heading=True,
        ),
    )
    robot_pos = torch.tensor([[10.0, 20.0, 30.0]])
    robot_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    windows.align(0, robot_pos, robot_quat)
    window = windows.get(0)

    torch.testing.assert_close(window.anchor_pos_w[0], robot_pos[0])
    torch.testing.assert_close(
        window.anchor_pos_w[1],
        torch.tensor([10.0, 21.0, 30.0]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        window.anchor_quat_w[0],
        robot_quat[0],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
