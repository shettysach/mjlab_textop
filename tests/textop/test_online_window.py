from __future__ import annotations

import torch
from builders import motion_block

from mjlab_textop.core.online.buffer import RollingMotionBuffer
from mjlab_textop.core.online.window import OnlineReferenceWindow


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
