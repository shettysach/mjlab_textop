from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from builders import fake_env, motion_block, write_mjlab_motion_npz

from mjlab_textop.core.feedback.observation import (
    OnlineObservationCfg,
    OnlineObservationState,
    make_torso_observation_camera,
)
from mjlab_textop.core.mdp.online_commands import (
    OnlineMotionCommand,
    OnlineMotionCommandCfg,
    OnlineObservationReporter,
    _contains_geom_pair,
    _find_collision_geom_ids,
    use_online_textop_motion_command,
)
from mjlab_textop.core.online.buffer import (
    MotionBlock,
    RollingMotionBuffer,
)
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.replay import (
    QueueOnlineSource,
    make_mjlab_npz_replay_source,
)
from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class _LiveTextOpOnlineSource:
    def __init__(self, blocks: list[MotionBlock]) -> None:
        self.blocks = list(blocks)

    def poll(self) -> MotionBlock | None:
        if not self.blocks:
            return None
        return self.blocks.pop(0)


class _RecordingObservationPublisher:
    def __init__(self) -> None:
        self.images = []

    def publish(self, *, image=None) -> None:
        self.images.append(image)


class _BlockingObservationPublisher:
    def __init__(
        self,
        *,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        self.started = started
        self.release = release
        self.publish_count = 0

    def publish(self, *, image=None) -> None:
        del image
        self.publish_count += 1
        self.started.set()
        assert self.release.wait(timeout=1.0)


class _RecordingDebugVisualizer:
    env_idx = 0
    show_all_envs = False

    def __init__(self) -> None:
        self.ghosts = []

    def get_env_indices(self, num_envs: int):
        del num_envs
        return [0]

    def add_ghost_mesh(
        self,
        qpos,
        *,
        model,
        mocap_pos=None,
        mocap_quat=None,
        alpha=0.5,
        label=None,
    ):
        self.ghosts.append(
            {
                "qpos": qpos.copy(),
                "model": model,
                "mocap_pos": mocap_pos,
                "mocap_quat": mocap_quat,
                "alpha": alpha,
                "label": label,
            }
        )


def test_rolling_buffer_reindexes_and_slices_first_five_frames() -> None:
    block = motion_block(frames=8)
    buffer = RollingMotionBuffer()

    buffer.append_block(block)
    joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = buffer.get_future(
        0, 5
    )

    assert stale_steps == 0
    assert joint_pos.shape == (5, 29)
    expected = block.joint_pos[:5, list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)
    np.testing.assert_allclose(
        joint_vel.cpu().numpy(),
        block.joint_vel[:5, list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)],
    )
    np.testing.assert_allclose(anchor_pos_w.cpu().numpy(), block.anchor_pos_w[:5])
    np.testing.assert_allclose(
        anchor_quat_w.cpu().numpy(),
        np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
    )


def test_rolling_buffer_overwrites_overlapping_block_frames() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(index=0, frames=8, offset=0.0))
    buffer.append_block(motion_block(index=4, frames=3, offset=5000.0))

    joint_pos, _, _, _, stale_steps = buffer.get_future(3, 4)

    assert stale_steps == 0
    expected_source = np.concatenate(
        [
            motion_block(index=0, frames=8, offset=0.0).joint_pos[3:4],
            motion_block(index=4, frames=3, offset=5000.0).joint_pos[:3],
        ],
        axis=0,
    )
    expected = expected_source[:, list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)


def test_rolling_buffer_requires_contiguous_start_window() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(index=1, frames=4))

    assert buffer.can_start(0, 5) is False
    assert buffer.can_start(1, 4) is True


def test_rolling_buffer_finds_earliest_contiguous_start_window() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(index=100, frames=3))

    assert buffer.earliest_start_frame(5) is None

    buffer.append_block(motion_block(index=103, frames=5))

    assert buffer.earliest_start_frame(5) == 100


def test_rolling_buffer_finds_latest_contiguous_start_window() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(index=100, frames=8))
    buffer.append_block(motion_block(index=200, frames=3))

    assert buffer.latest_start_frame(5) == 103

    buffer.append_block(motion_block(index=203, frames=5))

    assert buffer.latest_start_frame(5) == 203


def test_rolling_buffer_repeats_latest_available_frame_on_underrun() -> None:
    buffer = RollingMotionBuffer()
    block = motion_block(index=0, frames=5)
    buffer.append_block(block)

    joint_pos, _, _, _, stale_steps = buffer.get_future(3, 5)

    assert stale_steps == 3
    expected_source = np.stack(
        [
            block.joint_pos[3],
            block.joint_pos[4],
            block.joint_pos[4],
            block.joint_pos[4],
            block.joint_pos[4],
        ],
        axis=0,
    )
    expected = expected_source[:, list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)


def test_rolling_buffer_rejects_request_before_earliest_frame() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(index=10, frames=5))

    with pytest.raises(RuntimeError, match="at or before 0"):
        buffer.get_future(0, 5)


def test_rolling_buffer_discards_only_frames_behind_consumer() -> None:
    buffer = RollingMotionBuffer()
    buffer.append_block(motion_block(index=0, frames=8))
    buffer.discard_before(3)

    assert buffer.frame_count == 5
    assert buffer.can_start(0, 5) is False
    assert buffer.can_start(3, 5) is True


def test_rolling_buffer_rejects_wrong_joint_count() -> None:
    block = motion_block(frames=1)
    bad = MotionBlock(
        index=0,
        joint_pos=np.zeros((1, 28), dtype=np.float32),
        joint_vel=block.joint_vel,
        anchor_pos_w=block.anchor_pos_w,
        anchor_quat_w=block.anchor_quat_w,
    )

    with pytest.raises(ValueError, match="29 joints"):
        RollingMotionBuffer().append_block(bad)


def test_mjlab_npz_replay_source_chunks_and_round_trips_joint_order(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    joint_pos, _, _, _ = write_mjlab_motion_npz(path, frames=10)

    source = make_mjlab_npz_replay_source(path, block_size=8)
    assert source.fps == 50.0
    buffer = RollingMotionBuffer()
    while (block := source.poll()) is not None:
        buffer.append_block(block)

    round_trip_joint_pos, _, _, _, stale_steps = buffer.get_future(0, 10)

    assert stale_steps == 0
    np.testing.assert_allclose(round_trip_joint_pos.cpu().numpy(), joint_pos)


def test_online_command_polls_source_and_exposes_five_step_window() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(),
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


def test_online_command_debug_vis_skips_before_start() -> None:
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=QueueOnlineSource(), future_steps=5),
        fake_env(),
    )
    visualizer = _RecordingDebugVisualizer()

    command._debug_vis_impl(visualizer)

    assert visualizer.ghosts == []


def test_online_command_debug_vis_draws_reference_ghost_qpos() -> None:
    block = motion_block(frames=8)
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=QueueOnlineSource([block]),
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()
    visualizer = _RecordingDebugVisualizer()

    command._debug_vis_impl(visualizer)

    assert len(visualizer.ghosts) == 1
    ghost = visualizer.ghosts[0]
    assert ghost["alpha"] == 0.5
    assert ghost["label"] == "online_reference_0"
    np.testing.assert_allclose(ghost["qpos"][:3], command.anchor_pos_w[0].numpy())
    np.testing.assert_allclose(ghost["qpos"][3:7], command.anchor_quat_w[0].numpy())
    np.testing.assert_allclose(ghost["qpos"][7:], command.joint_pos[0].numpy())
    np.testing.assert_allclose(ghost["model"].geom_rgba[0], [0.5, 0.7, 0.5, 0.5])
    assert ghost["model"].geom_rgba[1, 3] == 0.0


def test_online_command_reuses_future_window_for_current_frame() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(),
    )
    command._update_command()

    calls = 0
    get_future = command.buffer.get_future

    def counting_get_future(*args, **kwargs):
        nonlocal calls
        calls += 1
        return get_future(*args, **kwargs)

    command.buffer.get_future = counting_get_future

    _ = command.future_joint_pos
    _ = command.future_joint_vel
    _ = command.future_anchor_pos_w
    _ = command.command

    assert calls == 1


def test_online_command_latches_collision_and_holds_last_safe_reference(
    monkeypatch,
) -> None:
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=QueueOnlineSource([motion_block(frames=8)]),
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()
    safe_joint_pos = command.joint_pos.clone()
    safe_anchor_pos = command.anchor_pos_w.clone()
    safe_anchor_quat = command.anchor_quat_w.clone()
    monkeypatch.setattr(
        OnlineMotionCommand,
        "_has_obstacle_collision",
        lambda self: True,
    )

    command._update_command()

    assert command.collision_stop is True
    assert command.current_frame == 0
    torch.testing.assert_close(
        command.future_joint_pos,
        safe_joint_pos[:, None, :].expand(-1, 5, -1),
    )
    torch.testing.assert_close(command.future_joint_vel, torch.zeros(1, 5, 29))
    torch.testing.assert_close(
        command.future_anchor_pos_w,
        safe_anchor_pos[:, None, :].expand(-1, 5, -1),
    )
    torch.testing.assert_close(
        command.future_anchor_quat_w,
        safe_anchor_quat[:, None, :].expand(-1, 5, -1),
    )

    monkeypatch.setattr(
        OnlineMotionCommand,
        "_has_obstacle_collision",
        lambda self: False,
    )
    command._update_command()

    assert command.collision_stop is True
    assert command.current_frame == 0


def test_online_command_clears_collision_latch_on_reset(monkeypatch) -> None:
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=QueueOnlineSource([motion_block(frames=8)]),
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()
    _ = command.future_joint_pos
    monkeypatch.setattr(
        OnlineMotionCommand,
        "_has_obstacle_collision",
        lambda self: True,
    )
    command._update_command()
    assert command.collision_stop is True

    command._resample_command(torch.tensor([0]))

    assert command.collision_stop is False
    assert command._collision_hold_window is None


def test_collision_geom_pair_matches_either_contact_order() -> None:
    contacts = torch.tensor([[5, 9], [2, 3], [7, 4]])
    robot_ids = torch.tensor([4, 5])
    obstacle_ids = torch.tensor([7, 9])

    assert _contains_geom_pair(
        contacts,
        contact_count=1,
        first_ids=robot_ids,
        second_ids=obstacle_ids,
    )
    assert _contains_geom_pair(
        contacts,
        contact_count=3,
        first_ids=robot_ids,
        second_ids=obstacle_ids,
    )
    assert not _contains_geom_pair(
        contacts,
        contact_count=0,
        first_ids=robot_ids,
        second_ids=obstacle_ids,
    )


def test_collision_geom_ids_select_robot_and_named_obstacles() -> None:
    geom_names = ["robot/torso_collision", "corridor_left_collision", "ground"]
    body_names = ["world", "robot/torso_link", "corridor_left", "ground"]
    model = SimpleNamespace(
        ngeom=3,
        geom_bodyid=np.array([1, 2, 3]),
        geom=lambda index: SimpleNamespace(name=geom_names[index]),
        body=lambda index: SimpleNamespace(name=body_names[index]),
    )

    robot_ids, obstacle_ids = _find_collision_geom_ids(
        model,
        entity_name="robot",
        obstacle_suffix="_collision",
        device="cpu",
    )

    torch.testing.assert_close(robot_ids, torch.tensor([0]))
    torch.testing.assert_close(obstacle_ids, torch.tensor([1]))


def test_online_command_exposes_startup_window_before_source_poll() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    assert command.future_joint_pos.shape == (1, 5, 29)
    assert command.future_joint_vel.shape == (1, 5, 29)
    assert command.future_anchor_pos_w.shape == (1, 5, 3)
    assert command.future_anchor_quat_w.shape == (1, 5, 4)
    assert command.buffer.frame_count == 0
    torch.testing.assert_close(command.future_joint_pos, torch.zeros(1, 5, 29))
    torch.testing.assert_close(
        command.future_anchor_pos_w,
        torch.tensor([[[10.0, 20.0, 30.0]]]).expand(1, 5, 3),
    )


def test_online_command_aligns_future_anchors_to_fixed_start_frame() -> None:
    source = QueueOnlineSource([motion_block(frames=8, offset=100.0)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    command._update_command()

    future_anchor_pos = command.future_anchor_pos_w[0]
    torch.testing.assert_close(future_anchor_pos[0], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(future_anchor_pos[1], torch.tensor([11.0, 20.0, 30.0]))


def test_online_command_fixed_start_alignment_preserves_reference_z_delta() -> None:
    block = motion_block(frames=8, offset=100.0)
    block.anchor_pos_w[:, 2] = np.arange(8, dtype=np.float32) + 2.0
    source = QueueOnlineSource([block])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    command._update_command()

    future_anchor_pos = command.future_anchor_pos_w[0]
    torch.testing.assert_close(future_anchor_pos[0], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(future_anchor_pos[1], torch.tensor([11.0, 20.0, 31.0]))


def test_online_command_fixed_start_alignment_preserves_future_xy_deltas() -> None:
    block = motion_block(frames=8)
    block.anchor_pos_w[:, :] = np.array(
        [
            [100.0, 200.0, 2.0],
            [100.0, 201.0, 3.0],
            [99.0, 201.0, 4.0],
            [99.0, 200.0, 5.0],
            [100.0, 200.0, 6.0],
            [101.0, 200.0, 7.0],
            [101.0, 201.0, 8.0],
            [100.0, 201.0, 9.0],
        ],
        dtype=np.float32,
    )
    block.anchor_quat_w[:] = np.array(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        dtype=np.float32,
    )
    source = QueueOnlineSource([block])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    command._update_command()
    command._update_command()

    expected_pos = torch.tensor(
        [
            [
                [10.0, 21.0, 31.0],
                [9.0, 21.0, 32.0],
                [9.0, 20.0, 33.0],
                [10.0, 20.0, 34.0],
                [11.0, 20.0, 35.0],
            ]
        ]
    )

    torch.testing.assert_close(command.future_anchor_pos_w, expected_pos)
    torch.testing.assert_close(
        command.future_anchor_quat_w,
        torch.tensor(block.anchor_quat_w[1:6])[None, :, :],
    )


def test_online_command_rejects_vectorized_envs() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])

    with pytest.raises(ValueError, match="one environment"):
        OnlineMotionCommand(
            OnlineMotionCommandCfg(source=source),
            fake_env(num_envs=2),
        )


def test_online_command_counts_consecutive_stale_windows() -> None:
    source = QueueOnlineSource([motion_block(frames=5)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()

    command._update_command()
    command._update_command()
    command._update_command()
    _ = command.future_joint_pos

    command._update_command()
    future = command.future_joint_pos
    command._update_metrics()

    assert command._consecutive_stale_steps > 0
    assert future.shape == (1, 5, 29)


def test_online_command_replay_allows_stale_windows_at_clip_end() -> None:
    source = QueueOnlineSource([motion_block(frames=5)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()

    for _ in range(4):
        command._update_command()
        _ = command.future_joint_pos

    assert command._consecutive_stale_steps > 0
    assert command.future_joint_pos.shape == (1, 5, 29)


def test_online_command_replay_does_not_evict_preloaded_frames() -> None:
    blocks = [
        motion_block(
            index=block_index * 64, frames=64, offset=float(block_index * 1000)
        )
        for block_index in range(16)
    ]
    source = QueueOnlineSource(blocks)
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
            max_poll_blocks=16,
        ),
        fake_env(),
    )

    command._update_command()

    assert command._started is True
    assert command.current_frame == 0
    assert command.buffer.earliest_index == 0
    assert command.buffer.frame_count == 1024
    assert command.future_joint_pos.shape == (1, 5, 29)


def test_online_command_rejects_replay_source_without_reset() -> None:
    with pytest.raises(TypeError, match="implement reset"):
        OnlineMotionCommandCfg(
            source=_LiveTextOpOnlineSource([motion_block(frames=8)]),
            source_mode="replay",
        )


def test_online_command_rejects_replay_source_fps_mismatch() -> None:
    source = QueueOnlineSource([motion_block(frames=8)], fps=25.0)

    with pytest.raises(ValueError, match="FPS must match env control rate"):
        OnlineMotionCommand(
            OnlineMotionCommandCfg(
                source=source,
                source_mode="replay",
                future_steps=5,
            ),
            fake_env(step_dt=0.02),
        )


def test_online_command_rejects_live_source_fps_mismatch() -> None:
    source = QueueOnlineSource([motion_block(frames=8)], fps=25.0)

    with pytest.raises(ValueError, match="FPS must match env control rate"):
        OnlineMotionCommand(
            OnlineMotionCommandCfg(
                source=source,
                source_mode="live",
                future_steps=5,
            ),
            fake_env(step_dt=0.02),
        )


def test_online_command_cfg_with_live_source_cfg_is_deepcopyable() -> None:
    cfg = OnlineMotionCommandCfg(
        live_source_cfg=SocketSourceCfg(
            host="127.0.0.1",
            port=8765,
            fps=50.0,
            max_queue_blocks=4,
        ),
        source_mode="live",
        future_steps=5,
    )

    copied = copy.deepcopy(cfg)

    assert copied.live_source_cfg == cfg.live_source_cfg
    assert copied.source is None


def test_online_command_creates_live_socket_source_from_cfg(monkeypatch) -> None:
    created = []

    class _FakeSocketSource:
        fps = 50.0
        diagnostics = SimpleNamespace(
            queue_depth=0,
            blocks_received=0,
            blocks_dropped=0,
            bad_messages=0,
        )

        def __init__(self, cfg):
            self.cfg = cfg
            self.started = False
            created.append(self)

        def start(self) -> None:
            self.started = True

        def poll(self):
            return None

    monkeypatch.setattr(
        "mjlab_textop.core.mdp.online_commands.SocketOnlineSource",
        _FakeSocketSource,
    )
    live_source_cfg = SocketSourceCfg(port=8765)

    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            live_source_cfg=live_source_cfg,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )

    assert len(created) == 1
    assert created[0].cfg == live_source_cfg
    assert created[0].started is True
    assert command.source is created[0]


def test_online_command_updates_live_diagnostics_metrics() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    source.diagnostics = SimpleNamespace(
        queue_depth=3,
        blocks_received=4,
        blocks_dropped=1,
        bad_messages=2,
    )
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=source, future_steps=5),
        fake_env(),
    )

    command._update_command()
    command._update_metrics()

    assert command.metrics["online_started"].item() == 1.0
    assert command.metrics["online_current_frame"].item() == 0.0
    assert command.metrics["online_latest_frame"].item() == 7.0
    assert command.metrics["online_lag_frames"].item() == 7.0
    assert command.metrics["online_queue_depth"].item() == 3.0
    assert command.metrics["online_blocks_received"].item() == 4.0
    assert command.metrics["online_blocks_dropped"].item() == 1.0
    assert command.metrics["online_bad_messages"].item() == 2.0


def test_online_command_publishes_observations_with_images_on_interval(
    monkeypatch,
) -> None:
    publisher = _RecordingObservationPublisher()
    monkeypatch.setattr(
        "mjlab_textop.core.feedback.online_reporter.OnlineObservationReporter._render_image",
        lambda self: np.zeros((1, 1, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "mjlab_textop.core.feedback.online_reporter.encode_render_image_jpeg",
        lambda image: b"jpeg",
    )
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=QueueOnlineSource([motion_block(frames=8)]),
            future_steps=5,
            observation=OnlineObservationCfg(
                publisher=publisher,
                publish_interval=2,
            ),
        ),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    command._update_command()
    command._update_metrics()
    _wait_for_observation_publish(command.observation_reporter)
    command._update_command()
    command._update_metrics()
    command._update_command()
    command._update_metrics()
    _wait_for_observation_publish(command.observation_reporter)

    assert [image.data for image in publisher.images] == [b"jpeg", b"jpeg"]


def test_online_command_does_not_create_observation_reporter_by_default() -> None:
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=QueueOnlineSource([motion_block(frames=8)]),
            future_steps=5,
        ),
        fake_env(),
    )

    assert command.cfg.observation is None
    assert command.observation_reporter is None


def test_online_observation_reporter_drops_observation_while_publish_inflight(
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    publisher = _BlockingObservationPublisher(started=started, release=release)
    render_calls = []

    monkeypatch.setattr(
        "mjlab_textop.core.feedback.online_reporter.OnlineObservationReporter._render_image",
        lambda self: render_calls.append(len(render_calls)) or np.zeros(
            (1, 1, 3),
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        "mjlab_textop.core.feedback.online_reporter.encode_render_image_jpeg",
        lambda image: b"jpeg",
    )
    reporter = OnlineObservationReporter(
        OnlineObservationCfg(
            publisher=publisher,
            publish_interval=2,
        ),
        fake_env(),
    )

    reporter.maybe_publish(_observation_state(frame=0))
    assert started.wait(timeout=1.0)
    reporter.maybe_publish(_observation_state(frame=2))

    assert len(render_calls) == 1
    assert publisher.publish_count == 1

    release.set()
    _wait_for_reporter_publish(reporter)
    reporter.maybe_publish(_observation_state(frame=3))
    reporter.maybe_publish(_observation_state(frame=4))
    _wait_for_reporter_publish(reporter)

    assert len(render_calls) == 2
    assert publisher.publish_count == 2


def test_online_observation_reporter_uses_observation_camera(monkeypatch) -> None:
    calls = {}

    class FakeOffscreenRenderer:
        def __init__(
            self,
            *,
            model,
            cfg,
            scene,
            sim_model,
            expanded_fields,
        ) -> None:
            del model, scene, sim_model, expanded_fields
            calls["cfg"] = cfg
            self._cam = SimpleNamespace(azimuth=cfg.azimuth)

        def initialize(self) -> None:
            calls["initialized"] = True

        def update(self, data, debug_vis_callback=None) -> None:
            del data, debug_vis_callback
            calls["updated"] = True
            calls["azimuth"] = self._cam.azimuth

        def render(self):
            return "rendered"

    monkeypatch.setattr(
        "mjlab_textop.core.feedback.online_reporter.OffscreenRenderer",
        FakeOffscreenRenderer,
    )
    env_viewer_camera = make_torso_observation_camera(width=999, height=999)
    observation_camera = make_torso_observation_camera(width=123, height=77)
    yaw_90_quat = torch.tensor(
        [[[0.7071068, 0.0, 0.0, 0.7071068]]],
        dtype=torch.float32,
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(viewer=env_viewer_camera),
        scene={
            "robot": SimpleNamespace(
                body_names=["torso_link"],
                data=SimpleNamespace(body_link_quat_w=yaw_90_quat),
            )
        },
        sim=SimpleNamespace(
            mj_model=object(),
            model=object(),
            data=object(),
            expanded_fields=set(),
        ),
    )
    reporter = OnlineObservationReporter(
        OnlineObservationCfg(camera=observation_camera),
        env,
    )

    assert reporter._render_image() == "rendered"
    assert calls["cfg"] is observation_camera
    assert calls["cfg"] is not env_viewer_camera
    assert calls["initialized"] is True
    assert calls["updated"] is True
    assert calls["azimuth"] == pytest.approx(90.0)


def _wait_for_observation_publish(reporter: OnlineObservationReporter | None) -> None:
    assert reporter is not None
    _wait_for_reporter_publish(reporter)


def _wait_for_reporter_publish(reporter: OnlineObservationReporter) -> None:
    assert reporter._publish_future is not None
    reporter._publish_future.result(timeout=1.0)
    reporter._collect_publish_result()


def _observation_state(*, frame: int) -> OnlineObservationState:
    return OnlineObservationState(
        frame=frame,
        started=True,
        latest_index=frame,
        lag_frames=0,
        buffer_frames=0,
        stale_steps=0,
        consecutive_stale_steps=0,
        robot_anchor_pos_w=torch.tensor([0.0, 0.0, 0.0]),
        robot_anchor_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )


def test_online_command_replay_reset_rewinds_source() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    env = fake_env()
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
        ),
        env,
    )
    command._update_command()

    assert command.buffer.frame_count == 8

    command._resample_command(torch.tensor([0]))

    assert command.buffer.frame_count == 8
    assert command.current_frame == 0
    assert command._started is True
    assert command.future_joint_pos.shape == (1, 5, 29)
    robot = env.scene["robot"]
    torch.testing.assert_close(
        robot.written_joint_pos,
        command.future_joint_pos[:, 0],
    )
    torch.testing.assert_close(
        robot.written_joint_vel,
        command.future_joint_vel[:, 0],
    )
    _, _, raw_anchor_pos, raw_anchor_quat, _ = command.buffer.get_future(
        command.current_frame, 1
    )
    expected_root = torch.cat(
        [command._fixed_start_reference_pos(raw_anchor_pos), raw_anchor_quat],
        dim=-1,
    )
    torch.testing.assert_close(robot.written_root_state[:, :7], expected_root)
    torch.testing.assert_close(robot.written_root_state[:, 7:], torch.zeros(1, 6))
    torch.testing.assert_close(robot.reset_env_ids, torch.tensor([0]))


def test_online_command_live_reset_does_not_rewind_source() -> None:
    source = _LiveTextOpOnlineSource([motion_block(frames=8)])
    env = fake_env()
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
            startup_timeout_steps=1,
        ),
        env,
    )
    command._update_command()

    assert command.buffer.frame_count == 8

    command._resample_command(torch.tensor([0]))

    assert command.buffer.frame_count == 8
    assert command.current_frame == 3
    assert command._started is True
    assert command.future_joint_pos.shape == (1, 5, 29)
    robot = env.scene["robot"]
    torch.testing.assert_close(
        robot.written_joint_pos,
        command.future_joint_pos[:, 0],
    )
    torch.testing.assert_close(robot.reset_env_ids, torch.tensor([0]))


def test_online_command_live_attaches_to_earliest_full_future_window() -> None:
    source = _LiveTextOpOnlineSource(
        [
            motion_block(index=100, frames=3),
            motion_block(index=103, frames=5),
        ]
    )
    env = fake_env(robot_anchor_pos=(10.0, 20.0, 30.0))
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        env,
    )

    command._update_command()

    assert command._started is True
    assert command.current_frame == 100
    assert command._has_started_once is True
    assert command._last_stale_steps == 0
    robot = env.scene["robot"]
    torch.testing.assert_close(
        robot.written_joint_pos,
        command.future_joint_pos[:, 0],
    )
    torch.testing.assert_close(
        robot.written_joint_vel,
        command.future_joint_vel[:, 0],
    )
    _, _, raw_anchor_pos, raw_anchor_quat, _ = command.buffer.get_future(
        command.current_frame, 1
    )
    expected_root = torch.cat(
        [command._fixed_start_reference_pos(raw_anchor_pos), raw_anchor_quat],
        dim=-1,
    )
    torch.testing.assert_close(robot.written_root_state[:, :7], expected_root)
    torch.testing.assert_close(robot.written_root_state[:, 7:], torch.zeros(1, 6))
    torch.testing.assert_close(robot.reset_env_ids, torch.tensor([0]))


def test_online_command_live_reset_before_first_start_uses_initial_window() -> None:
    source = _LiveTextOpOnlineSource(
        [
            motion_block(index=100, frames=8),
            motion_block(index=108, frames=8),
        ]
    )
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )

    command._resample_command(torch.tensor([0]))

    assert command._started is True
    assert command._has_started_once is True
    assert command.current_frame == 100
    assert command._last_stale_steps == 0


def test_online_command_live_reset_rejects_non_contiguous_stream() -> None:
    source = _LiveTextOpOnlineSource([motion_block(index=0, frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()

    source.blocks.extend(
        [
            motion_block(index=100, frames=3),
            motion_block(index=103, frames=5),
        ]
    )
    with pytest.raises(RuntimeError, match="Non-contiguous RobotMDAR live stream"):
        command._resample_command(torch.tensor([0]))


def test_online_command_live_pauses_before_advancing_into_stale_window() -> None:
    source = _LiveTextOpOnlineSource([motion_block(index=0, frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()

    for _ in range(4):
        command._update_command()

    assert command.current_frame == 3
    assert command.buffer.latest_index == 7
    assert command.future_joint_pos.shape == (1, 5, 29)
    assert command._last_stale_steps == 0

    source.blocks.append(motion_block(index=8, frames=1))
    command._update_command()

    assert command.current_frame == 4


def test_online_command_live_rejects_stream_jump_ahead() -> None:
    source = _LiveTextOpOnlineSource([motion_block(index=0, frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()

    for _ in range(3):
        command._update_command()

    source.blocks.extend(
        [
            motion_block(index=100, frames=3),
            motion_block(index=103, frames=5),
        ]
    )
    with pytest.raises(RuntimeError, match="Non-contiguous RobotMDAR live stream"):
        command._update_command()


def test_online_command_live_prunes_frames_behind_current_frame() -> None:
    source = _LiveTextOpOnlineSource([motion_block(index=0, frames=8)])
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()
    command._update_command()

    assert command.current_frame == 1
    assert command.buffer.earliest_index == 1


def test_online_command_live_polling_uses_hysteresis() -> None:
    command = OnlineMotionCommand(
        OnlineMotionCommandCfg(source=QueueOnlineSource(), source_mode="live"),
        fake_env(),
    )
    command.buffer.append_block(motion_block(index=0, frames=151))

    assert command._should_poll_live_source() is False

    command.current_frame = 101

    assert command._should_poll_live_source() is True


def test_use_online_textop_motion_command_preserves_injected_source() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    env_cfg = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(entity_name="robot", anchor_body_name="pelvis")
        }
    )

    use_online_textop_motion_command(env_cfg, source=source)

    assert env_cfg.commands["motion"].source is source
    assert env_cfg.commands["motion"].source_mode == "live"


def test_use_online_textop_motion_command_preserves_existing_debug_vis() -> None:
    env_cfg = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(
                entity_name="robot",
                anchor_body_name="pelvis",
                debug_vis=True,
            )
        }
    )

    use_online_textop_motion_command(env_cfg)

    assert env_cfg.commands["motion"].debug_vis is True


def test_use_online_textop_motion_command_can_override_debug_vis() -> None:
    env_cfg = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(
                entity_name="robot",
                anchor_body_name="pelvis",
                debug_vis=True,
            )
        }
    )

    use_online_textop_motion_command(env_cfg, debug_vis=False)

    assert env_cfg.commands["motion"].debug_vis is False


def test_use_online_textop_motion_command_can_disable_reference_reset() -> None:
    env_cfg = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(entity_name="robot", anchor_body_name="pelvis")
        }
    )

    use_online_textop_motion_command(
        env_cfg,
        source_mode="replay",
        reset_robot_to_reference=False,
    )

    assert env_cfg.commands["motion"].reset_robot_to_reference is False
