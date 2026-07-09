from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.mdp.offline_commands import (
    OfflineMotionCommand,
    OfflineMotionCommandCfg,
    make_future_time_steps,
    textop_motion_command_cfg_from,
    use_textop_motion_command,
)
from mjlab_textop.core.onnx_policy import OnnxPolicyRunner
from mjlab_textop.scripts import commands as commands_module
from mjlab_textop.scripts.commands import (
    ObservationParams,
    PlayLiveCommand,
    play_live_textop_motion,
)
from mjlab_textop.scripts.utils import ResolvedPolicy, resolve_policy


def test_make_future_time_steps_clamps_at_end() -> None:
    time_steps = torch.tensor([0, 7, 9], dtype=torch.long)

    future = make_future_time_steps(
        time_steps,
        future_steps=5,
        time_step_total=10,
    )

    assert future.tolist() == [
        [0, 1, 2, 3, 4],
        [7, 8, 9, 9, 9],
        [9, 9, 9, 9, 9],
    ]


def test_resolve_policy_accepts_checkpoint_file(tmp_path) -> None:
    checkpoint_file = tmp_path / "model.pt"
    checkpoint_file.write_text("checkpoint")

    policy = resolve_policy(
        checkpoint_file=str(checkpoint_file),
        onnx_file=None,
    )

    assert policy.runner_cls is MotionTrackingOnPolicyRunner
    assert policy.file == checkpoint_file.resolve()


def test_resolve_policy_accepts_onnx_file(tmp_path) -> None:
    onnx_file = tmp_path / "latest.onnx"
    onnx_file.write_text("onnx")

    policy = resolve_policy(
        checkpoint_file=None,
        onnx_file=str(onnx_file),
    )

    assert policy.runner_cls is OnnxPolicyRunner
    assert policy.file == onnx_file.resolve()


def test_resolve_policy_rejects_missing_policy() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_policy(checkpoint_file=None, onnx_file=None)


def test_resolve_policy_rejects_multiple_policies(tmp_path) -> None:
    checkpoint_file = tmp_path / "model.pt"
    onnx_file = tmp_path / "latest.onnx"
    checkpoint_file.write_text("checkpoint")
    onnx_file.write_text("onnx")

    with pytest.raises(ValueError, match="exactly one"):
        resolve_policy(
            checkpoint_file=str(checkpoint_file),
            onnx_file=str(onnx_file),
        )


def test_play_live_without_images_uses_mjlab_run_play(monkeypatch, tmp_path) -> None:
    calls = {}

    monkeypatch.setitem(
        commands_module.LIVE_TASK_REGISTRY,
        "default",
        lambda **kwargs: _fake_register_task(calls, kwargs),
    )
    monkeypatch.setattr(
        "mjlab_textop.scripts.commands.run_play",
        lambda task_name, play_cfg: calls.update(run_play=(task_name, play_cfg)),
    )
    policy_file = tmp_path / "policy.pt"
    policy_file.write_text("checkpoint")
    play_live_textop_motion(
        PlayLiveCommand(checkpoint_file=str(policy_file)),
        policy=ResolvedPolicy(MotionTrackingOnPolicyRunner, policy_file),
    )

    task_name, play_cfg = calls["run_play"]
    assert task_name == "task"
    assert play_cfg.video is False
    assert play_cfg.video_width is None
    assert play_cfg.video_height is None
    assert calls["task_kwargs"]["runner_cls"] is MotionTrackingOnPolicyRunner
    assert calls["task_kwargs"]["reference_debug_vis"] is False
    assert calls["task_kwargs"]["observation"] is None


def test_play_live_defaults_reference_debug_vis_off() -> None:
    assert PlayLiveCommand().reference_debug_vis is False
    assert PlayLiveCommand().observation is None
    assert ObservationParams().url == "http://127.0.0.1:8766/observation"


def test_play_live_with_images_does_not_enable_video_recording(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    monkeypatch.setitem(
        commands_module.LIVE_TASK_REGISTRY,
        "default",
        lambda **kwargs: _fake_register_task(calls, kwargs),
    )
    monkeypatch.setattr(
        "mjlab_textop.scripts.commands.run_play",
        lambda task_name, play_cfg: calls.update(run_play=(task_name, play_cfg)),
    )
    policy_file = tmp_path / "policy.pt"
    policy_file.write_text("checkpoint")
    play_live_textop_motion(
        PlayLiveCommand(
            checkpoint_file=str(policy_file),
            observation=ObservationParams(
                url="http://127.0.0.1:8766/observation",
            ),
        ),
        policy=ResolvedPolicy(MotionTrackingOnPolicyRunner, policy_file),
    )

    task_name, play_cfg = calls["run_play"]
    assert task_name == "task"
    assert play_cfg.video is False
    assert play_cfg.video_width == 320
    assert play_cfg.video_height == 240
    assert calls["task_kwargs"]["observation"].publisher is not None
    observation_camera = calls["task_kwargs"]["observation"].camera
    assert observation_camera.origin_type == observation_camera.OriginType.ASSET_BODY
    assert observation_camera.entity_name == "robot"
    assert observation_camera.body_name == "torso_link"
    assert observation_camera.width == 320
    assert observation_camera.height == 240
    assert observation_camera.distance == 2.0
    assert observation_camera.azimuth == 0.0
    assert observation_camera.elevation == -15.0


def test_play_live_uses_custom_observation_camera_geometry(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    monkeypatch.setitem(
        commands_module.LIVE_TASK_REGISTRY,
        "default",
        lambda **kwargs: _fake_register_task(calls, kwargs),
    )
    monkeypatch.setattr(
        "mjlab_textop.scripts.commands.run_play",
        lambda task_name, play_cfg: calls.update(run_play=(task_name, play_cfg)),
    )
    policy_file = tmp_path / "policy.pt"
    policy_file.write_text("checkpoint")
    play_live_textop_motion(
        PlayLiveCommand(
            checkpoint_file=str(policy_file),
            observation=ObservationParams(
                url="http://127.0.0.1:8766/observation",
                camera_distance=1.25,
                camera_azimuth=175.0,
                camera_elevation=-8.0,
            ),
        ),
        policy=ResolvedPolicy(MotionTrackingOnPolicyRunner, policy_file),
    )

    observation_camera = calls["task_kwargs"]["observation"].camera
    assert observation_camera.distance == 1.25
    assert observation_camera.azimuth == 175.0
    assert observation_camera.elevation == -8.0


def test_straight_live_uses_straight_task_registration(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    monkeypatch.setitem(
        commands_module.LIVE_TASK_REGISTRY,
        "straight",
        lambda **kwargs: _fake_register_task(calls, kwargs),
    )
    monkeypatch.setattr(
        "mjlab_textop.scripts.commands.run_play",
        lambda task_name, play_cfg: calls.update(run_play=(task_name, play_cfg)),
    )
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    play_live_textop_motion(
        PlayLiveCommand(
            task="straight",
            onnx_file=str(onnx_file),
            observation=ObservationParams(
                url="http://127.0.0.1:8766/observation",
            ),
        ),
        policy=ResolvedPolicy(OnnxPolicyRunner, onnx_file),
    )

    task_name, play_cfg = calls["run_play"]
    assert task_name == "task"
    assert play_cfg.checkpoint_file == str(onnx_file)
    assert calls["task_kwargs"]["runner_cls"] is OnnxPolicyRunner
    assert calls["task_kwargs"]["source_mode"] == "live"
    assert calls["task_kwargs"]["observation"].publisher is not None


def test_blocked_straight_live_uses_blocked_straight_task_registration(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    monkeypatch.setitem(
        commands_module.LIVE_TASK_REGISTRY,
        "blocked-straight",
        lambda **kwargs: _fake_register_task(calls, kwargs),
    )
    monkeypatch.setattr(
        "mjlab_textop.scripts.commands.run_play",
        lambda task_name, play_cfg: calls.update(run_play=(task_name, play_cfg)),
    )
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    play_live_textop_motion(
        PlayLiveCommand(
            task="blocked-straight",
            onnx_file=str(onnx_file),
            observation=ObservationParams(
                url="http://127.0.0.1:8766/observation",
            ),
        ),
        policy=ResolvedPolicy(OnnxPolicyRunner, onnx_file),
    )

    task_name, play_cfg = calls["run_play"]
    assert task_name == "task"
    assert play_cfg.checkpoint_file == str(onnx_file)
    assert calls["task_kwargs"]["runner_cls"] is OnnxPolicyRunner
    assert calls["task_kwargs"]["source_mode"] == "live"
    assert calls["task_kwargs"]["observation"].publisher is not None


def test_side_goals_live_uses_side_goals_task_registration(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}

    monkeypatch.setitem(
        commands_module.LIVE_TASK_REGISTRY,
        "side-goals",
        lambda **kwargs: _fake_register_task(calls, kwargs),
    )
    monkeypatch.setattr(
        "mjlab_textop.scripts.commands.run_play",
        lambda task_name, play_cfg: calls.update(run_play=(task_name, play_cfg)),
    )
    onnx_file = tmp_path / "policy.onnx"
    onnx_file.write_text("onnx")

    play_live_textop_motion(
        PlayLiveCommand(
            task="side-goals",
            onnx_file=str(onnx_file),
            observation=ObservationParams(
                url="http://127.0.0.1:8766/observation",
            ),
        ),
        policy=ResolvedPolicy(OnnxPolicyRunner, onnx_file),
    )

    task_name, play_cfg = calls["run_play"]
    assert task_name == "task"
    assert play_cfg.checkpoint_file == str(onnx_file)
    assert calls["task_kwargs"]["runner_cls"] is OnnxPolicyRunner
    assert calls["task_kwargs"]["source_mode"] == "live"
    assert calls["task_kwargs"]["observation"].publisher is not None


def _fake_register_task(calls: dict, kwargs: dict) -> str:
    calls["task_kwargs"] = kwargs
    return "task"


def test_textop_motion_command_cfg_rejects_invalid_future_steps() -> None:
    with pytest.raises(ValueError, match="future_steps must be positive"):
        OfflineMotionCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
            motion_file="/tmp/motion.npz",
            anchor_body_name="pelvis",
            body_names=("pelvis",),
            entity_name="robot",
            future_steps=0,
        )


def test_online_textop_motion_command_cfg_rejects_invalid_future_steps() -> None:
    from mjlab_textop.core.mdp.online_commands import OnlineMotionCommandCfg

    with pytest.raises(ValueError, match="future_steps must be positive"):
        OnlineMotionCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
            entity_name="robot",
            anchor_body_name="pelvis",
            future_steps=0,
        )


def test_textop_motion_command_cfg_is_motion_command_cfg() -> None:
    assert issubclass(OfflineMotionCommandCfg, MotionCommandCfg)
    assert issubclass(OfflineMotionCommand, MotionCommand)


def test_textop_motion_command_cfg_from_copies_motion_cfg_fields() -> None:
    cfg = MotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file="/tmp/motion.npz",
        anchor_body_name="pelvis",
        body_names=("pelvis", "torso_link"),
        entity_name="robot",
        pose_range={"x": (-0.1, 0.1)},
        sampling_mode="start",
    )

    textop_cfg = textop_motion_command_cfg_from(cfg, future_steps=7)

    assert isinstance(textop_cfg, OfflineMotionCommandCfg)
    assert textop_cfg.motion_file == cfg.motion_file
    assert textop_cfg.anchor_body_name == cfg.anchor_body_name
    assert textop_cfg.body_names == cfg.body_names
    assert textop_cfg.entity_name == cfg.entity_name
    assert textop_cfg.pose_range == cfg.pose_range
    assert textop_cfg.sampling_mode == "start"
    assert textop_cfg.future_steps == 7

    cfg.pose_range["x"] = (-1.0, 1.0)
    assert textop_cfg.pose_range["x"] == (-0.1, 0.1)


def test_use_textop_motion_command_replaces_motion_cfg() -> None:
    cfg = MotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file="/tmp/motion.npz",
        anchor_body_name="pelvis",
        body_names=("pelvis",),
        entity_name="robot",
    )
    env_cfg = SimpleNamespace(commands={"motion": cfg})

    use_textop_motion_command(env_cfg, future_steps=3)

    assert isinstance(env_cfg.commands["motion"], OfflineMotionCommandCfg)
    assert env_cfg.commands["motion"].future_steps == 3


def test_use_textop_motion_command_rejects_non_motion_cfg() -> None:
    env_cfg = SimpleNamespace(commands={"motion": object()})

    with pytest.raises(TypeError, match="MotionCommandCfg"):
        use_textop_motion_command(env_cfg)


def test_textop_motion_command_future_reference_properties() -> None:
    command = object.__new__(OfflineMotionCommand)
    command.cfg = OfflineMotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file="/tmp/motion.npz",
        anchor_body_name="pelvis",
        body_names=("pelvis", "torso_link"),
        entity_name="robot",
        future_steps=5,
    )
    command.time_steps = torch.tensor([0, 3], dtype=torch.long)
    command.motion_anchor_body_index = 1
    command.motion = SimpleNamespace(
        time_step_total=4,
        joint_pos=torch.arange(4 * 29, dtype=torch.float32).reshape(4, 29),
        joint_vel=torch.arange(4 * 29, dtype=torch.float32).reshape(4, 29) + 1000.0,
        body_pos_w=torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        body_quat_w=torch.arange(4 * 2 * 4, dtype=torch.float32).reshape(4, 2, 4),
    )
    command._env = SimpleNamespace(
        scene=SimpleNamespace(
            env_origins=torch.tensor(
                [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
                dtype=torch.float32,
            )
        )
    )

    assert command.future_time_steps.tolist() == [
        [0, 1, 2, 3, 3],
        [3, 3, 3, 3, 3],
    ]
    assert command.future_joint_pos.shape == (2, 5, 29)
    assert command.future_joint_vel.shape == (2, 5, 29)
    assert command.future_anchor_pos_w.shape == (2, 5, 3)
    assert command.future_anchor_quat_w.shape == (2, 5, 4)

    expected_anchor_pos = (
        command.motion.body_pos_w[command.future_time_steps, 1]
        + command._env.scene.env_origins[:, None, :]
    )
    torch.testing.assert_close(command.future_anchor_pos_w, expected_anchor_pos)
