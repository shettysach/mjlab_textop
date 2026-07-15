from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.feedback.observation import (
    HttpObservationPublisher,
    OnlineObservationCfg,
    make_torso_observation_camera,
)
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.scripts.utils import (
    ResolvedPolicy,
)
from mjlab_textop.tasks.registration import TextOpTask, register_task


@dataclass(kw_only=True)
class NormalizeCommand:
    input_motion_file: str = field(default=tyro.MISSING)
    output_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None


# --


@dataclass(kw_only=True)
class PlayLiveCommand:
    task: TextOpTask = "default"
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    onnx_provider: Literal["cpu", "cuda"] = "cpu"
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    max_queue_blocks: int = 32
    reset_robot_to_reference: bool = True
    reference_debug_vis: bool = False
    observation: ObservationParams | None = None


@dataclass(kw_only=True)
class ObservationParams:
    url: str = "http://127.0.0.1:8766/observation"
    timeout_sec: float = 1.0
    every_frames: int = 20
    image_width: int = 320
    image_height: int = 240
    camera_distance: float = 2.0
    camera_azimuth: float = 0.0
    camera_elevation: float = -15.0


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    task_name = register_task(
        cfg.task,
        runner_cls=policy.runner_cls,
        onnx_provider=policy.onnx_provider,
        live_source_cfg=SocketSourceCfg(
            host=cfg.host,
            port=cfg.port,
            max_queue_blocks=cfg.max_queue_blocks,
        ),
        source_mode="live",
        num_envs=cfg.num_envs,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
        reference_debug_vis=cfg.reference_debug_vis,
        observation=_make_online_observation(cfg),
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        video_width=cfg.observation.image_width if cfg.observation else None,
        video_height=cfg.observation.image_height if cfg.observation else None,
    )
    run_play(task_name, play_cfg)


def _make_online_observation(cfg: PlayLiveCommand) -> OnlineObservationCfg | None:
    if cfg.observation is None:
        return None

    publisher = HttpObservationPublisher(
        url=cfg.observation.url,
        timeout_sec=cfg.observation.timeout_sec,
    )
    camera = make_torso_observation_camera(
        width=cfg.observation.image_width,
        height=cfg.observation.image_height,
        distance=cfg.observation.camera_distance,
        azimuth=cfg.observation.camera_azimuth,
        elevation=cfg.observation.camera_elevation,
    )

    return OnlineObservationCfg(
        publisher=publisher,
        publish_interval=cfg.observation.every_frames,
        camera=camera,
    )


# --


@dataclass(kw_only=True)
class PlayOnlineCommand:
    motion_file: str = field(default=tyro.MISSING)
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    onnx_provider: Literal["cpu", "cuda"] = "cpu"
    device: str = "cuda:0"
    num_envs: int = 1
    block_size: int = 8
    reset_robot_to_reference: bool = True


def play_online_textop_motion(
    cfg: PlayOnlineCommand,
    *,
    motion_file: Path,
    policy: ResolvedPolicy,
) -> None:
    source = make_mjlab_npz_replay_source(motion_file, block_size=cfg.block_size)
    task_name = register_task(
        "default",
        runner_cls=policy.runner_cls,
        onnx_provider=policy.onnx_provider,
        source=source,
        source_mode="replay",
        num_envs=cfg.num_envs,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
    )
    run_play(task_name, play_cfg)
