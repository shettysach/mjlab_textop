from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, TypeAlias

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.feedback.observation import (
    HttpObservationPublisher,
    ObservationMode,
    OnlineObservationCfg,
    make_torso_observation_camera,
)
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.scripts.utils import (
    ResolvedPolicy,
)
from tasks.catalog import TaskSet
from tasks.registration import register_task


@dataclass(kw_only=True)
class NormalizeCommand:
    input_motion_file: str = field(default=tyro.MISSING)
    output_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None


# --


@dataclass(kw_only=True)
class ObservationParams:
    url: str = "http://127.0.0.1:8766/observation"
    timeout_sec: float = 1.0
    mode: ObservationMode = "requested"
    every_frames: int = 20
    image_size: tuple[int, int] = (320, 240)
    camera_distance: float = 2.0
    camera_azimuth: float = 0.0
    camera_elevation: float = -15.0


ObservationConfig: TypeAlias = (
    Annotated[
        ObservationParams,
        tyro.conf.subcommand("obs", prefix_name=False),
    ]
    | None
)


@dataclass(kw_only=True)
class PlayLiveCommand:
    task: TaskSet | None = None
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    max_queue_blocks: int = 8
    reset_robot_to_reference: bool = True
    ref_vis: bool = False
    obs: ObservationConfig = None


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    image_size = cfg.obs.image_size if cfg.obs else (None, None)
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
        reference_debug_vis=cfg.ref_vis,
        observation=_make_online_observation(cfg),
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        video_width=image_size[0],
        video_height=image_size[1],
    )
    run_play(task_name, play_cfg)


def _make_online_observation(cfg: PlayLiveCommand) -> OnlineObservationCfg | None:
    if cfg.obs is None:
        return None

    publisher = HttpObservationPublisher(
        url=cfg.obs.url,
        timeout_sec=cfg.obs.timeout_sec,
    )
    camera = make_torso_observation_camera(
        width=cfg.obs.image_size[0],
        height=cfg.obs.image_size[1],
        distance=cfg.obs.camera_distance,
        azimuth=cfg.obs.camera_azimuth,
        elevation=cfg.obs.camera_elevation,
    )

    return OnlineObservationCfg(
        publisher=publisher,
        mode=cfg.obs.mode,
        every_frames=cfg.obs.every_frames,
        camera=camera,
    )


# --


@dataclass(kw_only=True)
class PlayOnlineCommand:
    motion_file: str = field(default=tyro.MISSING)
    checkpoint_file: str | None = None
    onnx_file: str | None = None
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
        None,
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
