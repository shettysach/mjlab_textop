from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.feedback.observation import (
    HttpObservationPublisher,
    HttpObservationPublisherCfg,
    OnlineTextOpObservationCfg,
    make_torso_observation_camera,
)
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.scripts.utils import (
    ResolvedPolicy,
    register_blocked_straight_play_task,
    register_straight_play_task,
    register_textop_play_task,
    register_turn_play_task,
)
from mjlab_textop.tasks import register_tasks

TextOpLiveTask = Literal["straight", "blocked-straight", "turn"]


@dataclass(kw_only=True)
class NormalizeCommand:
    input_motion_file: str = field(default=tyro.MISSING)
    output_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None


# --


@dataclass(kw_only=True)
class PlayLiveCommand:
    task: TextOpLiveTask | None = None
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    future_steps: int = TEXTOP_FUTURE_STEPS
    fps: float = 50.0
    max_queue_blocks: int = 32
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )
    reset_robot_to_reference: bool = True
    reference_debug_vis: bool = True
    observation_url: str | None = None
    observation_timeout_sec: float = 1.0
    observation_every_frames: int = 5
    observation_image_width: int | None = 320
    observation_image_height: int | None = 240
    observation_camera_distance: float = 2.0
    observation_camera_azimuth: float = 0.0
    observation_camera_elevation: float = -15.0


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    register_tasks()
    task_name = _live_task_registry()[cfg.task](
        policy=policy,
        live_source_cfg=_make_live_source_cfg(cfg),
        source_mode="live",
        future_steps=cfg.future_steps,
        num_envs=cfg.num_envs,
        anchor_alignment=cfg.anchor_alignment,
        observation=_make_online_observation(cfg),
        reset_robot_to_reference=cfg.reset_robot_to_reference,
        reference_debug_vis=cfg.reference_debug_vis,
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        video_width=cfg.observation_image_width if cfg.observation_url else None,
        video_height=cfg.observation_image_height if cfg.observation_url else None,
    )
    run_play(task_name, play_cfg)


def _live_task_registry() -> dict[TextOpLiveTask | None, Callable[..., str]]:
    return {
        None: register_textop_play_task,
        "straight": register_straight_play_task,
        "blocked-straight": register_blocked_straight_play_task,
        "turn": register_turn_play_task,
    }


def _make_live_source_cfg(cfg: PlayLiveCommand) -> SocketTextOpSourceCfg:
    return SocketTextOpSourceCfg(
        host=cfg.host,
        port=cfg.port,
        fps=cfg.fps,
        max_queue_blocks=cfg.max_queue_blocks,
    )


def _make_online_observation(cfg: PlayLiveCommand) -> OnlineTextOpObservationCfg:
    observation_publisher_cfg = (
        HttpObservationPublisherCfg(
            url=cfg.observation_url,
            timeout_sec=cfg.observation_timeout_sec,
        )
        if cfg.observation_url is not None
        else None
    )
    observation_publisher = (
        HttpObservationPublisher(observation_publisher_cfg)
        if observation_publisher_cfg is not None
        else None
    )
    return OnlineTextOpObservationCfg(
        publisher=observation_publisher,
        publish_interval=cfg.observation_every_frames,
        camera=make_torso_observation_camera(
            width=cfg.observation_image_width or 320,
            height=cfg.observation_image_height or 240,
            distance=cfg.observation_camera_distance,
            azimuth=cfg.observation_camera_azimuth,
            elevation=cfg.observation_camera_elevation,
        ),
    )


# --


@dataclass(kw_only=True)
class PlayOnlineCommand:
    motion_file: str = field(default=tyro.MISSING)
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    device: str = "cuda:0"
    num_envs: int = 1
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 8
    reset_robot_to_reference: bool = True
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_online_textop_motion(
    cfg: PlayOnlineCommand,
    *,
    motion_file: Path,
    policy: ResolvedPolicy,
) -> None:
    register_tasks()
    source = make_mjlab_npz_replay_source(motion_file, block_size=cfg.block_size)
    task_name = register_textop_play_task(
        policy=policy,
        source=source,
        source_mode="replay",
        future_steps=cfg.future_steps,
        num_envs=cfg.num_envs,
        anchor_alignment=cfg.anchor_alignment,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
    )
    run_play(task_name, play_cfg)
