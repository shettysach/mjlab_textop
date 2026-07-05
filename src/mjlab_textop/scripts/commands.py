from __future__ import annotations

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
    TaskRegistrar,
    register_generic_play_task,
)
from mjlab_textop.tasks import register_tasks
from mjlab_textop.tasks.blocked_straight.registration import (
    register_blocked_straight_task,
)
from mjlab_textop.tasks.online_textop.registration import register_online_textop_task
from mjlab_textop.tasks.straight.registration import register_straight_task
from mjlab_textop.tasks.turn.registration import register_turn_task

TextOpLiveTask = Literal["default", "straight", "blocked-straight", "turn"]

LIVE_TASK_REGISTRY: dict[TextOpLiveTask, TaskRegistrar] = {
    "default": register_online_textop_task,
    "straight": register_straight_task,
    "blocked-straight": register_blocked_straight_task,
    "turn": register_turn_task,
}


@dataclass(kw_only=True)
class NormalizeCommand:
    input_motion_file: str = field(default=tyro.MISSING)
    output_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None


# --


@dataclass(kw_only=True)
class PlayLiveCommand:
    task: TextOpLiveTask = "default"
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
    reference_debug_vis: bool = False
    observation: ObservationParams | None = None


@dataclass(kw_only=True)
class ObservationParams:
    url: str = "http://127.0.0.1:8766/observation"
    timeout_sec: float = 1.0
    every_frames: int = 5
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
    register_tasks()
    task_name = _register_live_task(cfg, policy=policy)
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        video_width=cfg.observation.image_width if cfg.observation else None,
        video_height=cfg.observation.image_height if cfg.observation else None,
    )
    run_play(task_name, play_cfg)


def _register_live_task(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> str:
    live_source_cfg = _make_live_source_cfg(cfg)
    observation = _make_online_observation(cfg)
    return register_generic_play_task(
        task_registrar=LIVE_TASK_REGISTRY[cfg.task],
        policy=policy,
        live_source_cfg=live_source_cfg,
        source_mode="live",
        future_steps=cfg.future_steps,
        num_envs=cfg.num_envs,
        anchor_alignment=cfg.anchor_alignment,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
        reference_debug_vis=cfg.reference_debug_vis,
        observation=observation,
    )


def _make_live_source_cfg(cfg: PlayLiveCommand) -> SocketTextOpSourceCfg:
    return SocketTextOpSourceCfg(
        host=cfg.host,
        port=cfg.port,
        fps=cfg.fps,
        max_queue_blocks=cfg.max_queue_blocks,
    )


def _make_online_observation(cfg: PlayLiveCommand) -> OnlineTextOpObservationCfg:
    if cfg.observation is None:
        return OnlineTextOpObservationCfg()

    observation_publisher_cfg = HttpObservationPublisherCfg(
        url=cfg.observation.url,
        timeout_sec=cfg.observation.timeout_sec,
    )
    observation_publisher = HttpObservationPublisher(observation_publisher_cfg)

    return OnlineTextOpObservationCfg(
        publisher=observation_publisher,
        publish_interval=cfg.observation.every_frames,
        camera=make_torso_observation_camera(
            width=cfg.observation.image_width,
            height=cfg.observation.image_height,
            distance=cfg.observation.camera_distance,
            azimuth=cfg.observation.camera_azimuth,
            elevation=cfg.observation.camera_elevation,
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
    task_name = register_generic_play_task(
        task_registrar=register_online_textop_task,
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
