from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.feedback.observation import (
    HttpObservationPublisher,
    OnlineTextOpObservationCfg,
    make_torso_observation_camera,
)
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.scripts.utils import (
    ResolvedPolicy,
    TaskRegistrar,
)
from mjlab_textop.tasks.blocked_straight.registration import (
    register_blocked_straight_task,
)
from mjlab_textop.tasks.online_textop.registration import register_online_textop_task
from mjlab_textop.tasks.side_goals.registration import register_side_goals_task
from mjlab_textop.tasks.straight.registration import register_straight_task
from mjlab_textop.tasks.turn.registration import register_turn_task

TextOpLiveTask = Literal[
    "default", "straight", "blocked-straight", "side-goals", "turn"
]

LIVE_TASK_REGISTRY: dict[TextOpLiveTask, TaskRegistrar] = {
    "default": register_online_textop_task,
    "straight": register_straight_task,
    "blocked-straight": register_blocked_straight_task,
    "side-goals": register_side_goals_task,
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
    task_name = LIVE_TASK_REGISTRY[cfg.task](
        runner_cls=policy.runner_cls,
        live_source_cfg=SocketTextOpSourceCfg(
            host=cfg.host,
            port=cfg.port,
            fps=cfg.fps,
            max_queue_blocks=cfg.max_queue_blocks,
        ),
        source_mode="live",
        future_steps=cfg.future_steps,
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


def _make_online_observation(cfg: PlayLiveCommand) -> OnlineTextOpObservationCfg | None:
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

    return OnlineTextOpObservationCfg(
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
    device: str = "cuda:0"
    num_envs: int = 1
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 8
    reset_robot_to_reference: bool = True


def play_online_textop_motion(
    cfg: PlayOnlineCommand,
    *,
    motion_file: Path,
    policy: ResolvedPolicy,
) -> None:
    source = make_mjlab_npz_replay_source(motion_file, block_size=cfg.block_size)
    task_name = register_online_textop_task(
        runner_cls=policy.runner_cls,
        source=source,
        source_mode="replay",
        future_steps=cfg.future_steps,
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
