from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
)
from mjlab_textop.core.online.live_registry import (
    register_live_textop_source,
    unregister_live_textop_source,
)
from mjlab_textop.core.online.replay import (
    load_sliced_mjlab_npz_blocks,
    make_mjlab_npz_replay_source,
)
from mjlab_textop.core.online.source import QueueTextOpOnlineSource, TextOpMotionBlock
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import ensure_textop_task_registered
from mjlab_textop.scripts.utils import ResolvedPolicy, register_textop_play_task

SQUARE_PHASES: tuple[tuple[str, int], ...] = (
    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 30),
    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 30),
    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 30),
    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 120),
)


@dataclass(kw_only=True)
class NormalizeCommand:
    input_motion_file: str = field(default=tyro.MISSING)
    output_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None


# --


@dataclass(kw_only=True)
class PlayLiveCommand:
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    fps: float = 50.0
    max_queue_blocks: int = 32
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    ensure_textop_task_registered()
    source = SocketTextOpOnlineSource(
        SocketTextOpSourceCfg(
            host=cfg.host,
            port=cfg.port,
            fps=cfg.fps,
            max_queue_blocks=cfg.max_queue_blocks,
        )
    )
    source.start()
    source_key = register_live_textop_source(source)
    try:
        task_name = register_textop_play_task(
            policy=policy,
            source_key=source_key,
            source_mode="live",
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
        )
        play_cfg = PlayConfig(
            agent="trained",
            checkpoint_file=str(policy.file),
            num_envs=cfg.num_envs,
            device=cfg.device,
            viewer=cfg.viewer,
        )
        run_play(task_name, play_cfg)
    finally:
        unregister_live_textop_source(source_key)
        source.close()


# --


@dataclass(kw_only=True)
class PlayOnlineCommand:
    motion_file: str = field(default=tyro.MISSING)
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
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
    ensure_textop_task_registered()
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
        viewer=cfg.viewer,
    )
    run_play(task_name, play_cfg)


# --


@dataclass(kw_only=True)
class PlaySquareCommand:
    walk_motion_file: str = field(default=tyro.MISSING)
    turn_motion_file: str = field(default=tyro.MISSING)
    stand_motion_file: str = field(default=tyro.MISSING)
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 30
    reset_robot_to_reference: bool = True
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def build_square_sequence_blocks(
    motion_files: dict[str, Path],
    phases: tuple[tuple[str, int], ...] = SQUARE_PHASES,
    *,
    block_size: int,
) -> tuple[list[TextOpMotionBlock], int]:
    blocks: list[TextOpMotionBlock] = []
    next_frame_index = 0

    for phase_index, (prompt, frames) in enumerate(phases):
        if prompt not in motion_files:
            raise KeyError(f"No motion file configured for prompt: {prompt!r}")

        phase_blocks = load_sliced_mjlab_npz_blocks(
            motion_files[prompt],
            frames=frames,
            start_index=next_frame_index,
            block_size=block_size,
        )
        blocks.extend(phase_blocks)
        print(
            f"[square] phase={phase_index} "
            f"prompt={prompt!r} "
            f"frames={frames} "
            f"start_index={next_frame_index} "
            f"blocks={len(phase_blocks)}"
        )
        next_frame_index += frames

    return blocks, next_frame_index


def play_square_textop_motion(
    cfg: PlaySquareCommand,
    *,
    walk_motion_file: Path,
    turn_motion_file: Path,
    stand_motion_file: Path,
    policy: ResolvedPolicy,
) -> None:
    ensure_textop_task_registered()
    blocks, total_frames = build_square_sequence_blocks(
        {
            "walk forward": walk_motion_file,
            "turn left": turn_motion_file,
            "stand still": stand_motion_file,
        },
        block_size=cfg.block_size,
    )
    print(f"[square] queued total_frames={total_frames} blocks={len(blocks)}")

    source = QueueTextOpOnlineSource(blocks)
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
        viewer=cfg.viewer,
    )
    run_play(task_name, play_cfg)
