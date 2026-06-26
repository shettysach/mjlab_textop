from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.contract import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
)
from mjlab_textop.core.online.live_registry import (
    register_live_textop_source,
    unregister_live_textop_source,
)
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.core.task import (
    ensure_textop_task_registered,
    register_online_textop_onnx_task,
)


@dataclass(kw_only=True)
class PlayOnlineOnnxCommand:
    motion_file: str = field(default=tyro.MISSING)
    policy_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 8
    max_stale_steps: int = 25
    reset_robot_to_reference: bool = True
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


@dataclass(kw_only=True)
class PlayLiveOnnxCommand:
    policy_file: str = field(default=tyro.MISSING)
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    fps: float = 50.0
    max_queue_blocks: int = 32
    max_stale_steps: int = 25
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_online_textop_onnx(
    cfg: PlayOnlineOnnxCommand,
    *,
    motion_file: Path,
    policy_file: Path,
) -> None:
    ensure_textop_task_registered()
    source = make_mjlab_npz_replay_source(motion_file, block_size=cfg.block_size)
    task_name = register_online_textop_onnx_task(
        source=source,
        source_mode="replay",
        future_steps=cfg.future_steps,
        num_envs=cfg.num_envs,
        anchor_alignment=cfg.anchor_alignment,
        max_stale_steps=cfg.max_stale_steps,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
    )

    run_textop_onnx_play(
        task_name=task_name,
        policy_file=policy_file,
        device=cfg.device,
        num_envs=cfg.num_envs,
        viewer=cfg.viewer,
    )


def play_live_textop_onnx(
    cfg: PlayLiveOnnxCommand,
    *,
    policy_file: Path,
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
        task_name = register_online_textop_onnx_task(
            source_key=source_key,
            source_mode="live",
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
            max_stale_steps=cfg.max_stale_steps,
        )

        run_textop_onnx_play(
            task_name=task_name,
            policy_file=policy_file,
            device=cfg.device,
            num_envs=cfg.num_envs,
            viewer=cfg.viewer,
        )
    finally:
        unregister_live_textop_source(source_key)
        source.close()


def run_textop_onnx_play(
    *,
    task_name: str,
    policy_file: Path,
    device: str,
    num_envs: int,
    viewer: Literal["auto", "native", "viser"],
) -> None:
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy_file),
        num_envs=num_envs,
        device=device,
        viewer=viewer,
    )
    run_play(task_name, play_cfg)
