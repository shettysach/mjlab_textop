from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_vla.textop.contract import TEXTOP_FUTURE_STEPS
from mjlab_vla.textop.online.live import SocketTextOpOnlineSource, SocketTextOpSourceCfg
from mjlab_vla.textop.task import (
    ensure_textop_task_registered,
    register_online_textop_task,
)


@dataclass(kw_only=True)
class PlayLiveCommand:
    checkpoint_file: str = field(default=tyro.MISSING)
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


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    checkpoint_file: Path,
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
    try:
        task_name = register_online_textop_task(
            source=source,
            source_mode="live",
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
            max_stale_steps=cfg.max_stale_steps,
        )

        play_cfg = PlayConfig(
            agent="trained",
            checkpoint_file=str(checkpoint_file),
            num_envs=cfg.num_envs,
            device=cfg.device,
            viewer=cfg.viewer,
        )
        run_play(task_name, play_cfg)
    finally:
        source.close()
