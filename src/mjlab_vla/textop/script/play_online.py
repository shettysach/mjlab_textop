from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import tyro
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_vla.textop.contract import TEXTOP_FUTURE_STEPS
from mjlab_vla.textop.online import make_mjlab_npz_replay_source
from mjlab_vla.textop.task import (
    ONLINE_TEXTOP_TASK_NAME,
    ensure_textop_task_registered,
    make_online_textop_g1_flat_tracking_env_cfg,
)


@dataclass(kw_only=True)
class PlayOnlineCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    checkpoint_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 8
    max_stale_steps: int = 25
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_online_textop_motion(
    cfg: PlayOnlineCommand,
    *,
    motion_file: Path,
    checkpoint_file: Path,
) -> None:
    ensure_textop_task_registered()
    source = make_mjlab_npz_replay_source(motion_file, block_size=cfg.block_size)
    task_name = f"{ONLINE_TEXTOP_TASK_NAME}-Replay-{uuid4().hex}"

    env_cfg = make_online_textop_g1_flat_tracking_env_cfg(
        play=True,
        future_steps=cfg.future_steps,
        source=source,
        anchor_alignment=cfg.anchor_alignment,
        max_stale_steps=cfg.max_stale_steps,
    )
    env_cfg.scene.num_envs = cfg.num_envs

    register_mjlab_task(
        task_id=task_name,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )

    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(checkpoint_file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        viewer=cfg.viewer,
    )
    run_play(task_name, play_cfg)
