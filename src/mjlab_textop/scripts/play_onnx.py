from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

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
from mjlab_textop.core.onnx_policy import TextOpOnnxPolicy
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
    configure_torch_backends()

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_name, play=True)
    agent_cfg = load_rl_cfg(task_name)
    env_cfg.scene.num_envs = num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = TextOpOnnxPolicy(policy_file)
    try:
        resolved_viewer = _resolve_viewer(viewer)
        if resolved_viewer == "native":
            NativeMujocoViewer(wrapped_env, policy).run()
        elif resolved_viewer == "viser":
            ViserPlayViewer(wrapped_env, policy, checkpoint_manager=None).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    finally:
        wrapped_env.close()


def _resolve_viewer(viewer: Literal["auto", "native", "viser"]) -> str:
    if viewer != "auto":
        return viewer

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"
