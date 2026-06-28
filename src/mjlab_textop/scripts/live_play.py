from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mjlab_textop.core.feedback.image import (
    ObservationImageStore,
    encode_rgb_frame_as_jpeg_base64,
)


@dataclass(frozen=True)
class FeedbackImageCaptureCfg:
    store: ObservationImageStore
    every_steps: int = 20
    width: int = 320
    height: int = 240
    jpeg_quality: int = 60

    def __post_init__(self) -> None:
        if self.every_steps <= 0:
            raise ValueError(f"every_steps must be positive, got {self.every_steps}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError(
                f"jpeg_quality must be in [1, 95], got {self.jpeg_quality}"
            )


class FeedbackImageRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        capture_cfg: FeedbackImageCaptureCfg,
        clip_actions: float | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self.capture_cfg = capture_cfg
        self._capture_step = 0

    def step(self, actions: torch.Tensor):
        result = super().step(actions)
        self._capture_step += 1
        if self._capture_step % self.capture_cfg.every_steps == 0:
            frame = self.unwrapped.render()
            if frame is not None:
                self.capture_cfg.store.set_latest(
                    encode_rgb_frame_as_jpeg_base64(
                        frame,
                        width=self.capture_cfg.width,
                        height=self.capture_cfg.height,
                        quality=self.capture_cfg.jpeg_quality,
                        frame_index=self._capture_step,
                    )
                )
        return result


def run_live_play(
    task_id: str,
    cfg: PlayConfig,
    *,
    image_capture_cfg: FeedbackImageCaptureCfg | None = None,
) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if image_capture_cfg is not None:
        env_cfg.viewer.width = image_capture_cfg.width
        env_cfg.viewer.height = image_capture_cfg.height
    else:
        if cfg.video_height is not None:
            env_cfg.viewer.height = cfg.video_height
        if cfg.video_width is not None:
            env_cfg.viewer.width = cfg.video_width

    resume_path = _resolve_resume_path(cfg)
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")

    render_mode = "rgb_array" if image_capture_cfg is not None else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if image_capture_cfg is None:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = FeedbackImageRslRlVecEnvWrapper(
            env,
            capture_cfg=image_capture_cfg,
            clip_actions=agent_cfg.clip_actions,
        )

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(resume_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    resolved_viewer = _resolve_viewer(cfg.viewer)
    try:
        if resolved_viewer == "native":
            NativeMujocoViewer(env, policy).run()
        elif resolved_viewer == "viser":
            ViserPlayViewer(env, policy).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    finally:
        env.close()


def _resolve_resume_path(cfg: PlayConfig) -> Path:
    if cfg.agent != "trained":
        raise ValueError("play-live only supports trained agents")
    if cfg.checkpoint_file is None:
        raise ValueError("play-live requires a local checkpoint_file")
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    return resume_path


def _resolve_viewer(viewer: Literal["auto", "native", "viser"]) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"
