from __future__ import annotations

import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from mjlab_textop.core.online.live import textop_block_to_ndjson_message
from mjlab_textop.core.online.source import MotionBlock
from mjlab_textop.core.robotmdar import (
    robotmdar_motion_dict_to_block,
    slice_motion_dict_tail,
)
from mjlab_textop.core.schema import TEXTOP_FPS

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompt"
DEFAULT_VLM_SYSTEM_PROMPT_FILE = PROMPT_DIR / "SYSTEM.md"
DEFAULT_VLM_USER_PROMPT_FILE = PROMPT_DIR / "USER.md"


class PromptController(Protocol):
    @property
    def should_stop(self) -> bool: ...

    @property
    def input_active(self) -> bool: ...

    @property
    def log_suffix(self) -> str: ...

    @property
    def recovery_epoch(self) -> int: ...

    def choose_prompt(self, *, block_count: int) -> str: ...

    def on_block_sent(self, *, block_count: int) -> None: ...


class RobotMdarGeneratorArgs(Protocol):
    """The small CLI/configuration surface consumed by the runtime loader."""

    ckpt: str | Path
    datadir: str | Path
    skeleton_asset_root: str | Path
    device: str
    guidance_scale: float


@dataclass(frozen=True)
class RobotMdarRuntime:
    torch: Any
    OmegaConf: Any
    instantiate: Callable[..., Any]
    seed: Any
    ClassifierFreeWrapper: type
    generate_next_motion: Callable[..., Any]
    load_and_freeze_clip: Callable[..., Any]
    encode_text: Callable[..., Any]
    get_zero_abs_pose: Callable[..., Any]
    get_zero_feature: Callable[..., Any]


@dataclass(kw_only=True)
class RobotMdarGenerator:
    runtime: RobotMdarRuntime
    clip_model: Any
    val_data: Any
    vae: Any
    cfg_denoiser: Any
    diffusion: Any
    history_motion: Any
    history_len: int
    future_len: int
    abs_pose: Any

    def next_block(
        self,
        *,
        prompt: str,
        index: int,
        guidance_scale: float,
        recovery_epoch: int = 0,
    ) -> MotionBlock:
        future_motion, motion_dict, self.abs_pose = generate_motion_block(
            runtime=self.runtime,
            clip_model=self.clip_model,
            vae=self.vae,
            cfg_denoiser=self.cfg_denoiser,
            diffusion=self.diffusion,
            val_data=self.val_data,
            history_motion=self.history_motion,
            abs_pose=self.abs_pose,
            prompt=prompt,
            future_len=self.future_len,
            guidance_scale=guidance_scale,
        )
        self.history_motion = future_motion[:, -self.history_len :, :]
        return robotmdar_motion_dict_to_block(
            slice_motion_dict_tail(motion_dict, self.future_len),
            index=index,
            prompt=prompt,
            recovery_epoch=recovery_epoch,
        )


@dataclass(frozen=True)
class StreamConfig:
    guidance_scale: float
    log_every_blocks: int


def load_robotmdar_runtime() -> RobotMdarRuntime:
    try:
        import torch
        from hydra.utils import instantiate  # ty:ignore[unresolved-import]
        from omegaconf import OmegaConf  # ty:ignore[unresolved-import]
        from robotmdar.dtype import seed  # ty:ignore[unresolved-import]
        from robotmdar.dtype.motion import (  # ty:ignore[unresolved-import]
            get_zero_abs_pose,
            get_zero_feature,
        )
        from robotmdar.eval.generate_dar import (  # ty:ignore[unresolved-import]
            ClassifierFreeWrapper,
            generate_next_motion,
        )
        from robotmdar.model.clip import (  # ty:ignore[unresolved-import]
            encode_text,
            load_and_freeze_clip,
        )
    except ImportError as exc:
        raise ImportError(
            "RobotMDAR commands must be run in the TextOp/RobotMDAR environment."
        ) from exc
    return RobotMdarRuntime(
        torch=torch,
        instantiate=instantiate,
        OmegaConf=OmegaConf,
        seed=seed,
        get_zero_abs_pose=get_zero_abs_pose,
        get_zero_feature=get_zero_feature,
        ClassifierFreeWrapper=ClassifierFreeWrapper,
        generate_next_motion=generate_next_motion,
        encode_text=encode_text,
        load_and_freeze_clip=load_and_freeze_clip,
    )


def register_hydra_resolvers(OmegaConf) -> None:
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver(
            "hydra",
            lambda key: str(Path.cwd()) if key == "runtime.cwd" else "",
        )
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda fmt: datetime.now().strftime(fmt),
        )


def make_robotmdar_generator(
    args: RobotMdarGeneratorArgs,
    *,
    log_dir_name: str,
) -> RobotMdarGenerator:
    runtime = load_robotmdar_runtime()
    register_hydra_resolvers(runtime.OmegaConf)

    cfg = runtime.OmegaConf.load(Path(args.ckpt).parent / ".hydra" / "config.yaml")
    _configure_robotmdar_cfg(cfg, args=args, log_dir_name=log_dir_name)

    runtime.seed.set(cfg.seed)
    clip_model = runtime.load_and_freeze_clip("ViT-B/32", device=args.device)
    val_data = runtime.instantiate(cfg.data.val)
    vae = runtime.instantiate(cfg.vae)
    denoiser = runtime.instantiate(cfg.denoiser)
    schedule_sampler = runtime.instantiate(cfg.diffusion.schedule_sampler)
    diffusion = schedule_sampler.diffusion
    vae.eval()
    denoiser.eval()

    manager = runtime.instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)

    history_len = int(cfg.data.history_len)
    history_motion = val_data.normalize(
        runtime.get_zero_feature()
        .to(args.device)
        .reshape(1, 1, -1)
        .repeat(1, history_len, 1)
    )
    return RobotMdarGenerator(
        runtime=runtime,
        clip_model=clip_model,
        val_data=val_data,
        vae=vae,
        cfg_denoiser=runtime.ClassifierFreeWrapper(denoiser),
        diffusion=diffusion,
        history_motion=history_motion,
        history_len=history_len,
        future_len=int(cfg.data.future_len),
        abs_pose=runtime.get_zero_abs_pose((1,), device=args.device),
    )


def generate_motion_block(
    *,
    runtime: RobotMdarRuntime,
    clip_model: Any,
    vae: Any,
    cfg_denoiser: Any,
    diffusion: Any,
    val_data: Any,
    history_motion: Any,
    abs_pose: Any,
    prompt: str,
    future_len: int,
    guidance_scale: float,
) -> tuple[Any, Any, Any]:
    with runtime.torch.no_grad():
        text_embedding = runtime.encode_text(clip_model, [prompt]).float()
        return runtime.generate_next_motion(
            vae=vae,
            denoiser=cfg_denoiser,
            diffusion=diffusion,
            val_data=val_data,
            text_embedding=text_embedding,
            history_motion=history_motion,
            abs_pose=abs_pose,
            future_len=future_len,
            use_full_sample=True,
            guidance_scale=guidance_scale,
            ret_fk=True,
            ret_fk_full=False,
        )


def stream_robotmdar_blocks(
    *,
    conn: socket.socket,
    generator: RobotMdarGenerator,
    prompt_controller: PromptController,
    cfg: StreamConfig,
    log_message: Callable[[str], None],
    prompt_source: Callable[[PromptController], str],
    after_prompt: Callable[[PromptController], None] | None = None,
) -> None:
    frame_index = 0
    next_send_time = time.monotonic()
    block_count = 0

    while not prompt_controller.should_stop:
        block_start_time = time.monotonic()
        current_prompt = prompt_controller.choose_prompt(block_count=block_count)
        if after_prompt is not None:
            after_prompt(prompt_controller)

        block = generator.next_block(
            prompt=current_prompt,
            index=frame_index,
            guidance_scale=cfg.guidance_scale,
            recovery_epoch=prompt_controller.recovery_epoch,
        )
        conn.sendall(textop_block_to_ndjson_message(block).encode("utf-8"))
        # Start asynchronous planner work only after motion generation so a
        # colocated VLM can use the real-time pacing window instead of
        # contending with RobotMDAR for the GPU.
        prompt_controller.on_block_sent(block_count=block_count)

        block_frames = block.joint_pos.shape[0]
        frame_index += block_frames
        block_count += 1

        sleep_seconds = log_stream_timing(
            prompt_controller=prompt_controller,
            cfg=cfg,
            log_message=log_message,
            prompt_source=prompt_source,
            block_count=block_count,
            frame_index=frame_index,
            block_frames=block_frames,
            block_start_time=block_start_time,
            next_send_time=next_send_time,
            prompt=current_prompt,
        )
        next_send_time += block_frames / TEXTOP_FPS
        time.sleep(max(0.0, sleep_seconds))


def log_stream_timing(
    *,
    prompt_controller: PromptController,
    cfg: StreamConfig,
    log_message: Callable[[str], None],
    prompt_source: Callable[[PromptController], str],
    block_count: int,
    frame_index: int,
    block_frames: int,
    block_start_time: float,
    next_send_time: float,
    prompt: str,
) -> float:
    block_duration = block_frames / TEXTOP_FPS
    sleep_seconds = next_send_time + block_duration - time.monotonic()
    if (
        cfg.log_every_blocks > 0
        and block_count % cfg.log_every_blocks == 0
        and not prompt_controller.input_active
    ):
        generation_ms = (time.monotonic() - block_start_time) * 1000.0
        lag_ms = max(0.0, -sleep_seconds * 1000.0)
        log_message(
            "stream "
            f"block={block_count} frame={frame_index} "
            f"prompt={prompt!r} "
            f"prompt_source={prompt_source(prompt_controller)} "
            f"gen_ms={generation_ms:.1f} "
            f"lag_ms={lag_ms:.1f}"
            f"{prompt_controller.log_suffix}"
        )
    return sleep_seconds


def read_prompt_path(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def _configure_robotmdar_cfg(
    cfg: Any,
    *,
    args: RobotMdarGeneratorArgs,
    log_dir_name: str,
) -> None:
    cfg.device = args.device
    cfg.ckpt.dar = args.ckpt
    cfg.train.manager.device = args.device
    cfg.train.manager.save_dir = str(Path.cwd() / "logs" / log_dir_name)
    cfg.train.manager.platform._target_ = "robotmdar.train.train_platforms.NoPlatform"
    cfg.data.datadir = args.datadir
    cfg.skeleton.asset.assetRoot = args.skeleton_asset_root
    cfg.data.val.split = "none"
    cfg.data.val.batch_size = 1
    cfg.use_full_sample = True
    cfg.guidance_scale = args.guidance_scale
