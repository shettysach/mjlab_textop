from __future__ import annotations

import socket
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robotmdar_textop.logging import format_stream_status
from robotmdar_textop.motion import (
    robotmdar_motion_dict_to_block,
    slice_motion_dict_tail,
)
from textop_protocol.motion import MotionBlock, MotionFrames, StreamControl
from textop_protocol.motion_stream import textop_block_to_wire
from textop_protocol.timing import FPS, FUTURE_STEPS

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
DEFAULT_VLM_SYSTEM_PROMPT_FILE = Path("TASK.md")
DEFAULT_VLM_USER_PROMPT_FILE = PROMPT_DIR / "USER.md"
INVARIANT_CONTROLLER_PROMPT = PROMPT_DIR / "INVARIANT.md"

_TEXT_EMBEDDING_CACHE_SIZE = 16


@dataclass(frozen=True)
class BlockPlan:
    prompt: str
    source: str
    recovery_epoch: int = 0
    request_id: int | None = None
    reset_pacing: bool = False


class PromptController(Protocol):
    @property
    def should_stop(self) -> bool: ...

    @property
    def input_active(self) -> bool: ...

    @property
    def log_suffix(self) -> str: ...

    def next_plan(self, *, block_count: int) -> BlockPlan: ...


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
    _last_block: MotionBlock | None = field(default=None, init=False, repr=False)
    _text_embeddings: OrderedDict[str, Any] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

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
            vae=self.vae,
            cfg_denoiser=self.cfg_denoiser,
            diffusion=self.diffusion,
            val_data=self.val_data,
            history_motion=self.history_motion,
            abs_pose=self.abs_pose,
            text_embedding=self._text_embedding(prompt),
            future_len=self.future_len,
            guidance_scale=guidance_scale,
        )
        self.history_motion = future_motion[:, -self.history_len :, :]
        block = robotmdar_motion_dict_to_block(
            slice_motion_dict_tail(motion_dict, self.future_len),
            index=index,
            prompt=prompt,
            recovery_epoch=recovery_epoch,
        )
        self._last_block = block
        return block

    def observation_block(
        self,
        *,
        index: int,
        prompt: str,
        recovery_epoch: int,
        request_id: int,
    ) -> MotionBlock:
        if self._last_block is None:
            raise RuntimeError("Cannot request an observation before generating motion")
        previous = self._last_block

        def repeat_last(value: np.ndarray) -> np.ndarray:
            return np.repeat(value[-1:], FUTURE_STEPS, axis=0)

        joint_pos = repeat_last(previous.joint_pos)
        return MotionBlock(
            index=index,
            motion=MotionFrames(
                joint_pos=joint_pos,
                joint_vel=np.zeros_like(joint_pos),
                anchor_pos_w=repeat_last(previous.anchor_pos_w),
                anchor_quat_w=repeat_last(previous.anchor_quat_w),
            ),
            control=StreamControl(
                prompt=prompt,
                recovery_epoch=recovery_epoch,
                request_id=request_id,
            ),
        )

    def _text_embedding(self, prompt: str) -> Any:
        try:
            cached = self._text_embeddings[prompt]
        except KeyError:
            pass
        else:
            self._text_embeddings.move_to_end(prompt)
            return cached

        with self.runtime.torch.no_grad():
            embedding = self.runtime.encode_text(self.clip_model, [prompt]).float()

        if len(self._text_embeddings) >= _TEXT_EMBEDDING_CACHE_SIZE:
            self._text_embeddings.popitem(last=False)
        self._text_embeddings[prompt] = embedding
        return embedding


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
    vae: Any,
    cfg_denoiser: Any,
    diffusion: Any,
    val_data: Any,
    history_motion: Any,
    abs_pose: Any,
    text_embedding: Any,
    future_len: int,
    guidance_scale: float,
) -> tuple[Any, Any, Any]:
    with runtime.torch.no_grad():
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
    after_plan: Callable[[], None] | None = None,
) -> None:
    frame_index = 0
    next_send_time = time.monotonic()
    block_count = 0
    previous_command: tuple[str, str] | None = None

    while not prompt_controller.should_stop:
        plan = prompt_controller.next_plan(block_count=block_count)
        if plan.reset_pacing:
            next_send_time = time.monotonic()
        block_start_time = time.monotonic()
        current_prompt = plan.prompt
        current_source = plan.source
        current_command = (current_prompt, current_source)
        command_changed = current_command != previous_command
        previous_command = current_command
        if after_plan is not None:
            after_plan()

        block = (
            generator.observation_block(
                prompt=current_prompt,
                index=frame_index,
                recovery_epoch=plan.recovery_epoch,
                request_id=plan.request_id,
            )
            if plan.request_id is not None
            else generator.next_block(
                prompt=current_prompt,
                index=frame_index,
                guidance_scale=cfg.guidance_scale,
                recovery_epoch=plan.recovery_epoch,
            )
        )
        conn.sendall(textop_block_to_wire(block))

        block_frames = block.joint_pos.shape[0]
        frame_index += block_frames
        block_count += 1

        sleep_seconds = log_stream_timing(
            prompt_controller=prompt_controller,
            cfg=cfg,
            log_message=log_message,
            source=current_source,
            command_changed=command_changed,
            block_count=block_count,
            frame_index=frame_index,
            block_frames=block_frames,
            block_start_time=block_start_time,
            next_send_time=next_send_time,
            prompt=current_prompt,
        )
        next_send_time += block_frames / FPS
        time.sleep(max(0.0, sleep_seconds))


def log_stream_timing(
    *,
    prompt_controller: PromptController,
    cfg: StreamConfig,
    log_message: Callable[[str], None],
    source: str,
    command_changed: bool,
    block_count: int,
    frame_index: int,
    block_frames: int,
    block_start_time: float,
    next_send_time: float,
    prompt: str,
) -> float:
    block_duration = block_frames / FPS
    sleep_seconds = next_send_time + block_duration - time.monotonic()
    periodic_log_due = (
        cfg.log_every_blocks > 0 and block_count % cfg.log_every_blocks == 0
    )
    if (command_changed or periodic_log_due) and not prompt_controller.input_active:
        generation_ms = (time.monotonic() - block_start_time) * 1000.0
        lag_ms = max(0.0, -sleep_seconds * 1000.0)
        log_message(
            format_stream_status(
                block_count=block_count,
                frame_index=frame_index,
                prompt=prompt,
                source=source,
                generation_ms=generation_ms,
                lag_ms=lag_ms,
                block_frames=block_frames,
                suffix=prompt_controller.log_suffix,
            )
        )
    return sleep_seconds


def read_prompt_path(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")


def compose_system_prompt(task_prompt: str, *, include_invariant: bool = False) -> str:
    task_prompt = task_prompt.strip()
    if not include_invariant:
        return task_prompt
    invariant_prompt = read_prompt_path(INVARIANT_CONTROLLER_PROMPT).strip()
    return f"{invariant_prompt}\n\n{task_prompt}"


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
