from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mjlab_textop.core.online.live import textop_block_to_ndjson_message
from mjlab_textop.core.robotmdar import (
    robotmdar_motion_dict_to_block,
    slice_motion_dict_tail,
)
from mjlab_textop.robotmdar.feedback import HttpObservationReceiver
from mjlab_textop.robotmdar.planner.manual import PromptState
from mjlab_textop.robotmdar.planner.vlm import OpenAIChatPromptSelector
from mjlab_textop.robotmdar.produce import (
    DEFAULT_VLM_SYSTEM_PROMPT_FILE,
    DEFAULT_VLM_USER_PROMPT_FILE,
    _generate_motion_block,
    _load_robotmdar_runtime,
    _read_prompt_path,
    _register_hydra_resolvers,
)


def run_debug(args: argparse.Namespace) -> None:
    runtime = _load_robotmdar_runtime()

    _register_hydra_resolvers(runtime.OmegaConf)
    cfg = runtime.OmegaConf.load(Path(args.ckpt).parent / ".hydra" / "config.yaml")
    cfg.device = args.device
    cfg.ckpt.dar = args.ckpt
    cfg.train.manager.device = args.device
    cfg.train.manager.save_dir = str(Path.cwd() / "logs" / "robotmdar_debug")
    cfg.train.manager.platform._target_ = "robotmdar.train.train_platforms.NoPlatform"
    cfg.data.datadir = args.datadir
    cfg.skeleton.asset.assetRoot = args.skeleton_asset_root
    cfg.data.val.split = "none"
    cfg.data.val.batch_size = 1
    cfg.use_full_sample = True
    cfg.guidance_scale = args.guidance_scale

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
    cfg_denoiser = runtime.ClassifierFreeWrapper(denoiser)

    future_len = int(cfg.data.future_len)
    history_len = int(cfg.data.history_len)
    history_motion = val_data.normalize(
        runtime.get_zero_feature()
        .to(args.device)
        .reshape(1, 1, -1)
        .repeat(1, history_len, 1)
    )
    abs_pose = runtime.get_zero_abs_pose((1,), device=args.device)

    receiver = HttpObservationReceiver(
        host=args.observation_listen_host,
        port=args.observation_listen_port,
        path=args.observation_path,
    )
    selector = OpenAIChatPromptSelector(
        base_url=args.vlm_base_url,
        model=args.vlm_model,
        system_prompt=_read_prompt_path(args.vlm_system_prompt),
        user_prompt=_read_prompt_path(args.vlm_user_prompt),
        timeout_sec=args.vlm_timeout_sec,
        max_tokens=args.vlm_max_tokens,
        include_history=args.vlm_history,
    )
    prompt = PromptState(text=args.prompt)
    input_thread = threading.Thread(
        target=_prompt_loop,
        args=(prompt, receiver, selector, args.debug_dir),
        daemon=True,
    )

    receiver.start()
    input_thread.start()
    _log_debug_message("Using manual prompt stream with ?-triggered VLM debug.")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            _log_debug_message(f"Waiting for MJLab consumer on {args.host}:{args.port}")
            conn, addr = server.accept()
            _log_debug_message(f"MJLab consumer connected from {addr}")
            with conn:
                _run_debug_stream(
                    conn=conn,
                    args=args,
                    runtime=runtime,
                    clip_model=clip_model,
                    val_data=val_data,
                    vae=vae,
                    cfg_denoiser=cfg_denoiser,
                    diffusion=diffusion,
                    history_motion=history_motion,
                    history_len=history_len,
                    future_len=future_len,
                    abs_pose=abs_pose,
                    prompt=prompt,
                )
    except KeyboardInterrupt:
        _log_debug_message("Stopping RobotMDAR debug producer.")
    finally:
        prompt.stop = True
        receiver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream RobotMDAR manual prompts to MJLab and query the VLM with ?.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--skeleton-asset-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--prompt", default="stand")
    parser.add_argument("--observation-listen-host", default="127.0.0.1")
    parser.add_argument("--observation-listen-port", type=int, required=True)
    parser.add_argument("--observation-path", default="/observation")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:9379")
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument(
        "--vlm-system-prompt",
        type=Path,
        default=DEFAULT_VLM_SYSTEM_PROMPT_FILE,
    )
    parser.add_argument(
        "--vlm-user-prompt",
        type=Path,
        default=DEFAULT_VLM_USER_PROMPT_FILE,
    )
    parser.add_argument("--vlm-timeout-sec", type=float, default=30.0)
    parser.add_argument("--vlm-max-tokens", type=int, default=256)
    parser.add_argument(
        "--vlm-history",
        action="store_true",
        help="Send previous VLM-selected prompts back on later VLM requests.",
    )
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--log-every-blocks", type=int, default=20)
    args = parser.parse_args()
    if args.vlm_timeout_sec <= 0:
        raise ValueError(
            f"--vlm-timeout-sec must be positive, got {args.vlm_timeout_sec}"
        )
    if args.vlm_max_tokens <= 0:
        raise ValueError(
            f"--vlm-max-tokens must be positive, got {args.vlm_max_tokens}"
        )
    return args


def main() -> None:
    run_debug(parse_args())


def _run_debug_stream(
    *,
    conn: socket.socket,
    args: argparse.Namespace,
    runtime: Any,
    clip_model: Any,
    val_data: Any,
    vae: Any,
    cfg_denoiser: Any,
    diffusion: Any,
    history_motion: Any,
    history_len: int,
    future_len: int,
    abs_pose: Any,
    prompt: PromptState,
) -> None:
    frame_index = 0
    next_send_time = time.monotonic()
    block_count = 0

    while not prompt.stop:
        block_start_time = time.monotonic()
        current_prompt = prompt.text
        future_motion, motion_dict, abs_pose = _generate_motion_block(
            runtime=runtime,
            clip_model=clip_model,
            vae=vae,
            cfg_denoiser=cfg_denoiser,
            diffusion=diffusion,
            val_data=val_data,
            history_motion=history_motion,
            abs_pose=abs_pose,
            prompt=current_prompt,
            future_len=future_len,
            guidance_scale=args.guidance_scale,
        )
        history_motion = future_motion[:, -history_len:, :]
        block = robotmdar_motion_dict_to_block(
            slice_motion_dict_tail(motion_dict, future_len),
            index=frame_index,
        )
        conn.sendall(
            textop_block_to_ndjson_message(block, fps=args.fps).encode("utf-8")
        )

        block_frames = block.joint_pos.shape[0]
        frame_index += block_frames
        block_count += 1

        block_duration = block_frames / args.fps
        sleep_seconds = next_send_time + block_duration - time.monotonic()
        if (
            args.log_every_blocks > 0
            and block_count % args.log_every_blocks == 0
            and not prompt.input_active
        ):
            generation_ms = (time.monotonic() - block_start_time) * 1000.0
            lag_ms = max(0.0, -sleep_seconds * 1000.0)
            _log_debug_message(
                "stream "
                f"block={block_count} frame={frame_index} "
                f"prompt={current_prompt!r} "
                f"gen_ms={generation_ms:.1f} "
                f"lag_ms={lag_ms:.1f}"
            )
        next_send_time += block_duration
        time.sleep(max(0.0, sleep_seconds))


def _prompt_loop(
    prompt: PromptState,
    receiver: HttpObservationReceiver,
    selector: OpenAIChatPromptSelector,
    debug_dir: Path | None,
) -> None:
    while not prompt.stop:
        try:
            prompt.input_active = True
            text = input("Enter prompt (? for VLM, q to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            prompt.stop = True
            return
        finally:
            prompt.input_active = False
        if text == "?":
            _query_vlm(receiver, selector, debug_dir)
        elif text.lower() in {"q", "quit", "exit"}:
            prompt.stop = True
        elif text:
            prompt.text = text


def _query_vlm(
    receiver: HttpObservationReceiver,
    selector: OpenAIChatPromptSelector,
    debug_dir: Path | None = None,
) -> None:
    observation = receiver.latest()
    if observation is None:
        _log_debug_message("vlm_debug_error No observation received yet.")
        return

    if debug_dir is not None:
        _save_vlm_debug_image(observation, debug_dir)
    _log_debug_message("vlm_debug_request")
    try:
        selection = selector.choose_prompt_with_debug(observation=observation)
    except Exception as exc:
        _log_debug_message(f"vlm_debug_error {type(exc).__name__}: {exc}")
        return
    if selection.reasoning is not None:
        _log_debug_message(f"vlm_debug_reasoning {selection.reasoning}")
    _log_debug_message(f"vlm_debug_response {selection.prompt}")


def _save_vlm_debug_image(observation: Any, debug_dir: Path) -> None:
    if observation.image_bytes is None or observation.image_mime_type is None:
        _log_debug_message("vlm_debug_image none")
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    image_path = _next_vlm_debug_image_path(debug_dir)
    image_path.write_bytes(observation.image_bytes)
    _log_debug_message(f"vlm_debug_image {image_path}")


def _next_vlm_debug_image_path(debug_dir: Path) -> Path:
    index = 1
    while True:
        image_path = debug_dir / f"vlm_observation_{index:06d}.jpg"
        if not image_path.exists():
            return image_path
        index += 1


def _log_debug_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
