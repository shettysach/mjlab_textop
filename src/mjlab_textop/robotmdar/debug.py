from __future__ import annotations

import argparse
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab_textop.robotmdar.feedback import HttpObservationReceiver
from mjlab_textop.robotmdar.planner.manual import PromptState
from mjlab_textop.robotmdar.planner.vlm import OpenAIChatPromptSelector
from mjlab_textop.robotmdar.runtime import (
    DEFAULT_VLM_SYSTEM_PROMPT_FILE,
    DEFAULT_VLM_USER_PROMPT_FILE,
    StreamConfig,
    make_robotmdar_generator,
    read_prompt_path,
    stream_robotmdar_blocks,
)


def run_debug(args: argparse.Namespace) -> None:
    generator = make_robotmdar_generator(args, log_dir_name="robotmdar_debug")
    receiver = HttpObservationReceiver(
        host=args.observation_listen_host,
        port=args.observation_listen_port,
        path=args.observation_path,
    )
    selector = OpenAIChatPromptSelector(
        base_url=args.vlm_base_url,
        model=args.vlm_model,
        system_prompt=read_prompt_path(args.vlm_system_prompt),
        user_prompt=read_prompt_path(args.vlm_user_prompt),
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
                stream_robotmdar_blocks(
                    conn=conn,
                    generator=generator,
                    prompt_controller=PromptStateController(prompt),
                    cfg=StreamConfig(
                        guidance_scale=args.guidance_scale,
                        log_every_blocks=args.log_every_blocks,
                    ),
                    log_message=_log_debug_message,
                    prompt_source=lambda _controller: "manual",
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


@dataclass
class PromptStateController:
    prompt: PromptState

    @property
    def should_stop(self) -> bool:
        return self.prompt.stop

    @property
    def input_active(self) -> bool:
        return self.prompt.input_active

    @property
    def log_suffix(self) -> str:
        return ""

    def choose_prompt(self, *, block_count: int) -> str:
        del block_count
        return self.prompt.text


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
