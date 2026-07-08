# ty:ignore[unresolved-import]

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from mjlab_textop.robotmdar.feedback import HttpObservationReceiver
from mjlab_textop.robotmdar.planner.manual import ManualPromptPlanner
from mjlab_textop.robotmdar.planner.vlm import (
    OpenAIChatPromptSelector,
    VlmPromptPlanner,
)
from mjlab_textop.robotmdar.runtime import (
    DEFAULT_VLM_SYSTEM_PROMPT_FILE,
    DEFAULT_VLM_USER_PROMPT_FILE,
    StreamConfig,
    log_stream_timing,
    make_robotmdar_generator,
    read_prompt_path,
    stream_robotmdar_blocks,
)


def run_producer(args: argparse.Namespace) -> None:
    generator = make_robotmdar_generator(args, log_dir_name="robotmdar_producer")
    planner = make_prompt_planner(args)
    planner.start()
    if isinstance(planner, VlmPromptPlanner):
        _log_producer_message("Using VLM planner.")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            _log_producer_message(
                f"Waiting for MJLab consumer on {args.host}:{args.port}"
            )
            conn, addr = server.accept()
            _log_producer_message(f"MJLab consumer connected from {addr}")
            with conn:
                stream_robotmdar_blocks(
                    conn=conn,
                    generator=generator,
                    prompt_controller=planner,
                    cfg=StreamConfig(
                        fps=args.fps,
                        guidance_scale=args.guidance_scale,
                        log_every_blocks=args.log_every_blocks,
                    ),
                    log_message=_log_producer_message,
                    prompt_source=_prompt_source,
                    after_prompt=lambda controller: _log_vlm_reasoning_if_available(
                        planner=controller,
                        args=args,
                    ),
                )
    except KeyboardInterrupt:
        _log_producer_message("Stopping RobotMDAR producer.")
    finally:
        planner.request_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream RobotMDAR text-to-motion blocks to MJLab over NDJSON TCP.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--skeleton-asset-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--planner",
        choices=("manual", "vlm"),
        default="manual",
    )
    parser.add_argument("--prompt", default="stand")
    parser.add_argument("--observation-listen-host", default="127.0.0.1")
    parser.add_argument("--observation-listen-port", type=int, default=None)
    parser.add_argument("--observation-path", default="/observation")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:9379")
    parser.add_argument("--vlm-model", default=None)
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
    parser.add_argument(
        "--vlm-reasoning",
        action="store_true",
        help="Print VLM reasoning when the server returns it.",
    )
    parser.add_argument("--query-every-blocks", type=int, default=20)
    parser.add_argument("--log-every-blocks", type=int, default=20)
    args = parser.parse_args()
    if args.planner == "vlm" and args.observation_listen_port is None:
        raise ValueError(
            f"--observation-listen-port is required with --planner {args.planner}"
        )
    if args.planner == "vlm" and args.query_every_blocks <= 0:
        raise ValueError(
            f"--query-every-blocks must be positive, got {args.query_every_blocks}"
        )
    if args.vlm_timeout_sec <= 0:
        raise ValueError(
            f"--vlm-timeout-sec must be positive, got {args.vlm_timeout_sec}"
        )
    if args.vlm_max_tokens <= 0:
        raise ValueError(
            f"--vlm-max-tokens must be positive, got {args.vlm_max_tokens}"
        )
    if args.planner == "vlm" and not args.vlm_model:
        raise ValueError(f"--vlm-model is required with --planner {args.planner}")
    return args


def main() -> None:
    run_producer(parse_args())


def _log_producer_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _log_block_timing(
    *,
    planner: ManualPromptPlanner | VlmPromptPlanner,
    args: argparse.Namespace,
    block_count: int,
    frame_index: int,
    block_frames: int,
    block_start_time: float,
    next_send_time: float,
    prompt: str,
) -> float:
    return log_stream_timing(
        prompt_controller=planner,
        cfg=StreamConfig(
            fps=args.fps,
            guidance_scale=getattr(args, "guidance_scale", 0.0),
            log_every_blocks=args.log_every_blocks,
        ),
        log_message=_log_producer_message,
        prompt_source=_prompt_source,
        block_count=block_count,
        frame_index=frame_index,
        block_frames=block_frames,
        block_start_time=block_start_time,
        next_send_time=next_send_time,
        prompt=prompt,
    )


def _log_vlm_reasoning_if_available(
    *,
    planner: ManualPromptPlanner | VlmPromptPlanner,
    args: argparse.Namespace,
) -> None:
    if not getattr(args, "vlm_reasoning", False):
        return
    if not isinstance(planner, VlmPromptPlanner):
        return
    reasoning = planner.consume_pending_reasoning()
    if reasoning is not None:
        _log_producer_message(f"vlm_reasoning {reasoning}")


def _prompt_source(
    planner: ManualPromptPlanner | VlmPromptPlanner,
) -> str:
    if isinstance(planner, VlmPromptPlanner):
        return planner.current_prompt_source
    return "manual"


def make_prompt_planner(
    args: argparse.Namespace,
) -> ManualPromptPlanner | VlmPromptPlanner:
    if args.planner == "vlm":
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
        return VlmPromptPlanner(
            feedback=receiver,
            selector=selector,
            initial_prompt=args.prompt,
            query_every_blocks=args.query_every_blocks,
        )
    return ManualPromptPlanner(args.prompt)


if __name__ == "__main__":
    main()
