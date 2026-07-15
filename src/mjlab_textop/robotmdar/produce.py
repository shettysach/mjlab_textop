from __future__ import annotations

import argparse
import socket
import sys

from mjlab_textop.robotmdar.args import (
    add_generator_arguments,
    add_stream_arguments,
    add_vlm_arguments,
    validate_vlm_arguments,
)
from mjlab_textop.robotmdar.feedback import HttpObservationReceiver
from mjlab_textop.robotmdar.planner.manual import ManualPromptPlanner
from mjlab_textop.robotmdar.planner.vlm import (
    OpenAIChatPromptSelector,
    VlmPromptPlanner,
)
from mjlab_textop.robotmdar.runtime import (
    PromptController,
    StreamConfig,
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
    add_generator_arguments(parser)
    add_stream_arguments(parser)
    parser.add_argument(
        "--planner",
        choices=("manual", "vlm"),
        default="manual",
    )
    parser.add_argument("--prompt", default="stand")
    add_vlm_arguments(parser, require_model=False, require_observation_port=False)
    parser.add_argument(
        "--vlm-reasoning",
        action="store_true",
        help="Print VLM reasoning when the server returns it.",
    )
    parser.add_argument("--query-every-blocks", type=int, default=20)
    parser.add_argument(
        "--command-hold-blocks",
        type=int,
        default=4,
        help="Generate each received command before activating a queued follow-up.",
    )
    args = parser.parse_args()
    if args.planner == "vlm" and args.query_every_blocks <= 0:
        raise ValueError(
            f"--query-every-blocks must be positive, got {args.query_every_blocks}"
        )
    if args.command_hold_blocks <= 0:
        raise ValueError(
            f"--command-hold-blocks must be positive, got {args.command_hold_blocks}"
        )
    if args.planner == "vlm":
        validate_vlm_arguments(args, planner_name=f"--planner {args.planner}")
    return args


def main() -> None:
    run_producer(parse_args())


def _log_producer_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _log_vlm_reasoning_if_available(
    *,
    planner: PromptController,
    args: argparse.Namespace,
) -> None:
    if not args.vlm_reasoning:
        return
    if not isinstance(planner, VlmPromptPlanner):
        return
    reasoning = planner.consume_pending_reasoning()
    if reasoning is not None:
        _log_producer_message(f"vlm_reasoning {reasoning}")


def _prompt_source(
    planner: PromptController,
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
            command_hold_blocks=args.command_hold_blocks,
        )
    return ManualPromptPlanner(
        args.prompt,
        command_hold_blocks=args.command_hold_blocks,
    )


if __name__ == "__main__":
    main()
