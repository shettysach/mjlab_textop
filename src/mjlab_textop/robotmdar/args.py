from __future__ import annotations

import argparse
from pathlib import Path

from mjlab_textop.robotmdar.runtime import (
    DEFAULT_VLM_SYSTEM_PROMPT_FILE,
    DEFAULT_VLM_USER_PROMPT_FILE,
)


def add_generator_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments required to construct a RobotMDAR generator."""
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--skeleton-asset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=5.0)


def add_stream_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-every-blocks", type=int, default=20)


def add_vlm_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_model: bool,
    require_observation_port: bool,
) -> None:
    parser.add_argument("--observation-listen-host", default="127.0.0.1")
    parser.add_argument(
        "--observation-listen-port",
        type=int,
        required=require_observation_port,
        default=None,
    )
    parser.add_argument("--observation-path", default="/observation")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:9379")
    parser.add_argument("--vlm-model", required=require_model, default=None)
    parser.add_argument(
        "--vlm-system-prompt",
        type=Path,
        default=DEFAULT_VLM_SYSTEM_PROMPT_FILE,
        help="Task prompt appended to the invariant controller prompt (default: TASK.md).",
    )
    parser.add_argument(
        "--vlm-user-prompt",
        type=Path,
        default=DEFAULT_VLM_USER_PROMPT_FILE,
    )
    parser.add_argument("--vlm-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--vlm-history-length",
        type=int,
        default=5,
        help=(
            "Maximum number of user-image turns in each VLM request, including "
            "the current turn (default: 5)."
        ),
    )


def validate_vlm_arguments(args: argparse.Namespace, *, planner_name: str) -> None:
    if args.vlm_timeout_sec <= 0:
        raise ValueError(
            f"--vlm-timeout-sec must be positive, got {args.vlm_timeout_sec}"
        )
    if args.vlm_history_length <= 0:
        raise ValueError(
            f"--vlm-history-length must be positive, got {args.vlm_history_length}"
        )
    if args.observation_listen_port is None:
        raise ValueError(f"--observation-listen-port is required with {planner_name}")
    if not args.vlm_model:
        raise ValueError(f"--vlm-model is required with {planner_name}")
