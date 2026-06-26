from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import tyro

from mjlab_textop.core.normalize import normalize
from mjlab_textop.scripts.normalize import NormalizeCommand
from mjlab_textop.scripts.play_live import PlayLiveCommand, play_live_textop_motion
from mjlab_textop.scripts.play_online import (
    PlayOnlineCommand,
    play_online_textop_motion,
)
from mjlab_textop.scripts.play_onnx import (
    PlayLiveOnnxCommand,
    PlayOnlineOnnxCommand,
    play_live_textop_onnx,
    play_online_textop_onnx,
)

TextOpCommand: TypeAlias = (
    NormalizeCommand
    | PlayOnlineCommand
    | PlayLiveCommand
    | PlayOnlineOnnxCommand
    | PlayLiveOnnxCommand
)

TextOpCommandType = tyro.extras.subcommand_type_from_defaults(
    {
        "normalize": NormalizeCommand(),
        "play-online": PlayOnlineCommand(),
        "play-live": PlayLiveCommand(),
        "play-online-onnx": PlayOnlineOnnxCommand(),
        "play-live-onnx": PlayLiveOnnxCommand(),
    },
)


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def verify_resolved(resolved: Path, label: str) -> Path:
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def verify_path(path: str, label: str) -> Path:
    return verify_resolved(resolve_path(path), label)


def run_command(cfg: TextOpCommand) -> None:
    match cfg:
        case NormalizeCommand():
            input_motion_file = verify_path(cfg.input_motion_file, "input motion file")
            output_motion_file = resolve_path(cfg.output_motion_file)
            normalize(
                input_motion_file,
                output_motion_file,
                device=cfg.device,
                max_frames=cfg.max_frames,
            )
            return

        case PlayOnlineCommand():
            motion_file = verify_path(
                cfg.motion_file,
                "Normalized motion file",
            )
            checkpoint_file = verify_path(
                cfg.checkpoint_file,
                "Checkpoint file",
            )
            play_online_textop_motion(
                cfg,
                motion_file=motion_file,
                checkpoint_file=checkpoint_file,
            )
            return

        case PlayLiveCommand():
            checkpoint_file = verify_path(
                cfg.checkpoint_file,
                "Checkpoint file",
            )
            play_live_textop_motion(
                cfg,
                checkpoint_file=checkpoint_file,
            )
            return

        case PlayOnlineOnnxCommand():
            motion_file = verify_path(
                cfg.motion_file,
                "Motion file",
            )
            policy_file = verify_path(
                cfg.policy_file,
                "ONNX policy file",
            )
            play_online_textop_onnx(
                cfg,
                motion_file=motion_file,
                policy_file=policy_file,
            )
            return

        case PlayLiveOnnxCommand():
            policy_file = verify_path(
                cfg.policy_file,
                "ONNX policy file",
            )
            play_live_textop_onnx(
                cfg,
                policy_file=policy_file,
            )
            return


def main() -> None:
    run_command(tyro.cli(TextOpCommandType))


if __name__ == "__main__":
    main()
