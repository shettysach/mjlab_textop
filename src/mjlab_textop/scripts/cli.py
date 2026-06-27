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
from mjlab_textop.scripts.policy import resolve_policy, verify_resolved

TextOpCommand: TypeAlias = NormalizeCommand | PlayOnlineCommand | PlayLiveCommand

TextOpCommandType = tyro.extras.subcommand_type_from_defaults(
    {
        "normalize": NormalizeCommand(),
        "play-online": PlayOnlineCommand(),
        "play-live": PlayLiveCommand(),
    },
)


def run_command(cfg: TextOpCommand) -> None:
    match cfg:
        case NormalizeCommand():
            input_motion_file = verify_resolved(
                Path(cfg.input_motion_file).expanduser().resolve(),
                "input motion file",
            )
            output_motion_file = Path(cfg.output_motion_file).expanduser().resolve()
            normalize(
                input_motion_file,
                output_motion_file,
                device=cfg.device,
                max_frames=cfg.max_frames,
            )
            return

        case PlayOnlineCommand():
            motion_file = verify_resolved(
                Path(cfg.motion_file).expanduser().resolve(),
                "Normalized motion file",
            )
            policy = resolve_policy(
                checkpoint_file=cfg.checkpoint_file,
                onnx_file=cfg.onnx_file,
            )
            play_online_textop_motion(
                cfg,
                motion_file=motion_file,
                policy=policy,
            )
            return

        case PlayLiveCommand():
            policy = resolve_policy(
                checkpoint_file=cfg.checkpoint_file,
                onnx_file=cfg.onnx_file,
            )
            play_live_textop_motion(
                cfg,
                policy=policy,
            )
            return


def main() -> None:
    run_command(tyro.cli(TextOpCommandType))


if __name__ == "__main__":
    main()
