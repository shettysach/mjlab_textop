from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mjlab_textop.core.robotmdar_record import save_robotmdar_raw_record
from mjlab_textop.core.schema import TEXTOP_FPS
from mjlab_textop.robotmdar.runtime import make_robotmdar_generator


def run_record(args: argparse.Namespace) -> None:
    generator = make_robotmdar_generator(args, log_dir_name="robotmdar_record")
    recorded_blocks = []
    frame_index = 0
    for block_index in range(args.num_blocks):
        block_start_time = time.monotonic()
        block = generator.next_block(
            prompt=args.prompt,
            index=frame_index,
            guidance_scale=args.guidance_scale,
        )
        recorded_blocks.append(block)
        frame_index += block.joint_pos.shape[0]

        if args.log_every_blocks > 0 and (block_index + 1) % args.log_every_blocks == 0:
            generation_ms = (time.monotonic() - block_start_time) * 1000.0
            print(
                "record "
                f"block={block_index + 1}/{args.num_blocks} "
                f"frame={frame_index} prompt={args.prompt!r} "
                f"gen_ms={generation_ms:.1f}",
                file=sys.stderr,
            )

    save_robotmdar_raw_record(
        args.output,
        recorded_blocks,
        fps=TEXTOP_FPS,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
    )
    print(
        f"Recorded {len(recorded_blocks)} RobotMDAR blocks "
        f"({frame_index} frames) to {args.output}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a raw RobotMDAR reference record without MJLab live play.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--skeleton-asset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--prompt", default="walk")
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-every-blocks", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_blocks <= 0:
        raise ValueError(f"--num-blocks must be positive, got {args.num_blocks}")
    run_record(args)


if __name__ == "__main__":
    main()
