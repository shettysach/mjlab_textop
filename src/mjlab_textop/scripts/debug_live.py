from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab_textop.core.motion import load_mjlab_motion
from mjlab_textop.core.online.live import parse_textop_block_message
from mjlab_textop.core.online.source import TextOpMotionBlock
from mjlab_textop.core.robotmdar import save_textop_motion_blocks_as_mjlab_npz
from mjlab_textop.core.robotmdar_record import save_robotmdar_raw_record


@dataclass(kw_only=True)
class DebugLiveCommand:
    host: str = "127.0.0.1"
    port: int = 8765
    fps: float = 50.0
    num_blocks: int = 1
    timeout_seconds: float = 10.0
    output_dir: str = "/tmp/mjlab_textop_live_debug"
    compare_motion_file: str | None = None


def debug_live_textop_stream(
    cfg: DebugLiveCommand,
    *,
    compare_motion_file: Path | None = None,
) -> None:
    if cfg.num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {cfg.num_blocks}")
    if cfg.timeout_seconds <= 0.0:
        raise ValueError(
            f"timeout_seconds must be positive, got {cfg.timeout_seconds}"
        )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks, stream_fps = _receive_live_blocks(
        host=cfg.host,
        port=cfg.port,
        default_fps=cfg.fps,
        num_blocks=cfg.num_blocks,
        timeout_seconds=cfg.timeout_seconds,
    )

    raw_path = save_robotmdar_raw_record(
        output_dir / "live_raw_textop.npz",
        blocks,
        fps=stream_fps,
        prompt="",
        guidance_scale=0.0,
        source="play-live-debug",
    )
    replay_path = output_dir / "live_mjlab_replay.npz"
    save_textop_motion_blocks_as_mjlab_npz(replay_path, blocks, fps=stream_fps)

    report = _build_report(
        blocks,
        fps=stream_fps,
        replay_path=replay_path,
        raw_path=raw_path,
        compare_motion_file=compare_motion_file,
    )
    report_path = output_dir / "live_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nWrote debug files to {output_dir}")


def _receive_live_blocks(
    *,
    host: str,
    port: int,
    default_fps: float,
    num_blocks: int,
    timeout_seconds: float,
) -> tuple[list[TextOpMotionBlock], float]:
    blocks: list[TextOpMotionBlock] = []
    stream_fps = float(default_fps)

    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        with sock.makefile("rb") as reader:
            while len(blocks) < num_blocks:
                line = reader.readline()
                if not line:
                    raise RuntimeError(
                        f"Socket closed after receiving {len(blocks)} block(s)"
                    )
                block, stream_fps = parse_textop_block_message(
                    line,
                    default_fps=stream_fps,
                )
                blocks.append(block)

    return blocks, stream_fps


def _build_report(
    blocks: list[TextOpMotionBlock],
    *,
    fps: float,
    replay_path: Path,
    raw_path: Path,
    compare_motion_file: Path | None,
) -> str:
    motion = load_mjlab_motion(replay_path)
    frame_index = _frame_indices(blocks)
    lines = [
        "Live TextOp Debug Report",
        "",
        f"blocks: {len(blocks)}",
        f"fps: {fps:g}",
        f"raw_textop_npz: {raw_path}",
        f"mjlab_replay_npz: {replay_path}",
        f"frame_index_start: {int(frame_index[0])}",
        f"frame_index_stop: {int(frame_index[-1])}",
        f"contiguous_frames: {_is_contiguous(frame_index)}",
        f"joint_pos_shape: {motion.joint_pos.shape}",
        f"joint_vel_shape: {motion.joint_vel.shape}",
        f"body_pos_w_shape: {motion.body_pos_w.shape}",
        f"body_quat_w_shape: {motion.body_quat_w.shape}",
        f"first_anchor_pos_w: {_format_array(motion.root_pos_w[0])}",
        f"anchor_z_min_max: {_format_array(_min_max(motion.root_pos_w[:, 2]))}",
        f"quat_norm_min_max: {_format_array(_min_max(np.linalg.norm(motion.root_quat_w, axis=-1)))}",
    ]

    if compare_motion_file is not None:
        lines.extend(["", *_compare_replay_motion(motion, compare_motion_file)])

    return "\n".join(lines)


def _compare_replay_motion(live_motion, compare_motion_file: Path) -> list[str]:
    expected = load_mjlab_motion(compare_motion_file)
    frames = min(live_motion.num_frames, expected.num_frames)
    if frames == 0:
        return [f"compare_motion_file: {compare_motion_file}", "compare_frames: 0"]

    live_root_pos = live_motion.root_pos_w[:frames]
    expected_root_pos = expected.root_pos_w[:frames]
    live_root_quat = live_motion.root_quat_w[:frames]
    expected_root_quat = expected.root_quat_w[:frames]

    return [
        f"compare_motion_file: {compare_motion_file}",
        f"compare_frames: {frames}",
        f"compare_fps: live={live_motion.fps} expected={expected.fps}",
        _max_abs_diff_line(
            "joint_pos",
            live_motion.joint_pos[:frames],
            expected.joint_pos[:frames],
        ),
        _max_abs_diff_line(
            "joint_vel",
            live_motion.joint_vel[:frames],
            expected.joint_vel[:frames],
        ),
        _max_abs_diff_line("root_pos_w", live_root_pos, expected_root_pos),
        _max_abs_diff_line("root_quat_w", live_root_quat, expected_root_quat),
        f"expected_first_root_pos_w: {_format_array(expected_root_pos[0])}",
        f"live_first_root_pos_w: {_format_array(live_root_pos[0])}",
    ]


def _max_abs_diff_line(name: str, lhs: np.ndarray, rhs: np.ndarray) -> str:
    return f"{name}_max_abs_diff: {float(np.max(np.abs(lhs - rhs))):.6g}"


def _frame_indices(blocks: list[TextOpMotionBlock]) -> np.ndarray:
    indices = []
    for block in blocks:
        indices.extend(range(block.index, block.index + block.joint_pos.shape[0]))
    return np.asarray(indices, dtype=np.int64)


def _is_contiguous(frame_index: np.ndarray) -> bool:
    if frame_index.shape[0] <= 1:
        return True
    return bool(np.all(np.diff(frame_index) == 1))


def _min_max(value: np.ndarray) -> np.ndarray:
    return np.asarray([np.min(value), np.max(value)], dtype=np.float32)


def _format_array(value: np.ndarray) -> str:
    return np.array2string(
        np.asarray(value, dtype=np.float32),
        precision=5,
        separator=", ",
    )
