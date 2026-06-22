from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from mjlab_vla.textop.contract import (
    TEXTOP_G1_JOINT_COUNT,
    TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
    TEXTOP_ROOT_BODY_INDEX,
)

MJLAB_TO_TEXTOP_G1_JOINT_INDEX: tuple[int, ...] = tuple(
    int(i) for i in np.argsort(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
)


@dataclass(frozen=True)
class TextOpMotionBlock:
    """Block of online TextOp reference motion frames.

    Joint arrays are in TextOp/IsaacLab order. The rolling buffer converts them
    to MJLab order when appending.
    """

    index: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


class TextOpOnlineSource(Protocol):
    def poll(self) -> TextOpMotionBlock | None:
        """Return the next available block, or None when no block is ready."""


class QueueTextOpOnlineSource:
    def __init__(self, blocks: list[TextOpMotionBlock] | None = None) -> None:
        self._blocks: deque[TextOpMotionBlock] = deque(blocks or [])

    def append(self, block: TextOpMotionBlock) -> None:
        self._blocks.append(block)

    def poll(self) -> TextOpMotionBlock | None:
        if not self._blocks:
            return None
        return self._blocks.popleft()


def make_mjlab_npz_replay_source(
    path: str | Path,
    *,
    block_size: int = 8,
) -> QueueTextOpOnlineSource:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    data = np.load(Path(path))
    joint_pos_mjlab = _validate_joint_array("joint_pos", data["joint_pos"])
    joint_vel_mjlab = _validate_joint_array("joint_vel", data["joint_vel"])
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)

    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must be shaped [T, B, 3], got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(
            f"body_quat_w must be shaped [T, B, 4], got {body_quat_w.shape}"
        )
    if body_pos_w.shape[:2] != body_quat_w.shape[:2]:
        raise ValueError(
            "body_pos_w/body_quat_w frame-body shapes differ: "
            f"{body_pos_w.shape[:2]} vs {body_quat_w.shape[:2]}"
        )
    if body_pos_w.shape[0] != joint_pos_mjlab.shape[0]:
        raise ValueError(
            f"body_pos_w frame count {body_pos_w.shape[0]} differs from "
            f"joint_pos frame count {joint_pos_mjlab.shape[0]}"
        )
    if body_pos_w.shape[1] <= TEXTOP_ROOT_BODY_INDEX:
        raise ValueError(
            f"body arrays must include root body index {TEXTOP_ROOT_BODY_INDEX}, "
            f"got {body_pos_w.shape[1]} bodies"
        )

    joint_pos_textop = joint_pos_mjlab[:, MJLAB_TO_TEXTOP_G1_JOINT_INDEX]
    joint_vel_textop = joint_vel_mjlab[:, MJLAB_TO_TEXTOP_G1_JOINT_INDEX]
    anchor_pos_w = body_pos_w[:, TEXTOP_ROOT_BODY_INDEX]
    anchor_quat_w = body_quat_w[:, TEXTOP_ROOT_BODY_INDEX]

    blocks = []
    for start in range(0, joint_pos_textop.shape[0], block_size):
        stop = min(start + block_size, joint_pos_textop.shape[0])
        blocks.append(
            TextOpMotionBlock(
                index=start,
                joint_pos=joint_pos_textop[start:stop],
                joint_vel=joint_vel_textop[start:stop],
                anchor_pos_w=anchor_pos_w[start:stop],
                anchor_quat_w=anchor_quat_w[start:stop],
            )
        )
    return QueueTextOpOnlineSource(blocks)


class TextOpRollingMotionBuffer:
    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        max_frames: int | None = 512,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self.device = torch.device(device)
        self.max_frames = max_frames
        self._joint_pos: dict[int, torch.Tensor] = {}
        self._joint_vel: dict[int, torch.Tensor] = {}
        self._anchor_pos_w: dict[int, torch.Tensor] = {}
        self._anchor_quat_w: dict[int, torch.Tensor] = {}
        self._latest_index: int | None = None

    @property
    def latest_index(self) -> int | None:
        return self._latest_index

    @property
    def frame_count(self) -> int:
        return len(self._joint_pos)

    def clear(self) -> None:
        self._joint_pos.clear()
        self._joint_vel.clear()
        self._anchor_pos_w.clear()
        self._anchor_quat_w.clear()
        self._latest_index = None

    def append_block(self, block: TextOpMotionBlock) -> None:
        joint_pos = _validate_joint_array("joint_pos", block.joint_pos)
        joint_vel = _validate_joint_array("joint_vel", block.joint_vel)
        anchor_pos_w = _validate_anchor_array("anchor_pos_w", block.anchor_pos_w, 3)
        anchor_quat_w = _normalize_quat(
            _validate_anchor_array("anchor_quat_w", block.anchor_quat_w, 4)
        )

        if block.index < 0:
            raise ValueError(
                f"TextOp block index must be non-negative, got {block.index}"
            )
        if joint_vel.shape[0] != joint_pos.shape[0]:
            raise ValueError(
                f"joint_vel frame count {joint_vel.shape[0]} differs from "
                f"joint_pos frame count {joint_pos.shape[0]}"
            )
        for name, value in (
            ("anchor_pos_w", anchor_pos_w),
            ("anchor_quat_w", anchor_quat_w),
        ):
            if value.shape[0] != joint_pos.shape[0]:
                raise ValueError(
                    f"{name} frame count {value.shape[0]} differs from "
                    f"joint_pos frame count {joint_pos.shape[0]}"
                )

        joint_pos = _reindex_textop_joints_to_mjlab(joint_pos)
        joint_vel = _reindex_textop_joints_to_mjlab(joint_vel)

        for offset in range(joint_pos.shape[0]):
            frame = block.index + offset
            self._joint_pos[frame] = _to_tensor(joint_pos[offset], self.device)
            self._joint_vel[frame] = _to_tensor(joint_vel[offset], self.device)
            self._anchor_pos_w[frame] = _to_tensor(anchor_pos_w[offset], self.device)
            self._anchor_quat_w[frame] = _to_tensor(anchor_quat_w[offset], self.device)

        block_latest = block.index + joint_pos.shape[0] - 1
        self._latest_index = (
            block_latest
            if self._latest_index is None
            else max(self._latest_index, block_latest)
        )
        self._evict_old_frames()

    def can_start(self, frame: int, future_steps: int) -> bool:
        return all(
            (frame + offset) in self._joint_pos for offset in range(future_steps)
        )

    def get_future(
        self,
        frame: int,
        future_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        if not self._joint_pos:
            raise RuntimeError("Online TextOp buffer has no frames")

        stale_steps = 0
        frames = []
        for offset in range(future_steps):
            requested = frame + offset
            resolved = self._resolve_frame(requested)
            if resolved != requested:
                stale_steps += 1
            frames.append(resolved)

        return (
            torch.stack([self._joint_pos[idx] for idx in frames], dim=0),
            torch.stack([self._joint_vel[idx] for idx in frames], dim=0),
            torch.stack([self._anchor_pos_w[idx] for idx in frames], dim=0),
            torch.stack([self._anchor_quat_w[idx] for idx in frames], dim=0),
            stale_steps,
        )

    def _resolve_frame(self, frame: int) -> int:
        if frame in self._joint_pos:
            return frame

        available = [idx for idx in self._joint_pos if idx <= frame]
        if available:
            return max(available)

        raise RuntimeError(f"No available online TextOp frame at or before {frame}")

    def _evict_old_frames(self) -> None:
        if self.max_frames is None or self._latest_index is None:
            return
        first_kept = self._latest_index - self.max_frames + 1
        for frame in list(self._joint_pos):
            if frame < first_kept:
                del self._joint_pos[frame]
                del self._joint_vel[frame]
                del self._anchor_pos_w[frame]
                del self._anchor_quat_w[frame]


def _validate_joint_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be shaped [T, 29], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if array.shape[1] != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            f"{name} must have {TEXTOP_G1_JOINT_COUNT} joints, got {array.shape[1]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_anchor_array(name: str, value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be shaped [T, {width}], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("anchor_quat_w contains zero-norm entries")
    return (quat / norm).astype(np.float32)


def _reindex_textop_joints_to_mjlab(values: np.ndarray) -> np.ndarray:
    return values[:, TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX]


def _to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)
