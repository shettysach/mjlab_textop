from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mjlab_textop.core.motion import (
    normalize_quat,
    validate_frame_vector_array,
    validate_g1_joint_frames,
)


@dataclass(frozen=True)
class MotionBlock:
    """Block of online TextOp reference motion frames.

    Joint arrays are in TextOp/IsaacLab order. The rolling buffer converts them
    to MJLab order when appending.
    """

    index: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray
    prompt: str | None = None
    recovery_epoch: int = 0


def validate_motion_block(block: MotionBlock) -> MotionBlock:
    joint_pos = validate_g1_joint_frames("joint_pos", block.joint_pos)
    joint_vel = validate_g1_joint_frames("joint_vel", block.joint_vel)
    anchor_pos_w = validate_frame_vector_array("anchor_pos_w", block.anchor_pos_w, 3)
    anchor_quat_w = normalize_quat(
        validate_frame_vector_array("anchor_quat_w", block.anchor_quat_w, 4)
    )

    if block.index < 0:
        raise ValueError(f"Block index must be non-negative, got {block.index}")
    if block.prompt is not None and (
        not isinstance(block.prompt, str) or not block.prompt.strip()
    ):
        raise ValueError("Block prompt must be a non-empty string or None")
    if (
        not isinstance(block.recovery_epoch, int)
        or isinstance(block.recovery_epoch, bool)
        or block.recovery_epoch < 0
    ):
        raise ValueError("Block recovery_epoch must be a non-negative integer")
    for name, value in (
        ("joint_vel", joint_vel),
        ("anchor_pos_w", anchor_pos_w),
        ("anchor_quat_w", anchor_quat_w),
    ):
        if value.shape[0] != joint_pos.shape[0]:
            raise ValueError(
                f"{name} frame count {value.shape[0]} differs from "
                f"joint_pos frame count {joint_pos.shape[0]}"
            )

    return MotionBlock(
        index=block.index,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        anchor_pos_w=anchor_pos_w,
        anchor_quat_w=anchor_quat_w,
        prompt=block.prompt,
        recovery_epoch=block.recovery_epoch,
    )


class OnlineSource(Protocol):
    def poll(self) -> MotionBlock | None:
        """Return the next available block, or None when no block is ready."""


@runtime_checkable
class ResettableOnlineSource(OnlineSource, Protocol):
    def reset(self) -> None:
        """Reset a finite source to its initial frame."""


class QueueOnlineSource:
    def __init__(
        self,
        blocks: list[MotionBlock] | None = None,
    ) -> None:
        self._initial_blocks = tuple(blocks or [])
        self._blocks: deque[MotionBlock] = deque(self._initial_blocks)

    def append(self, block: MotionBlock) -> None:
        self._blocks.append(block)

    def poll(self) -> MotionBlock | None:
        if not self._blocks:
            return None
        return self._blocks.popleft()

    def reset(self) -> None:
        self._blocks = deque(self._initial_blocks)
