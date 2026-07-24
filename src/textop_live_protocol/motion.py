from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from textop_live_protocol.g1 import G1_JOINT_COUNT


@dataclass(frozen=True)
class MotionFrames:
    """Numerical TextOp reference frames, independent of stream control."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


@dataclass(frozen=True)
class StreamControl:
    """Producer metadata used to coordinate prompts and collision recovery."""

    prompt: str | None = None
    recovery_epoch: int = 0


@dataclass(frozen=True)
class MotionBlock:
    """Indexed stream envelope containing motion frames and control metadata.

    Joint arrays are in TextOp/IsaacLab order. The rolling buffer converts them
    to MJLab order when appending.
    """

    index: int
    motion: MotionFrames
    control: StreamControl = field(default_factory=StreamControl)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.motion.joint_pos

    @property
    def joint_vel(self) -> np.ndarray:
        return self.motion.joint_vel

    @property
    def anchor_pos_w(self) -> np.ndarray:
        return self.motion.anchor_pos_w

    @property
    def anchor_quat_w(self) -> np.ndarray:
        return self.motion.anchor_quat_w


def validate_motion_block(block: MotionBlock) -> MotionBlock:
    joint_pos = validate_g1_joint_frames("joint_pos", block.joint_pos)
    joint_vel = validate_g1_joint_frames("joint_vel", block.joint_vel)
    anchor_pos_w = validate_frame_vector_array("anchor_pos_w", block.anchor_pos_w, 3)
    anchor_quat_w = normalize_quat(
        validate_frame_vector_array("anchor_quat_w", block.anchor_quat_w, 4)
    )

    if block.index < 0:
        raise ValueError(f"Block index must be non-negative, got {block.index}")
    if block.control.prompt is not None and (
        not isinstance(block.control.prompt, str) or not block.control.prompt.strip()
    ):
        raise ValueError("Block prompt must be a non-empty string or None")
    if (
        not isinstance(block.control.recovery_epoch, int)
        or isinstance(block.control.recovery_epoch, bool)
        or block.control.recovery_epoch < 0
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
        motion=MotionFrames(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        ),
        control=block.control,
    )


def validate_g1_joint_frames(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != G1_JOINT_COUNT:
        raise ValueError(
            f"{name} must be shaped [T, {G1_JOINT_COUNT}], got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    return array


def validate_frame_vector_array(
    name: str,
    value: np.ndarray,
    width: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be shaped [T, {width}], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    return array


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / norm).astype(np.float32)
