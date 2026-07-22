from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab_textop.core.motion import (
    reindex_textop_g1_joints_to_mjlab,
)
from mjlab_textop.core.online.source import (
    MotionBlock,
    validate_motion_block,
)


@dataclass(frozen=True)
class BufferedMotionFrame:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    anchor_pos_w: torch.Tensor
    anchor_quat_w: torch.Tensor


class RollingMotionBuffer:
    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self._frames: dict[int, BufferedMotionFrame] = {}
        self._latest_index: int | None = None

    @property
    def latest_index(self) -> int | None:
        return self._latest_index

    @property
    def earliest_index(self) -> int | None:
        if not self._frames:
            return None
        return min(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._latest_index = None

    def append_block(self, block: MotionBlock) -> None:
        block = validate_motion_block(block)

        joint_pos = reindex_textop_g1_joints_to_mjlab(block.motion.joint_pos)
        joint_vel = reindex_textop_g1_joints_to_mjlab(block.motion.joint_vel)

        joint_pos_tensor = torch.as_tensor(
            joint_pos, dtype=torch.float32, device=self.device
        )
        joint_vel_tensor = torch.as_tensor(
            joint_vel, dtype=torch.float32, device=self.device
        )
        anchor_pos_w_tensor = torch.as_tensor(
            block.motion.anchor_pos_w,
            dtype=torch.float32,
            device=self.device,
        )
        anchor_quat_w_tensor = torch.as_tensor(
            block.motion.anchor_quat_w,
            dtype=torch.float32,
            device=self.device,
        )

        for offset in range(joint_pos.shape[0]):
            frame = block.index + offset
            self._frames[frame] = BufferedMotionFrame(
                joint_pos=joint_pos_tensor[offset],
                joint_vel=joint_vel_tensor[offset],
                anchor_pos_w=anchor_pos_w_tensor[offset],
                anchor_quat_w=anchor_quat_w_tensor[offset],
            )

        block_latest = block.index + joint_pos.shape[0] - 1
        self._latest_index = (
            block_latest
            if self._latest_index is None
            else max(self._latest_index, block_latest)
        )

    def discard_before(self, frame: int) -> None:
        """Discard frames that the live consumer can no longer request."""

        for index in tuple(self._frames):
            if index < frame:
                del self._frames[index]

    def can_start(self, frame: int, future_steps: int) -> bool:
        return all(
            (frame + offset) in self._frames for offset in range(future_steps)
        )

    def earliest_start_frame(self, future_steps: int) -> int | None:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        for frame in sorted(self._frames):
            if self.can_start(frame, future_steps):
                return frame
        return None

    def latest_start_frame(self, future_steps: int) -> int | None:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        for frame in sorted(self._frames, reverse=True):
            if self.can_start(frame, future_steps):
                return frame
        return None

    def get_future(
        self,
        frame: int,
        future_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {future_steps}")
        if not self._frames:
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
            torch.stack([self._frames[idx].joint_pos for idx in frames], dim=0),
            torch.stack([self._frames[idx].joint_vel for idx in frames], dim=0),
            torch.stack([self._frames[idx].anchor_pos_w for idx in frames], dim=0),
            torch.stack(
                [self._frames[idx].anchor_quat_w for idx in frames], dim=0
            ),
            stale_steps,
        )

    def _resolve_frame(self, frame: int) -> int:
        if frame in self._frames:
            return frame

        available = [idx for idx in self._frames if idx <= frame]
        if available:
            return max(available)

        raise RuntimeError(f"No available online TextOp frame at or before {frame}")
