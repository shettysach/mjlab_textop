from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
)

from mjlab_textop.core.online.buffer import RollingMotionBuffer


@dataclass(frozen=True)
class FutureWindow:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    anchor_pos_w: torch.Tensor
    anchor_quat_w: torch.Tensor
    stale_steps: int


class OnlineReferenceWindow:
    """Own future-window assembly, caching, staleness, and anchor alignment."""

    def __init__(
        self,
        buffer: RollingMotionBuffer,
        *,
        num_envs: int,
        device: torch.device | str,
        future_steps: int,
    ) -> None:
        self.buffer = buffer
        self.future_steps = future_steps
        self.reference_start_anchor_pos_w = torch.zeros(num_envs, 3, device=device)
        self.robot_start_anchor_pos_w = torch.zeros(num_envs, 3, device=device)
        self.last_stale_steps = 0
        self.consecutive_stale_steps = 0
        self._last_stale_frame: int | None = None
        self._cache_frame: int | None = None
        self._cache: FutureWindow | None = None

    def reset(self) -> None:
        self.last_stale_steps = 0
        self.consecutive_stale_steps = 0
        self._last_stale_frame = None
        self.reference_start_anchor_pos_w.zero_()
        self.robot_start_anchor_pos_w.zero_()
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache_frame = None
        self._cache = None

    def cached_for(self, frame: int) -> FutureWindow | None:
        return self._cache if self._cache_frame == frame else None

    def align(self, frame: int, robot_anchor_pos_w: torch.Tensor) -> None:
        _, _, anchor_pos_w, _, _ = self.buffer.get_future(frame, 1)
        self.reference_start_anchor_pos_w = anchor_pos_w.expand(
            robot_anchor_pos_w.shape[0], -1
        )
        self.robot_start_anchor_pos_w = robot_anchor_pos_w.clone()
        self.clear_cache()

    def translate_anchor(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        """Place the raw reference origin at the robot's startup anchor."""
        return (
            self.robot_start_anchor_pos_w[0]
            + anchor_pos_w
            - self.reference_start_anchor_pos_w[0]
        )

    def reference_root_velocity(self, frame: int, *, dt: float) -> torch.Tensor:
        """Estimate world-frame root velocity from adjacent reference poses."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        pair_start = frame
        if not self.buffer.can_start(pair_start, 2):
            pair_start = frame - 1
        if pair_start < 0 or not self.buffer.can_start(pair_start, 2):
            return torch.zeros(6, device=self.buffer.device)

        _, _, anchor_pos_w, anchor_quat_w, _ = self.buffer.get_future(pair_start, 2)
        linear_velocity_w = (anchor_pos_w[1] - anchor_pos_w[0]) / dt
        relative_quat = quat_mul(
            anchor_quat_w[1:2],
            quat_conjugate(anchor_quat_w[0:1]),
        )
        angular_velocity_w = axis_angle_from_quat(relative_quat)[0] / dt
        return torch.cat([linear_velocity_w, angular_velocity_w], dim=-1)

    def get(self, frame: int) -> FutureWindow:
        cached = self.cached_for(frame)
        if cached is not None:
            return cached

        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = (
            self.buffer.get_future(frame, self.future_steps)
        )
        window = FutureWindow(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=self.translate_anchor(anchor_pos_w),
            anchor_quat_w=anchor_quat_w,
            stale_steps=stale_steps,
        )
        self.last_stale_steps = stale_steps
        if self._last_stale_frame != frame:
            if stale_steps > 0:
                self.consecutive_stale_steps += 1
            else:
                self.consecutive_stale_steps = 0
            self._last_stale_frame = frame

        # Missing future frames are clamped by the rolling buffer. Retain the
        # stale count so live deployments can surface underruns without aborting.
        self._cache_frame = frame
        self._cache = window
        return window
