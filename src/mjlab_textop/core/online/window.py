from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

from mjlab_textop.core.kinematics import (
    differentiate_positions,
    differentiate_quaternions,
)
from mjlab_textop.core.online.buffer import RollingMotionBuffer
from mjlab_textop.trackers.spec import ReferenceWindowSpec


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
        spec: ReferenceWindowSpec | None = None,
        future_steps: int | None = None,
    ) -> None:
        if spec is not None and future_steps is not None:
            raise ValueError("Pass either spec or future_steps, not both")
        if spec is None:
            if future_steps is None:
                raise ValueError("Reference window spec is required")
            spec = ReferenceWindowSpec(frame_offsets=tuple(range(future_steps)))

        self.buffer = buffer
        self.spec = spec
        self.reference_start_anchor_pos_w = torch.zeros(num_envs, 3, device=device)
        self.robot_start_anchor_pos_w = torch.zeros(num_envs, 3, device=device)
        self.heading_alignment_quat_w = torch.zeros(num_envs, 4, device=device)
        self.heading_alignment_quat_w[:, 0] = 1.0
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
        self.heading_alignment_quat_w.zero_()
        self.heading_alignment_quat_w[:, 0] = 1.0
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache_frame = None
        self._cache = None

    def cached_for(self, frame: int) -> FutureWindow | None:
        return self._cache if self._cache_frame == frame else None

    def align(
        self,
        frame: int,
        robot_anchor_pos_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor | None = None,
    ) -> None:
        _, _, anchor_pos_w, anchor_quat_w, _ = self.buffer.get_future(frame, 1)
        self.reference_start_anchor_pos_w = anchor_pos_w.expand(
            robot_anchor_pos_w.shape[0], -1
        )
        self.robot_start_anchor_pos_w = robot_anchor_pos_w.clone()
        self.heading_alignment_quat_w.zero_()
        self.heading_alignment_quat_w[:, 0] = 1.0
        if self.spec.align_heading:
            if robot_anchor_quat_w is None:
                raise ValueError(
                    "Robot anchor orientation is required for heading alignment"
                )
            reference_heading = yaw_quat(anchor_quat_w).expand_as(robot_anchor_quat_w)
            self.heading_alignment_quat_w = quat_mul(
                yaw_quat(robot_anchor_quat_w),
                quat_inv(reference_heading),
            )
        self.clear_cache()

    def translate_anchor(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        """Place and orient the raw reference path at the robot's startup pose."""
        relative_pos_w = anchor_pos_w - self.reference_start_anchor_pos_w[0]
        alignment = self.heading_alignment_quat_w[0].expand(relative_pos_w.shape[0], -1)
        return self.robot_start_anchor_pos_w[0] + quat_apply(
            alignment,
            relative_pos_w,
        )

    def align_anchor_quat(self, anchor_quat_w: torch.Tensor) -> torch.Tensor:
        alignment = self.heading_alignment_quat_w[0].expand_as(anchor_quat_w)
        return quat_mul(alignment, anchor_quat_w)

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
        anchor_pos_w = self.translate_anchor(anchor_pos_w)
        anchor_quat_w = self.align_anchor_quat(anchor_quat_w)
        linear_velocity_w = differentiate_positions(anchor_pos_w, dt=dt)[0]
        angular_velocity_w = differentiate_quaternions(anchor_quat_w, dt=dt)[0]
        return torch.cat([linear_velocity_w, angular_velocity_w], dim=-1)

    def get(self, frame: int) -> FutureWindow:
        cached = self.cached_for(frame)
        if cached is not None:
            return cached

        span = self.spec.required_span
        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = (
            self.buffer.get_future(frame, span)
        )
        index = torch.tensor(
            self.spec.frame_offsets,
            dtype=torch.long,
            device=self.buffer.device,
        )
        window = FutureWindow(
            joint_pos=joint_pos.index_select(0, index),
            joint_vel=joint_vel.index_select(0, index),
            anchor_pos_w=self.translate_anchor(anchor_pos_w).index_select(0, index),
            anchor_quat_w=self.align_anchor_quat(anchor_quat_w).index_select(0, index),
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
