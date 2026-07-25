from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab_textop.core.online.window import FutureWindow
from textop_protocol.motion import MotionBlock


@dataclass
class CollisionRecovery:
    """State machine for collision latching and recovery-block acceptance."""

    active: bool = False
    epoch: int = 0
    contact_active: bool = False
    hold_window: FutureWindow | None = None

    def collision_edge(self, in_collision: bool) -> bool:
        if not in_collision:
            self.contact_active = False
            return False
        if self.contact_active:
            return False
        self.contact_active = True
        return True

    def activate(self, safe_window: FutureWindow) -> int:
        self.active = True
        self.epoch += 1
        self.hold_window = make_stationary_window(safe_window)
        return self.epoch

    def accepts(self, block: MotionBlock) -> bool:
        return (
            block.control.prompt is not None
            and block.control.prompt.strip().lower() == "stand"
            and block.control.recovery_epoch == self.epoch
        )

    def complete(self) -> None:
        self.active = False
        self.hold_window = None

    def reset(self) -> None:
        self.active = False
        self.contact_active = False
        self.hold_window = None


def make_stationary_window(window: FutureWindow) -> FutureWindow:
    future_steps = window.joint_pos.shape[0]
    return FutureWindow(
        joint_pos=window.joint_pos[0].repeat(future_steps, 1),
        joint_vel=torch.zeros_like(window.joint_vel),
        anchor_pos_w=window.anchor_pos_w[0].repeat(future_steps, 1),
        anchor_quat_w=window.anchor_quat_w[0].repeat(future_steps, 1),
        stale_steps=0,
    )
