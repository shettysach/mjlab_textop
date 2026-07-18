from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mjlab_textop.core.schema import MJLAB_G1_JOINT_NAMES
from mjlab_textop.trackers.sonic.constants import (
    SONIC_CONTROL_HZ,
    SONIC_JOINT_COUNT,
)

SONIC_ACTION_FILTER_CUTOFF_HZ = 5.0


@dataclass(frozen=True)
class SonicActionBounds:
    """Normalized MJLab-order actions that keep position targets within limits."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        expected = (SONIC_JOINT_COUNT,)
        if self.lower.shape != expected or self.upper.shape != expected:
            raise ValueError(
                f"SONIC action bounds must be shaped {expected}, got "
                f"{tuple(self.lower.shape)} and {tuple(self.upper.shape)}"
            )
        if self.lower.device != self.upper.device:
            raise ValueError("SONIC action bounds must be on the same device")
        if torch.any(self.lower > self.upper):
            raise ValueError("SONIC lower action bounds exceed upper bounds")


class SonicActionPostprocessor:
    """Limit unsafe targets and suppress high-frequency decoder output."""

    def __init__(
        self,
        bounds: SonicActionBounds | None,
        *,
        cutoff_hz: float = SONIC_ACTION_FILTER_CUTOFF_HZ,
    ) -> None:
        if not 0.0 < cutoff_hz < 0.5 * SONIC_CONTROL_HZ:
            raise ValueError(
                "SONIC action filter cutoff must be between zero and the "
                f"{0.5 * SONIC_CONTROL_HZ:g} Hz Nyquist frequency"
            )
        self.bounds = bounds
        self.alpha = 1.0 - math.exp(
            -2.0 * math.pi * cutoff_hz / SONIC_CONTROL_HZ
        )
        self._previous: torch.Tensor | None = None

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != SONIC_JOINT_COUNT:
            raise ValueError(
                "SONIC action must be shaped "
                f"[B, {SONIC_JOINT_COUNT}], got {tuple(action.shape)}"
            )

        safe_action = action
        if self.bounds is not None:
            if self.bounds.lower.device != action.device:
                raise ValueError(
                    "SONIC action and bounds must be on the same device, got "
                    f"{action.device} and {self.bounds.lower.device}"
                )
            safe_action = torch.clamp(
                action,
                min=self.bounds.lower,
                max=self.bounds.upper,
            )

        previous = self._previous
        if (
            previous is not None
            and previous.shape == safe_action.shape
            and previous.device == safe_action.device
            and previous.dtype == safe_action.dtype
        ):
            safe_action = previous + self.alpha * (safe_action - previous)

        self._previous = safe_action.detach().clone()
        return safe_action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() > 0:
            # Released SONIC models support a single environment, so any
            # environment reset invalidates the complete filter state.
            self._previous = None


def resolve_sonic_action_bounds(env: Any) -> SonicActionBounds:
    """Derive normalized policy bounds from MJLab's live G1 action term."""

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    target_names = tuple(action_term.target_names)
    if target_names != MJLAB_G1_JOINT_NAMES:
        raise ValueError(
            "SONIC requires the joint-position action term in MJLab G1 order, "
            f"got {target_names}"
        )

    scale = _first_action_row(
        action_term.scale,
        width=SONIC_JOINT_COUNT,
        device=robot.data.default_joint_pos.device,
        label="scale",
    )
    if torch.any(scale <= 0.0):
        raise ValueError("SONIC action scales must be positive")
    offset = _first_action_row(
        action_term.offset,
        width=SONIC_JOINT_COUNT,
        device=scale.device,
        label="offset",
    )

    target_ids = action_term.target_ids
    limits = robot.data.soft_joint_pos_limits[0, target_ids]
    encoder_bias = robot.data.encoder_bias[0, target_ids]
    lower = (limits[:, 0] + encoder_bias - offset) / scale
    upper = (limits[:, 1] + encoder_bias - offset) / scale
    return SonicActionBounds(lower=lower, upper=upper)


def _first_action_row(
    value: float | torch.Tensor,
    *,
    width: int,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    if isinstance(value, (float, int)):
        return torch.full((width,), float(value), device=device)
    if value.ndim == 1 and value.shape == (width,):
        return value.detach().clone()
    if value.ndim == 2 and value.shape[1] == width and value.shape[0] > 0:
        return value[0].detach().clone()
    raise ValueError(
        f"SONIC action {label} must be scalar, [{width}], or [B, {width}], "
        f"got {tuple(value.shape)}"
    )
