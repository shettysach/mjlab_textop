from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_box_minus


def differentiate_positions(positions_w: torch.Tensor, *, dt: float) -> torch.Tensor:
    """Differentiate a world-frame position sequence along its first axis."""
    _validate_sequence(positions_w, width=3, dt=dt)
    if positions_w.shape[0] < 2:
        return torch.zeros_like(positions_w)
    return torch.gradient(positions_w, spacing=dt, dim=0)[0]


def differentiate_quaternions(
    quaternions_w: torch.Tensor, *, dt: float
) -> torch.Tensor:
    """Differentiate a world-frame wxyz quaternion sequence into angular velocity."""
    _validate_sequence(quaternions_w, width=4, dt=dt)
    frame_count = quaternions_w.shape[0]
    angular_velocity_w = torch.zeros(
        (frame_count, 3),
        dtype=quaternions_w.dtype,
        device=quaternions_w.device,
    )
    if frame_count < 2:
        return angular_velocity_w

    angular_velocity_w[0] = (
        quat_box_minus(quaternions_w[1:2], quaternions_w[0:1])[0] / dt
    )
    angular_velocity_w[-1] = (
        quat_box_minus(quaternions_w[-1:], quaternions_w[-2:-1])[0] / dt
    )
    if frame_count > 2:
        angular_velocity_w[1:-1] = quat_box_minus(
            quaternions_w[2:], quaternions_w[:-2]
        ) / (2.0 * dt)
    return angular_velocity_w


def _validate_sequence(value: torch.Tensor, *, width: int, dt: float) -> None:
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"Expected sequence shaped [T, {width}], got {value.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
