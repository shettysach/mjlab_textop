from __future__ import annotations

from dataclasses import dataclass
from math import cos
from typing import Any

import torch
from mjlab.utils.lab_api.math import matrix_from_quat


@dataclass(frozen=True)
class FallDetectionCfg:
    min_anchor_height: float | None = 0.35
    min_anchor_up_z: float | None = cos(1.2)


@dataclass(frozen=True)
class FallDetectionResult:
    fallen: bool
    reason: str | None = None


def detect_anchor_fall(
    *,
    anchor_pos_w: Any,
    anchor_quat_w: Any,
    cfg: FallDetectionCfg,
) -> FallDetectionResult:
    pos = torch.as_tensor(anchor_pos_w).detach().flatten()
    quat = torch.as_tensor(anchor_quat_w).detach().flatten()

    if cfg.min_anchor_height is not None and pos[2].item() < cfg.min_anchor_height:
        return FallDetectionResult(
            fallen=True,
            reason=f"anchor_height_below_{cfg.min_anchor_height:g}",
        )

    if cfg.min_anchor_up_z is not None:
        quat_norm = torch.linalg.vector_norm(quat)
        if quat_norm.item() <= 0.0:
            return FallDetectionResult(fallen=True, reason="invalid_anchor_quat")
        up_z = matrix_from_quat(quat / quat_norm)[2, 2].item()
        if up_z < cfg.min_anchor_up_z:
            return FallDetectionResult(
                fallen=True, reason=f"anchor_tilt_below_{cfg.min_anchor_up_z:g}"
            )

    return FallDetectionResult(fallen=False)
