from __future__ import annotations

from dataclasses import dataclass
from math import cos
from typing import Any


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
    pos = _to_float_list(anchor_pos_w)
    quat = _to_float_list(anchor_quat_w)

    if cfg.min_anchor_height is not None and pos[2] < cfg.min_anchor_height:
        return FallDetectionResult(
            fallen=True,
            reason=(
                f"anchor_height_below_{cfg.min_anchor_height:g}"
            ),
        )

    if cfg.min_anchor_up_z is not None:
        up_z = _anchor_up_z(quat)
        if up_z < cfg.min_anchor_up_z:
            return FallDetectionResult(
                fallen=True,
                reason=f"anchor_tilt_below_{cfg.min_anchor_up_z:g}",
            )

    return FallDetectionResult(fallen=False)


def _anchor_up_z(quat_wxyz: list[float]) -> float:
    w, x, y, z = quat_wxyz
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        return 1.0
    x /= norm
    y /= norm
    return 1.0 - 2.0 * (x * x + y * y)


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in value]
