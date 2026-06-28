from __future__ import annotations

import pytest
import torch

from mjlab_textop.core.feedback.fall import FallDetectionCfg, detect_anchor_fall


def test_detect_anchor_fall_by_height() -> None:
    result = detect_anchor_fall(
        anchor_pos_w=torch.tensor([0.0, 0.0, 0.2]),
        anchor_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        cfg=FallDetectionCfg(min_anchor_height=0.35, min_anchor_up_z=None),
    )

    assert result.fallen is True
    assert result.reason == "anchor_height_below_0.35"


def test_detect_anchor_fall_by_tilt() -> None:
    result = detect_anchor_fall(
        anchor_pos_w=torch.tensor([0.0, 0.0, 1.0]),
        anchor_quat_w=torch.tensor([0.7071068, 0.7071068, 0.0, 0.0]),
        cfg=FallDetectionCfg(min_anchor_height=None, min_anchor_up_z=0.5),
    )

    assert result.fallen is True
    assert result.reason == "anchor_tilt_below_0.5"


def test_detect_anchor_fall_allows_upright_pose() -> None:
    result = detect_anchor_fall(
        anchor_pos_w=torch.tensor([0.0, 0.0, 0.8]),
        anchor_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        cfg=FallDetectionCfg(min_anchor_height=0.35, min_anchor_up_z=0.5),
    )

    assert result.fallen is False
    assert result.reason is None


def test_detect_anchor_fall_treats_zero_quaternion_as_invalid() -> None:
    result = detect_anchor_fall(
        anchor_pos_w=torch.tensor([0.0, 0.0, 0.8]),
        anchor_quat_w=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        cfg=FallDetectionCfg(min_anchor_height=0.35, min_anchor_up_z=0.5),
    )

    assert result.fallen is True
    assert result.reason == "invalid_anchor_quat"


def test_detect_anchor_fall_rejects_wrong_vector_shapes() -> None:
    with pytest.raises(ValueError, match="anchor_pos_w must contain 3 values"):
        detect_anchor_fall(
            anchor_pos_w=torch.tensor([0.0, 0.0]),
            anchor_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            cfg=FallDetectionCfg(),
        )

    with pytest.raises(ValueError, match="anchor_quat_w must contain 4 values"):
        detect_anchor_fall(
            anchor_pos_w=torch.tensor([0.0, 0.0, 0.8]),
            anchor_quat_w=torch.tensor([1.0, 0.0, 0.0]),
            cfg=FallDetectionCfg(),
        )
