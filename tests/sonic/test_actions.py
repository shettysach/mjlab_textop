from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab_textop.core.schema import MJLAB_G1_JOINT_NAMES
from mjlab_textop.trackers.sonic.actions import (
    SonicActionBounds,
    SonicActionPostprocessor,
    resolve_sonic_action_bounds,
)


def test_action_postprocessor_clamps_then_filters() -> None:
    bounds = SonicActionBounds(
        lower=torch.full((29,), -1.0),
        upper=torch.full((29,), 1.0),
    )
    postprocessor = SonicActionPostprocessor(bounds, cutoff_hz=5.0)

    first = postprocessor(torch.full((1, 29), 2.0))
    second = postprocessor(torch.full((1, 29), -2.0))

    torch.testing.assert_close(first, torch.ones(1, 29))
    expected = 1.0 + postprocessor.alpha * (-1.0 - 1.0)
    torch.testing.assert_close(second, torch.full((1, 29), expected))


def test_action_postprocessor_reset_discards_filter_history() -> None:
    postprocessor = SonicActionPostprocessor(bounds=None, cutoff_hz=5.0)
    postprocessor(torch.ones(1, 29))

    postprocessor.reset(torch.tensor([0]))
    action = postprocessor(torch.full((1, 29), -1.0))

    torch.testing.assert_close(action, torch.full((1, 29), -1.0))


def test_resolve_action_bounds_accounts_for_scale_offset_and_encoder_bias() -> None:
    scale = torch.full((1, 29), 2.0)
    offset = torch.full((1, 29), 0.5)
    encoder_bias = torch.full((1, 29), 0.1)
    limits = torch.stack(
        [
            torch.full((1, 29), -0.9),
            torch.full((1, 29), 1.1),
        ],
        dim=-1,
    )
    action_term = SimpleNamespace(
        target_names=list(MJLAB_G1_JOINT_NAMES),
        target_ids=torch.arange(29),
        scale=scale,
        offset=offset,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros(1, 29),
            encoder_bias=encoder_bias,
            soft_joint_pos_limits=limits,
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            get_term=lambda name: action_term if name == "joint_pos" else None
        ),
    )

    bounds = resolve_sonic_action_bounds(env)

    torch.testing.assert_close(bounds.lower, torch.full((29,), -0.65))
    torch.testing.assert_close(bounds.upper, torch.full((29,), 0.35))


def test_action_postprocessor_rejects_invalid_cutoff() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SonicActionPostprocessor(bounds=None, cutoff_hz=25.0)
