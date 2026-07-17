from __future__ import annotations

import pytest
import torch

from mjlab_textop.trackers.sonic.constants import (
    SONIC_DECODER_INPUT_DIM,
    SONIC_ENCODER_INPUT_DIM,
    SONIC_RAW_OBSERVATION_DIM,
    SONIC_TOKEN_DIM,
)
from mjlab_textop.trackers.sonic.inputs import (
    ENCODER_REFERENCE_ANCHOR_ORI,
    ENCODER_REFERENCE_JOINT_POS,
    ENCODER_REFERENCE_JOINT_VEL,
    RAW_BASE_ANG_VEL,
    RAW_GRAVITY,
    RAW_JOINT_POS,
    RAW_JOINT_VEL,
    RAW_LAST_ACTION,
    RAW_REFERENCE_ANCHOR_ORI,
    RAW_REFERENCE_JOINT_POS,
    RAW_REFERENCE_JOINT_VEL,
    SonicInputBuilder,
)

DECODER_BASE_ANG_VEL = slice(64, 94)
DECODER_JOINT_POS = slice(94, 384)
DECODER_JOINT_VEL = slice(384, 674)
DECODER_LAST_ACTION = slice(674, 964)
DECODER_GRAVITY = slice(964, 994)


def test_encoder_input_matches_released_low_latency_layout() -> None:
    builder = SonicInputBuilder()
    actor_obs = torch.arange(
        SONIC_RAW_OBSERVATION_DIM,
        dtype=torch.float32,
    ).unsqueeze(0)

    encoder_input = builder.build_encoder_input(actor_obs)

    assert encoder_input.shape == (1, SONIC_ENCODER_INPUT_DIM)
    torch.testing.assert_close(
        encoder_input[:, ENCODER_REFERENCE_JOINT_POS],
        actor_obs[:, RAW_REFERENCE_JOINT_POS],
    )
    torch.testing.assert_close(
        encoder_input[:, ENCODER_REFERENCE_JOINT_VEL],
        actor_obs[:, RAW_REFERENCE_JOINT_VEL],
    )
    torch.testing.assert_close(
        encoder_input[:, ENCODER_REFERENCE_ANCHOR_ORI],
        actor_obs[:, RAW_REFERENCE_ANCHOR_ORI],
    )
    assert torch.count_nonzero(encoder_input[:, 644:]) == 0


def test_decoder_input_zero_pads_then_appends_history_oldest_first() -> None:
    builder = SonicInputBuilder()
    token = torch.arange(SONIC_TOKEN_DIM, dtype=torch.float32).unsqueeze(0)
    first_obs = _actor_observation(1.0)
    second_obs = _actor_observation(2.0)

    first = builder.build_decoder_input(token, first_obs)
    second = builder.build_decoder_input(token, second_obs)

    assert first.shape == (1, SONIC_DECODER_INPUT_DIM)
    torch.testing.assert_close(first[:, :SONIC_TOKEN_DIM], token)
    _assert_history(first, leading_zeros=9, values=(1.0,))
    _assert_history(second, leading_zeros=8, values=(1.0, 2.0))


def test_decoder_history_can_be_reset() -> None:
    builder = SonicInputBuilder()
    token = torch.zeros(1, SONIC_TOKEN_DIM)
    builder.build_decoder_input(token, _actor_observation(3.0))

    builder.reset()
    decoder_input = builder.build_decoder_input(token, _actor_observation(4.0))

    _assert_history(decoder_input, leading_zeros=9, values=(4.0,))


def test_decoder_history_accepts_raw_policy_action_feedback() -> None:
    builder = SonicInputBuilder()
    token = torch.zeros(1, SONIC_TOKEN_DIM)
    actor_obs = _actor_observation(1.0)
    raw_policy_action = torch.arange(29, dtype=torch.float32).unsqueeze(0)

    decoder_input = builder.build_decoder_input(
        token,
        actor_obs,
        last_policy_action=raw_policy_action,
    )

    action_history = decoder_input[:, DECODER_LAST_ACTION].reshape(1, 10, 29)
    assert torch.count_nonzero(action_history[:, :-1]) == 0
    torch.testing.assert_close(action_history[:, -1], raw_policy_action)


@pytest.mark.parametrize(
    ("actor_shape", "token_shape"),
    [
        ((1, SONIC_RAW_OBSERVATION_DIM - 1), (1, SONIC_TOKEN_DIM)),
        ((1, SONIC_RAW_OBSERVATION_DIM), (1, SONIC_TOKEN_DIM - 1)),
    ],
)
def test_input_builder_rejects_incompatible_shapes(
    actor_shape: tuple[int, int],
    token_shape: tuple[int, int],
) -> None:
    builder = SonicInputBuilder()
    actor_obs = torch.zeros(actor_shape)
    token = torch.zeros(token_shape)

    with pytest.raises(ValueError, match="SONIC"):
        builder.build_decoder_input(token, actor_obs)


def _actor_observation(value: float) -> torch.Tensor:
    actor_obs = torch.zeros(1, SONIC_RAW_OBSERVATION_DIM)
    actor_obs[:, RAW_BASE_ANG_VEL] = value
    actor_obs[:, RAW_JOINT_POS] = value
    actor_obs[:, RAW_JOINT_VEL] = value
    actor_obs[:, RAW_LAST_ACTION] = value
    actor_obs[:, RAW_GRAVITY] = value
    return actor_obs


def _assert_history(
    decoder_input: torch.Tensor,
    *,
    leading_zeros: int,
    values: tuple[float, ...],
) -> None:
    for history_slice, width in (
        (DECODER_BASE_ANG_VEL, 3),
        (DECODER_JOINT_POS, 29),
        (DECODER_JOINT_VEL, 29),
        (DECODER_LAST_ACTION, 29),
        (DECODER_GRAVITY, 3),
    ):
        history = decoder_input[:, history_slice].reshape(1, 10, width)
        assert torch.count_nonzero(history[:, :leading_zeros]) == 0
        for index, value in enumerate(values, start=leading_zeros):
            torch.testing.assert_close(
                history[:, index],
                torch.full((1, width), value),
            )
