from __future__ import annotations

import torch

from mjlab_textop.trackers.sonic.constants import (
    SONIC_DECODER_INPUT_DIM,
    SONIC_ENCODER_INPUT_DIM,
    SONIC_HISTORY_FRAMES,
    SONIC_JOINT_COUNT,
    SONIC_RAW_OBSERVATION_DIM,
    SONIC_TOKEN_DIM,
)

RAW_REFERENCE_JOINT_POS = slice(0, 290)
RAW_REFERENCE_JOINT_VEL = slice(290, 580)
RAW_REFERENCE_ANCHOR_ORI = slice(580, 640)
RAW_BASE_ANG_VEL = slice(640, 643)
RAW_JOINT_POS = slice(643, 672)
RAW_JOINT_VEL = slice(672, 701)
RAW_LAST_ACTION = slice(701, 730)
RAW_GRAVITY = slice(730, 733)

ENCODER_MODE = slice(0, 4)
ENCODER_REFERENCE_JOINT_POS = slice(4, 294)
ENCODER_REFERENCE_JOINT_VEL = slice(294, 584)
ENCODER_REFERENCE_ANCHOR_ORI = slice(584, 644)

PROPRIO_BASE_ANG_VEL = slice(0, 3)
PROPRIO_JOINT_POS = slice(3, 32)
PROPRIO_JOINT_VEL = slice(32, 61)
PROPRIO_LAST_ACTION = slice(61, 90)
PROPRIO_GRAVITY = slice(90, 93)


class SonicInputBuilder:
    """Build released low-latency SONIC encoder and decoder inputs."""

    def __init__(self) -> None:
        self._history: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._history is None:
            return
        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids] = 0.0

    def build_encoder_input(self, actor_obs: torch.Tensor) -> torch.Tensor:
        self._validate_actor_obs(actor_obs)
        encoder_input = actor_obs.new_zeros(
            actor_obs.shape[0],
            SONIC_ENCODER_INPUT_DIM,
        )
        # G1 joint-reference encoder mode is zero. The remaining three mode
        # slots and all unused teleop/SMPL modalities intentionally stay zero.
        encoder_input[:, ENCODER_MODE] = 0.0
        encoder_input[:, ENCODER_REFERENCE_JOINT_POS] = actor_obs[
            :, RAW_REFERENCE_JOINT_POS
        ]
        encoder_input[:, ENCODER_REFERENCE_JOINT_VEL] = actor_obs[
            :, RAW_REFERENCE_JOINT_VEL
        ]
        encoder_input[:, ENCODER_REFERENCE_ANCHOR_ORI] = actor_obs[
            :, RAW_REFERENCE_ANCHOR_ORI
        ]
        return encoder_input

    def build_decoder_input(
        self,
        token: torch.Tensor,
        actor_obs: torch.Tensor,
        *,
        last_policy_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_actor_obs(actor_obs)
        if token.ndim != 2 or token.shape != (
            actor_obs.shape[0],
            SONIC_TOKEN_DIM,
        ):
            raise ValueError(
                "SONIC token must be shaped "
                f"[{actor_obs.shape[0]}, {SONIC_TOKEN_DIM}], "
                f"got {tuple(token.shape)}"
            )

        current = actor_obs[:, RAW_BASE_ANG_VEL.start : RAW_GRAVITY.stop]
        if last_policy_action is not None:
            self._validate_last_policy_action(last_policy_action, actor_obs)
            current = current.clone()
            current[:, PROPRIO_LAST_ACTION] = last_policy_action
        self._append_history(current)
        assert self._history is not None
        history = self._history

        decoder_input = torch.cat(
            [
                token,
                history[:, :, PROPRIO_BASE_ANG_VEL].reshape(actor_obs.shape[0], -1),
                history[:, :, PROPRIO_JOINT_POS].reshape(actor_obs.shape[0], -1),
                history[:, :, PROPRIO_JOINT_VEL].reshape(actor_obs.shape[0], -1),
                history[:, :, PROPRIO_LAST_ACTION].reshape(actor_obs.shape[0], -1),
                history[:, :, PROPRIO_GRAVITY].reshape(actor_obs.shape[0], -1),
            ],
            dim=-1,
        )
        if decoder_input.shape[1] != SONIC_DECODER_INPUT_DIM:
            raise AssertionError(
                "Internal SONIC decoder layout error: "
                f"expected {SONIC_DECODER_INPUT_DIM}, got {decoder_input.shape[1]}"
            )
        return decoder_input

    def _append_history(self, current: torch.Tensor) -> None:
        expected_shape = (
            current.shape[0],
            SONIC_HISTORY_FRAMES,
            current.shape[1],
        )
        if (
            self._history is None
            or self._history.shape != expected_shape
            or self._history.device != current.device
            or self._history.dtype != current.dtype
        ):
            self._history = current.new_zeros(expected_shape)
        else:
            self._history[:, :-1] = self._history[:, 1:].clone()
        self._history[:, -1] = current

    @staticmethod
    def _validate_actor_obs(actor_obs: torch.Tensor) -> None:
        if actor_obs.ndim != 2 or actor_obs.shape[1] != SONIC_RAW_OBSERVATION_DIM:
            raise ValueError(
                "SONIC actor observation must be shaped "
                f"[B, {SONIC_RAW_OBSERVATION_DIM}], got {tuple(actor_obs.shape)}"
            )

    @staticmethod
    def _validate_last_policy_action(
        value: torch.Tensor,
        actor_obs: torch.Tensor,
    ) -> None:
        expected_shape = (actor_obs.shape[0], SONIC_JOINT_COUNT)
        if value.shape != expected_shape:
            raise ValueError(
                "SONIC last policy action must be shaped "
                f"{expected_shape}, got {tuple(value.shape)}"
            )
        if value.device != actor_obs.device:
            raise ValueError(
                "SONIC last policy action and actor observation must be on the "
                f"same device, got {value.device} and {actor_obs.device}"
            )
