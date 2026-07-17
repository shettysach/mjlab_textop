from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from mjlab_textop.core.schema import (
    ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
    MJLAB_G1_JOINT_NAMES,
)
from mjlab_textop.trackers.onnx import OnnxModelSpec, OnnxTensorModel
from mjlab_textop.trackers.sonic.constants import (
    SONIC_DECODER_INPUT_DIM,
    SONIC_ENCODER_INPUT_DIM,
    SONIC_JOINT_COUNT,
    SONIC_TOKEN_DIM,
)
from mjlab_textop.trackers.sonic.inputs import SonicInputBuilder

SONIC_ENCODER_FILENAME = "model_encoder.onnx"
SONIC_DECODER_FILENAME = "model_decoder.onnx"
SONIC_OBSERVATION_CONFIG_FILENAME = "observation_config.yaml"
SONIC_LOW_LATENCY_OBSERVATION_NAMES = (
    "token_state",
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
    "encoder_mode_4",
    "motion_joint_positions_10frame_step1",
    "motion_joint_velocities_10frame_step1",
    "motion_anchor_orientation_10frame_step1",
    "motion_anchor_orientation",
    "motion_joint_positions_lowerbody_10frame_step1",
    "motion_joint_velocities_lowerbody_10frame_step1",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "smpl_joints_4frame_step1",
    "smpl_anchor_orientation_4frame_step1",
    "motion_joint_positions_wrists_4frame_step1",
)
_CONFIG_NAME_PATTERN = re.compile(
    r"""^\s*-\s+name:\s*["']?([^"'#\s]+)""",
    flags=re.MULTILINE,
)
FROZEN_WRIST_JOINT_INDEX = tuple(
    index for index, name in enumerate(MJLAB_G1_JOINT_NAMES) if "_wrist_" in name
)


@dataclass(frozen=True)
class SonicModelBundle:
    directory: Path
    encoder: Path
    decoder: Path
    observation_config: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> SonicModelBundle:
        resolved = Path(directory).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"SONIC model directory does not exist: {resolved}")

        files = {
            "encoder": resolved / SONIC_ENCODER_FILENAME,
            "decoder": resolved / SONIC_DECODER_FILENAME,
            "observation_config": resolved / SONIC_OBSERVATION_CONFIG_FILENAME,
        }
        missing = [path.name for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"SONIC model directory {resolved} is missing: {', '.join(missing)}"
            )
        _validate_low_latency_observation_config(files["observation_config"])
        return cls(directory=resolved, **files)


class SonicLowLatencyPolicy:
    """Released low-latency SONIC encoder/decoder for one MJLab G1."""

    def __init__(
        self,
        bundle: SonicModelBundle,
        *,
        device: str = "cpu",
        env: Any | None = None,
    ) -> None:
        base_env = getattr(env, "unwrapped", env)
        num_envs = getattr(base_env, "num_envs", 1)
        if num_envs != 1:
            raise ValueError(
                "The released SONIC ONNX models require exactly one environment, "
                f"got {num_envs}"
            )

        self.encoder = OnnxTensorModel(
            bundle.encoder,
            device=device,
            spec=OnnxModelSpec(
                input_name="obs_dict",
                input_width=SONIC_ENCODER_INPUT_DIM,
                output_name="encoded_tokens",
                output_width=SONIC_TOKEN_DIM,
                batch_size=1,
            ),
        )
        self.decoder = OnnxTensorModel(
            bundle.decoder,
            device=device,
            spec=OnnxModelSpec(
                input_name="obs_dict",
                input_width=SONIC_DECODER_INPUT_DIM,
                output_name="action",
                output_width=SONIC_JOINT_COUNT,
                batch_size=1,
            ),
        )
        self.device = self.decoder.device
        self.input_builder = SonicInputBuilder()
        self._env = base_env
        self._last_episode_steps: torch.Tensor | None = None
        self._last_policy_action: torch.Tensor | None = None
        self._sonic_to_mjlab: dict[torch.device, torch.Tensor] = {}
        self._wrist_index: dict[torch.device, torch.Tensor] = {}

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = obs["actor"]
        if not isinstance(actor_obs, torch.Tensor):
            raise TypeError("SONIC actor observation must be a tensor")
        self._reset_finished_histories()
        encoder_input = self.input_builder.build_encoder_input(actor_obs)
        token = self.encoder(encoder_input)
        last_policy_action = self._last_policy_action
        if last_policy_action is None:
            last_policy_action = actor_obs.new_zeros(
                actor_obs.shape[0],
                SONIC_JOINT_COUNT,
            )
        decoder_input = self.input_builder.build_decoder_input(
            token,
            actor_obs,
            last_policy_action=last_policy_action,
        )
        raw_action = self.decoder(decoder_input)
        # SONIC feeds the unmodified policy output back into its action history.
        # Wrist targets are frozen only on the separate action sent to MJLab.
        self._last_policy_action = raw_action.detach().clone()
        action = raw_action.index_select(
            -1,
            self._joint_order_index(raw_action.device),
        )
        action[:, self._frozen_wrist_index(action.device)] = 0.0
        return action

    def reset(self) -> None:
        self.input_builder.reset()
        self._last_episode_steps = None
        self._last_policy_action = None

    def _reset_finished_histories(self) -> None:
        if self._env is None or not hasattr(self._env, "episode_length_buf"):
            return
        episode_steps = self._env.episode_length_buf
        if self._last_episode_steps is not None:
            reset_ids = torch.nonzero(
                episode_steps < self._last_episode_steps,
                as_tuple=False,
            ).flatten()
            if reset_ids.numel():
                self.input_builder.reset(reset_ids)
                if self._last_policy_action is not None:
                    self._last_policy_action[reset_ids] = 0.0
        self._last_episode_steps = episode_steps.clone()

    def _joint_order_index(self, device: torch.device) -> torch.Tensor:
        index = self._sonic_to_mjlab.get(device)
        if index is None:
            index = torch.tensor(
                ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
                dtype=torch.long,
                device=device,
            )
            self._sonic_to_mjlab[device] = index
        return index

    def _frozen_wrist_index(self, device: torch.device) -> torch.Tensor:
        index = self._wrist_index.get(device)
        if index is None:
            index = torch.tensor(
                FROZEN_WRIST_JOINT_INDEX,
                dtype=torch.long,
                device=device,
            )
            self._wrist_index[device] = index
        return index


class SonicOnnxPolicyRunner:
    """MJLab runner adapter for the released low-latency SONIC model bundle."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        del train_cfg, log_dir
        self.env = env
        self.device = device
        self.policy: SonicLowLatencyPolicy | None = None

    def load(self, path: str | Path, *_args: Any, **_kwargs: Any) -> None:
        bundle = SonicModelBundle.from_directory(path)
        self.policy = SonicLowLatencyPolicy(
            bundle,
            device=self.device,
            env=self.env,
        )

    def get_inference_policy(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> SonicLowLatencyPolicy:
        if self.policy is None:
            raise RuntimeError("SONIC ONNX policy has not been loaded")
        return self.policy


def _validate_low_latency_observation_config(path: Path) -> None:
    names = tuple(_CONFIG_NAME_PATTERN.findall(path.read_text()))
    expected = SONIC_LOW_LATENCY_OBSERVATION_NAMES
    if names[: len(expected)] != expected:
        raise ValueError(
            f"{path} is not the released low-latency SONIC observation config"
        )
