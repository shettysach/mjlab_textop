from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import mjlab_textop.trackers.sonic.onnx as sonic_onnx
from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
from mjlab_textop.trackers.sonic.constants import (
    SONIC_DECODER_INPUT_DIM,
    SONIC_ENCODER_INPUT_DIM,
    SONIC_RAW_OBSERVATION_DIM,
    SONIC_TOKEN_DIM,
)
from mjlab_textop.trackers.sonic.onnx import (
    FROZEN_WRIST_JOINT_INDEX,
    SONIC_LOW_LATENCY_OBSERVATION_NAMES,
    SonicLowLatencyPolicy,
    SonicModelBundle,
    SonicOnnxPolicyRunner,
)

DECODER_LAST_ACTION = slice(674, 964)


class _FakeModel:
    instances: list[_FakeModel] = []

    def __init__(self, model_file, *, device, spec) -> None:
        self.model_file = Path(model_file)
        self.device = torch.device(device)
        self.spec = spec
        self.calls: list[torch.Tensor] = []
        self.__class__.instances.append(self)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.clone())
        if self.spec.output_width == SONIC_TOKEN_DIM:
            return value.new_zeros(value.shape[0], SONIC_TOKEN_DIM)
        return torch.arange(29, device=value.device, dtype=torch.float32).unsqueeze(0)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    for filename in (
        sonic_onnx.SONIC_ENCODER_FILENAME,
        sonic_onnx.SONIC_DECODER_FILENAME,
    ):
        (tmp_path / filename).write_text(filename)
    config = "\n".join(
        f'- name: "{name}"' for name in SONIC_LOW_LATENCY_OBSERVATION_NAMES
    )
    (tmp_path / sonic_onnx.SONIC_OBSERVATION_CONFIG_FILENAME).write_text(config)
    return tmp_path


def test_model_bundle_requires_complete_released_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model_encoder.onnx"):
        SonicModelBundle.from_directory(tmp_path)


def test_model_bundle_rejects_an_incompatible_observation_config(
    model_dir: Path,
) -> None:
    (model_dir / sonic_onnx.SONIC_OBSERVATION_CONFIG_FILENAME).write_text(
        '- name: "token_state"\n'
    )

    with pytest.raises(ValueError, match="not the released low-latency"):
        SonicModelBundle.from_directory(model_dir)


def test_policy_runs_encoder_decoder_and_reindexes_action(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: Path,
) -> None:
    _FakeModel.instances.clear()
    monkeypatch.setattr(sonic_onnx, "OnnxTensorModel", _FakeModel)
    bundle = SonicModelBundle.from_directory(model_dir)
    policy = SonicLowLatencyPolicy(bundle)
    actor_obs = torch.zeros(1, SONIC_RAW_OBSERVATION_DIM)
    observations = TensorDict({"actor": actor_obs}, batch_size=[1])

    action = policy(observations)

    encoder, decoder = _FakeModel.instances
    assert encoder.spec.input_width == SONIC_ENCODER_INPUT_DIM
    assert decoder.spec.input_width == SONIC_DECODER_INPUT_DIM
    assert encoder.calls[0].shape == (1, SONIC_ENCODER_INPUT_DIM)
    assert decoder.calls[0].shape == (1, SONIC_DECODER_INPUT_DIM)
    expected = torch.arange(29, dtype=torch.float32)[
        list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
    ].unsqueeze(0)
    expected[:, list(FROZEN_WRIST_JOINT_INDEX)] = 0.0
    torch.testing.assert_close(action, expected)


def test_policy_feedback_keeps_unmodified_decoder_action(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: Path,
) -> None:
    _FakeModel.instances.clear()
    monkeypatch.setattr(sonic_onnx, "OnnxTensorModel", _FakeModel)
    policy = SonicLowLatencyPolicy(SonicModelBundle.from_directory(model_dir))
    observations = _observations(0.0)

    policy(observations)
    policy(observations)

    decoder_input = _FakeModel.instances[1].calls[-1]
    action_history = decoder_input[:, DECODER_LAST_ACTION].reshape(1, 10, 29)
    expected = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(action_history[:, -1], expected)


def test_runner_loads_policy_bundle(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: Path,
) -> None:
    _FakeModel.instances.clear()
    monkeypatch.setattr(sonic_onnx, "OnnxTensorModel", _FakeModel)
    runner = SonicOnnxPolicyRunner(env=None, train_cfg={}, device="cpu")

    with pytest.raises(RuntimeError, match="has not been loaded"):
        runner.get_inference_policy()

    runner.load(model_dir)

    assert isinstance(runner.get_inference_policy(), SonicLowLatencyPolicy)


def test_policy_rejects_multiple_environments(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: Path,
) -> None:
    monkeypatch.setattr(sonic_onnx, "OnnxTensorModel", _FakeModel)
    bundle = SonicModelBundle.from_directory(model_dir)

    with pytest.raises(ValueError, match="exactly one environment"):
        SonicLowLatencyPolicy(
            bundle,
            env=SimpleNamespace(num_envs=2),
        )


def test_policy_clears_history_after_environment_reset(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: Path,
) -> None:
    _FakeModel.instances.clear()
    monkeypatch.setattr(sonic_onnx, "OnnxTensorModel", _FakeModel)
    env = SimpleNamespace(
        num_envs=1,
        episode_length_buf=torch.tensor([5]),
    )
    policy = SonicLowLatencyPolicy(
        SonicModelBundle.from_directory(model_dir),
        env=env,
    )

    policy(_observations(1.0))
    env.episode_length_buf[:] = 6
    policy(_observations(2.0))
    env.episode_length_buf[:] = 0
    policy(_observations(3.0))

    decoder_input = _FakeModel.instances[1].calls[-1]
    base_angular_velocity_history = decoder_input[:, 64:94].reshape(1, 10, 3)
    assert torch.count_nonzero(base_angular_velocity_history[:, :-1]) == 0
    torch.testing.assert_close(
        base_angular_velocity_history[:, -1],
        torch.full((1, 3), 3.0),
    )
    last_action_history = decoder_input[:, DECODER_LAST_ACTION].reshape(
        1,
        10,
        29,
    )
    assert torch.count_nonzero(last_action_history) == 0


def _observations(value: float) -> TensorDict:
    actor_obs = torch.full((1, SONIC_RAW_OBSERVATION_DIM), value)
    return TensorDict({"actor": actor_obs}, batch_size=[1])
