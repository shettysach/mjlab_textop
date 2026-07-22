from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import mjlab_textop.core.onnx_policy as onnx_policy
from mjlab_textop.core.onnx_policy import OnnxPolicy, OnnxPolicyRunner
from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


def _observations(actor: torch.Tensor) -> TensorDict:
    return TensorDict({"actor": actor}, batch_size=[actor.shape[0]])


class _FakeSession:
    def __init__(self, _policy_file: str, sess_options=None, providers=None):
        self.output = np.arange(29, dtype=np.float32).reshape(1, 29)
        self.inputs = [SimpleNamespace(name="actor_obs")]
        self.outputs = [SimpleNamespace(name="action")]
        self.providers = providers
        self.sess_options = sess_options
        self.received: np.ndarray | None = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return [
            provider[0] if isinstance(provider, tuple) else provider
            for provider in self.providers
        ]

    def io_binding(self):
        return SimpleNamespace()

    def run(self, _output_names, inputs):
        self.received = inputs["actor_obs"]
        return [np.repeat(self.output, self.received.shape[0], axis=0)]


class _RecordingBinding:
    def __init__(self) -> None:
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

    def clear_binding_inputs(self) -> None:
        self.inputs.clear()

    def clear_binding_outputs(self) -> None:
        self.outputs.clear()

    def bind_input(self, **kwargs) -> None:
        self.inputs.append(kwargs)

    def bind_output(self, **kwargs) -> None:
        self.outputs.append(kwargs)


class _RecordingCudaSession:
    def __init__(self) -> None:
        self.run_count = 0

    def run_with_iobinding(self, _binding) -> None:
        self.run_count += 1


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, name: str, value: str) -> None:
        self.entries[name] = value


def _install_fake_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_providers: list[str] | None = None,
) -> None:
    fake_ort = SimpleNamespace(
        InferenceSession=_FakeSession,
        SessionOptions=_FakeSessionOptions,
        get_available_providers=lambda: (
            available_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ),
    )
    monkeypatch.setattr(onnx_policy, "ort", fake_ort)


def test_textop_onnx_policy_reindexes_action_to_mjlab_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))
    obs = torch.zeros(2, 431)

    action = policy(_observations(obs))

    expected_one = torch.arange(29, dtype=torch.float32)[
        list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
    ]
    assert action.shape == (2, 29)
    torch.testing.assert_close(action, expected_one.repeat(2, 1))
    assert policy.session.providers == ["CPUExecutionProvider"]
    assert policy.session.received is not None
    assert policy.session.received.dtype == np.float32
    assert policy.session.received.shape == (2, 431)


def test_textop_onnx_policy_remains_cpu_only_when_runner_requests_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"), device="cuda:1")

    assert policy.device == torch.device("cpu")
    assert policy.session.providers == ["CPUExecutionProvider"]


def test_textop_onnx_policy_configures_cuda_provider_and_torch_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    stream = SimpleNamespace(cuda_stream=12345)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    policy = OnnxPolicy(
        Path("latest.onnx"),
        device="cuda:1",
        execution_provider="cuda",
    )

    assert policy.device == torch.device("cuda:1")
    assert policy.session.providers == [
        (
            "CUDAExecutionProvider",
            {"device_id": 1, "user_compute_stream": "12345"},
        )
    ]
    assert policy.session.sess_options.entries == {
        "session.disable_cpu_ep_fallback": "1"
    }


def test_textop_onnx_policy_rejects_cuda_when_gpu_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(
        monkeypatch,
        available_providers=["CPUExecutionProvider"],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        OnnxPolicy(
            Path("latest.onnx"),
            device="cuda:0",
            execution_provider="cuda",
        )


def test_textop_onnx_policy_rejects_cuda_provider_with_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)

    with pytest.raises(RuntimeError, match="requires an MJLab CUDA device"):
        OnnxPolicy(
            Path("latest.onnx"),
            device="cpu",
            execution_provider="cuda",
        )


def test_cuda_run_binds_torch_buffers_and_reuses_output() -> None:
    policy = object.__new__(OnnxPolicy)
    policy.device = torch.device("cpu")
    policy.device_id = 0
    policy.input_name = "actor_obs"
    policy.output_name = "action"
    policy._binding = _RecordingBinding()
    policy._output = None
    policy.session = _RecordingCudaSession()
    obs = torch.zeros(2, 431)

    first = policy._run_cuda(obs)
    second = policy._run_cuda(obs)

    assert first.data_ptr() == second.data_ptr()
    assert first.shape == (2, 29)
    assert policy.session.run_count == 2
    assert policy._binding.inputs == [
        {
            "name": "actor_obs",
            "device_type": "cuda",
            "device_id": 0,
            "element_type": np.float32,
            "shape": (2, 431),
            "buffer_ptr": obs.data_ptr(),
        }
    ]
    assert policy._binding.outputs[0]["name"] == "action"
    assert policy._binding.outputs[0]["shape"] == (2, 29)
    assert policy._binding.outputs[0]["buffer_ptr"] == first.data_ptr()


def test_textop_onnx_policy_accepts_actor_observation_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))

    action = policy(_observations(torch.zeros(1, 431)))

    assert action.shape == (1, 29)


def test_textop_onnx_policy_rejects_unbatched_obs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))

    with pytest.raises(RuntimeError, match=r"\[N, 431\]"):
        policy(TensorDict({"actor": torch.zeros(431)}, batch_size=[]))


def test_textop_onnx_policy_rejects_wrong_obs_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))

    with pytest.raises(RuntimeError, match="Expected ONNX obs dim 431"):
        policy(_observations(torch.zeros(1, 430)))


def test_textop_onnx_runner_loads_inference_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    runner = OnnxPolicyRunner(env=None, train_cfg={}, device="cpu")

    with pytest.raises(RuntimeError, match="has not been loaded"):
        runner.get_inference_policy()

    runner.load(Path("latest.onnx"), load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy(device="cpu")

    assert isinstance(policy, OnnxPolicy)
    assert policy(_observations(torch.zeros(1, 431))).shape == (1, 29)


def test_onnx_runner_reads_execution_provider_from_config() -> None:
    runner = OnnxPolicyRunner(
        env=None,
        train_cfg={"onnx_execution_provider": "cuda"},
        device="cuda:0",
    )

    assert runner.execution_provider == "cuda"
