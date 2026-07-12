from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_textop.core.onnx_policy import (
    OnnxPolicy,
    OnnxPolicyRunner,
    _check_cuda_iobinding_obs,
    _cuda_device_id,
)
from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class _FakeSession:
    def __init__(self, _policy_file: str, providers=None):
        self.output = np.arange(29, dtype=np.float32).reshape(1, 29)
        self.inputs = [SimpleNamespace(name="actor_obs")]
        self.outputs = [SimpleNamespace(name="action")]
        self.providers = providers
        self.received: np.ndarray | None = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, _output_names, inputs):
        self.received = inputs["actor_obs"]
        return [np.repeat(self.output, self.received.shape[0], axis=0)]


def _install_fake_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = SimpleNamespace(
        InferenceSession=_FakeSession,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


def _install_fake_cuda_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _NullContext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=1234),
    )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def test_textop_onnx_policy_reindexes_action_to_mjlab_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))
    obs = torch.zeros(2, 431)

    action = policy(obs)

    expected_one = torch.arange(29, dtype=torch.float32)[
        list(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
    ]
    assert action.shape == (2, 29)
    torch.testing.assert_close(action, expected_one.repeat(2, 1))
    assert policy.session.providers == ["CPUExecutionProvider"]
    assert policy.session.received is not None
    assert policy.session.received.dtype == np.float32
    assert policy.session.received.shape == (2, 431)


def test_textop_onnx_policy_uses_cuda_provider_for_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    _install_fake_cuda_stream(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"), device="cuda:1")

    assert policy.session.providers == [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 1,
                "user_compute_stream": "1234",
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


def test_textop_onnx_policy_requires_cuda_provider_for_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ort = SimpleNamespace(
        InferenceSession=_FakeSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError, match="CUDA provider is not available"):
        OnnxPolicy(Path("latest.onnx"), device="cuda:0")


def test_textop_onnx_policy_accepts_cpu_obs_for_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    _install_fake_cuda_stream(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"), device="cuda:0")

    with pytest.raises(RuntimeError, match="Expected ONNX CUDA obs on cuda:0"):
        policy(torch.zeros(1, 431))


def test_textop_onnx_policy_uses_zero_cuda_device_id_for_plain_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    _install_fake_cuda_stream(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"), device="cuda")

    assert policy.onnx_device == torch.device("cuda:0")
    assert policy.session.providers == [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "user_compute_stream": "1234",
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


def test_cuda_device_id_is_an_integer() -> None:
    assert _cuda_device_id(torch.device("cuda")) == 0
    assert _cuda_device_id(torch.device("cuda:1")) == 1


def test_textop_onnx_policy_accepts_actor_observation_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))

    action = policy({"actor": torch.zeros(1, 431)})

    assert action.shape == (1, 29)


def test_textop_onnx_policy_rejects_unbatched_obs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))

    with pytest.raises(RuntimeError, match=r"\[N, 431\]"):
        policy(torch.zeros(431))


def test_textop_onnx_policy_rejects_wrong_obs_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = OnnxPolicy(Path("latest.onnx"))

    with pytest.raises(RuntimeError, match="Expected ONNX obs dim 431"):
        policy(torch.zeros(1, 430))


def test_cuda_iobinding_check_rejects_wrong_dtype() -> None:
    obs = torch.zeros(1, 431, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="dtype float32"):
        _check_cuda_iobinding_obs(obs, expected_device=torch.device("cpu"))


def test_cuda_iobinding_check_rejects_noncontiguous_obs() -> None:
    obs = torch.zeros(2, 862, dtype=torch.float32)[:, ::2]

    with pytest.raises(RuntimeError, match="contiguous"):
        _check_cuda_iobinding_obs(obs, expected_device=torch.device("cpu"))


def test_cuda_iobinding_check_rejects_grad_obs() -> None:
    obs = torch.zeros(1, 431, dtype=torch.float32, requires_grad=True)

    with pytest.raises(RuntimeError, match="detached"):
        _check_cuda_iobinding_obs(obs, expected_device=torch.device("cpu"))


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
    assert policy({"actor": torch.zeros(1, 431)}).shape == (1, 29)
