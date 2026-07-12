from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class OnnxPolicy:
    """Run an ONNX actor and convert its action to MJLab joint order."""

    def __init__(self, policy_file: Path, device: str = "cpu"):
        import onnxruntime as ort

        self.onnx_device = torch.device(device)
        if self.onnx_device.type == "cuda" and self.onnx_device.index is None:
            self.onnx_device = torch.device("cuda:0")
        providers = _onnx_providers_for_device(ort, self.onnx_device)
        self.session = ort.InferenceSession(str(policy_file), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.textop_to_mjlab = torch.tensor(
            ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
            dtype=torch.long,
        )

    def __call__(self, obs: torch.Tensor | Any) -> torch.Tensor:
        obs = _actor_obs(obs)
        if obs.ndim != 2:
            raise RuntimeError(
                f"Expected batched ONNX obs shaped [N, 431], got {obs.shape}"
            )
        if obs.shape[-1] != 431:
            raise RuntimeError(f"Expected ONNX obs dim 431, got {obs.shape[-1]}")

        action_textop = self._run(obs)
        index = self.textop_to_mjlab.to(obs.device)
        action_mjlab = action_textop.index_select(-1, index)

        if action_mjlab.ndim != 2:
            raise RuntimeError(
                f"Expected batched ONNX action shaped [N, 29], got {action_mjlab.shape}"
            )
        if action_mjlab.shape[-1] != 29:
            raise RuntimeError(
                f"Expected ONNX action dim 29, got {action_mjlab.shape[-1]}"
            )

        return action_mjlab

    def _run(self, obs: torch.Tensor) -> torch.Tensor:
        if self.onnx_device.type == "cuda":
            return self._run_cuda_iobinding(obs)
        return self._run_cpu(obs)

    def _run_cpu(self, obs: torch.Tensor) -> torch.Tensor:
        obs_device = obs.device
        obs = obs.detach()
        if obs.device.type != "cpu" or obs.dtype != torch.float32:
            obs = obs.to(device="cpu", dtype=torch.float32)
        if not obs.is_contiguous():
            obs = obs.contiguous()

        action_textop_np = self.session.run(None, {self.input_name: obs.numpy()})[0]
        return torch.from_numpy(action_textop_np).to(obs_device)

    def _run_cuda_iobinding(self, obs: torch.Tensor) -> torch.Tensor:
        _check_cuda_iobinding_obs(obs, expected_device=self.onnx_device)
        action_textop = torch.empty(
            (obs.shape[0], 29),
            device=obs.device,
            dtype=torch.float32,
        )
        binding = self.session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=_cuda_device_id(obs.device),
            element_type=np.float32,
            shape=tuple(obs.shape),
            buffer_ptr=obs.data_ptr(),
        )
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=_cuda_device_id(obs.device),
            element_type=np.float32,
            shape=tuple(action_textop.shape),
            buffer_ptr=action_textop.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        return action_textop


class OnnxPolicyRunner:
    """Runner adapter so MJLab's play script can load an ONNX policy."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        del env, train_cfg, log_dir
        self.device = device
        self.policy: OnnxPolicy | None = None

    def load(self, path: str | Path, *_args: Any, **_kwargs: Any) -> None:
        self.policy = OnnxPolicy(Path(path), device=self.device)

    def get_inference_policy(self, *_args: Any, **_kwargs: Any) -> OnnxPolicy:
        if self.policy is None:
            raise RuntimeError("ONNX policy has not been loaded")
        return self.policy


def _actor_obs(obs: torch.Tensor | Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs

    try:
        actor_obs = obs["actor"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "Expected observation to be a tensor or contain an 'actor' tensor"
        ) from None

    if not isinstance(actor_obs, torch.Tensor):
        raise RuntimeError(
            f"Expected observation to be a tensor, got {type(actor_obs).__name__}"
        )
    return actor_obs


def _check_cuda_iobinding_obs(
    obs: torch.Tensor,
    *,
    expected_device: torch.device,
) -> None:
    if obs.device != expected_device:
        raise RuntimeError(
            f"Expected ONNX CUDA obs on {expected_device}, got {obs.device}"
        )
    if obs.dtype != torch.float32:
        raise RuntimeError(f"Expected ONNX CUDA obs dtype float32, got {obs.dtype}")
    if not obs.is_contiguous():
        raise RuntimeError("Expected ONNX CUDA obs to be contiguous")
    if obs.requires_grad:
        raise RuntimeError("Expected ONNX CUDA obs to be detached")


def _onnx_providers_for_device(ort: Any, torch_device: torch.device) -> list[Any]:
    if torch_device.type == "cpu":
        return ["CPUExecutionProvider"]
    if torch_device.type != "cuda":
        raise RuntimeError(f"Unsupported ONNX Runtime device: {torch_device}")

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "ONNX Runtime CUDA provider is not available. Install with the cu128 "
            "extra and verify CUDA libraries are visible to onnxruntime-gpu."
        )

    with torch.cuda.device(torch_device):
        stream_ptr = torch.cuda.current_stream(torch_device).cuda_stream

    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": _cuda_device_id(torch_device),
                "user_compute_stream": str(stream_ptr),
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


def _cuda_device_id(device: torch.device) -> int:
    return 0 if device.index is None else device.index
