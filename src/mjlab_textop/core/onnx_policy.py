from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX

OnnxExecutionProvider = Literal["cpu", "cuda"]


class OnnxPolicy:
    """Run an ONNX actor and convert its action to MJLab joint order."""

    def __init__(
        self,
        policy_file: Path,
        device: str = "cpu",
        *,
        execution_provider: OnnxExecutionProvider = "cpu",
    ) -> None:
        self.execution_provider = execution_provider
        ort = _load_onnxruntime()

        if execution_provider == "cuda":
            self.device, self.device_id, self._stream = _resolve_cuda_device(
                ort,
                torch.device(device),
            )
            session_options = ort.SessionOptions()
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
            self.session = ort.InferenceSession(
                str(policy_file),
                sess_options=session_options,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": self.device_id,
                            "user_compute_stream": str(self._stream.cuda_stream),
                        },
                    )
                ],
            )
            self._binding = self.session.io_binding()
        else:
            self.device = torch.device("cpu")
            self.session = ort.InferenceSession(
                str(policy_file),
                providers=["CPUExecutionProvider"],
            )
            self._binding = None

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._output: torch.Tensor | None = None
        self._textop_to_mjlab: dict[torch.device, torch.Tensor] = {}

    def __call__(self, obs: torch.Tensor | Any) -> torch.Tensor:
        obs = _actor_obs(obs)
        if obs.ndim != 2:
            raise RuntimeError(
                f"Expected batched ONNX obs shaped [N, 431], got {obs.shape}"
            )
        if obs.shape[-1] != 431:
            raise RuntimeError(f"Expected ONNX obs dim 431, got {obs.shape[-1]}")

        action_textop = (
            self._run_cuda(obs)
            if self.execution_provider == "cuda"
            else self._run_cpu(obs)
        )
        action_mjlab = action_textop.index_select(
            -1,
            self._joint_order_index(action_textop.device),
        )

        if action_mjlab.ndim != 2:
            raise RuntimeError(
                f"Expected batched ONNX action shaped [N, 29], got {action_mjlab.shape}"
            )
        if action_mjlab.shape[-1] != 29:
            raise RuntimeError(
                f"Expected ONNX action dim 29, got {action_mjlab.shape[-1]}"
            )
        return action_mjlab

    def _run_cpu(self, obs: torch.Tensor) -> torch.Tensor:
        output_device = obs.device
        obs = obs.detach().to(device="cpu", dtype=torch.float32).contiguous()
        action = self.session.run(None, {self.input_name: obs.numpy()})[0]
        return torch.from_numpy(action).to(output_device)

    def _run_cuda(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.device != self.device:
            raise RuntimeError(
                "CUDA ONNX policy requires observations on the policy device: "
                f"expected {self.device}, got {obs.device}"
            )

        obs = obs.detach().to(dtype=torch.float32).contiguous()
        output_shape = (int(obs.shape[0]), 29)
        if self._output is None or self._output.shape != output_shape:
            self._output = torch.empty(
                output_shape,
                dtype=torch.float32,
                device=self.device,
            )

        assert self._binding is not None
        self._binding.clear_binding_inputs()
        self._binding.clear_binding_outputs()
        self._binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(obs.shape),
            buffer_ptr=obs.data_ptr(),
        )
        self._binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=output_shape,
            buffer_ptr=self._output.data_ptr(),
        )
        self.session.run_with_iobinding(self._binding)
        return self._output

    def _joint_order_index(self, device: torch.device) -> torch.Tensor:
        index = self._textop_to_mjlab.get(device)
        if index is None:
            index = torch.tensor(
                ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
                dtype=torch.long,
                device=device,
            )
            self._textop_to_mjlab[device] = index
        return index


class OnnxPolicyRunner:
    """Runner adapter so MJLab's play script can load an ONNX policy."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        del env, log_dir
        self.device = device
        self.execution_provider: OnnxExecutionProvider = train_cfg.get(
            "onnx_execution_provider",
            "cpu",
        )
        self.policy: OnnxPolicy | None = None

    def load(self, path: str | Path, *_args: Any, **_kwargs: Any) -> None:
        self.policy = OnnxPolicy(
            Path(path),
            device=self.device,
            execution_provider=self.execution_provider,
        )

    def get_inference_policy(self, *_args: Any, **_kwargs: Any) -> OnnxPolicy:
        if self.policy is None:
            raise RuntimeError("ONNX policy has not been loaded")
        return self.policy


def _resolve_cuda_device(
    ort: Any,
    requested_device: torch.device,
) -> tuple[torch.device, int, Any]:
    if requested_device.type != "cuda":
        raise RuntimeError(
            f"CUDA ONNX execution requires an MJLab CUDA device, got {requested_device}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ONNX execution requested, but Torch has no CUDA")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDA ONNX execution requested, but ONNX Runtime has no "
            "CUDAExecutionProvider. Install the cu128 extra with onnxruntime-gpu."
        )

    device_id = (
        requested_device.index
        if requested_device.index is not None
        else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_id)
    return device, device_id, torch.cuda.current_stream(device_id)


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


def _load_onnxruntime() -> Any:
    """Load the optional runtime only when an ONNX policy is constructed."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ImportError(
            "ONNX policy execution requires a working onnxruntime installation. "
            "Task configuration and non-ONNX code can run without it."
        ) from exc
