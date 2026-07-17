from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict

from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class TextOpOnnxPolicy:
    """Run a TextOp ONNX actor and convert its action to MJLab joint order."""

    def __init__(
        self,
        policy_file: Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        if self.device.type == "cuda":
            self.device_id = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            self.device = torch.device("cuda", self.device_id)
            self._stream = torch.cuda.current_stream(self.device_id)
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
            self.session = ort.InferenceSession(
                str(policy_file),
                providers=["CPUExecutionProvider"],
            )
            self._binding = None

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._output: torch.Tensor | None = None
        self._textop_to_mjlab: dict[torch.device, torch.Tensor] = {}

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = cast(torch.Tensor, obs["actor"])
        action = (
            self._run_cuda(actor_obs)
            if self.device.type == "cuda"
            else self._run_cpu(actor_obs)
        )
        return action.index_select(
            -1,
            self._joint_order_index(action.device),
        )

    def _run_cpu(self, obs: torch.Tensor) -> torch.Tensor:
        cpu_obs = obs.detach().to(device="cpu", dtype=torch.float32).contiguous()
        action = self.session.run(None, {self.input_name: cpu_obs.numpy()})[0]
        return torch.from_numpy(action).to(obs.device)

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

        binding: Any = self._binding
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(obs.shape),
            buffer_ptr=obs.data_ptr(),
        )
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=output_shape,
            buffer_ptr=self._output.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
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


class TextOpOnnxPolicyRunner:
    """MJLab runner adapter for a TextOp ONNX policy."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        del env, train_cfg, log_dir
        self.device = device
        self.policy: TextOpOnnxPolicy | None = None

    def load(self, path: str | Path, *_args: Any, **_kwargs: Any) -> None:
        self.policy = TextOpOnnxPolicy(Path(path), device=self.device)

    def get_inference_policy(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> TextOpOnnxPolicy:
        if self.policy is None:
            raise RuntimeError("TextOp ONNX policy has not been loaded")
        return self.policy
