from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class OnnxPolicy:
    """Run an ONNX actor and convert its action to MJLab joint order."""

    def __init__(self, policy_file: Path, device: str = "cpu"):
        # RobotMDAR's ONNX policy is intentionally CPU-only. CUDA execution has
        # been unstable in deployment, while inference output is copied back to
        # the observation device below.
        del device
        self.device = torch.device("cpu")

        ort = _load_onnxruntime()
        self.session = ort.InferenceSession(
            str(policy_file),
            providers=["CPUExecutionProvider"],
        )
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
        return self._run_cpu(obs)

    def _run_cpu(self, obs: torch.Tensor) -> torch.Tensor:
        obs_device = obs.device
        obs = obs.detach()
        if obs.device.type != "cpu" or obs.dtype != torch.float32:
            obs = obs.to(device="cpu", dtype=torch.float32)
        obs = obs.contiguous()

        action_textop_np = self.session.run(None, {self.input_name: obs.numpy()})[0]
        return torch.from_numpy(action_textop_np).to(obs_device)

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


def _load_onnxruntime() -> Any:
    """Load the optional runtime only when an ONNX policy is constructed."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ImportError(
            "ONNX policy execution requires a working onnxruntime installation. "
            "Task configuration and non-ONNX code can run without it."
        ) from exc
