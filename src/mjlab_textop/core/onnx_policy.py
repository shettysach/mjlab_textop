from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mjlab_textop.core.schema import TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class TextOpOnnxPolicy:
    """Run a TextOp ONNX actor and convert its action to MJLab joint order."""

    def __init__(self, policy_file: Path, device: str = "cpu"):
        import onnxruntime as ort

        self.onnx_device = torch.device(device)
        providers = _onnx_providers_for_device(ort, self.onnx_device)
        self.session = ort.InferenceSession(str(policy_file), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.textop_to_mjlab = torch.tensor(
            TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
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

        action_textop = self._run_cpu(obs)
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

    def _run_cpu(self, obs: torch.Tensor) -> torch.Tensor:
        obs_device = obs.device
        obs = obs.detach()
        if obs.device.type != "cpu" or obs.dtype != torch.float32:
            obs = obs.to(device="cpu", dtype=torch.float32)
        if not obs.is_contiguous():
            obs = obs.contiguous()

        action_textop_np = self.session.run(None, {self.input_name: obs.numpy()})[0]
        return torch.from_numpy(action_textop_np).to(obs_device)


class CustomOnnxPolicyRunner:
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
        self.policy: TextOpOnnxPolicy | None = None

    def load(self, path: str | Path, *_args: Any, **_kwargs: Any) -> None:
        self.policy = TextOpOnnxPolicy(Path(path), device=self.device)

    def get_inference_policy(self, *_args: Any, **_kwargs: Any) -> TextOpOnnxPolicy:
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
            "Expected ONNX observation to be a tensor or contain an 'actor' tensor"
        ) from None

    if not isinstance(actor_obs, torch.Tensor):
        raise RuntimeError(
            f"Expected ONNX actor observation to be a tensor, got "
            f"{type(actor_obs).__name__}"
        )
    return actor_obs


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

    return [
        ("CUDAExecutionProvider", {"device_id": torch_device.index}),
        "CPUExecutionProvider",
    ]
