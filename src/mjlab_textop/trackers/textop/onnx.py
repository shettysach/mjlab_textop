from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from mjlab_textop.core.schema import ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
from mjlab_textop.trackers.onnx import OnnxModelSpec, OnnxTensorModel


class TextOpOnnxPolicy:
    """Run a TextOp ONNX actor and convert its action to MJLab joint order."""

    def __init__(
        self,
        policy_file: Path,
        device: str = "cpu",
    ) -> None:
        self.model = OnnxTensorModel(
            policy_file,
            device=device,
            spec=OnnxModelSpec(output_width=29),
        )
        self.device = self.model.device
        self.session = self.model.session
        self._textop_to_mjlab: dict[torch.device, torch.Tensor] = {}

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = obs["actor"]
        if not isinstance(actor_obs, torch.Tensor):
            raise TypeError("TextOp actor observation must be a tensor")
        action = self.model(actor_obs)
        return action.index_select(
            -1,
            self._joint_order_index(action.device),
        )

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
