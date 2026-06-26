from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mjlab_textop.core.contract import TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX


class TextOpOnnxPolicy:
    """Run a TextOp ONNX actor and convert its action to MJLab joint order."""

    def __init__(self, policy_file: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(policy_file))
        self.input_name = self.session.get_inputs()[0].name
        self.textop_to_mjlab = torch.tensor(
            TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX,
            dtype=torch.long,
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2:
            raise RuntimeError(
                f"Expected batched ONNX obs shaped [N, 431], got {obs.shape}"
            )
        if obs.shape[-1] != 431:
            raise RuntimeError(f"Expected ONNX obs dim 431, got {obs.shape[-1]}")

        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        action_textop_np = self.session.run(None, {self.input_name: obs_np})[0]

        action_textop = torch.from_numpy(action_textop_np).to(obs.device)
        index = self.textop_to_mjlab.to(obs.device)
        action_mjlab = action_textop.index_select(-1, index)

        if action_mjlab.ndim != 2:
            raise RuntimeError(
                f"Expected batched ONNX action shaped [N, 29], "
                f"got {action_mjlab.shape}"
            )
        if action_mjlab.shape[-1] != 29:
            raise RuntimeError(f"Expected ONNX action dim 29, got {action_mjlab.shape[-1]}")

        return action_mjlab
