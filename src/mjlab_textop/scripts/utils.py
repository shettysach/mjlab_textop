from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.onnx_policy import (
    OnnxExecutionProvider,
    OnnxPolicyRunner,
)
from tasks.registration import PolicyRunnerCls


def verify_resolved(resolved: Path, label: str) -> Path:
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


@dataclass(frozen=True)
class ResolvedPolicy:
    runner_cls: PolicyRunnerCls
    file: Path
    onnx_provider: OnnxExecutionProvider = "cpu"


def resolve_policy(
    checkpoint_file: str | Path | None,
    onnx_file: str | Path | None,
    device: str = "cpu",
) -> ResolvedPolicy:
    if checkpoint_file is not None and onnx_file is not None:
        raise ValueError("Pass exactly one of --checkpoint-file or --onnx-file")

    if checkpoint_file is not None:
        return ResolvedPolicy(
            MotionTrackingOnPolicyRunner,
            verify_resolved(
                Path(checkpoint_file).expanduser().resolve(),
                "Checkpoint file",
            ),
        )

    if onnx_file is not None:
        return ResolvedPolicy(
            OnnxPolicyRunner,
            verify_resolved(
                Path(onnx_file).expanduser().resolve(),
                "ONNX policy file",
            ),
            onnx_provider=_onnx_provider_for_device(device),
        )

    raise ValueError("Pass exactly one of --checkpoint-file or --onnx-file")


def _onnx_provider_for_device(device: str) -> OnnxExecutionProvider:
    device_type = device.partition(":")[0]
    if device_type == "cuda":
        return "cuda"
    if device_type == "cpu":
        return "cpu"
    raise ValueError(f"ONNX policy requires a CPU or CUDA device, got {device!r}")
