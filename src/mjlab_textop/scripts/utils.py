from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.feedback.observation import (
    OnlineTextOpObservationCfg,
)
from mjlab_textop.core.mdp.online_commands import TextOpOnlineSourceMode
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.onnx_policy import OnnxPolicyRunner

PolicyRunnerCls = type[MotionTrackingOnPolicyRunner] | type[OnnxPolicyRunner]


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


def resolve_policy(
    checkpoint_file: str | Path | None,
    onnx_file: str | Path | None,
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
        )

    raise ValueError("Pass exactly one of --checkpoint-file or --onnx-file")


class TaskRegistrar(Protocol):
    def __call__(
        self,
        *,
        runner_cls: PolicyRunnerCls,
        source: TextOpOnlineSource | None = None,
        live_source_cfg: SocketTextOpSourceCfg | None = None,
        source_mode: TextOpOnlineSourceMode,
        future_steps: int,
        num_envs: int,
        anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
            "align_to_robot_start"
        ),
        reset_robot_to_reference: bool = True,
        reference_debug_vis: bool | None = None,
        observation: OnlineTextOpObservationCfg | None = None,
    ) -> str: ...
