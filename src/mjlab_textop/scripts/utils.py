from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab_textop.trackers.sonic import (
    SONIC_LOW_LATENCY_TRACKER,
    SonicModelBundle,
)
from mjlab_textop.trackers.spec import TrackerSpec
from mjlab_textop.trackers.textop.specs import (
    TEXTOP_ONNX_TRACKER,
    TEXTOP_PYTORCH_TRACKER,
)


def verify_resolved(resolved: Path, label: str) -> Path:
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


@dataclass(frozen=True)
class ResolvedTracker:
    spec: TrackerSpec
    artifact: Path


def resolve_tracker(
    checkpoint_file: str | Path | None,
    onnx_file: str | Path | None,
    sonic_model_dir: str | Path | None = None,
) -> ResolvedTracker:
    selected = sum(
        value is not None for value in (checkpoint_file, onnx_file, sonic_model_dir)
    )
    if selected != 1:
        raise ValueError(
            "Pass exactly one of --checkpoint-file, --onnx-file, or --sonic-model-dir"
        )

    if checkpoint_file is not None:
        return ResolvedTracker(
            spec=TEXTOP_PYTORCH_TRACKER,
            artifact=verify_resolved(
                Path(checkpoint_file).expanduser().resolve(),
                "Checkpoint file",
            ),
        )

    if onnx_file is not None:
        return ResolvedTracker(
            spec=TEXTOP_ONNX_TRACKER,
            artifact=verify_resolved(
                Path(onnx_file).expanduser().resolve(),
                "ONNX policy file",
            ),
        )

    assert sonic_model_dir is not None
    bundle = SonicModelBundle.from_directory(sonic_model_dir)
    return ResolvedTracker(
        spec=SONIC_LOW_LATENCY_TRACKER,
        artifact=bundle.directory,
    )
