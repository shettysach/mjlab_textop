from __future__ import annotations

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.trackers.spec import ReferenceWindowSpec, TrackerSpec
from mjlab_textop.trackers.textop.config import (
    configure_textop_onnx_tracker,
    configure_textop_pytorch_tracker,
)
from mjlab_textop.trackers.textop.onnx import TextOpOnnxPolicyRunner

TEXTOP_PYTORCH_TRACKER = TrackerSpec(
    name="textop-pytorch",
    runner_cls=MotionTrackingOnPolicyRunner,
    configure_env=configure_textop_pytorch_tracker,
    reference_window=ReferenceWindowSpec(frame_offsets=(0, 1, 2, 3, 4)),
)

TEXTOP_ONNX_TRACKER = TrackerSpec(
    name="textop-onnx",
    runner_cls=TextOpOnnxPolicyRunner,
    configure_env=configure_textop_onnx_tracker,
    reference_window=ReferenceWindowSpec(frame_offsets=(0, 1, 2, 3, 4)),
)
