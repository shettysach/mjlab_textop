from __future__ import annotations

from mjlab_textop.trackers.sonic.config import configure_sonic_tracker
from mjlab_textop.trackers.sonic.constants import SONIC_REFERENCE_FRAMES
from mjlab_textop.trackers.sonic.onnx import SonicOnnxPolicyRunner
from mjlab_textop.trackers.spec import ReferenceWindowSpec, TrackerSpec

SONIC_LOW_LATENCY_TRACKER = TrackerSpec(
    name="sonic-low-latency",
    runner_cls=SonicOnnxPolicyRunner,
    configure_env=configure_sonic_tracker,
    reference_window=ReferenceWindowSpec(
        frame_offsets=tuple(range(SONIC_REFERENCE_FRAMES)),
        align_heading=True,
    ),
)
