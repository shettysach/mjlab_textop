from __future__ import annotations

from typing import Literal

OnlineSourceMode = Literal["replay", "live"]

ONLINE_METRIC_NAMES = (
    "online_buffer_frames",
    "online_stale_steps",
    "online_consecutive_stale_steps",
    "online_current_frame",
    "online_latest_frame",
    "online_lag_frames",
    "online_started",
    "online_queue_depth",
    "online_blocks_received",
    "online_bad_messages",
    "online_collision_stop",
)
