from __future__ import annotations

from dataclasses import dataclass

from mjlab_textop.core.online.buffer import RollingMotionBuffer


@dataclass
class OnlineReferenceClock:
    """Own the consumer frame and live-buffer polling hysteresis."""

    current_frame: int
    started: bool = False
    has_started_once: bool = False
    startup_wait_steps: int = 0
    live_polling_paused: bool = False

    def reset_runtime(self) -> None:
        self.startup_wait_steps = 0
        self.live_polling_paused = False

    def startup_frame(
        self,
        buffer: RollingMotionBuffer,
        *,
        source_mode: str,
        future_steps: int,
    ) -> int | None:
        if source_mode == "live":
            return buffer.earliest_start_frame(future_steps)
        return (
            self.current_frame
            if buffer.can_start(self.current_frame, future_steps)
            else None
        )

    def resample_live_frame(
        self,
        buffer: RollingMotionBuffer,
        *,
        future_steps: int,
    ) -> int | None:
        if not self.has_started_once:
            return buffer.earliest_start_frame(future_steps)
        return buffer.latest_start_frame(future_steps)

    def should_poll_live(
        self,
        buffer: RollingMotionBuffer,
        *,
        low_watermark: int,
        high_watermark: int,
    ) -> bool:
        latest_index = buffer.latest_index
        if latest_index is None:
            self.live_polling_paused = False
            return True

        future_lead = latest_index - self.current_frame
        if self.live_polling_paused:
            if future_lead >= low_watermark:
                return False
            self.live_polling_paused = False
        elif future_lead >= high_watermark:
            self.live_polling_paused = True
            return False
        return True

    def can_advance_live(
        self,
        buffer: RollingMotionBuffer,
        *,
        future_steps: int,
    ) -> bool:
        if not buffer.can_start(self.current_frame, future_steps):
            raise RuntimeError(
                "Lost active live reference window: "
                f"current={self.current_frame}, "
                f"earliest={buffer.earliest_index}, "
                f"latest={buffer.latest_index}, "
                f"future_steps={future_steps}"
            )
        return buffer.can_start(self.current_frame + 1, future_steps)
