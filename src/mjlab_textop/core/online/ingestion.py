from __future__ import annotations

from dataclasses import replace
from typing import Callable

from mjlab_textop.core.online.buffer import RollingMotionBuffer
from mjlab_textop.core.online.source import OnlineSource
from textop_live_protocol.motion import MotionBlock


class OnlineBlockIngestor:
    """Poll a source and map producer block indices into the consumer timeline."""

    def __init__(
        self,
        source: OnlineSource,
        buffer: RollingMotionBuffer,
        *,
        live: bool,
    ) -> None:
        self.source = source
        self.buffer = buffer
        self.live = live
        self.index_offset = 0
        self._recovery_aligned = False

    def poll(
        self,
        *,
        max_blocks: int,
        should_poll: Callable[[], bool] | None = None,
    ) -> int:
        """Append normal source blocks and return the number appended."""
        appended = 0
        for _ in range(max_blocks):
            if self.live and should_poll is not None and not should_poll():
                break
            block = self.source.poll()
            if block is None:
                break
            if self.live:
                block = self._map_live_block(block)
                self._validate_contiguous(block)
            self.buffer.append_block(block)
            appended += 1
        return appended

    def begin_recovery(self) -> None:
        self._recovery_aligned = False

    def poll_recovery(
        self,
        *,
        max_blocks: int,
        current_frame: int,
        future_steps: int,
        accepts: Callable[[MotionBlock], bool],
    ) -> bool:
        """Append accepted recovery blocks until a complete window is available."""
        for _ in range(max_blocks):
            block = self.source.poll()
            if block is None:
                return False
            if not accepts(block):
                continue

            if not self._recovery_aligned:
                self.index_offset = current_frame - block.index
                self._recovery_aligned = True
            self.buffer.append_block(self._with_index_offset(block))
            if self.buffer.can_start(current_frame, future_steps):
                return True
        return False

    def _map_live_block(self, block: MotionBlock) -> MotionBlock:
        return self._with_index_offset(block)

    def _with_index_offset(self, block: MotionBlock) -> MotionBlock:
        return replace(block, index=block.index + self.index_offset)

    def _validate_contiguous(self, block: MotionBlock) -> None:
        latest_index = self.buffer.latest_index
        if latest_index is not None and block.index != latest_index + 1:
            raise RuntimeError(
                "Non-contiguous RobotMDAR live stream: "
                f"expected block index {latest_index + 1}, got {block.index}"
            )
