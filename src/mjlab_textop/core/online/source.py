from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

from textop_live_protocol.motion import (
    MotionBlock,
    MotionFrames,
    StreamControl,
    validate_motion_block,
)

__all__ = [
    "MotionBlock",
    "MotionFrames",
    "OnlineSource",
    "QueueOnlineSource",
    "ResettableOnlineSource",
    "StreamControl",
    "validate_motion_block",
]


class OnlineSource(Protocol):
    def poll(self) -> MotionBlock | None:
        """Return the next available block, or None when no block is ready."""


@runtime_checkable
class ResettableOnlineSource(OnlineSource, Protocol):
    def reset(self) -> None:
        """Reset a finite source to its initial frame."""


class QueueOnlineSource:
    def __init__(
        self,
        blocks: list[MotionBlock] | None = None,
    ) -> None:
        self._initial_blocks = tuple(blocks or [])
        self._blocks: deque[MotionBlock] = deque(self._initial_blocks)

    def append(self, block: MotionBlock) -> None:
        self._blocks.append(block)

    def poll(self) -> MotionBlock | None:
        if not self._blocks:
            return None
        return self._blocks.popleft()

    def reset(self) -> None:
        self._blocks = deque(self._initial_blocks)
