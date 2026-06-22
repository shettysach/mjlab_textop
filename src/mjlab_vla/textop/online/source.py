from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TextOpMotionBlock:
    """Block of online TextOp reference motion frames.

    Joint arrays are in TextOp/IsaacLab order. The rolling buffer converts them
    to MJLab order when appending.
    """

    index: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


class TextOpOnlineSource(Protocol):
    def poll(self) -> TextOpMotionBlock | None:
        """Return the next available block, or None when no block is ready."""


@runtime_checkable
class ResettableTextOpOnlineSource(TextOpOnlineSource, Protocol):
    def reset(self) -> None:
        """Reset a finite source to its initial frame."""


class QueueTextOpOnlineSource:
    def __init__(self, blocks: list[TextOpMotionBlock] | None = None) -> None:
        self._initial_blocks = tuple(blocks or [])
        self._blocks: deque[TextOpMotionBlock] = deque(self._initial_blocks)

    def append(self, block: TextOpMotionBlock) -> None:
        self._blocks.append(block)

    def poll(self) -> TextOpMotionBlock | None:
        if not self._blocks:
            return None
        return self._blocks.popleft()

    def reset(self) -> None:
        self._blocks = deque(self._initial_blocks)
