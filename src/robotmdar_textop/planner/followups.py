from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


def command_followups(command: str) -> list[str]:
    return _stand_followup(command, transient_words={"left", "right"})


def vlm_command_followups(command: str) -> list[str]:
    return _stand_followup(command, transient_words={"left", "right", "walk"})


def _stand_followup(command: str, *, transient_words: set[str]) -> list[str]:
    words = set(command.lower().split())
    if words & transient_words:
        return ["stand"]
    return []


@dataclass(frozen=True)
class ActiveCommand:
    text: str
    source: str


class CommandSequencer:
    """Own command duration and deterministic follow-up activation."""

    def __init__(
        self,
        initial: str,
        *,
        hold_blocks: int,
        followups: Callable[[str], list[str]] = command_followups,
    ) -> None:
        if hold_blocks <= 0:
            raise ValueError(f"hold_blocks must be positive, got {hold_blocks}")
        self.hold_blocks = hold_blocks
        self.current = ActiveCommand(initial, "initial")
        self._followups = followups
        self._started_block: int | None = None
        self._pending: deque[ActiveCommand] = deque()

    @property
    def busy(self) -> bool:
        return self._started_block is not None or bool(self._pending)

    def activate(
        self,
        command: str,
        *,
        source: str,
        block_count: int,
        replace: bool = False,
    ) -> ActiveCommand:
        if replace:
            self._pending.clear()
        self.current = ActiveCommand(command, source)
        self._started_block = block_count
        self._pending.extend(
            ActiveCommand(followup, "followup") for followup in self._followups(command)
        )
        return self.current

    def advance(self, block_count: int) -> tuple[ActiveCommand, bool]:
        if (
            self._started_block is not None
            and block_count - self._started_block < self.hold_blocks
        ):
            return self.current, False
        if self._pending:
            self.current = self._pending.popleft()
            self._started_block = block_count
            return self.current, True
        self._started_block = None
        return self.current, False

    def override(self, command: str, *, source: str) -> ActiveCommand:
        self._pending.clear()
        self._started_block = None
        self.current = ActiveCommand(command, source)
        return self.current

    def release(self) -> None:
        self._started_block = None
