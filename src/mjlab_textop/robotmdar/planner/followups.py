from __future__ import annotations

from collections import deque


def command_followups(command: str) -> list[str]:
    command = command.lower().strip()
    if "left" in command or "right" in command:
        return ["stand"]
    return []


class FollowupCommandQueue:
    def __init__(self) -> None:
        self._commands: deque[str] = deque()

    def receive(self, command: str) -> None:
        self._commands.append(command)
        self._commands.extend(command_followups(command))

    def next(self) -> str | None:
        if self._commands:
            return self._commands.popleft()
        return None

    def __bool__(self) -> bool:
        return bool(self._commands)
