from __future__ import annotations

from typing import Protocol, runtime_checkable

from mjlab.managers.recorder_manager import RecorderTerm


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


class OnlineTextOpCleanup(RecorderTerm):
    """Close online command resources when the MJLab environment closes."""

    def close(self) -> None:
        command_name = self.cfg.params["command_name"]
        command = self._env.command_manager.get_term(command_name)
        if not isinstance(command, Closeable):
            raise TypeError(f"Command {command_name!r} does not implement close()")
        command.close()
