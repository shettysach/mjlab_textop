from __future__ import annotations

import threading
from dataclasses import dataclass

from mjlab_textop.robotmdar.planner.followups import FollowupCommandQueue


@dataclass
class PromptState:
    text: str
    stop: bool = False
    input_active: bool = False
    revision: int = 0


class ManualPromptPlanner:
    def __init__(self, initial_prompt: str, *, command_hold_blocks: int = 1) -> None:
        if command_hold_blocks <= 0:
            raise ValueError(
                f"command_hold_blocks must be positive, got {command_hold_blocks}"
            )
        self.prompt = PromptState(text=initial_prompt)
        self.command_hold_blocks = command_hold_blocks
        self._thread: threading.Thread | None = None
        self._commands = FollowupCommandQueue()
        self._current_prompt = initial_prompt
        self._command_started_block: int | None = None
        self._last_prompt_text: str | None = None
        self._last_prompt_revision = -1

    @property
    def should_stop(self) -> bool:
        return self.prompt.stop

    @property
    def input_active(self) -> bool:
        return self.prompt.input_active

    @property
    def log_suffix(self) -> str:
        return "\nEnter text prompt (or q to exit): "

    def start(self) -> None:
        self._thread = threading.Thread(
            target=_prompt_loop,
            args=(self.prompt,),
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self.prompt.stop = True

    def choose_prompt(self, *, block_count: int) -> str:
        if (
            self.prompt.revision != self._last_prompt_revision
            or self.prompt.text != self._last_prompt_text
        ):
            self._last_prompt_revision = self.prompt.revision
            self._last_prompt_text = self.prompt.text
            self._commands.receive(self.prompt.text, replace=True)
            command = self._commands.next()
            assert command is not None
            self._current_prompt = command
            self._command_started_block = block_count
            return self._current_prompt

        if (
            self._command_started_block is not None
            and block_count - self._command_started_block < self.command_hold_blocks
        ):
            return self._current_prompt

        if self._commands:
            command = self._commands.next()
            assert command is not None
            self._current_prompt = command
            self._command_started_block = block_count

        return self._current_prompt


def _prompt_loop(prompt: PromptState) -> None:
    while not prompt.stop:
        try:
            prompt.input_active = True
            text = input("Enter text prompt (or q to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            prompt.stop = True
            return
        finally:
            prompt.input_active = False
        if text.lower() in {"q", "quit", "exit"}:
            prompt.stop = True
        elif text:
            prompt.text = text
            prompt.revision += 1
