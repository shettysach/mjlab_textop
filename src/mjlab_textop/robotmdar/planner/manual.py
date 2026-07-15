from __future__ import annotations

import threading
from dataclasses import dataclass

from mjlab_textop.robotmdar.planner.followups import CommandSequencer


@dataclass
class PromptState:
    text: str
    stop: bool = False
    input_active: bool = False
    revision: int = 0


class ManualPromptPlanner:
    def __init__(self, initial_prompt: str, *, command_hold_blocks: int = 1) -> None:
        self.prompt = PromptState(text=initial_prompt)
        self._thread: threading.Thread | None = None
        self._sequencer = CommandSequencer(
            initial_prompt, hold_blocks=command_hold_blocks
        )
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
            return self._sequencer.activate(
                self.prompt.text,
                source="manual",
                block_count=block_count,
                replace=True,
            ).text

        command, _ = self._sequencer.advance(block_count)
        return command.text


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
