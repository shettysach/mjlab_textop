from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass
class PromptState:
    text: str
    stop: bool = False
    input_active: bool = False


@dataclass(frozen=True)
class ScheduledPromptPhase:
    text: str
    frames: int


@dataclass(frozen=True)
class PlannerContext:
    frame_index: int
    block_count: int


@dataclass(frozen=True)
class GeneratedBlockInfo:
    prompt: str
    start_frame: int
    frames: int
    block_count: int


class ScheduledPromptSource:
    def __init__(self, phases: tuple[ScheduledPromptPhase, ...]) -> None:
        if not phases:
            raise ValueError("Scheduled prompt source requires at least one phase")
        self.phases = phases
        self.phase_index = 0
        self.phase_elapsed_frames = 0

    @property
    def text(self) -> str:
        return self.phases[self.phase_index].text

    def advance(self, frames: int) -> bool:
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")

        self.phase_elapsed_frames += frames
        while self.phase_elapsed_frames >= self.phases[self.phase_index].frames:
            self.phase_elapsed_frames -= self.phases[self.phase_index].frames
            self.phase_index += 1
            if self.phase_index >= len(self.phases):
                self.phase_index = len(self.phases) - 1
                self.phase_elapsed_frames = self.phases[self.phase_index].frames
                return True
        return False


class PromptPlanner(Protocol):
    @property
    def should_stop(self) -> bool:
        ...

    @property
    def input_active(self) -> bool:
        ...

    @property
    def log_suffix(self) -> str:
        ...

    def start(self) -> None:
        ...

    def request_stop(self) -> None:
        ...

    def choose_prompt(self, context: PlannerContext) -> str:
        ...

    def on_block_sent(self, info: GeneratedBlockInfo) -> None:
        ...


class ManualPromptPlanner:
    def __init__(self, initial_prompt: str) -> None:
        self.prompt = PromptState(text=initial_prompt)
        self._thread: threading.Thread | None = None

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

    def choose_prompt(self, context: PlannerContext) -> str:
        return self.prompt.text

    def on_block_sent(self, info: GeneratedBlockInfo) -> None:
        return


class ScheduledPromptPlanner:
    def __init__(
        self,
        phases: tuple[ScheduledPromptPhase, ...],
        *,
        stop_on_complete: bool,
    ) -> None:
        self.phases = phases
        self.source = ScheduledPromptSource(phases)
        self.stop_on_complete = stop_on_complete
        self._stop = False

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def log_suffix(self) -> str:
        return ""

    def start(self) -> None:
        return

    def request_stop(self) -> None:
        self._stop = True

    def choose_prompt(self, context: PlannerContext) -> str:
        return self.source.text

    def on_block_sent(self, info: GeneratedBlockInfo) -> None:
        schedule_finished = self.source.advance(info.frames)
        if schedule_finished and self.stop_on_complete:
            self._stop = True


def make_prompt_planner(args) -> PromptPlanner:
    if args.schedule:
        return ScheduledPromptPlanner(
            parse_prompt_schedule(args.schedule, repeat=args.schedule_repeat),
            stop_on_complete=args.schedule_stop,
        )
    return ManualPromptPlanner(args.prompt)


def parse_prompt_schedule(
    schedule: str,
    *,
    repeat: int = 1,
) -> tuple[ScheduledPromptPhase, ...]:
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}")

    phases = []
    for raw_entry in schedule.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Schedule entry must be formatted as 'prompt:frames': {entry!r}"
            )
        prompt, raw_frames = entry.rsplit(":", 1)
        prompt = prompt.strip()
        raw_frames = raw_frames.strip()
        if not prompt:
            raise ValueError(f"Schedule entry has empty prompt: {entry!r}")
        try:
            frames = int(raw_frames)
        except ValueError as exc:
            raise ValueError(f"Invalid frame count in schedule entry: {entry!r}") from exc
        if frames <= 0:
            raise ValueError(
                f"Schedule frame count must be positive in entry: {entry!r}"
            )
        phases.append(ScheduledPromptPhase(text=prompt, frames=frames))

    if not phases:
        raise ValueError("Schedule must contain at least one prompt phase")
    return tuple(phases * repeat)


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
