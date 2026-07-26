from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol

from robotmdar_textop.feedback import FeedbackObservation
from robotmdar_textop.planner.followups import (
    CommandSequencer,
    vlm_command_followups,
)
from robotmdar_textop.runtime import BlockPlan


class ObservationProvider(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def latest(self) -> FeedbackObservation | None: ...

    def wait_for_checkpoint(self, checkpoint_id: int) -> FeedbackObservation: ...


@dataclass(frozen=True)
class VlmPromptSelection:
    prompt: str
    reasoning: str | None
    response: dict[str, Any]


@dataclass(frozen=True)
class VlmExecutionContext:
    previous_decision: str | None
    executed_sequence: tuple[str, ...]
    current_command: str


class VlmPromptSelector(Protocol):
    def choose_prompt_with_debug(
        self,
        *,
        observation: FeedbackObservation,
        execution_context: VlmExecutionContext | None = None,
    ) -> VlmPromptSelection: ...


@dataclass(frozen=True)
class _VlmUserTurn:
    prompt: str
    image_data_url: str | None


@dataclass(frozen=True)
class _VlmConversationTurn:
    user: _VlmUserTurn
    assistant_prompt: str
    assistant_reasoning: str | None


class VlmPromptPlanner:
    def __init__(
        self,
        *,
        feedback: ObservationProvider,
        selector: VlmPromptSelector,
        initial_prompt: str,
        command_hold_blocks: int = 1,
    ) -> None:
        self.feedback = feedback
        self.selector = selector
        self.current_prompt = initial_prompt
        self._current_source = "initial"
        self.last_error: str | None = None
        self._pending_reasoning: str | None = None
        self._stop = False
        self._initial_sequence_pending = True
        self._awaiting_checkpoint: int | None = None
        self._next_checkpoint_id = 1
        self._collision_recovery = False
        self._recovery_epoch = 0
        self._last_ack_checkpoint: int | None = None
        self._last_ack_source_frame: int | None = None
        self._last_vlm_ms: float | None = None
        self._last_vlm_decision: str | None = None
        self._executed_since_query: list[str] = []
        self._sequencer = CommandSequencer(
            initial_prompt,
            hold_blocks=command_hold_blocks,
            followups=vlm_command_followups,
        )

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def log_suffix(self) -> str:
        state = "awaiting_ack" if self._awaiting_checkpoint is not None else "idle"
        suffix = f" vlm={state}"
        if self._awaiting_checkpoint is not None:
            suffix += f" checkpoint={self._awaiting_checkpoint}"
        if self._last_ack_checkpoint is not None:
            suffix += (
                f" last_ack=(checkpoint: {self._last_ack_checkpoint}, "
                f"source_frame: {self._last_ack_source_frame})"
            )
        if self._last_vlm_ms is not None:
            suffix += f" vlm_ms={self._last_vlm_ms:.1f}"
        if self.last_error is not None:
            suffix += f" error={self.last_error!r}"
        return suffix

    def start(self) -> None:
        self.feedback.start()

    def request_stop(self) -> None:
        self._stop = True
        self.feedback.close()

    def next_plan(self, *, block_count: int) -> BlockPlan:
        reset_pacing = self._awaiting_checkpoint is not None
        selection = self._complete_checkpoint()
        checkpoint_id = self._advance_prompt(
            block_count=block_count,
            selection=selection,
        )
        plan = BlockPlan(
            prompt=self.current_prompt,
            source=self._current_source,
            recovery_epoch=(
                self._recovery_epoch if self._collision_recovery else 0
            ),
            checkpoint_id=checkpoint_id,
            reset_pacing=reset_pacing,
        )
        if checkpoint_id is None and (
            not self._executed_since_query
            or self._executed_since_query[-1] != plan.prompt
        ):
            self._executed_since_query.append(plan.prompt)
        return plan

    def _complete_checkpoint(self) -> VlmPromptSelection | None:
        checkpoint_id = self._awaiting_checkpoint
        if checkpoint_id is None:
            return None

        observation = self.feedback.wait_for_checkpoint(checkpoint_id)
        self._awaiting_checkpoint = None
        if observation.collision_stop:
            self._enter_collision_recovery(observation.recovery_epoch)
            return None

        self._last_ack_checkpoint = checkpoint_id
        self._last_ack_source_frame = observation.source_frame
        execution_context = VlmExecutionContext(
            previous_decision=self._last_vlm_decision,
            executed_sequence=tuple(self._executed_since_query)
            or (self.current_prompt,),
            current_command=self.current_prompt,
        )
        self._executed_since_query.clear()
        started = monotonic()
        try:
            selection = self.selector.choose_prompt_with_debug(
                observation=observation,
                execution_context=execution_context,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop = True
            raise RuntimeError(f"VLM request failed: {self.last_error}") from exc
        self._last_vlm_ms = (monotonic() - started) * 1000.0

        latest = self.feedback.latest()
        if latest is not None and latest.collision_stop:
            self._enter_collision_recovery(latest.recovery_epoch)
            return None
        self.last_error = None
        return selection

    def _advance_prompt(
        self,
        *,
        block_count: int,
        selection: VlmPromptSelection | None,
    ) -> int | None:
        observation = self.feedback.latest()
        if observation is not None and observation.collision_stop:
            self._enter_collision_recovery(observation.recovery_epoch)
            return None
        if self._collision_recovery:
            self._collision_recovery = False
            self._sequencer.release()
            command = self._sequencer.activate(
                "stand",
                source="followup",
                block_count=block_count,
                replace=True,
            )
            self._set_current(command.text, command.source)
            return None

        if selection is not None:
            self._last_vlm_decision = selection.prompt
            self._pending_reasoning = selection.reasoning
            command = self._sequencer.activate(
                selection.prompt,
                source="vlm",
                block_count=block_count,
            )
            self._set_current(command.text, command.source)
            return None

        if self._initial_sequence_pending:
            self._initial_sequence_pending = False
            command = self._sequencer.activate(
                self.current_prompt,
                source="initial",
                block_count=block_count,
            )
            self._set_current(command.text, command.source)
            return None

        command_was_active = self._sequencer.busy
        command, changed = self._sequencer.advance(block_count)
        if changed:
            self._set_current(command.text, command.source)
            return None
        if command_was_active and not self._sequencer.busy:
            checkpoint_id = self._next_checkpoint_id
            self._next_checkpoint_id += 1
            self._awaiting_checkpoint = checkpoint_id
            self._set_current(self.current_prompt, "checkpoint")
            return checkpoint_id
        return None

    def _enter_collision_recovery(self, recovery_epoch: int) -> None:
        if not self._collision_recovery:
            self._collision_recovery = True
            self._awaiting_checkpoint = None
            self._executed_since_query.clear()
            command = self._sequencer.override("stand", source="collision_recovery")
            self._set_current(command.text, command.source)
        self._recovery_epoch = recovery_epoch

    def _set_current(self, prompt: str, source: str) -> None:
        self.current_prompt = prompt
        self._current_source = source

    def consume_pending_reasoning(self) -> str | None:
        reasoning = self._pending_reasoning
        self._pending_reasoning = None
        return reasoning


class OpenAIChatPromptSelector:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout_sec: float = 30.0,
        history_length: int = 5,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if not user_prompt:
            raise ValueError("user_prompt must be a non-empty string")
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.timeout_sec = timeout_sec
        self.history_length = history_length
        self._history: deque[_VlmConversationTurn] = deque(maxlen=history_length - 1)

    def choose_prompt(
        self,
        *,
        observation: FeedbackObservation,
        execution_context: VlmExecutionContext | None = None,
    ) -> str:
        return self.choose_prompt_with_debug(
            observation=observation,
            execution_context=execution_context,
        ).prompt

    def choose_prompt_with_debug(
        self,
        *,
        observation: FeedbackObservation,
        execution_context: VlmExecutionContext | None = None,
    ) -> VlmPromptSelection:
        current_user = _make_user_turn(
            self.user_prompt,
            observation,
            execution_context=execution_context,
        )
        response = self._post_json(
            _make_chat_completions_payload(
                current_user=current_user,
                history=self._history,
                model=self.model,
                system_prompt=self.system_prompt,
            )
        )
        choice = response["choices"][0]
        message = choice["message"]
        prompt = message["content"]
        reasoning = _extract_reasoning(choice)
        self._history.append(
            _VlmConversationTurn(
                user=current_user,
                assistant_prompt=prompt,
                assistant_reasoning=reasoning,
            )
        )
        return VlmPromptSelection(
            prompt=prompt,
            reasoning=reasoning,
            response=response,
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))


def _make_chat_completions_payload(
    *,
    current_user: _VlmUserTurn,
    history: Iterable[_VlmConversationTurn],
    model: str,
    system_prompt: str,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    for turn in history:
        messages.append(_make_user_message(turn.user))
        messages.append(_make_assistant_message(turn))
    messages.append(_make_user_message(current_user))
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }


def _make_user_turn(
    user_prompt: str,
    observation: FeedbackObservation,
    *,
    execution_context: VlmExecutionContext | None = None,
) -> _VlmUserTurn:
    image_data_url = None
    if observation.image_bytes is not None and observation.image_mime_type is not None:
        image_data_url = _image_data_url(
            observation.image_bytes,
            observation.image_mime_type,
        )
    prompt = (
        user_prompt
        if execution_context is None
        else _execution_context_prompt(execution_context, user_prompt=user_prompt)
    )
    return _VlmUserTurn(prompt=prompt, image_data_url=image_data_url)


def _execution_context_prompt(
    context: VlmExecutionContext,
    *,
    user_prompt: str,
) -> str:
    previous = context.previous_decision or "none"
    sequence = " -> ".join(context.executed_sequence) or context.current_command
    return (
        f"Previous decision: {previous}\n"
        f"Executed sequence: {sequence}\n"
        f"Current command: {context.current_command}\n\n"
        f"{user_prompt}"
    )


def _make_user_message(turn: _VlmUserTurn) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": turn.prompt}]
    if turn.image_data_url is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": turn.image_data_url},
            }
        )
    return {
        "role": "user",
        "content": content,
    }


def _make_assistant_message(turn: _VlmConversationTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"type": "text", "text": turn.assistant_prompt}],
    }
    if turn.assistant_reasoning is not None:
        message["reasoning_content"] = turn.assistant_reasoning
    return message


def _image_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{b64encode(data).decode('ascii')}"


def _extract_reasoning(choice: dict[str, Any]) -> str | None:
    message = choice.get("message")
    candidates: list[Any] = []
    if isinstance(message, dict):
        candidates.extend(
            [
                message.get("reasoning"),
                message.get("reasoning_content"),
                message.get("thinking"),
            ]
        )
    candidates.extend(
        [
            choice.get("reasoning"),
            choice.get("reasoning_content"),
            choice.get("thinking"),
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None
