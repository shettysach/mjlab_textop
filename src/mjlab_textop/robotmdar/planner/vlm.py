from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from mjlab_textop.robotmdar.feedback import FeedbackObservation
from mjlab_textop.robotmdar.planner.followups import CommandSequencer


class ObservationProvider(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def latest(self) -> FeedbackObservation | None: ...


@dataclass(frozen=True)
class VlmPromptSelection:
    prompt: str
    reasoning: str | None
    response: dict[str, Any]


class VlmPromptPlanner:
    def __init__(
        self,
        *,
        feedback: ObservationProvider,
        selector: "OpenAIChatPromptSelector",
        initial_prompt: str,
        command_hold_blocks: int = 1,
    ) -> None:
        self.feedback = feedback
        self.selector = selector
        self.current_prompt = initial_prompt
        self.current_prompt_source = "initial"
        self.last_error: str | None = None
        self._pending_reasoning: str | None = None
        self._stop = False
        self._future: Future[VlmPromptSelection] | None = None
        self._future_epoch: int | None = None
        self._selection_epoch = 0
        self._collision_recovery = False
        self._recovery_epoch = 0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_query_block: int | None = None
        self._last_query_image_revision: int | None = None
        self._sequencer = CommandSequencer(
            initial_prompt, hold_blocks=command_hold_blocks
        )

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def recovery_epoch(self) -> int:
        return self._recovery_epoch if self._collision_recovery else 0

    @property
    def log_suffix(self) -> str:
        state = "inflight" if self._future is not None else "idle"
        suffix = f" vlm_state={state}"
        if self._last_query_block is not None:
            suffix += f" vlm_last_query_block={self._last_query_block}"
        if self._last_query_image_revision is not None:
            suffix += (
                f" vlm_last_query_image_revision={self._last_query_image_revision}"
            )
        if self.last_error is not None:
            suffix += f" vlm_last_error={self.last_error!r}"
        return suffix

    def start(self) -> None:
        self.feedback.start()

    def request_stop(self) -> None:
        self._stop = True
        try:
            self.feedback.close()
        finally:
            if self._future is not None:
                self._future.cancel()
            self._executor.shutdown(wait=True, cancel_futures=True)

    def choose_prompt(self, *, block_count: int) -> str:
        observation = self.feedback.latest()
        if observation is not None and observation.collision_stop:
            self._enter_collision_recovery(observation.recovery_epoch)
            return self.current_prompt
        if self._collision_recovery:
            self._collision_recovery = False
            self._sequencer.release()

        if self._collect_finished_request():
            command = self._sequencer.activate(
                self.current_prompt,
                source="vlm",
                block_count=block_count,
            )
            return self._set_current(command.text, command.source)

        command, changed = self._sequencer.advance(block_count)
        if changed:
            return self._set_current(command.text, command.source)
        return self.current_prompt

    def on_block_sent(self, *, block_count: int) -> None:
        if (
            self._stop
            or self._future is not None
            or self._collision_recovery
            or self._sequencer.busy
        ):
            return

        observation = self.feedback.latest()
        if observation is None or not self._is_unqueried_image(observation):
            return

        self._last_query_block = block_count
        self._last_query_image_revision = observation.image_revision
        self._future_epoch = self._selection_epoch
        self._future = self._executor.submit(
            self.selector.choose_prompt_with_debug,
            observation=observation,
        )

    def _enter_collision_recovery(self, recovery_epoch: int) -> None:
        if not self._collision_recovery:
            self._collision_recovery = True
            self._selection_epoch += 1
            if self._future is not None and self._future.cancel():
                self._future = None
                self._future_epoch = None
        self._recovery_epoch = recovery_epoch
        command = self._sequencer.override("stand", source="collision_recovery")
        self._set_current(command.text, command.source)

    def _set_current(self, prompt: str, source: str) -> str:
        self.current_prompt = prompt
        self.current_prompt_source = source
        return prompt

    def consume_pending_reasoning(self) -> str | None:
        reasoning = self._pending_reasoning
        self._pending_reasoning = None
        return reasoning

    def _collect_finished_request(self) -> bool:
        if self._future is None or not self._future.done():
            return False

        received_command = False
        try:
            selection = self._future.result()
            if self._future_epoch == self._selection_epoch:
                self.current_prompt = selection.prompt
                self._pending_reasoning = selection.reasoning
                received_command = True
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._future = None
            self._future_epoch = None
        return received_command

    def _is_unqueried_image(self, observation: FeedbackObservation) -> bool:
        if (
            observation.collision_stop
            or observation.image_bytes is None
            or observation.image_mime_type is None
        ):
            return False
        if self._last_query_image_revision is None:
            return True
        return observation.image_revision > self._last_query_image_revision


class OpenAIChatPromptSelector:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout_sec: float = 30.0,
        max_tokens: int = 32,
        include_history: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if not user_prompt:
            raise ValueError("user_prompt must be a non-empty string")
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self.include_history = include_history
        self.prompt_history: list[str] = []

    def choose_prompt(
        self,
        *,
        observation: FeedbackObservation,
    ) -> str:
        return self.choose_prompt_with_debug(observation=observation).prompt

    def choose_prompt_with_debug(
        self,
        *,
        observation: FeedbackObservation,
    ) -> VlmPromptSelection:
        response = self._post_json(
            _make_chat_completions_payload(
                observation=observation,
                prompt_history=(self.prompt_history if self.include_history else []),
                model=self.model,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                max_tokens=self.max_tokens,
            )
        )
        choice = response["choices"][0]
        message = choice["message"]
        prompt = message["content"]
        if self.include_history:
            self.prompt_history.append(prompt)
        return VlmPromptSelection(
            prompt=prompt,
            reasoning=_extract_reasoning(choice),
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
    observation: FeedbackObservation,
    prompt_history: list[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if observation.image_bytes is not None and observation.image_mime_type is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(
                        observation.image_bytes,
                        observation.image_mime_type,
                    )
                },
            }
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    messages.extend(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": prompt}],
        }
        for prompt in prompt_history
    )
    messages.append({"role": "user", "content": content})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


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
