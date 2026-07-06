from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from mjlab_textop.robotmdar.feedback import FeedbackObservation


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
        query_every_blocks: int,
    ) -> None:
        if query_every_blocks <= 0:
            raise ValueError(
                f"query_every_blocks must be positive, got {query_every_blocks}"
            )
        self.feedback = feedback
        self.selector = selector
        self.current_prompt = initial_prompt
        self.current_prompt_source = "initial"
        self.query_every_blocks = query_every_blocks
        self.last_error: str | None = None
        self._pending_reasoning: str | None = None
        self._stop = False
        self._future: Future[VlmPromptSelection] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_query_block: int | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def log_suffix(self) -> str:
        state = "inflight" if self._future is not None else "idle"
        suffix = f" vlm_state={state}"
        if self._last_query_block is not None:
            suffix += f" vlm_last_query_block={self._last_query_block}"
        if self.last_error is not None:
            suffix += f" vlm_last_error={self.last_error!r}"
        return suffix

    def start(self) -> None:
        self.feedback.start()

    def request_stop(self) -> None:
        self._stop = True
        self.feedback.close()
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def choose_prompt(self, *, block_count: int) -> str:
        self._collect_finished_request()
        observation = self.feedback.latest()
        if (
            not self._stop
            and self._future is None
            and observation is not None
            and self._should_query_selector(block_count)
        ):
            self._last_query_block = block_count
            self._future = self._executor.submit(
                self.selector.choose_prompt_with_debug,
                observation=observation,
            )
        return self.current_prompt

    def consume_pending_reasoning(self) -> str | None:
        reasoning = self._pending_reasoning
        self._pending_reasoning = None
        return reasoning

    def _collect_finished_request(self) -> None:
        if self._future is None or not self._future.done():
            return

        try:
            selection = self._future.result()
            self.current_prompt = selection.prompt
            self.current_prompt_source = "vlm"
            self._pending_reasoning = selection.reasoning
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._future = None

    def _should_query_selector(self, block_count: int) -> bool:
        if self._last_query_block is None:
            return True
        return block_count - self._last_query_block >= self.query_every_blocks


class OpenAIChatPromptSelector:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str | None = None,
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
        observation: FeedbackObservation | None,
    ) -> str:
        return self.choose_prompt_with_debug(observation=observation).prompt

    def choose_prompt_with_debug(
        self,
        *,
        observation: FeedbackObservation | None,
    ) -> VlmPromptSelection:
        response = self._post_json(
            _make_chat_completions_payload(
                observation=observation,
                prompt_history=(
                    self.prompt_history if self.include_history else []
                ),
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
    observation: FeedbackObservation | None,
    prompt_history: list[str],
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if (
        observation is not None
        and observation.image_bytes is not None
        and observation.image_mime_type is not None
    ):
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
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt is not None
        else []
    )
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
