from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

from mjlab_textop.robotmdar.feedback import FeedbackObservation

DescriptionCallback = Callable[[str], None]
DescriptionErrorCallback = Callable[[str], None]


class ObservationProvider(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def latest(self) -> FeedbackObservation | None: ...


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
        self.current_prompt_source = "fallback"
        self.query_every_blocks = query_every_blocks
        self.last_error: str | None = None
        self._stop = False
        self._future: Future[str] | None = None
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
        suffix = f" vlm={state}"
        if self.last_error is not None:
            suffix += f" vlm_error={self.last_error}"
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
                self.selector.choose_prompt,
                observation=observation,
            )
        return self.current_prompt

    def _collect_finished_request(self) -> None:
        if self._future is None or not self._future.done():
            return

        try:
            next_prompt = self._future.result().strip()
            if not next_prompt:
                self.last_error = "Empty"
                return
            self.current_prompt = next_prompt
            self.current_prompt_source = "prev"
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__
        finally:
            self._future = None

    def _should_query_selector(self, block_count: int) -> bool:
        if self._last_query_block is None:
            return True
        return block_count - self._last_query_block >= self.query_every_blocks


class VlmDescriptionPlanner:
    def __init__(
        self,
        *,
        feedback: ObservationProvider,
        describer: "OpenAIChatObservationDescriber",
        query_every_blocks: int,
        on_description: DescriptionCallback | None = None,
        on_error: DescriptionErrorCallback | None = None,
    ) -> None:
        if query_every_blocks <= 0:
            raise ValueError(
                f"query_every_blocks must be positive, got {query_every_blocks}"
            )
        self.feedback = feedback
        self.describer = describer
        self.query_every_blocks = query_every_blocks
        self.on_description = on_description
        self.on_error = on_error
        self.last_description: str | None = None
        self.last_error: str | None = None
        self._stop = False
        self._future: Future[str] | None = None
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
        suffix = f" vlm_describe={state}"
        if self.last_error is not None:
            suffix += f" vlm_describe_error={self.last_error}"
        return suffix

    def start(self) -> None:
        self.feedback.start()

    def request_stop(self) -> None:
        self._stop = True
        self.feedback.close()
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def tick(self, *, block_count: int) -> None:
        self._collect_finished_request()
        observation = self.feedback.latest()
        if (
            not self._stop
            and self._future is None
            and observation is not None
            and self._should_query_describer(block_count)
        ):
            self._last_query_block = block_count
            self._future = self._executor.submit(
                self.describer.describe,
                observation=observation,
            )

    def _collect_finished_request(self) -> None:
        if self._future is None or not self._future.done():
            return

        try:
            description = self._future.result().strip()
            if not description:
                self.last_error = "Empty"
                self._emit_error("Empty")
                return
            self.last_description = description
            self.last_error = None
            if self.on_description is not None:
                self.on_description(description)
        except Exception as exc:
            self.last_error = type(exc).__name__
            self._emit_error(type(exc).__name__)
        finally:
            self._future = None

    def _emit_error(self, error: str) -> None:
        if self.on_error is not None:
            self.on_error(error)

    def _should_query_describer(self, block_count: int) -> bool:
        if self._last_query_block is None:
            return True
        return block_count - self._last_query_block >= self.query_every_blocks


class DescribingPromptPlanner:
    def __init__(
        self,
        *,
        prompt_planner: Any,
        description_planner: VlmDescriptionPlanner,
    ) -> None:
        self.prompt_planner = prompt_planner
        self.description_planner = description_planner

    @property
    def should_stop(self) -> bool:
        return self.prompt_planner.should_stop or self.description_planner.should_stop

    @property
    def input_active(self) -> bool:
        return self.prompt_planner.input_active

    @property
    def log_suffix(self) -> str:
        return self.description_planner.log_suffix + self.prompt_planner.log_suffix

    def start(self) -> None:
        self.prompt_planner.start()
        self.description_planner.start()

    def request_stop(self) -> None:
        self.prompt_planner.request_stop()
        self.description_planner.request_stop()

    def choose_prompt(self, *, block_count: int) -> str:
        self.description_planner.tick(block_count=block_count)
        return self.prompt_planner.choose_prompt(block_count=block_count)


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

    def choose_prompt(
        self,
        *,
        observation: FeedbackObservation | None,
    ) -> str:
        response = self._post_json(
            _make_chat_completions_payload(
                image_bytes=None if observation is None else observation.image_bytes,
                image_mime_type=(
                    None if observation is None else observation.image_mime_type
                ),
                model=self.model,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                max_tokens=self.max_tokens,
            )
        )
        return str(response["choices"][0]["message"]["content"])

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAIChatObservationDescriber:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str | None = None,
        timeout_sec: float = 30.0,
        max_tokens: int = 256,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens

    def describe(
        self,
        *,
        observation: FeedbackObservation | None,
    ) -> str:
        response = self._post_json(
            _make_description_chat_completions_payload(
                state=_make_state_payload(observation=observation),
                image_bytes=None if observation is None else observation.image_bytes,
                image_mime_type=(
                    None if observation is None else observation.image_mime_type
                ),
                model=self.model,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
            )
        )
        return str(response["choices"][0]["message"]["content"])

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
    image_bytes: bytes | None,
    image_mime_type: str | None,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if image_bytes is not None and image_mime_type is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_bytes, image_mime_type)},
            }
        )
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt is not None
        else []
    )
    messages.append({"role": "user", "content": content})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _make_description_chat_completions_payload(
    *,
    state: dict[str, Any],
    image_bytes: bytes | None,
    image_mime_type: str | None,
    model: str,
    system_prompt: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    text = (
        "Describe the robot and scene in the observation. Include the robot pose, "
        "visible robot motion/state, nearby objects, goals, obstacles, and any "
        "important spatial relationships. Do not choose or suggest a robot command.\n"
        f"State: {json.dumps(state, separators=(',', ':'))}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if image_bytes is not None and image_mime_type is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_bytes, image_mime_type)},
            }
        )
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt is not None
        else []
    )
    messages.append({"role": "user", "content": content})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _make_state_payload(
    *,
    observation: FeedbackObservation | None,
) -> dict[str, Any]:
    if observation is None:
        return {
            "frame": 0,
            "started": False,
            "latest_frame": None,
            "lag_frames": 0,
            "buffer_frames": 0,
            "stale_steps": 0,
            "consecutive_stale_steps": 0,
            "robot_anchor_pos_w": [0.0, 0.0, 0.0],
            "robot_anchor_quat_w": [1.0, 0.0, 0.0, 0.0],
            "has_image": False,
        }
    return {
        "frame": int(observation.frame),
        "started": bool(observation.started),
        "latest_frame": (
            None if observation.latest_frame is None else int(observation.latest_frame)
        ),
        "lag_frames": int(observation.lag_frames),
        "buffer_frames": int(observation.buffer_frames),
        "stale_steps": int(observation.stale_steps),
        "consecutive_stale_steps": int(observation.consecutive_stale_steps),
        "robot_anchor_pos_w": [float(item) for item in observation.robot_anchor_pos_w],
        "robot_anchor_quat_w": [
            float(item) for item in observation.robot_anchor_quat_w
        ],
        "has_image": observation.image_bytes is not None
        and observation.image_mime_type is not None,
    }


def _image_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{b64encode(data).decode('ascii')}"
