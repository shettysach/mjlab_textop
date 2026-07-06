from __future__ import annotations

import json
import threading
import time
from argparse import Namespace

from mjlab_textop.robotmdar import produce
from mjlab_textop.robotmdar.feedback import (
    FeedbackObservation,
    parse_feedback_observation,
)
from mjlab_textop.robotmdar.planner import (
    ManualPromptPlanner,
    OpenAIChatPromptSelector,
    VlmPromptPlanner,
    VlmPromptSelection,
)


class _FakeObservationProvider:
    def __init__(self, observation: FeedbackObservation | None = None) -> None:
        self.observation = observation
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def latest(self) -> FeedbackObservation | None:
        return self.observation


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FailingSelector:
    def __init__(self) -> None:
        self.calls = 0
        self.finished = threading.Event()

    def choose_prompt_with_debug(self, **kwargs) -> VlmPromptSelection:
        del kwargs
        self.calls += 1
        self.finished.set()
        raise TimeoutError("vlm timed out")


class _FixedSelector:
    def __init__(self, prompt: str, reasoning: str | None = None) -> None:
        self.prompt = prompt
        self.reasoning = reasoning
        self.calls = 0
        self.finished = threading.Event()

    def choose_prompt_with_debug(self, **kwargs) -> VlmPromptSelection:
        del kwargs
        self.calls += 1
        self.finished.set()
        return VlmPromptSelection(
            prompt=self.prompt,
            reasoning=self.reasoning,
            response={},
        )


class _BlockingSelector:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def choose_prompt_with_debug(self, **kwargs) -> VlmPromptSelection:
        del kwargs
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=1)
        self.finished.set()
        return VlmPromptSelection(prompt=self.prompt, reasoning=None, response={})


def _wait_for(condition) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _observation(
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> FeedbackObservation:
    return FeedbackObservation(
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
    )


def _default_vlm_user_prompt() -> str:
    return produce._read_prompt_path(produce.DEFAULT_VLM_USER_PROMPT_FILE)


def test_parse_feedback_observation() -> None:
    observation = parse_feedback_observation(
        {
            "image": {
                "mime_type": "image/jpeg",
                "data": "anBlZyBieXRlcw==",
            },
        }
    )

    assert observation.image_bytes == b"jpeg bytes"
    assert observation.image_mime_type == "image/jpeg"


def test_manual_prompt_planner_uses_current_prompt_without_starting_thread() -> None:
    planner = ManualPromptPlanner("walk forward")

    assert planner.choose_prompt(block_count=0) == "walk forward"

    planner.prompt.text = "turn left"

    assert planner.choose_prompt(block_count=1) == "turn left"
    assert planner.should_stop is False
    assert planner.input_active is False
    assert "Enter text prompt" in planner.log_suffix


def test_vlm_planner_queries_selector_on_cadence() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("turn left")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=2,
    )

    planner.start()

    assert provider.started is True
    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert planner.current_prompt_source == "initial"
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "turn left"
    assert planner.current_prompt_source == "vlm"
    assert planner.choose_prompt(block_count=2) == "turn left"
    _wait_for(lambda: selector.calls == 2)

    planner.request_stop()

    assert provider.closed is True


def test_vlm_planner_does_not_block_while_selector_runs() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _BlockingSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=10,
    )

    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert selector.started.wait(timeout=1)
    assert planner.log_suffix == " vlm_state=inflight vlm_last_query_block=0"
    assert planner.choose_prompt(block_count=1) == "walk forward"
    assert selector.calls == 1

    selector.release.set()
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=2) == "turn right"
    assert planner.log_suffix == " vlm_state=idle vlm_last_query_block=0"

    planner.request_stop()


def test_vlm_planner_keeps_current_prompt_on_selector_errors() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FailingSelector()
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=3,
    )

    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "walk forward"
    assert selector.calls == 1
    assert planner.last_error == "TimeoutError: vlm timed out"
    assert planner.current_prompt_source == "initial"
    assert planner.log_suffix == " vlm_state=idle vlm_last_query_block=0 vlm_last_error='TimeoutError: vlm timed out'"

    planner.request_stop()


def test_vlm_planner_keeps_last_good_prompt_on_empty_selector_result() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("   ")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=2,
    )

    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "   "
    assert planner.last_error is None
    assert planner.current_prompt_source == "vlm"
    assert planner.log_suffix == " vlm_state=idle vlm_last_query_block=0"

    planner.request_stop()


def test_vlm_planner_recovers_after_empty_selector_result() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("   ")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
        query_every_blocks=1,
    )

    assert planner.choose_prompt(block_count=0) == "stand"
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "   "

    selector.prompt = " wave "
    selector.finished.clear()
    assert planner.choose_prompt(block_count=2) == "   "
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=3) == " wave "
    assert planner.last_error is None
    assert planner.current_prompt_source == "vlm"

    planner.request_stop()


def test_producer_log_prints_vlm_reasoning_once_when_enabled(monkeypatch) -> None:
    messages = []
    planner = VlmPromptPlanner(
        feedback=_FakeObservationProvider(_observation()),
        selector=_FixedSelector(
            "wave",
            reasoning="The robot is stable, so waving is feasible.",
        ),
        initial_prompt="stand",
        query_every_blocks=1,
    )

    monkeypatch.setattr(produce, "_log_producer_message", messages.append)

    assert planner.choose_prompt(block_count=0) == "stand"
    assert planner.selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "wave"

    args = Namespace(vlm_reasoning=True)
    produce._log_vlm_reasoning_if_available(planner=planner, args=args)
    produce._log_vlm_reasoning_if_available(planner=planner, args=args)

    assert messages == [
        "vlm_reasoning The robot is stable, so waving is feasible.",
    ]

    planner.request_stop()


def test_producer_log_includes_vlm_prompt_source(monkeypatch) -> None:
    messages = []
    planner = VlmPromptPlanner(
        feedback=_FakeObservationProvider(_observation()),
        selector=_FixedSelector("wave"),
        initial_prompt="stand",
        query_every_blocks=1,
    )

    monkeypatch.setattr(produce, "_log_producer_message", messages.append)

    produce._log_block_timing(
        planner=planner,
        args=Namespace(fps=50.0, log_every_blocks=1),
        block_count=1,
        frame_index=20,
        block_frames=20,
        block_start_time=time.monotonic(),
        next_send_time=time.monotonic(),
        prompt="stand",
    )

    assert "prompt_source=initial" in messages[0]
    assert "vlm_state=idle" in messages[0]

    planner.current_prompt_source = "vlm"
    produce._log_block_timing(
        planner=planner,
        args=Namespace(fps=50.0, log_every_blocks=1),
        block_count=2,
        frame_index=40,
        block_frames=20,
        block_start_time=time.monotonic(),
        next_send_time=time.monotonic(),
        prompt="wave",
    )

    assert "prompt_source=vlm" in messages[1]

    planner.request_stop()


def test_http_vlm_prompt_selector_posts_context_and_observation(monkeypatch) -> None:
    posted = {}

    def fake_urlopen(request, timeout):
        posted["url"] = request.full_url
        posted["timeout"] = timeout
        posted["payload"] = json.loads(request.data.decode("utf-8"))
        posted["content_type"] = request.headers["Content-type"]
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "wave",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        system_prompt="You are a motion planner.",
        user_prompt=_default_vlm_user_prompt(),
        timeout_sec=1.5,
        max_tokens=16,
    )

    prompt = selector.choose_prompt(
        observation=_observation(),
    )

    assert prompt == "wave"
    assert posted["url"] == "http://127.0.0.1:9379/v1/chat/completions"
    assert posted["timeout"] == 1.5
    assert posted["content_type"] == "application/json"
    assert posted["payload"]["model"] == "gemma-4-e2b-it"
    assert posted["payload"]["max_tokens"] == 16
    assert posted["payload"]["temperature"] == 0
    assert posted["payload"]["messages"][0]["role"] == "system"
    assert posted["payload"]["messages"][0]["content"][0]["text"] == (
        "You are a motion planner."
    )
    content = posted["payload"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _default_vlm_user_prompt()
    assert len(content) == 1


def test_http_vlm_prompt_selector_posts_image_from_observation_bytes(
    monkeypatch,
) -> None:
    posted = {}

    def fake_urlopen(request, timeout):
        del timeout
        posted["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "punch",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        user_prompt=_default_vlm_user_prompt(),
    )

    prompt = selector.choose_prompt(
        observation=_observation(
            image_bytes=b"jpeg bytes",
            image_mime_type="image/jpeg",
        ),
    )

    content = posted["payload"]["messages"][0]["content"]
    assert prompt == "punch"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _default_vlm_user_prompt()
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,anBlZyBieXRlcw=="},
    }


def test_http_vlm_prompt_selector_can_send_prompt_history(monkeypatch) -> None:
    posted = []
    responses = iter(
        [
            {"choices": [{"message": {"content": "walk"}}]},
            {"choices": [{"message": {"content": "stand"}}]},
        ]
    )

    def fake_urlopen(request, timeout):
        del timeout
        posted.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(next(responses))

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        system_prompt="You are a motion planner.",
        user_prompt=_default_vlm_user_prompt(),
        include_history=True,
    )

    assert selector.choose_prompt(observation=_observation()) == "walk"
    assert selector.choose_prompt(observation=_observation()) == "stand"

    assert [message["role"] for message in posted[0]["messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in posted[1]["messages"]] == [
        "system",
        "assistant",
        "user",
    ]
    assert posted[1]["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "walk"}],
    }


def test_http_vlm_prompt_selector_returns_raw_response(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'STOP. Clear location near pose.g39g}<|"|>',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        user_prompt=_default_vlm_user_prompt(),
    )

    assert (
        selector.choose_prompt(
            observation=_observation(),
        )
        == 'STOP. Clear location near pose.g39g}<|"|>'
    )


def test_http_vlm_prompt_selector_returns_debug_reasoning(monkeypatch) -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "content": "sidestep left",
                    "reasoning_content": "Obstacle is in front, so move laterally.",
                }
            }
        ]
    }

    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeResponse(raw_response)

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        user_prompt=_default_vlm_user_prompt(),
    )

    selection = selector.choose_prompt_with_debug(observation=_observation())

    assert selection.prompt == "sidestep left"
    assert selection.reasoning == "Obstacle is in front, so move laterally."
    assert selection.response == raw_response


def test_http_vlm_prompt_selector_returns_choice_reasoning(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "stop"},
                        "reasoning": "The path is blocked.",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        user_prompt=_default_vlm_user_prompt(),
    )

    selection = selector.choose_prompt_with_debug(observation=_observation())

    assert selection.prompt == "stop"
    assert selection.reasoning == "The path is blocked."


def test_make_prompt_planner_reads_vlm_prompt_files(tmp_path) -> None:
    system_prompt_file = tmp_path / "sys.md"
    user_prompt_file = tmp_path / "user.md"
    system_prompt_file.write_text("System file prompt.\n", encoding="utf-8")
    user_prompt_file.write_text("User file prompt.\n", encoding="utf-8")

    planner = produce.make_prompt_planner(
        Namespace(
            planner="vlm",
            prompt="stand",
            observation_listen_host="127.0.0.1",
            observation_listen_port=8766,
            observation_path="/observation",
            vlm_base_url="http://127.0.0.1:9379",
            vlm_model="gemma-4-e2b-it",
            vlm_system_prompt=system_prompt_file,
            vlm_user_prompt=user_prompt_file,
            vlm_timeout_sec=1.0,
            vlm_max_tokens=128,
            vlm_history=True,
            query_every_blocks=4,
        )
    )

    assert isinstance(planner, VlmPromptPlanner)
    assert planner.selector.system_prompt == "System file prompt.\n"
    assert planner.selector.user_prompt == "User file prompt.\n"
    assert planner.selector.include_history is True

    planner.request_stop()
