from __future__ import annotations

import json
import threading
import time
from argparse import Namespace
from types import SimpleNamespace

import pytest

from mjlab_textop.robotmdar import produce
from mjlab_textop.robotmdar.feedback import (
    FeedbackObservation,
    HttpObservationReceiver,
    parse_feedback_observation,
)
from mjlab_textop.robotmdar.planner import (
    ManualPromptPlanner,
    OpenAIChatPromptSelector,
    VlmPromptPlanner,
    VlmPromptSelection,
)
from mjlab_textop.robotmdar.planner.followups import command_followups
from mjlab_textop.robotmdar.runtime import (
    DEFAULT_VLM_USER_PROMPT_FILE,
    StreamConfig,
    log_stream_timing,
    read_prompt_path,
    stream_robotmdar_blocks,
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
        self.image_revisions: list[int] = []

    def choose_prompt_with_debug(
        self,
        *,
        observation: FeedbackObservation,
    ) -> VlmPromptSelection:
        self.calls += 1
        self.image_revisions.append(observation.image_revision)
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


def _choose_and_mark_block_sent(
    planner: VlmPromptPlanner,
    block_count: int,
) -> str:
    prompt = planner.choose_prompt(block_count=block_count)
    planner.on_block_sent(block_count=block_count)
    return prompt


def _observation(
    *,
    image_bytes: bytes | None = b"jpeg bytes",
    image_mime_type: str | None = "image/jpeg",
    image_revision: int = 1,
    collision_stop: bool = False,
    recovery_epoch: int = 0,
) -> FeedbackObservation:
    return FeedbackObservation(
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        image_revision=image_revision,
        collision_stop=collision_stop,
        recovery_epoch=recovery_epoch,
    )


def _default_vlm_user_prompt() -> str:
    return read_prompt_path(DEFAULT_VLM_USER_PROMPT_FILE)


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
    assert observation.image_revision == 1
    assert observation.collision_stop is False


def test_parse_collision_feedback_without_image() -> None:
    observation = parse_feedback_observation(
        {"collision_stop": True, "recovery_epoch": 7}
    )

    assert observation.image_bytes is None
    assert observation.image_revision == 0
    assert observation.collision_stop is True
    assert observation.recovery_epoch == 7


def test_observation_receiver_merges_images_without_clearing_collision() -> None:
    receiver = HttpObservationReceiver(port=8766)

    receiver.handle_post(b'{"collision_stop":true,"recovery_epoch":7}')
    receiver.handle_post(b'{"image":{"mime_type":"image/jpeg","data":"anBlZw=="}}')

    observation = receiver.latest()
    assert observation is not None
    assert observation.image_bytes == b"jpeg"
    assert observation.image_revision == 1
    assert observation.collision_stop is True
    assert observation.recovery_epoch == 7

    receiver.handle_post(b'{"collision_stop":false,"recovery_epoch":7}')

    observation = receiver.latest()
    assert observation is not None
    assert observation.image_bytes == b"jpeg"
    assert observation.image_revision == 1
    assert observation.collision_stop is False
    assert observation.recovery_epoch == 7

    receiver.handle_post(b'{"image":{"mime_type":"image/jpeg","data":"anBlZw=="}}')

    observation = receiver.latest()
    assert observation is not None
    assert observation.image_revision == 2


def test_manual_prompt_planner_uses_current_prompt_without_starting_thread() -> None:
    planner = ManualPromptPlanner("walk forward")

    assert planner.choose_prompt(block_count=0) == "walk forward"

    planner.prompt.text = "turn left"

    assert planner.choose_prompt(block_count=1) == "turn left"
    assert planner.should_stop is False
    assert planner.input_active is False
    assert "Enter text prompt" in planner.log_suffix


def test_manual_prompt_planner_locally_schedules_stand_after_lateral_command() -> None:
    planner = ManualPromptPlanner("step left", command_hold_blocks=2)

    assert planner.choose_prompt(block_count=0) == "step left"
    assert planner.choose_prompt(block_count=1) == "step left"
    assert planner.choose_prompt(block_count=2) == "stand"

    planner.prompt.text = "turn right"
    planner.prompt.revision += 1

    assert planner.choose_prompt(block_count=3) == "turn right"
    assert planner.choose_prompt(block_count=4) == "turn right"
    assert planner.choose_prompt(block_count=5) == "stand"


def test_manual_prompt_planner_accepts_repeated_manual_command() -> None:
    planner = ManualPromptPlanner("step left")

    assert planner.choose_prompt(block_count=0) == "step left"
    assert planner.choose_prompt(block_count=1) == "stand"

    planner.prompt.revision += 1

    assert planner.choose_prompt(block_count=2) == "step left"
    assert planner.choose_prompt(block_count=3) == "stand"


def test_command_followups_match_direction_words_only() -> None:
    assert command_followups("turn RIGHT") == ["stand"]
    assert command_followups("bright light") == []
    assert command_followups("move upright") == []
    assert command_followups("leftover motion") == []


def test_vlm_planner_queries_each_image_once() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("turn left")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    planner.start()

    assert provider.started is True
    assert _choose_and_mark_block_sent(planner, 0) == "walk forward"
    assert planner.current_prompt_source == "initial"
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 1) == "turn left"
    assert planner.current_prompt_source == "vlm"
    assert _choose_and_mark_block_sent(planner, 2) == "stand"
    assert planner.current_prompt_source == "followup"
    assert selector.calls == 1
    assert _choose_and_mark_block_sent(planner, 3) == "stand"

    assert selector.calls == 1
    provider.observation = _observation(image_revision=2)
    assert _choose_and_mark_block_sent(planner, 4) == "stand"
    _wait_for(lambda: selector.calls == 2)

    planner.request_stop()

    assert provider.closed is True


def test_vlm_planner_forces_stand_until_collision_recovery_clears() -> None:
    provider = _FakeObservationProvider(
        _observation(collision_stop=True, recovery_epoch=7)
    )
    selector = _FixedSelector("walk forward")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    assert _choose_and_mark_block_sent(planner, 0) == "stand"
    assert planner.current_prompt_source == "collision_recovery"
    assert planner.recovery_epoch == 7
    assert _choose_and_mark_block_sent(planner, 1) == "stand"
    assert selector.calls == 0

    provider.observation = _observation(image_revision=2, collision_stop=False)

    assert _choose_and_mark_block_sent(planner, 2) == "stand"
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 3) == "walk forward"

    planner.request_stop()


def test_vlm_planner_locally_schedules_stand_after_lateral_command() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("step RIGHT")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        command_hold_blocks=3,
    )

    assert _choose_and_mark_block_sent(planner, 0) == "walk forward"
    assert selector.finished.wait(timeout=1)

    assert _choose_and_mark_block_sent(planner, 1) == "step RIGHT"
    assert planner.current_prompt_source == "vlm"
    assert selector.calls == 1

    assert _choose_and_mark_block_sent(planner, 2) == "step RIGHT"
    assert _choose_and_mark_block_sent(planner, 3) == "step RIGHT"
    assert selector.calls == 1

    assert _choose_and_mark_block_sent(planner, 4) == "stand"
    assert planner.current_prompt_source == "followup"
    assert _choose_and_mark_block_sent(planner, 5) == "stand"
    assert _choose_and_mark_block_sent(planner, 6) == "stand"
    assert selector.calls == 1

    provider.observation = _observation(image_revision=2)
    assert _choose_and_mark_block_sent(planner, 7) == "stand"
    _wait_for(lambda: selector.calls == 2)

    planner.request_stop()


def test_vlm_planner_does_not_block_while_selector_runs() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _BlockingSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert not selector.started.wait(timeout=0.05)

    planner.on_block_sent(block_count=0)

    assert selector.started.wait(timeout=1)
    assert planner.log_suffix == (
        " vlm_state=inflight vlm_last_query_block=0 vlm_last_query_image_revision=1"
    )
    assert _choose_and_mark_block_sent(planner, 1) == "walk forward"
    assert selector.calls == 1

    selector.release.set()
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 2) == "turn right"
    assert planner.log_suffix == (
        " vlm_state=idle vlm_last_query_block=0 vlm_last_query_image_revision=1"
    )

    planner.request_stop()


def test_vlm_planner_coalesces_images_while_request_is_inflight() -> None:
    provider = _FakeObservationProvider(_observation(image_revision=1))
    selector = _BlockingSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    assert _choose_and_mark_block_sent(planner, 0) == "walk forward"
    assert selector.started.wait(timeout=1)

    provider.observation = _observation(image_revision=2)
    assert _choose_and_mark_block_sent(planner, 1) == "walk forward"
    provider.observation = _observation(image_revision=3)

    selector.release.set()
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 2) == "turn right"
    assert _choose_and_mark_block_sent(planner, 3) == "stand"
    assert _choose_and_mark_block_sent(planner, 4) == "stand"
    _wait_for(lambda: selector.calls == 2)

    assert selector.image_revisions == [1, 3]
    planner.request_stop()


def test_vlm_planner_ignores_observations_without_images() -> None:
    provider = _FakeObservationProvider(
        _observation(
            image_bytes=None,
            image_mime_type=None,
            image_revision=0,
        )
    )
    selector = _FixedSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    assert _choose_and_mark_block_sent(planner, 0) == "walk forward"
    assert selector.calls == 0

    planner.request_stop()


def test_vlm_planner_keeps_current_prompt_on_selector_errors() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FailingSelector()
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    assert _choose_and_mark_block_sent(planner, 0) == "walk forward"
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 1) == "walk forward"
    assert selector.calls == 1
    assert planner.last_error == "TimeoutError: vlm timed out"
    assert planner.current_prompt_source == "initial"
    assert (
        planner.log_suffix == " vlm_state=idle vlm_last_query_block=0"
        " vlm_last_query_image_revision=1"
        " vlm_last_error='TimeoutError: vlm timed out'"
    )

    planner.request_stop()


def test_vlm_planner_keeps_last_good_prompt_on_empty_selector_result() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("   ")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
    )

    assert _choose_and_mark_block_sent(planner, 0) == "walk forward"
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 1) == "   "
    assert planner.last_error is None
    assert planner.current_prompt_source == "vlm"
    assert planner.log_suffix == (
        " vlm_state=idle vlm_last_query_block=0 vlm_last_query_image_revision=1"
    )

    planner.request_stop()


def test_vlm_planner_recovers_after_empty_selector_result() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("   ")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
    )

    assert _choose_and_mark_block_sent(planner, 0) == "stand"
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 1) == "   "

    selector.prompt = " wave "
    selector.finished.clear()
    provider.observation = _observation(image_revision=2)
    assert _choose_and_mark_block_sent(planner, 2) == "   "
    assert selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 3) == " wave "
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
    )

    monkeypatch.setattr(produce, "_log_producer_message", messages.append)

    assert _choose_and_mark_block_sent(planner, 0) == "stand"
    assert planner.selector.finished.wait(timeout=1)
    assert _choose_and_mark_block_sent(planner, 1) == "wave"

    args = Namespace(vlm_reasoning=True)
    produce._log_vlm_reasoning_if_available(planner=planner, args=args)
    produce._log_vlm_reasoning_if_available(planner=planner, args=args)

    assert messages == [
        "vlm_reasoning The robot is stable, so waving is feasible.",
    ]

    planner.request_stop()


def test_stream_submits_planner_work_after_generation_and_send(monkeypatch) -> None:
    events = []

    class Controller:
        should_stop = False
        input_active = False
        log_suffix = ""
        recovery_epoch = 0

        def choose_prompt(self, *, block_count: int) -> str:
            events.append(("choose", block_count))
            return "stand"

        def on_block_sent(self, *, block_count: int) -> None:
            events.append(("planner", block_count))
            self.should_stop = True

    class Generator:
        def next_block(self, **kwargs):
            events.append(("generate", kwargs["index"]))
            return SimpleNamespace(joint_pos=SimpleNamespace(shape=(20,)))

    class Connection:
        def sendall(self, data: bytes) -> None:
            events.append(("send", data))

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.runtime.textop_block_to_ndjson_message",
        lambda _block: "block\n",
    )
    monkeypatch.setattr(
        "mjlab_textop.robotmdar.runtime.time.sleep", lambda _delay: None
    )

    stream_robotmdar_blocks(
        conn=Connection(),
        generator=Generator(),
        prompt_controller=Controller(),
        cfg=StreamConfig(guidance_scale=5.0, log_every_blocks=0),
        log_message=lambda _message: None,
        prompt_source=lambda _controller: "test",
    )

    assert events == [
        ("choose", 0),
        ("generate", 0),
        ("send", b"block\n"),
        ("planner", 0),
    ]


def test_producer_log_includes_vlm_prompt_source(monkeypatch) -> None:
    messages = []
    planner = VlmPromptPlanner(
        feedback=_FakeObservationProvider(_observation()),
        selector=_FixedSelector("wave"),
        initial_prompt="stand",
    )

    monkeypatch.setattr(produce, "_log_producer_message", messages.append)

    log_stream_timing(
        prompt_controller=planner,
        cfg=StreamConfig(guidance_scale=0.0, log_every_blocks=1),
        log_message=produce._log_producer_message,
        prompt_source=produce._prompt_source,
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
    log_stream_timing(
        prompt_controller=planner,
        cfg=StreamConfig(guidance_scale=0.0, log_every_blocks=1),
        log_message=produce._log_producer_message,
        prompt_source=produce._prompt_source,
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
    assert selector.history_length == 5

    prompt = selector.choose_prompt(
        observation=_observation(
            image_bytes=None,
            image_mime_type=None,
            image_revision=0,
        ),
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
        system_prompt="You are a motion planner.",
        user_prompt=_default_vlm_user_prompt(),
    )

    prompt = selector.choose_prompt(
        observation=_observation(
            image_bytes=b"jpeg bytes",
            image_mime_type="image/jpeg",
        ),
    )

    content = posted["payload"]["messages"][1]["content"]
    assert prompt == "punch"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _default_vlm_user_prompt()
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,anBlZyBieXRlcw=="},
    }


def test_http_vlm_prompt_selector_sends_bounded_complete_turns(monkeypatch) -> None:
    posted = []
    responses = iter(
        [
            {"choices": [{"message": {"content": "walk"}}]},
            {"choices": [{"message": {"content": "stand"}}]},
            {"choices": [{"message": {"content": "turn left"}}]},
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
        history_length=2,
    )

    assert (
        selector.choose_prompt(observation=_observation(image_bytes=b"first")) == "walk"
    )
    assert (
        selector.choose_prompt(observation=_observation(image_bytes=b"second"))
        == "stand"
    )
    assert (
        selector.choose_prompt(observation=_observation(image_bytes=b"third"))
        == "turn left"
    )

    assert [message["role"] for message in posted[0]["messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in posted[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert posted[1]["messages"][2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "walk"}],
    }
    assert posted[1]["messages"][1]["content"][1]["image_url"]["url"] == (
        "data:image/jpeg;base64,Zmlyc3Q="
    )
    assert [message["role"] for message in posted[2]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert posted[2]["messages"][1]["content"][1]["image_url"]["url"] == (
        "data:image/jpeg;base64,c2Vjb25k"
    )
    assert posted[2]["messages"][2]["content"][0]["text"] == "stand"
    assert posted[2]["messages"][3]["content"][1]["image_url"]["url"] == (
        "data:image/jpeg;base64,dGhpcmQ="
    )


def test_http_vlm_prompt_selector_history_length_one_is_stateless(monkeypatch) -> None:
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
        history_length=1,
    )

    selector.choose_prompt(observation=_observation(image_bytes=b"first"))
    selector.choose_prompt(observation=_observation(image_bytes=b"second"))

    assert [message["role"] for message in posted[0]["messages"]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in posted[1]["messages"]] == [
        "system",
        "user",
    ]


def test_http_vlm_prompt_selector_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="history_length must be positive"):
        OpenAIChatPromptSelector(
            base_url="http://127.0.0.1:9379",
            model="gemma-4-e2b-it",
            system_prompt="You are a motion planner.",
            user_prompt=_default_vlm_user_prompt(),
            history_length=0,
        )


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
        system_prompt="You are a motion planner.",
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
        system_prompt="You are a motion planner.",
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
        system_prompt="You are a motion planner.",
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
            vlm_history_length=5,
            command_hold_blocks=4,
        )
    )

    assert isinstance(planner, VlmPromptPlanner)
    assert planner.selector.system_prompt == "System file prompt.\n"
    assert planner.selector.user_prompt == "User file prompt.\n"
    assert planner.selector.history_length == 5

    planner.request_stop()
