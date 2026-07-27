from __future__ import annotations

import json
import threading
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
from builders import motion_block

from robotmdar_textop import produce
from robotmdar_textop.feedback import (
    FeedbackObservation,
    HttpObservationReceiver,
    parse_feedback_observation,
)
from robotmdar_textop.planner import (
    ManualPromptPlanner,
    OpenAIChatPromptSelector,
    VlmExecutionContext,
    VlmPromptPlanner,
    VlmPromptSelection,
)
from robotmdar_textop.planner.followups import (
    command_followups,
    vlm_command_followups,
)
from robotmdar_textop.runtime import (
    DEFAULT_VLM_USER_PROMPT_FILE,
    BlockPlan,
    StreamConfig,
    compose_system_prompt,
    read_prompt_path,
    stream_robotmdar_blocks,
)
from textop_protocol.motion import MotionBlock, StreamControl


class _FakeObservationProvider:
    def __init__(self, observation: FeedbackObservation | None = None) -> None:
        self.observation = observation
        self.started = False
        self.closed = False
        self.request: FeedbackObservation | None = None
        self.waited_for: list[int] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def latest(self) -> FeedbackObservation | None:
        return self.observation

    def acknowledge(
        self,
        request_id: int,
        *,
        image_revision: int | None = None,
        source_frame: int | None = None,
    ) -> FeedbackObservation:
        observation = _observation(
            image_revision=image_revision or request_id,
            request_id=request_id,
            source_frame=(source_frame if source_frame is not None else request_id * 8),
        )
        self.observation = observation
        self.request = observation
        return observation

    def wait_for_observation(self, request_id: int) -> FeedbackObservation:
        self.waited_for.append(request_id)
        if self.observation is not None and self.observation.collision_stop:
            return self.observation
        observation = self.request
        self.request = None
        assert observation is not None
        assert observation.request_id == request_id
        return observation


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
        self.execution_contexts: list[VlmExecutionContext] = []

    def choose_prompt_with_debug(self, **kwargs) -> VlmPromptSelection:
        self.execution_contexts.append(kwargs["execution_context"])
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
        execution_context: VlmExecutionContext | None = None,
    ) -> VlmPromptSelection:
        del execution_context
        self.calls += 1
        self.image_revisions.append(observation.image_revision)
        self.started.set()
        self.release.wait(timeout=1)
        self.finished.set()
        return VlmPromptSelection(prompt=self.prompt, reasoning=None, response={})


def _next_prompt(
    planner: VlmPromptPlanner,
    block_count: int,
) -> str:
    return planner.next_plan(block_count=block_count).prompt


def _observation(
    *,
    image_bytes: bytes | None = b"jpeg bytes",
    image_mime_type: str | None = "image/jpeg",
    image_revision: int = 1,
    collision_stop: bool = False,
    recovery_epoch: int = 0,
    request_id: int | None = None,
    source_frame: int | None = None,
) -> FeedbackObservation:
    return FeedbackObservation(
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        image_revision=image_revision,
        collision_stop=collision_stop,
        recovery_epoch=recovery_epoch,
        request_id=request_id,
        source_frame=source_frame,
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
        {
            "collision_stop": True,
            "recovery_epoch": 7,
        }
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


def test_observation_receiver_waits_for_exact_request() -> None:
    receiver = HttpObservationReceiver(port=8766)
    received: list[FeedbackObservation] = []
    thread = threading.Thread(
        target=lambda: received.append(receiver.wait_for_observation(7))
    )
    thread.start()

    receiver.handle_post(
        b'{"source_frame":40,"image":{"mime_type":"image/jpeg","data":"cGVyaW9kaWM="}}'
    )
    thread.join(timeout=0.05)
    assert thread.is_alive()

    receiver.handle_post(
        b'{"request_id":7,"source_frame":41,'
        b'"image":{"mime_type":"image/jpeg","data":"cmVxdWVzdA=="}}'
    )
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(received) == 1
    assert received[0].image_bytes == b"request"
    assert received[0].request_id == 7
    assert received[0].source_frame == 41


def test_observation_receiver_discards_request_preempted_by_collision() -> None:
    receiver = HttpObservationReceiver(port=8766)
    receiver.handle_post(
        b'{"request_id":7,"source_frame":41,'
        b'"image":{"mime_type":"image/jpeg","data":"cmVxdWVzdA=="}}'
    )
    receiver.handle_post(b'{"collision_stop":true,"recovery_epoch":3}')

    observation = receiver.wait_for_observation(7)

    assert observation.collision_stop is True
    assert receiver._requested_observation is None


def test_manual_prompt_planner_uses_current_prompt_without_starting_thread() -> None:
    planner = ManualPromptPlanner("walk forward")

    assert planner.next_plan(block_count=0).prompt == "walk forward"

    planner.prompt.text = "turn left"

    assert planner.next_plan(block_count=1).prompt == "turn left"
    assert planner.should_stop is False
    assert planner.input_active is False
    assert "Enter text prompt" in planner.log_suffix


def test_manual_prompt_planner_locally_schedules_stand_after_lateral_command() -> None:
    planner = ManualPromptPlanner("step left", command_hold_blocks=2)

    assert planner.next_plan(block_count=0).prompt == "step left"
    assert planner.next_plan(block_count=1).prompt == "step left"
    assert planner.next_plan(block_count=2).prompt == "stand"

    planner.prompt.text = "turn right"
    planner.prompt.revision += 1

    assert planner.next_plan(block_count=3).prompt == "turn right"
    assert planner.next_plan(block_count=4).prompt == "turn right"
    assert planner.next_plan(block_count=5).prompt == "stand"


def test_manual_prompt_planner_accepts_repeated_manual_command() -> None:
    planner = ManualPromptPlanner("step left")

    assert planner.next_plan(block_count=0).prompt == "step left"
    assert planner.next_plan(block_count=1).prompt == "stand"

    planner.prompt.revision += 1

    assert planner.next_plan(block_count=2).prompt == "step left"
    assert planner.next_plan(block_count=3).prompt == "stand"


def test_manual_prompt_planner_does_not_bound_walk_commands() -> None:
    planner = ManualPromptPlanner("walk", command_hold_blocks=1)

    assert planner.next_plan(block_count=0).prompt == "walk"
    assert planner.next_plan(block_count=1).prompt == "walk"
    assert planner.next_plan(block_count=2).prompt == "walk"


def test_command_followups_match_direction_words_only() -> None:
    assert command_followups("turn RIGHT") == ["stand"]
    assert command_followups("walk") == []
    assert vlm_command_followups("walk") == ["stand"]
    assert command_followups("bright light") == []
    assert command_followups("move upright") == []
    assert command_followups("leftover motion") == []


def test_vlm_planner_bounds_transient_command_then_emits_request() -> None:
    provider = _FakeObservationProvider(_observation(image_revision=99))
    selector = _FixedSelector("turn left")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk",
        command_hold_blocks=2,
    )

    planner.start()
    assert provider.started is True
    assert _next_prompt(planner, 0) == "walk"
    assert _next_prompt(planner, 1) == "walk"
    assert _next_prompt(planner, 2) == "stand"
    assert _next_prompt(planner, 3) == "stand"
    assert selector.calls == 0

    request = planner.next_plan(block_count=4)
    assert request.prompt == "stand"
    assert request.source == "requested"
    assert request.request_id == 1
    assert planner.log_suffix == " vlm=awaiting_observation request=1"
    assert selector.calls == 0

    provider.acknowledge(1, image_revision=100, source_frame=41)
    next_plan = planner.next_plan(block_count=5)
    assert provider.waited_for == [1]
    assert selector.calls == 1
    assert next_plan.prompt == "turn left"
    assert next_plan.source == "vlm"
    assert next_plan.reset_pacing is True
    assert "last_observation=(request: 1, source_frame: 41)" in planner.log_suffix

    planner.request_stop()
    assert provider.closed is True


def test_vlm_planner_queries_only_exact_request_observations() -> None:
    provider = _FakeObservationProvider(_observation(image_revision=1))
    selector = _FixedSelector("wave")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
    )

    assert _next_prompt(planner, 0) == "stand"
    provider.observation = _observation(image_revision=500)
    request = planner.next_plan(block_count=1)
    assert request.source == "requested"
    assert selector.calls == 0

    provider.acknowledge(1, image_revision=7, source_frame=88)
    planner.next_plan(block_count=2)

    assert selector.calls == 1
    assert selector.execution_contexts == [
        VlmExecutionContext(
            previous_decision=None,
            executed_sequence=("stand",),
            current_command="stand",
        )
    ]


def test_vlm_planner_reports_completed_followup_to_next_query() -> None:
    provider = _FakeObservationProvider()
    selector = _FixedSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
        command_hold_blocks=2,
    )

    assert _next_prompt(planner, 0) == "stand"
    assert _next_prompt(planner, 1) == "stand"
    assert planner.next_plan(block_count=2).source == "requested"
    provider.acknowledge(1)

    assert _next_prompt(planner, 3) == "turn right"
    assert _next_prompt(planner, 4) == "turn right"
    assert _next_prompt(planner, 5) == "stand"
    assert _next_prompt(planner, 6) == "stand"
    assert planner.next_plan(block_count=7).source == "requested"
    provider.acknowledge(2)
    planner.next_plan(block_count=8)

    assert selector.execution_contexts[1] == VlmExecutionContext(
        previous_decision="turn right",
        executed_sequence=("turn right", "stand"),
        current_command="stand",
    )


def test_vlm_planner_pauses_while_selector_runs() -> None:
    provider = _FakeObservationProvider()
    selector = _BlockingSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
    )

    assert _next_prompt(planner, 0) == "stand"
    assert planner.next_plan(block_count=1).source == "requested"
    provider.acknowledge(1)

    result: list[BlockPlan] = []
    thread = threading.Thread(
        target=lambda: result.append(planner.next_plan(block_count=2))
    )
    thread.start()
    assert selector.started.wait(timeout=1)
    assert thread.is_alive()
    assert selector.calls == 1

    selector.release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert len(result) == 1
    assert result[0].prompt == "turn right"
    assert result[0].reset_pacing is True


def test_vlm_planner_discards_selection_if_collision_arrives_during_query() -> None:
    provider = _FakeObservationProvider()
    selector = _BlockingSelector("walk")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
    )

    assert _next_prompt(planner, 0) == "stand"
    assert planner.next_plan(block_count=1).source == "requested"
    provider.acknowledge(1)

    result: list[BlockPlan] = []
    thread = threading.Thread(
        target=lambda: result.append(planner.next_plan(block_count=2))
    )
    thread.start()
    assert selector.started.wait(timeout=1)
    provider.observation = _observation(collision_stop=True, recovery_epoch=9)
    selector.release.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(result) == 1
    assert result[0].prompt == "stand"
    assert result[0].source == "collision_recovery"
    assert result[0].recovery_epoch == 9


def test_vlm_planner_recovers_from_collision_while_awaiting_observation() -> None:
    provider = _FakeObservationProvider()
    selector = _FixedSelector("walk")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
    )

    assert _next_prompt(planner, 0) == "stand"
    assert planner.next_plan(block_count=1).source == "requested"
    provider.observation = _observation(collision_stop=True, recovery_epoch=12)

    recovery = planner.next_plan(block_count=2)
    assert selector.calls == 0
    assert recovery.prompt == "stand"
    assert recovery.source == "collision_recovery"
    assert recovery.reset_pacing is True

    provider.observation = _observation(collision_stop=False, recovery_epoch=12)
    assert planner.next_plan(block_count=3).source == "followup"
    assert planner.next_plan(block_count=4).source == "requested"


def test_vlm_planner_fails_closed_on_selector_error() -> None:
    provider = _FakeObservationProvider()
    selector = _FailingSelector()
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="stand",
    )

    assert _next_prompt(planner, 0) == "stand"
    assert planner.next_plan(block_count=1).source == "requested"
    provider.acknowledge(1)

    with pytest.raises(RuntimeError, match="VLM request failed"):
        planner.next_plan(block_count=2)

    assert planner.should_stop is True
    assert planner.last_error == "TimeoutError: vlm timed out"
    assert "error='TimeoutError: vlm timed out'" in planner.log_suffix


def test_producer_log_prints_vlm_reasoning_once_when_enabled(monkeypatch) -> None:
    messages = []
    provider = _FakeObservationProvider()
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=_FixedSelector(
            "wave",
            reasoning="The robot is stable, so waving is feasible.",
        ),
        initial_prompt="stand",
    )

    monkeypatch.setattr(produce, "_log_producer_message", messages.append)

    assert _next_prompt(planner, 0) == "stand"
    assert planner.next_plan(block_count=1).source == "requested"
    provider.acknowledge(1)
    assert _next_prompt(planner, 2) == "wave"

    args = Namespace(vlm_reasoning=True)
    produce._log_vlm_reasoning_if_available(planner=planner, args=args)
    produce._log_vlm_reasoning_if_available(planner=planner, args=args)

    assert messages == [
        "vlm_reasoning The robot is stable, so waving is feasible.",
    ]


def test_stream_generates_and_sends_planned_block(
    monkeypatch,
) -> None:
    events = []

    class Controller:
        should_stop = False
        input_active = False
        log_suffix = ""

        def next_plan(self, *, block_count: int) -> BlockPlan:
            events.append(("plan", block_count))
            self.should_stop = True
            return BlockPlan(prompt="stand", source="test")

    class Generator:
        def next_block(self, **kwargs):
            events.append(("generate", kwargs["index"]))
            return SimpleNamespace(joint_pos=SimpleNamespace(shape=(20,)))

    class Connection:
        def sendall(self, data: bytes) -> None:
            events.append(("send", data))

    monkeypatch.setattr(
        "robotmdar_textop.runtime.textop_block_to_wire",
        lambda _block: b"block",
    )
    monkeypatch.setattr("robotmdar_textop.runtime.time.sleep", lambda _delay: None)

    stream_robotmdar_blocks(
        conn=Connection(),
        generator=Generator(),
        prompt_controller=Controller(),
        cfg=StreamConfig(guidance_scale=5.0, log_every_blocks=0),
        log_message=lambda _message: events.append(("log", None)),
    )

    assert events == [
        ("plan", 0),
        ("generate", 0),
        ("send", b"block"),
        ("log", None),
    ]


def test_stream_stops_generating_until_request_is_acknowledged(
    monkeypatch,
) -> None:
    wait_started = threading.Event()
    release_ack = threading.Event()

    class BlockingProvider(_FakeObservationProvider):
        def wait_for_observation(self, request_id: int) -> FeedbackObservation:
            self.waited_for.append(request_id)
            wait_started.set()
            assert release_ack.wait(timeout=1)
            return _observation(
                request_id=request_id,
                source_frame=17,
            )

    class Generator:
        def next_block(self, **kwargs) -> MotionBlock:
            return replace(
                motion_block(index=kwargs["index"]),
                control=StreamControl(prompt=kwargs["prompt"]),
            )

        def observation_block(self, **kwargs) -> MotionBlock:
            return replace(
                motion_block(index=kwargs["index"]),
                control=StreamControl(
                    prompt=kwargs["prompt"],
                    request_id=kwargs["request_id"],
                ),
            )

    provider = BlockingProvider()
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=_FixedSelector("wave"),
        initial_prompt="stand",
    )
    sent: list[MotionBlock] = []

    class Connection:
        def sendall(self, block: MotionBlock) -> None:
            sent.append(block)
            if len(sent) == 3:
                planner.request_stop()

    monkeypatch.setattr(
        "robotmdar_textop.runtime.textop_block_to_wire",
        lambda block: block,
    )
    monkeypatch.setattr("robotmdar_textop.runtime.time.sleep", lambda _delay: None)
    errors: list[BaseException] = []

    def run_stream() -> None:
        try:
            stream_robotmdar_blocks(
                conn=Connection(),
                generator=Generator(),
                prompt_controller=planner,
                cfg=StreamConfig(guidance_scale=5.0, log_every_blocks=0),
                log_message=lambda _message: None,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_stream)
    thread.start()
    assert wait_started.wait(timeout=1)

    assert len(sent) == 2
    assert sent[0].control.prompt == "stand"
    assert sent[0].control.request_id is None
    assert sent[1].control.request_id == 1
    assert thread.is_alive()

    release_ack.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert errors == []
    assert [block.control.prompt for block in sent] == ["stand", "stand", "wave"]


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
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        system_prompt="You are a motion planner.",
        user_prompt=_default_vlm_user_prompt(),
        timeout_sec=1.5,
    )
    assert selector.history_length == -1

    prompt = selector.choose_prompt(
        observation=_observation(
            image_bytes=None,
            image_mime_type=None,
            image_revision=0,
        ),
        execution_context=VlmExecutionContext(
            previous_decision="turn right",
            executed_sequence=("turn right", "stand"),
            current_command="stand",
        ),
    )

    assert prompt == "wave"
    assert posted["url"] == "http://127.0.0.1:9379/v1/chat/completions"
    assert posted["timeout"] == 1.5
    assert posted["content_type"] == "application/json"
    assert posted["payload"]["model"] == "gemma-4-e2b-it"
    assert "max_tokens" not in posted["payload"]
    assert posted["payload"]["temperature"] == 0
    assert posted["payload"]["messages"][0]["role"] == "system"
    assert posted["payload"]["messages"][0]["content"][0]["text"] == (
        "You are a motion planner."
    )
    content = posted["payload"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == (
        "Previous decision: turn right\n"
        "Executed sequence: turn right -> stand\n"
        "Current command: stand\n\n"
        f"{_default_vlm_user_prompt()}"
    )
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
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
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
            {
                "choices": [
                    {
                        "message": {
                            "content": "walk",
                            "reasoning_content": "The path ahead is open.",
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "stand"}}]},
            {"choices": [{"message": {"content": "turn left"}}]},
        ]
    )

    def fake_urlopen(request, timeout):
        del timeout
        posted.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(next(responses))

    monkeypatch.setattr(
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
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
        "reasoning_content": "The path ahead is open.",
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
    assert "reasoning_content" not in posted[2]["messages"][2]
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
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
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


def test_http_vlm_prompt_selector_default_history_is_unlimited(monkeypatch) -> None:
    posted = []

    def fake_urlopen(request, timeout):
        del timeout
        posted.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": "stand"}}]})

    monkeypatch.setattr(
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        system_prompt="You are a motion planner.",
        user_prompt=_default_vlm_user_prompt(),
    )

    for image_bytes in (b"first", b"second", b"third"):
        selector.choose_prompt(observation=_observation(image_bytes=image_bytes))

    assert [message["role"] for message in posted[2]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


def test_http_vlm_prompt_selector_rejects_empty_history() -> None:
    with pytest.raises(
        ValueError,
        match=r"history_length must be -1 \(unlimited\) or positive",
    ):
        OpenAIChatPromptSelector(
            base_url="http://127.0.0.1:9379",
            model="gemma-4-e2b-it",
            system_prompt="You are a motion planner.",
            user_prompt=_default_vlm_user_prompt(),
            history_length=0,
        )


def test_http_vlm_prompt_selector_rejects_history_below_unlimited_sentinel() -> None:
    with pytest.raises(
        ValueError,
        match=r"history_length must be -1 \(unlimited\) or positive",
    ):
        OpenAIChatPromptSelector(
            base_url="http://127.0.0.1:9379",
            model="gemma-4-e2b-it",
            system_prompt="You are a motion planner.",
            user_prompt=_default_vlm_user_prompt(),
            history_length=-2,
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
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
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
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
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
        "robotmdar_textop.planner.vlm.urllib.request.urlopen",
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
            vlm_history_length=5,
            command_hold_blocks=4,
        )
    )

    assert isinstance(planner, VlmPromptPlanner)
    assert planner.selector.system_prompt == compose_system_prompt(
        "System file prompt.\n"
    )
    assert planner.selector.user_prompt == "User file prompt.\n"
    assert planner.selector.history_length == 5

    planner.request_stop()
