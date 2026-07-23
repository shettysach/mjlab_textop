from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
from mcp.types import ImageContent, TextContent

from mjlab_textop.scout.config import ScoutConfig
from mjlab_textop.scout.runtime import ScoutRuntime
from mjlab_textop.scout.schemas import (
    CapturedView,
    SceneSummary,
    ScoutView,
    TaskInfo,
)
from mjlab_textop.scout.tools import ScoutTools
from tasks.catalog import TASKS, TaskDefinition


class _FakeModel:
    nbody = 2
    body_geomadr = np.array([0, 0])
    body_geomnum = np.array([0, 1])
    body_pos = np.array([[0.0, 0.0, 0.0], [6.0, -2.0, 0.0]])
    geom_type = np.array([6])
    geom_size = np.array([[1.0, 2.0, 0.1]])
    geom_rgba = np.array([[0.0, 1.0, 0.0, 1.0]])
    opt = SimpleNamespace(timestep=0.002)

    def body(self, index: int):
        return SimpleNamespace(name=("world", "goal")[index])

    def geom(self, index: int):
        return SimpleNamespace(name="goal_visual")


def test_scout_catalog_has_simple_objectives() -> None:
    assert set(TASKS) == {
        "straight",
        "blocked-straight",
        "side-goals",
        "turn",
        "portrait-corridors",
    }
    assert TASKS["straight"].objective == ("Reach and stand on the green region.")
    assert TASKS["portrait-corridors"].objective == (
        "Stand in front of the creator of Linux."
    )


def test_runtime_keeps_scene_and_renderer_on_one_thread(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    model = _FakeModel()
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=np.array([[0.0, 0.0, 1.0]]))
    )

    class FakeScene:
        def __init__(self, cfg, device):
            calls.append(("scene", threading.get_ident()))
            self.entities = {"robot": robot}

        def compile(self):
            return model

        def initialize(self, mj_model, sim_model, data):
            calls.append(("initialize", threading.get_ident()))

        def reset(self):
            calls.append(("reset", threading.get_ident()))

        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

        def __getitem__(self, name):
            return self.entities[name]

    class FakeSimulation:
        def __init__(self, num_envs, cfg, model, device):
            self.mj_model = model
            self.model = object()
            self.data = object()
            self.expanded_fields = set()

        def forward(self):
            calls.append(("forward", threading.get_ident()))

    class FakeRenderer:
        def __init__(self, **kwargs):
            calls.append(("renderer", threading.get_ident()))

        def initialize(self):
            calls.append(("renderer.initialize", threading.get_ident()))

        def update(self, data):
            calls.append(("renderer.update", threading.get_ident()))

        def render(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            calls.append(("renderer.close", threading.get_ident()))

    def fake_env_factory(**kwargs) -> Any:
        return SimpleNamespace(
            scene=SimpleNamespace(num_envs=1),
            sim=SimpleNamespace(),
        )

    definition = TaskDefinition(
        env_factory=fake_env_factory,
        objective="Reach the goal.",
    )
    monkeypatch.setattr(
        "mjlab_textop.scout.runtime.get_task",
        lambda task: ("straight", definition),
    )
    monkeypatch.setattr("mjlab_textop.scout.runtime.Scene", FakeScene)
    monkeypatch.setattr("mjlab_textop.scout.runtime.Simulation", FakeSimulation)
    monkeypatch.setattr("mjlab_textop.scout.runtime.OffscreenRenderer", FakeRenderer)

    runtime = ScoutRuntime(ScoutConfig(device="cpu", image_width=4, image_height=4))
    try:
        runtime.load_task("straight")
        summary = runtime.get_scene_summary()
        captured = runtime.capture_view("overview")
        runtime.capture_view("overhead")
    finally:
        runtime.close()

    assert summary.robot_position == (0.0, 0.0, 1.0)
    assert summary.bodies[0].name == "body_1"
    assert captured.image.startswith(b"\xff\xd8")
    worker_threads = {thread_id for _, thread_id in calls}
    assert len(worker_threads) == 1
    assert next(iter(worker_threads)) != threading.get_ident()


def test_capture_tool_returns_text_and_native_mcp_image() -> None:
    class FakeRuntime:
        def list_tasks(self) -> tuple[TaskInfo, ...]:
            return ()

        def load_task(self, task: str) -> TaskInfo:
            raise NotImplementedError

        def get_scene_summary(self) -> SceneSummary:
            raise NotImplementedError

        def capture_view(self, view: ScoutView = "agent") -> CapturedView:
            return CapturedView(
                task="straight",
                view=view,
                width=2,
                height=1,
                image=b"jpeg-data",
            )

        def close_task(self) -> None:
            pass

    content = ScoutTools(FakeRuntime()).capture_view("agent")

    assert isinstance(content[0], TextContent)
    assert json.loads(content[0].text) == {
        "task": "straight",
        "view": "agent",
        "width": 2,
        "height": 1,
        "mime_type": "image/jpeg",
    }
    assert isinstance(content[1], ImageContent)
    assert content[1].mimeType == "image/jpeg"
