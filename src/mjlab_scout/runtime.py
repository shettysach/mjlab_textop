from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.viewer import OffscreenRenderer, ViewerConfig

from mjlab_scout.config import ScoutConfig
from mjlab_scout.schemas import (
    CapturedView,
    ScoutView,
    TaskInfo,
)
from mjlab_textop.core.feedback.observation import (
    encode_render_image_jpeg,
    make_torso_observation_camera,
)
from tasks.catalog import TASKS, TaskDefinition, TaskSet, get_task

DEFAULT_VIEWS: tuple[ScoutView, ...] = ("agent", "overview", "overhead")
ResultT = TypeVar("ResultT")


@dataclass
class _LoadedTask:
    name: TaskSet
    definition: TaskDefinition
    scene: Scene
    sim: Simulation
    views: tuple[ScoutView, ...]


class ScoutRuntime:
    """Own one MJLab scene and its renderer on a dedicated thread."""

    def __init__(self, config: ScoutConfig | None = None) -> None:
        self.config = config or ScoutConfig()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scout")
        self._loaded: _LoadedTask | None = None
        self._renderer: OffscreenRenderer | None = None
        self._renderer_view: ScoutView | None = None
        self._closed = False

    def list_tasks(self) -> tuple[TaskInfo, ...]:
        return tuple(
            TaskInfo(
                name=name,
                objective=definition.objective,
                views=DEFAULT_VIEWS,
            )
            for name, definition in TASKS.items()
        )

    def load_task(self, task: str) -> TaskInfo:
        return self._submit(self._load_task, task)

    def capture_view(self, view: ScoutView = "overview") -> CapturedView:
        return self._submit(self._capture_view, view)

    def close_task(self) -> None:
        self._submit(self._close_task)

    def close(self) -> None:
        if self._closed:
            return
        self._submit(self._close_task)
        self._closed = True
        self._worker.shutdown(wait=True)

    def _submit(self, fn: Callable[..., ResultT], *args: Any) -> ResultT:
        if self._closed:
            raise RuntimeError("Scout runtime is closed")
        return self._worker.submit(fn, *args).result()

    def _load_task(self, task: str) -> TaskInfo:
        name, definition = get_task(task)
        self._close_task()

        with redirect_stdout(sys.stderr):
            # MCP stdio owns stdout; MJLab and Warp diagnostics belong on stderr.
            env_cfg = definition.env_factory(play=True)
            env_cfg.scene.num_envs = 1
            scene = Scene(env_cfg.scene, device=self.config.device)
            model = scene.compile()
            sim = Simulation(
                num_envs=1,
                cfg=env_cfg.sim,
                model=model,
                device=self.config.device,
            )
            scene.initialize(sim.mj_model, sim.model, sim.data)
            scene.reset()
            scene.write_data_to_sim()
            sim.forward()
            scene.update(sim.mj_model.opt.timestep)

        self._loaded = _LoadedTask(
            name=name,
            definition=definition,
            scene=scene,
            sim=sim,
            views=(*DEFAULT_VIEWS, *_corridor_views(sim.mj_model)),
        )
        return TaskInfo(
            name=name,
            objective=definition.objective,
            views=self._loaded.views,
        )

    def _capture_view(self, view: ScoutView) -> CapturedView:
        loaded = self._require_loaded()
        if view not in loaded.views:
            choices = ", ".join(loaded.views)
            raise ValueError(f"Unknown view {view!r}. Available: {choices}")

        if self._renderer_view != view:
            self._close_renderer()
            camera = self._camera_for(view, loaded)
            self._renderer = OffscreenRenderer(
                model=loaded.sim.mj_model,
                cfg=camera,
                scene=loaded.scene,
                sim_model=loaded.sim.model,
                expanded_fields=loaded.sim.expanded_fields,
            )
            with redirect_stdout(sys.stderr):
                self._renderer.initialize()
            self._renderer_view = view

        assert self._renderer is not None
        with redirect_stdout(sys.stderr):
            camera_name = None if view in DEFAULT_VIEWS else view
            self._renderer.update(loaded.sim.data, camera=camera_name)
            image = encode_render_image_jpeg(self._renderer.render())
        return CapturedView(
            task=loaded.name,
            view=view,
            width=self.config.image_width,
            height=self.config.image_height,
            image=image,
        )

    def _camera_for(self, view: ScoutView, loaded: _LoadedTask) -> ViewerConfig:
        if view not in DEFAULT_VIEWS:
            return ViewerConfig(
                width=self.config.image_width,
                height=self.config.image_height,
                max_extra_envs=0,
            )

        center, distance = _scene_frame(loaded.sim.mj_model)
        if view == "agent":
            return make_torso_observation_camera(
                width=self.config.image_width,
                height=self.config.image_height,
            )
        if view == "overview":
            return ViewerConfig(
                origin_type=ViewerConfig.OriginType.WORLD,
                lookat=(center[0], center[1], 0.75),
                distance=distance,
                azimuth=0.0,
                elevation=-38.0,
                width=self.config.image_width,
                height=self.config.image_height,
                max_extra_envs=0,
            )
        return ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(center[0], center[1], 0.0),
            distance=distance,
            azimuth=90.0,
            elevation=-89.0,
            width=self.config.image_width,
            height=self.config.image_height,
            max_extra_envs=0,
        )

    def _close_task(self) -> None:
        self._close_renderer()
        self._loaded = None

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            with redirect_stdout(sys.stderr):
                self._renderer.close()
        self._renderer = None
        self._renderer_view = None

    def _require_loaded(self) -> _LoadedTask:
        if self._loaded is None:
            raise RuntimeError("Load a task before inspecting the scene")
        return self._loaded


def _scene_frame(model: Any) -> tuple[tuple[float, float], float]:
    positions = []
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name or "/" in name or name == "terrain":
            continue
        if int(model.body_geomnum[body_id]) > 0:
            x, y, _ = model.body_pos[body_id]
            positions.append((float(x), float(y)))
    if not positions:
        return (0.0, 0.0), 8.0
    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    center = ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5)
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return center, max(8.0, span * 1.35)


def _corridor_views(model: Any) -> tuple[ScoutView, ...]:
    return tuple(
        name
        for camera_id in range(model.ncam)
        if (name := model.camera(camera_id).name) and name.startswith("corridor_")
    )
