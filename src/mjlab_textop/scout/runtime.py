from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import mujoco
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.viewer import OffscreenRenderer, ViewerConfig

from mjlab_textop.core.feedback.observation import encode_render_image_jpeg
from mjlab_textop.scout.config import ScoutConfig
from mjlab_textop.scout.schemas import (
    BodySummary,
    CapturedView,
    GeometrySummary,
    SceneSummary,
    ScoutView,
    TaskInfo,
)
from tasks.catalog import TASKS, TaskDefinition, TaskSet, get_task

AVAILABLE_VIEWS: tuple[ScoutView, ...] = ("agent", "overview", "overhead")
ResultT = TypeVar("ResultT")


@dataclass
class _LoadedTask:
    name: TaskSet
    definition: TaskDefinition
    scene: Scene
    sim: Simulation


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
            TaskInfo(name=name, objective=definition.objective)
            for name, definition in TASKS.items()
        )

    def load_task(self, task: str) -> TaskInfo:
        return self._submit(self._load_task, task)

    def get_scene_summary(self) -> SceneSummary:
        return self._submit(self._get_scene_summary)

    def capture_view(self, view: ScoutView = "agent") -> CapturedView:
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
        )
        return TaskInfo(name=name, objective=definition.objective)

    def _get_scene_summary(self) -> SceneSummary:
        loaded = self._require_loaded()
        robot = loaded.scene["robot"]
        robot_position = _vec3(robot.data.root_link_pos_w[0])
        bodies = _task_bodies(loaded.sim.mj_model)
        return SceneSummary(
            task=loaded.name,
            objective=loaded.definition.objective,
            robot_position=robot_position,
            bodies=bodies,
            available_views=AVAILABLE_VIEWS,
        )

    def _capture_view(self, view: ScoutView) -> CapturedView:
        if view not in AVAILABLE_VIEWS:
            choices = ", ".join(AVAILABLE_VIEWS)
            raise ValueError(f"Unknown view {view!r}. Available: {choices}")

        loaded = self._require_loaded()
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
            self._renderer.update(loaded.sim.data)
            image = encode_render_image_jpeg(self._renderer.render())
        return CapturedView(
            task=loaded.name,
            view=view,
            width=self.config.image_width,
            height=self.config.image_height,
            image=image,
        )

    def _camera_for(self, view: ScoutView, loaded: _LoadedTask) -> ViewerConfig:
        if view == "agent":
            return ViewerConfig(
                origin_type=ViewerConfig.OriginType.ASSET_BODY,
                entity_name="robot",
                body_name="torso_link",
                distance=3.0,
                azimuth=0.0,
                elevation=-12.0,
                width=self.config.image_width,
                height=self.config.image_height,
                max_extra_envs=0,
            )

        center, distance = _scene_frame(loaded.sim.mj_model)
        if view == "overview":
            return ViewerConfig(
                origin_type=ViewerConfig.OriginType.WORLD,
                lookat=(center[0], center[1], 0.75),
                distance=distance,
                azimuth=135.0,
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


def _task_bodies(model: Any) -> tuple[BodySummary, ...]:
    bodies = []
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name or "/" in name or name == "terrain":
            continue
        geom_start = int(model.body_geomadr[body_id])
        geom_stop = geom_start + int(model.body_geomnum[body_id])
        geometries = tuple(
            _geometry_summary(model, geom_id, index)
            for index, geom_id in enumerate(range(geom_start, geom_stop), 1)
        )
        if geometries:
            bodies.append(
                BodySummary(
                    # Asset names can encode task answers, as with portrait subjects.
                    name=f"body_{len(bodies) + 1}",
                    position=_vec3(model.body_pos[body_id]),
                    geometries=geometries,
                )
            )
    return tuple(bodies)


def _geometry_summary(model: Any, geom_id: int, index: int) -> GeometrySummary:
    kind = mujoco.mjtGeom(int(model.geom_type[geom_id])).name  # ty: ignore
    return GeometrySummary(
        name=f"geometry_{index}",
        kind=kind.removeprefix("mjGEOM_").lower(),
        size=tuple(float(value) for value in model.geom_size[geom_id]),
        rgba=_rgba(model.geom_rgba[geom_id]),
    )


def _scene_frame(model: Any) -> tuple[tuple[float, float], float]:
    positions = [body.position for body in _task_bodies(model)]
    if not positions:
        return (0.0, 0.0), 8.0
    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    center = ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5)
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return center, max(8.0, span * 1.35)


def _vec3(value: Any) -> tuple[float, float, float]:
    x, y, z = value
    return float(x), float(y), float(z)


def _rgba(value: Any) -> tuple[float, float, float, float]:
    red, green, blue, alpha = value
    return float(red), float(green), float(blue), float(alpha)
