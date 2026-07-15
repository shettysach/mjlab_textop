from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.viewer import OffscreenRenderer, ViewerConfig

from mjlab_textop.core.feedback.observation import (
    ObservationImage,
    ObservationPublisher,
    OnlineObservationCfg,
    OnlineObservationState,
    encode_render_image_jpeg,
)


class OnlineObservationReporter:
    def __init__(
        self,
        cfg: OnlineObservationCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.publisher = cfg.publisher
        self._last_publish_frame: int | None = None
        self._image_renderer: OffscreenRenderer | None = None
        self._publish_executor = ThreadPoolExecutor(max_workers=1)
        self._publish_future: Future[None] | None = None
        self._event_futures: deque[Future[None]] = deque()
        self.last_publish_error: str | None = None
        self._closed = False

    def maybe_publish(self, state: OnlineObservationState) -> None:
        if self._closed:
            return
        publisher = self.publisher
        current_frame = state.frame
        if publisher is None or not state.started:
            return
        self._collect_publish_result()
        if (
            self._last_publish_frame is not None
            and current_frame - self._last_publish_frame < self.cfg.publish_interval
        ):
            return
        if self._publish_future is not None:
            self._last_publish_frame = current_frame
            return

        rendered_image = _copy_rendered_image(self._render_image())
        self._publish_future = self._publish_executor.submit(
            self._encode_and_publish,
            publisher,
            rendered_image,
        )
        self._last_publish_frame = current_frame

    def _encode_and_publish(
        self,
        publisher: ObservationPublisher,
        rendered_image: Any,
    ) -> None:
        data = encode_render_image_jpeg(rendered_image)
        publisher.publish(
            image=ObservationImage(
                data=data,
                mime_type="image/jpeg",
            )
        )

    def publish_collision_stop(self, active: bool, *, recovery_epoch: int) -> None:
        if self.publisher is None or self._closed:
            return
        self._collect_event_result()
        self._event_futures.append(
            self._publish_executor.submit(
                self.publisher.publish,
                image=None,
                collision_stop=active,
                recovery_epoch=recovery_epoch,
            )
        )

    @property
    def _event_future(self) -> Future[None] | None:
        """Most recently queued event, retained for diagnostics."""
        return self._event_futures[-1] if self._event_futures else None

    def _collect_publish_result(self) -> None:
        if self._publish_future is None or not self._publish_future.done():
            return
        try:
            self._publish_future.result()
            self.last_publish_error = None
        except Exception as exc:
            self.last_publish_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._publish_future = None

    def _collect_event_result(self) -> None:
        while self._event_futures and self._event_futures[0].done():
            future = self._event_futures.popleft()
            try:
                future.result()
                self.last_publish_error = None
            except Exception as exc:
                self.last_publish_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._publish_executor.shutdown(wait=True, cancel_futures=True)
            self._collect_publish_result()
            self._collect_event_result()
        finally:
            if self._image_renderer is not None:
                self._image_renderer.close()
                self._image_renderer = None

    def _render_image(self):
        renderer = self._image_renderer
        env = self.env
        if renderer is None:
            renderer = OffscreenRenderer(
                model=env.sim.mj_model,
                cfg=self.cfg.camera,
                scene=env.scene,
                sim_model=env.sim.model,
                expanded_fields=env.sim.expanded_fields,
            )
            renderer.initialize()
            self._image_renderer = renderer

        debug_callback = (
            env.update_visualizers if hasattr(env, "update_visualizers") else None
        )
        self._sync_camera_orientation(renderer)
        renderer.update(env.sim.data, debug_vis_callback=debug_callback)
        return renderer.render()

    def _sync_camera_orientation(self, renderer: OffscreenRenderer) -> None:
        yaw_degrees = _body_yaw_degrees(self.env, self.cfg.camera)
        if yaw_degrees is None:
            return
        renderer._cam.azimuth = self.cfg.camera.azimuth + yaw_degrees


def _body_yaw_degrees(
    env: ManagerBasedRlEnv,
    camera_cfg: ViewerConfig,
) -> float | None:
    if camera_cfg.origin_type != camera_cfg.OriginType.ASSET_BODY:
        return None
    if camera_cfg.entity_name is None or camera_cfg.body_name is None:
        raise ValueError("ASSET_BODY observation camera requires entity_name/body_name")

    robot = env.scene[camera_cfg.entity_name]
    body_index = robot.body_names.index(camera_cfg.body_name)
    quat = robot.data.body_link_quat_w[int(camera_cfg.env_idx), body_index]
    _, _, yaw = euler_xyz_from_quat(quat.reshape(1, 4))
    return float(torch.rad2deg(yaw).item())


def _copy_rendered_image(image: Any) -> Any:
    copy = getattr(image, "copy", None)
    if callable(copy):
        return copy()
    return image
