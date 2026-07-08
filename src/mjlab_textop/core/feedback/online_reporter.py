from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.viewer import OffscreenRenderer, ViewerConfig

from mjlab_textop.core.feedback.observation import (
    ObservationImage,
    OnlineObservationState,
    OnlineTextOpObservationCfg,
    encode_render_image_jpeg,
    make_online_textop_observation,
)


class OnlineObservationReporter:
    def __init__(
        self,
        cfg: OnlineTextOpObservationCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.publisher = cfg.publisher
        self._last_publish_frame: int | None = None
        self._image_renderer: OffscreenRenderer | None = None

    def maybe_publish(self, state: OnlineObservationState) -> None:
        publisher = self.publisher
        current_frame = state.frame
        if publisher is None or not state.started:
            return
        if (
            self._last_publish_frame is not None
            and current_frame - self._last_publish_frame < self.cfg.publish_interval
        ):
            return

        image = self._render_observation_image()
        payload = make_online_textop_observation(state)
        publisher.publish(payload, image=image)
        self._last_publish_frame = current_frame

    def _render_observation_image(self) -> ObservationImage:
        data = encode_render_image_jpeg(self._render_image())
        return ObservationImage(
            data=data,
            mime_type="image/jpeg",
        )

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
