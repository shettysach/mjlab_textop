from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.viewer import OffscreenRenderer, ViewerConfig

from mjlab_textop.core.feedback.observation import (
    ObservationImage,
    OnlineObservationState,
    OnlineTextOpObservationCfg,
    encode_render_image_jpeg,
    make_online_textop_observation,
)
from mjlab_textop.core.online.buffer import (
    TextOpRollingMotionBuffer,
)
from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
)
from mjlab_textop.core.online.source import (
    QueueTextOpOnlineSource,
    ResettableTextOpOnlineSource,
    TextOpOnlineSource,
)
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS, TEXTOP_G1_JOINT_COUNT

TextOpOnlineSourceMode = Literal["replay", "live"]
ONLINE_METRIC_NAMES = (
    "online_buffer_frames",
    "online_stale_steps",
    "online_consecutive_stale_steps",
    "online_current_frame",
    "online_latest_frame",
    "online_lag_frames",
    "online_started",
    "online_queue_depth",
    "online_blocks_received",
    "online_blocks_dropped",
    "online_bad_messages",
)


@dataclass(frozen=True)
class TextOpFutureWindow:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    anchor_pos_w: torch.Tensor
    anchor_quat_w: torch.Tensor
    stale_steps: int


@dataclass(kw_only=True)
class OnlineTextOpMotionCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    entity_name: str = "robot"
    anchor_body_name: str = "pelvis"
    future_steps: int = TEXTOP_FUTURE_STEPS
    source: TextOpOnlineSource | None = None
    live_source_cfg: SocketTextOpSourceCfg | None = None
    source_mode: TextOpOnlineSourceMode = "live"
    start_frame: int = 0
    startup_timeout_steps: int = 250
    max_poll_blocks: int = 16
    max_buffer_frames: int | None = 512
    clear_buffer_on_reset: bool = True
    reset_robot_to_reference: bool = True
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )
    observation: OnlineTextOpObservationCfg = field(
        default_factory=OnlineTextOpObservationCfg
    )

    def __post_init__(self) -> None:
        if self.future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {self.future_steps}")
        if self.source_mode not in ("replay", "live"):
            raise ValueError(f"Unknown source_mode: {self.source_mode}")
        if self.source_mode == "replay" and not isinstance(
            self.source, ResettableTextOpOnlineSource
        ):
            raise TypeError("Replay online source must implement reset()")

    def build(self, env: ManagerBasedRlEnv) -> OnlineTextOpMotionCommand:
        return OnlineTextOpMotionCommand(self, env)


class OnlineTextOpMotionCommand(CommandTerm):
    cfg: OnlineTextOpMotionCommandCfg

    def __init__(self, cfg: OnlineTextOpMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.source = self._make_source()
        if self.num_envs != 1:
            raise ValueError(
                f"Online TextOp supports one environment in v1, got {self.num_envs}"
            )
        if self.cfg.start_frame < 0:
            raise ValueError(
                f"start_frame must be non-negative, got {self.cfg.start_frame}"
            )
        self._validate_source_fps(env)

        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        max_buffer_frames = (
            None if self.cfg.source_mode == "replay" else self.cfg.max_buffer_frames
        )
        self.buffer = TextOpRollingMotionBuffer(
            device=self.device,
            max_frames=max_buffer_frames,
        )
        self.current_frame = int(self.cfg.start_frame)
        self._started = False
        self._has_started_once = False
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame: int | None = None
        self.observation_reporter = OnlineObservationReporter(cfg.observation, env)
        self._anchor_pos_offset_w = torch.zeros(3, device=self.device)
        self._future_cache_frame: int | None = None
        self._future_cache: TextOpFutureWindow | None = None
        self._init_metrics()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.future_joint_pos[:, 0]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.future_joint_vel[:, 0]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.future_anchor_pos_w[:, 0]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.future_anchor_quat_w[:, 0]

    @property
    def future_joint_pos(self) -> torch.Tensor:
        return self._future_window().joint_pos.unsqueeze(0)

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self._future_window().joint_vel.unsqueeze(0)

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return self._future_window().anchor_pos_w.unsqueeze(0)

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self._future_window().anchor_quat_w.unsqueeze(0)

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    def _init_metrics(self) -> None:
        for name in ONLINE_METRIC_NAMES:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _set_metric(self, name: str, value: float | int | bool) -> None:
        self.metrics[name][:] = float(value)

    def _update_metrics(self) -> None:
        latest_index = self.buffer.latest_index
        lag_frames = 0 if latest_index is None else latest_index - self.current_frame
        self._set_metric("online_buffer_frames", self.buffer.frame_count)
        self._set_metric("online_stale_steps", self._last_stale_steps)
        self._set_metric(
            "online_consecutive_stale_steps",
            self._consecutive_stale_steps,
        )
        self._set_metric("online_current_frame", self.current_frame)
        self._set_metric(
            "online_latest_frame",
            -1 if latest_index is None else latest_index,
        )
        self._set_metric("online_lag_frames", lag_frames)
        self._set_metric("online_started", self._started)
        diagnostics = getattr(self.source, "diagnostics", None)
        if diagnostics is not None:
            self._set_metric(
                "online_queue_depth",
                getattr(diagnostics, "queue_depth", 0),
            )
            self._set_metric(
                "online_blocks_received",
                getattr(diagnostics, "blocks_received", 0),
            )
            self._set_metric(
                "online_blocks_dropped",
                getattr(diagnostics, "blocks_dropped", 0),
            )
            self._set_metric(
                "online_bad_messages",
                getattr(diagnostics, "bad_messages", 0),
            )
        self.observation_reporter.maybe_publish(
            OnlineObservationState(
                frame=self.current_frame,
                started=self._started,
                latest_index=latest_index,
                lag_frames=lag_frames,
                buffer_frames=self.buffer.frame_count,
                stale_steps=self._last_stale_steps,
                consecutive_stale_steps=self._consecutive_stale_steps,
                robot_anchor_pos_w=self.robot_anchor_pos_w[0],
                robot_anchor_quat_w=self.robot_anchor_quat_w[0],
            )
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._reset_runtime_counters()
        self._started = False

        if self.cfg.source_mode == "replay":
            if self.cfg.clear_buffer_on_reset:
                self.buffer.clear()
            source = cast(ResettableTextOpOnlineSource, self.source)
            source.reset()
            self._poll_source()

            self.current_frame = int(self.cfg.start_frame)
            if self.buffer.can_start(self.current_frame, self.cfg.future_steps):
                if self.cfg.reset_robot_to_reference:
                    self._reset_robot_to_reference(env_ids)
                self._started = True
            return

        if self.cfg.source_mode == "live":
            self._poll_source()
            live_start_frame = self._live_start_or_resync_frame()
            if live_start_frame is None:
                return
            self.current_frame = live_start_frame
            self._align_reference_anchor()
            if self.cfg.reset_robot_to_reference:
                self._reset_robot_to_reference(env_ids)
            self._started = True
            self._has_started_once = True
            return

    def _reset_runtime_counters(self) -> None:
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame = None
        self._anchor_pos_offset_w.zero_()
        self._clear_future_cache()

    def _update_command(self) -> None:
        self._poll_source()

        if not self._started:
            start_frame = self._startup_start_frame()
            if start_frame is not None:
                self.current_frame = start_frame
                self._align_reference_anchor()
                if self.cfg.reset_robot_to_reference:
                    env_ids = torch.arange(self.num_envs, device=self.device)
                    self._reset_robot_to_reference(env_ids)
                self._started = True
                if self.cfg.source_mode == "live":
                    self._has_started_once = True
                return

            self._startup_wait_steps += 1
            if self._startup_wait_steps > self.cfg.startup_timeout_steps:
                raise RuntimeError(
                    "Online TextOp buffer did not receive enough contiguous "
                    f"frames for future_steps={self.cfg.future_steps}"
                )
            return

        # V1 assumes one MJLab command update corresponds to one TextOp source
        # frame. RobotMDAR/TextOpDeploy commonly runs at 50 Hz; add explicit
        # source-FPS resampling before using streams at a different control rate.
        if self.cfg.source_mode == "live" and not self._can_advance_live_frame():
            return
        self.current_frame += 1
        self._clear_future_cache()

    def _poll_source(self) -> None:
        for _ in range(self.cfg.max_poll_blocks):
            block = self.source.poll()
            if block is None:
                return
            self.buffer.append_block(block)
            self._clear_future_cache()

    def _startup_start_frame(self) -> int | None:
        if self.cfg.source_mode == "live":
            return self._initial_live_start_frame()
        if self.buffer.can_start(self.current_frame, self.cfg.future_steps):
            return self.current_frame
        return None

    def _initial_live_start_frame(self) -> int | None:
        return self.buffer.earliest_start_frame(self.cfg.future_steps)

    def _resync_live_start_frame(self) -> int | None:
        return self.buffer.latest_start_frame(self.cfg.future_steps)

    def _live_start_or_resync_frame(self) -> int | None:
        if not self._has_started_once:
            return self._initial_live_start_frame()
        return self._resync_live_start_frame()

    def _can_advance_live_frame(self) -> bool:
        latest_index = self.buffer.latest_index
        latest_start_frame = self._resync_live_start_frame()
        if latest_index is None or latest_start_frame is None:
            return False

        if not self.buffer.can_start(self.current_frame, self.cfg.future_steps):
            if latest_start_frame > self.current_frame:
                self.current_frame = latest_start_frame
                self._align_reference_anchor()
            return False

        next_frame = self.current_frame + 1
        if self.buffer.can_start(next_frame, self.cfg.future_steps):
            return True
        if latest_start_frame > self.current_frame:
            self.current_frame = latest_start_frame
            self._align_reference_anchor()
        return False

    def _future_window(self) -> TextOpFutureWindow:
        if not self._started:
            return self._startup_future_window()
        if (
            self._future_cache is not None
            and self._future_cache_frame == self.current_frame
        ):
            return self._future_cache

        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = (
            self.buffer.get_future(self.current_frame, self.cfg.future_steps)
        )
        window = TextOpFutureWindow(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w + self._anchor_pos_offset_w[None, :],
            anchor_quat_w=anchor_quat_w,
            stale_steps=stale_steps,
        )
        self._last_stale_steps = stale_steps
        if self._last_stale_frame != self.current_frame:
            if stale_steps > 0:
                self._consecutive_stale_steps += 1
            else:
                self._consecutive_stale_steps = 0
            self._last_stale_frame = self.current_frame
        # Clamp stale future frames for now. Keep tracking consecutive stale
        # windows so live deployments can surface underruns without aborting.
        self._future_cache_frame = self.current_frame
        self._future_cache = window
        return window

    def _startup_future_window(self) -> TextOpFutureWindow:
        dtype = self.robot_anchor_pos_w.dtype
        joint_shape = (self.cfg.future_steps, TEXTOP_G1_JOINT_COUNT)
        joint_pos = torch.zeros(joint_shape, device=self.device, dtype=dtype)
        joint_vel = torch.zeros(joint_shape, device=self.device, dtype=dtype)
        anchor_pos_w = self.robot_anchor_pos_w[0].expand(self.cfg.future_steps, -1)
        anchor_quat_w = self.robot_anchor_quat_w[0].expand(self.cfg.future_steps, -1)
        return TextOpFutureWindow(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
            stale_steps=0,
        )

    def _clear_future_cache(self) -> None:
        self._future_cache_frame = None
        self._future_cache = None

    def _align_reference_anchor(self) -> None:
        if self.cfg.anchor_alignment == "direct_world":
            self._anchor_pos_offset_w.zero_()
            self._clear_future_cache()
            return

        _, _, anchor_pos_w, _, _ = self.buffer.get_future(self.current_frame, 1)
        self._anchor_pos_offset_w = self.robot_anchor_pos_w[0] - anchor_pos_w[0]
        self._clear_future_cache()

    def _reset_robot_to_reference(self, env_ids: torch.Tensor) -> None:
        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, _ = self.buffer.get_future(
            self.current_frame,
            1,
        )
        joint_pos = joint_pos[0].repeat(len(env_ids), 1)
        joint_vel = joint_vel[0].repeat(len(env_ids), 1)
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root_pos = (anchor_pos_w[0] + self._anchor_pos_offset_w).repeat(len(env_ids), 1)
        root_quat = anchor_quat_w[0].repeat(len(env_ids), 1)
        root_vel = torch.zeros(
            len(env_ids), 6, device=self.device, dtype=root_pos.dtype
        )
        root_state = torch.cat([root_pos, root_quat, root_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _validate_source_fps(self, env: ManagerBasedRlEnv) -> None:
        fps = getattr(self.source, "fps", None)
        if fps is None:
            return
        expected_fps = 1.0 / float(env.step_dt)
        if abs(float(fps) - expected_fps) > 1.0e-4:
            raise ValueError(
                "Replay TextOp source FPS must match env control rate: "
                f"{float(fps):g} != {expected_fps:g}"
            )

    def _make_source(self) -> TextOpOnlineSource:
        if self.cfg.source is not None:
            return self.cfg.source
        if self.cfg.source_mode == "live" and self.cfg.live_source_cfg is not None:
            source = SocketTextOpOnlineSource(self.cfg.live_source_cfg)
            source.start()
            return source
        return QueueTextOpOnlineSource()


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
            env.update_visualizers
            if hasattr(env, "update_visualizers")
            else None
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


def use_online_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode = "live",
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    observation: OnlineTextOpObservationCfg | None = None,
) -> None:
    motion_cfg = env_cfg.commands[command_name]
    entity_name = getattr(motion_cfg, "entity_name", "robot")
    anchor_body_name = getattr(motion_cfg, "anchor_body_name", "pelvis")
    if source is None and source_mode == "replay":
        source = QueueTextOpOnlineSource()

    env_cfg.commands[command_name] = OnlineTextOpMotionCommandCfg(
        entity_name=entity_name,
        anchor_body_name=anchor_body_name,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation or OnlineTextOpObservationCfg(),
    )
