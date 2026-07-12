from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer

from mjlab_textop.core.feedback.observation import (
    OnlineObservationCfg,
    OnlineObservationState,
)
from mjlab_textop.core.feedback.online_reporter import OnlineObservationReporter
from mjlab_textop.core.mdp.online_reference_debug import OnlineReferenceGhost
from mjlab_textop.core.mdp.online_types import (
    ONLINE_METRIC_NAMES,
    FutureWindow,
    OnlineSourceMode,
)
from mjlab_textop.core.online.buffer import (
    RollingMotionBuffer,
)
from mjlab_textop.core.online.live import (
    SocketOnlineSource,
    SocketSourceCfg,
)
from mjlab_textop.core.online.source import (
    OnlineSource,
    QueueOnlineSource,
    ResettableOnlineSource,
)
from mjlab_textop.core.schema import FUTURE_STEPS, G1_JOINT_COUNT

LIVE_BUFFER_LOW_WATERMARK_FRAMES = 50
LIVE_BUFFER_HIGH_WATERMARK_FRAMES = 150


@dataclass(kw_only=True)
class OnlineMotionCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    entity_name: str = "robot"
    anchor_body_name: str = "pelvis"
    future_steps: int = FUTURE_STEPS
    source: OnlineSource | None = None
    live_source_cfg: SocketSourceCfg | None = None
    source_mode: OnlineSourceMode = "live"
    start_frame: int = 0
    startup_timeout_steps: int = 250
    max_poll_blocks: int = 16
    clear_buffer_on_reset: bool = True
    reset_robot_to_reference: bool = True
    observation: OnlineObservationCfg | None = None

    def __post_init__(self) -> None:
        if self.future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {self.future_steps}")
        if self.source_mode not in ("replay", "live"):
            raise ValueError(f"Unknown source_mode: {self.source_mode}")
        if self.source_mode == "replay" and not isinstance(
            self.source, ResettableOnlineSource
        ):
            raise TypeError("Replay online source must implement reset()")

    def build(self, env: ManagerBasedRlEnv) -> OnlineMotionCommand:
        return OnlineMotionCommand(self, env)


class OnlineMotionCommand(CommandTerm):
    cfg: OnlineMotionCommandCfg

    def __init__(self, cfg: OnlineMotionCommandCfg, env: ManagerBasedRlEnv):
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
        self.buffer = RollingMotionBuffer(device=self.device)
        self.current_frame = int(self.cfg.start_frame)
        self._started = False
        self._has_started_once = False
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame: int | None = None
        self._live_polling_paused = False
        self.observation_reporter = (
            None
            if cfg.observation is None
            else OnlineObservationReporter(cfg.observation, env)
        )
        self._reference_start_anchor_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._robot_start_anchor_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._future_cache_frame: int | None = None
        self._future_cache: FutureWindow | None = None
        self._reference_ghost = OnlineReferenceGhost(env, self.robot)
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
        if self.observation_reporter is not None:
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
            source = cast(ResettableOnlineSource, self.source)
            source.reset()
            self._poll_source()

            self.current_frame = int(self.cfg.start_frame)
            if self.buffer.can_start(self.current_frame, self.cfg.future_steps):
                self._align_reference_anchor()
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
        self._live_polling_paused = False
        self._reference_start_anchor_pos_w.zero_()
        self._robot_start_anchor_pos_w.zero_()
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
        if self.cfg.source_mode == "live":
            self.buffer.discard_before(self.current_frame)
        self._clear_future_cache()

    def _poll_source(self) -> None:
        for _ in range(self.cfg.max_poll_blocks):
            if self.cfg.source_mode == "live" and not self._should_poll_live_source():
                return
            block = self.source.poll()
            if block is None:
                return
            if self.cfg.source_mode == "live":
                latest_index = self.buffer.latest_index
                if latest_index is not None and block.index != latest_index + 1:
                    raise RuntimeError(
                        "Non-contiguous RobotMDAR live stream: "
                        f"expected block index {latest_index + 1}, got "
                        f"{block.index}"
                    )
            self.buffer.append_block(block)
            self._clear_future_cache()

    def _should_poll_live_source(self) -> bool:
        latest_index = self.buffer.latest_index
        if latest_index is None:
            self._live_polling_paused = False
            return True

        future_lead = latest_index - self.current_frame
        if self._live_polling_paused:
            if future_lead >= LIVE_BUFFER_LOW_WATERMARK_FRAMES:
                return False
            self._live_polling_paused = False
        elif future_lead >= LIVE_BUFFER_HIGH_WATERMARK_FRAMES:
            self._live_polling_paused = True
            return False
        return True

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
        if not self.buffer.can_start(self.current_frame, self.cfg.future_steps):
            raise RuntimeError(
                "Lost active live reference window: "
                f"current={self.current_frame}, "
                f"earliest={self.buffer.earliest_index}, "
                f"latest={self.buffer.latest_index}, "
                f"future_steps={self.cfg.future_steps}"
            )

        next_frame = self.current_frame + 1
        return self.buffer.can_start(next_frame, self.cfg.future_steps)

    def _future_window(self) -> FutureWindow:
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

        anchor_pos_w = self._fixed_start_reference_pos(anchor_pos_w)
        # anchor_pos_w = self._zero_future_anchor_pos_w(anchor_pos_w)

        window = FutureWindow(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w,
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

    def _startup_future_window(self) -> FutureWindow:
        dtype = self.robot_anchor_pos_w.dtype
        joint_shape = (self.cfg.future_steps, G1_JOINT_COUNT)
        joint_pos = torch.zeros(joint_shape, device=self.device, dtype=dtype)
        joint_vel = torch.zeros(joint_shape, device=self.device, dtype=dtype)
        anchor_pos_w = self.robot_anchor_pos_w[0].expand(self.cfg.future_steps, -1)
        anchor_quat_w = self.robot_anchor_quat_w[0].expand(self.cfg.future_steps, -1)
        return FutureWindow(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
            stale_steps=0,
        )

    def _clear_future_cache(self) -> None:
        self._future_cache_frame = None
        self._future_cache = None

    # Place the raw reference origin at the robot's startup anchor.
    def _fixed_start_reference_pos(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        return (
            self._robot_start_anchor_pos_w[0]
            + anchor_pos_w
            - self._reference_start_anchor_pos_w[0]
        )

    # FIX: Meant to fix the drift, see /notes/DRIFT.md
    # Match TextOpDeploy's unconditional zero anchor-position observation.
    def _zero_future_anchor_pos_w(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        return self.robot_anchor_pos_w[0].expand_as(anchor_pos_w)

    # FIX: Meant to fix the drift, see /notes/DRIFT.md
    # XY is re-anchored to the current robot anchor.
    # Z preserves reference height motion, offset to the robot's start height.
    def _aligned_reference_pos(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        aligned_pos_w = anchor_pos_w.clone()

        reference_start_xy_w = anchor_pos_w[0, :2]
        reference_delta_xy_w = anchor_pos_w[:, :2] - reference_start_xy_w

        aligned_pos_w[:, :2] = self.robot_anchor_pos_w[0, :2] + reference_delta_xy_w
        aligned_pos_w[:, 2] = (
            self._robot_start_anchor_pos_w[0, 2]
            + anchor_pos_w[:, 2]
            - self._reference_start_anchor_pos_w[0, 2]
        )

        return aligned_pos_w

    def _align_reference_anchor(self) -> None:
        _, _, anchor_pos_w, _, _ = self.buffer.get_future(
            self.current_frame,
            1,
        )
        self._reference_start_anchor_pos_w = anchor_pos_w.expand(self.num_envs, -1)
        self._robot_start_anchor_pos_w = self.robot_anchor_pos_w.clone()
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

        root_pos = self._fixed_start_reference_pos(anchor_pos_w)
        root_pos = root_pos[0].repeat(len(env_ids), 1)
        root_quat = anchor_quat_w[0].repeat(len(env_ids), 1)
        root_vel = torch.zeros(
            len(env_ids), 6, device=self.device, dtype=root_pos.dtype
        )
        root_state = torch.cat([root_pos, root_quat, root_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        if not self._started:
            return

        self._reference_ghost.draw(
            visualizer,
            num_envs=self.num_envs,
            joint_pos=self.joint_pos,
            anchor_pos_w=self.future_anchor_pos_w[:, 0],
            anchor_quat_w=self.anchor_quat_w,
        )

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

    def _make_source(self) -> OnlineSource:
        if self.cfg.source is not None:
            return self.cfg.source
        if self.cfg.source_mode == "live" and self.cfg.live_source_cfg is not None:
            source = SocketOnlineSource(self.cfg.live_source_cfg)
            source.start()
            return source
        return QueueOnlineSource()


def use_online_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
    future_steps: int = FUTURE_STEPS,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
) -> None:
    motion_cfg = env_cfg.commands[command_name]
    entity_name = getattr(motion_cfg, "entity_name", "robot")
    anchor_body_name = getattr(motion_cfg, "anchor_body_name", "pelvis")
    if debug_vis is None:
        debug_vis = bool(getattr(motion_cfg, "debug_vis", False))
    if source is None and source_mode == "replay":
        source = QueueOnlineSource()

    env_cfg.commands[command_name] = OnlineMotionCommandCfg(
        entity_name=entity_name,
        anchor_body_name=anchor_body_name,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation,
        debug_vis=debug_vis,
    )
