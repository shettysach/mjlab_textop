from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_vla.textop.contract import TEXTOP_FUTURE_STEPS
from mjlab_vla.textop.online import (
    QueueTextOpOnlineSource,
    TextOpOnlineSource,
    TextOpRollingMotionBuffer,
)


@dataclass(kw_only=True)
class OnlineTextOpMotionCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    entity_name: str = "robot"
    anchor_body_name: str = "pelvis"
    future_steps: int = TEXTOP_FUTURE_STEPS
    source: TextOpOnlineSource = field(default_factory=QueueTextOpOnlineSource)
    start_frame: int = 0
    startup_timeout_steps: int = 250
    max_stale_steps: int = 25
    max_poll_blocks: int = 16
    max_buffer_frames: int | None = 512

    def build(self, env: ManagerBasedRlEnv) -> OnlineTextOpMotionCommand:
        return OnlineTextOpMotionCommand(self, env)


class OnlineTextOpMotionCommand(CommandTerm):
    cfg: OnlineTextOpMotionCommandCfg

    def __init__(self, cfg: OnlineTextOpMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if self.num_envs != 1:
            raise ValueError(
                f"Online TextOp supports one environment in v1, got {self.num_envs}"
            )
        if self.cfg.future_steps <= 0:
            raise ValueError(
                f"future_steps must be positive, got {self.cfg.future_steps}"
            )
        if self.cfg.start_frame < 0:
            raise ValueError(
                f"start_frame must be non-negative, got {self.cfg.start_frame}"
            )

        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.buffer = TextOpRollingMotionBuffer(
            device=self.device,
            max_frames=self.cfg.max_buffer_frames,
        )
        self.current_frame = int(self.cfg.start_frame)
        self._started = False
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame: int | None = None

        self.metrics["online_buffer_frames"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_stale_steps"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_consecutive_stale_steps"] = torch.zeros(
            self.num_envs, device=self.device
        )

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
        return self._future()[0].unsqueeze(0)

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self._future()[1].unsqueeze(0)

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return self._future()[2].unsqueeze(0)

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self._future()[3].unsqueeze(0)

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    def _update_metrics(self) -> None:
        self.metrics["online_buffer_frames"][:] = float(self.buffer.frame_count)
        self.metrics["online_stale_steps"][:] = float(self._last_stale_steps)
        self.metrics["online_consecutive_stale_steps"][:] = float(
            self._consecutive_stale_steps
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.current_frame = int(self.cfg.start_frame)
        self._started = False
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame = None

    def _update_command(self) -> None:
        self._poll_source()

        if not self._started:
            if self.buffer.can_start(self.current_frame, self.cfg.future_steps):
                self._started = True
                return

            self._startup_wait_steps += 1
            if self._startup_wait_steps > self.cfg.startup_timeout_steps:
                raise RuntimeError(
                    "Online TextOp buffer did not receive enough contiguous "
                    f"frames for future_steps={self.cfg.future_steps}"
                )
            return

        self.current_frame += 1

    def _poll_source(self) -> None:
        for _ in range(self.cfg.max_poll_blocks):
            block = self.cfg.source.poll()
            if block is None:
                return
            self.buffer.append_block(block)

    def _future(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._started:
            raise RuntimeError("Online TextOp command is waiting for startup frames")

        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = (
            self.buffer.get_future(self.current_frame, self.cfg.future_steps)
        )
        self._last_stale_steps = stale_steps
        if self._last_stale_frame != self.current_frame:
            if stale_steps > 0:
                self._consecutive_stale_steps += 1
            else:
                self._consecutive_stale_steps = 0
            self._last_stale_frame = self.current_frame
        if self._consecutive_stale_steps > self.cfg.max_stale_steps:
            raise RuntimeError(
                "Online TextOp future window exceeded max consecutive stale "
                f"steps: {self._consecutive_stale_steps} > {self.cfg.max_stale_steps}"
            )
        return joint_pos, joint_vel, anchor_pos_w, anchor_quat_w


def use_online_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
) -> None:
    motion_cfg = env_cfg.commands[command_name]
    entity_name = getattr(motion_cfg, "entity_name", "robot")
    anchor_body_name = getattr(motion_cfg, "anchor_body_name", "pelvis")
    kwargs = {}
    if source is not None:
        kwargs["source"] = source

    env_cfg.commands[command_name] = OnlineTextOpMotionCommandCfg(
        entity_name=entity_name,
        anchor_body_name=anchor_body_name,
        future_steps=future_steps,
        # TODO: **kwargs,
    )
