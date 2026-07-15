from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from mjlab_textop.core.schema import FUTURE_STEPS


def make_future_time_steps(
    time_steps: torch.Tensor,
    *,
    time_step_total: int,
) -> torch.Tensor:
    offsets = torch.arange(
        FUTURE_STEPS,
        dtype=torch.long,
        device=time_steps.device,
    )
    future = time_steps[:, None] + offsets[None, :]
    return torch.clamp(future, max=time_step_total - 1)


@dataclass(kw_only=True)
class OfflineMotionCommandCfg(MotionCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> OfflineMotionCommand:
        return OfflineMotionCommand(self, env)


class OfflineMotionCommand(MotionCommand):
    cfg: OfflineMotionCommandCfg

    @property
    def future_time_steps(self) -> torch.Tensor:
        return make_future_time_steps(
            self.time_steps,
            time_step_total=self.motion.time_step_total,
        )

    @property
    def future_joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.future_time_steps]

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.future_time_steps]

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[
                self.future_time_steps,
                self.motion_anchor_body_index,
            ]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[
            self.future_time_steps,
            self.motion_anchor_body_index,
        ]


def textop_motion_command_cfg_from(
    cfg: MotionCommandCfg,
) -> OfflineMotionCommandCfg:
    kwargs = {
        field.name: copy.deepcopy(getattr(cfg, field.name))
        for field in fields(MotionCommandCfg)
    }
    return OfflineMotionCommandCfg(**kwargs)


def use_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
) -> None:
    motion_cfg = env_cfg.commands[command_name]
    if not isinstance(motion_cfg, MotionCommandCfg):
        raise TypeError(
            f"Expected env_cfg.commands[{command_name!r}] to be MotionCommandCfg, "
            f"got {type(motion_cfg).__name__}"
        )

    env_cfg.commands[command_name] = textop_motion_command_cfg_from(
        motion_cfg,
    )
