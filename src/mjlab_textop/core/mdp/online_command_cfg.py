from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTermCfg

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_types import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import (
    OnlineSource,
    QueueOnlineSource,
    ResettableOnlineSource,
)

if TYPE_CHECKING:
    from mjlab_textop.core.mdp.online_commands import OnlineMotionCommand


@dataclass(kw_only=True)
class OnlineMotionCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    entity_name: str = "robot"
    anchor_body_name: str = "pelvis"
    source: OnlineSource | None = None
    live_source_cfg: SocketSourceCfg | None = None
    source_mode: OnlineSourceMode = "live"
    start_frame: int = 0
    startup_timeout_steps: int = 250
    max_poll_blocks: int = 16
    clear_buffer_on_reset: bool = True
    reset_robot_to_reference: bool = True
    observation: OnlineObservationCfg | None = None
    collision_stop_geom_suffix: str | None = "_collision"

    def __post_init__(self) -> None:
        if self.source_mode not in ("replay", "live"):
            raise ValueError(f"Unknown source_mode: {self.source_mode}")
        if self.source_mode == "replay" and not isinstance(
            self.source, ResettableOnlineSource
        ):
            raise TypeError("Replay online source must implement reset()")
        if self.collision_stop_geom_suffix == "":
            raise ValueError("collision_stop_geom_suffix must be non-empty or None")

    def build(self, env: ManagerBasedRlEnv) -> OnlineMotionCommand:
        from mjlab_textop.core.mdp.online_commands import OnlineMotionCommand

        return OnlineMotionCommand(self, env)


def use_online_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
    collision_stop_geom_suffix: str | None = "_collision",
) -> OnlineMotionCommandCfg:
    motion_cfg = env_cfg.commands[command_name]
    entity_name = motion_cfg.entity_name
    anchor_body_name = motion_cfg.anchor_body_name
    if debug_vis is None:
        debug_vis = motion_cfg.debug_vis
    if source is None and source_mode == "replay":
        source = QueueOnlineSource()

    online_cfg = OnlineMotionCommandCfg(
        entity_name=entity_name,
        anchor_body_name=anchor_body_name,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation,
        collision_stop_geom_suffix=collision_stop_geom_suffix,
        debug_vis=debug_vis,
    )
    env_cfg.commands[command_name] = online_cfg
    return online_cfg
