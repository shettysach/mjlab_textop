from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.tasks.blocked_straight.assets import (
    make_blocked_straight_spec_fn,
)
from mjlab_textop.tasks.goal_task import configure_goal_task
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_env_cfg,
)


@dataclass(frozen=True)
class BlockedStraightTaskCfg:
    goal_pos_w: tuple[float, float, float] = (24.0, 0.0, 0.0)
    goal_size: float = 18.0
    obstacle_pos_xy: tuple[float, float] = (12.0, 0.0)
    obstacle_size_xy: tuple[float, float] = (1.5, 8.0)
    success_radius: float = 0.25
    stop_trigger_radius: float = 0.55
    speed_threshold: float = 0.10
    hold_time_s: float = 1.0
    timeout_s: float = 20.0


BLOCKED_STRAIGHT_TASK_CFG = BlockedStraightTaskCfg()


def make_blocked_straight_g1_env_cfg(
    *,
    play: bool = True,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
    policy_format: Literal["pt", "onnx"] = "pt",
    task_cfg: BlockedStraightTaskCfg = BLOCKED_STRAIGHT_TASK_CFG,
):
    cfg = make_online_textop_g1_env_cfg(
        play=play,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
        policy_format=policy_format,
    )
    return _configure_blocked_straight_cfg(cfg, task_cfg=task_cfg)


def _configure_blocked_straight_cfg(
    cfg,
    *,
    task_cfg: BlockedStraightTaskCfg,
):
    cfg.scene.num_envs = 1
    cfg.scene.spec_fn = make_blocked_straight_spec_fn(
        goal_pos_w=task_cfg.goal_pos_w,
        size=task_cfg.goal_size,
        obstacle_pos_xy=task_cfg.obstacle_pos_xy,
        obstacle_size_xy=task_cfg.obstacle_size_xy,
    )
    configure_goal_task(
        cfg,
        prefix="blocked_straight",
        goal_pos_w=task_cfg.goal_pos_w,
        success_radius=task_cfg.success_radius,
        stop_trigger_radius=task_cfg.stop_trigger_radius,
        speed_threshold=task_cfg.speed_threshold,
        hold_time_s=task_cfg.hold_time_s,
        timeout_s=task_cfg.timeout_s,
        overshoot_margin=1.0,
    )

    return cfg
