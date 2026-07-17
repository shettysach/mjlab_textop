from __future__ import annotations

from dataclasses import dataclass

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.tasks.goal_task import configure_goal_task
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_env_cfg,
)
from mjlab_textop.tasks.side_goals.assets import make_side_goals_spec_fn
from mjlab_textop.trackers.spec import TrackerSpec
from mjlab_textop.trackers.textop.specs import TEXTOP_PYTORCH_TRACKER


@dataclass(frozen=True)
class SideGoalsTaskCfg:
    blue_goal_pos_w: tuple[float, float, float] = (0.0, 5.0, 0.0)
    green_goal_pos_w: tuple[float, float, float] = (0.0, -5.0, 0.0)
    goal_size: float = 8.0
    arena_size: float = 18.0
    success_radius: float = 0.25
    stop_trigger_radius: float = 0.55
    speed_threshold: float = 0.10
    hold_time_s: float = 1.0
    timeout_s: float = 20.0


SIDE_GOALS_TASK_CFG = SideGoalsTaskCfg()


def make_side_goals_g1_env_cfg(
    *,
    play: bool = True,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
    tracker: TrackerSpec = TEXTOP_PYTORCH_TRACKER,
    task_cfg: SideGoalsTaskCfg = SIDE_GOALS_TASK_CFG,
):
    cfg = make_online_textop_g1_env_cfg(
        play=play,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
        tracker=tracker,
    )
    return _configure_side_goals_cfg(cfg, task_cfg=task_cfg)


def _configure_side_goals_cfg(
    cfg,
    *,
    task_cfg: SideGoalsTaskCfg,
):
    cfg.scene.num_envs = 1
    cfg.scene.spec_fn = make_side_goals_spec_fn(
        blue_goal_pos_w=task_cfg.blue_goal_pos_w,
        green_goal_pos_w=task_cfg.green_goal_pos_w,
        goal_size=task_cfg.goal_size,
        arena_size=task_cfg.arena_size,
    )
    configure_goal_task(
        cfg,
        prefix="side_goals",
        goal_pos_w=task_cfg.green_goal_pos_w,
        success_radius=task_cfg.success_radius,
        stop_trigger_radius=task_cfg.stop_trigger_radius,
        speed_threshold=task_cfg.speed_threshold,
        hold_time_s=task_cfg.hold_time_s,
        timeout_s=task_cfg.timeout_s,
        primary_metric_name="",
        extra_distance_metrics={
            "side_goals_green_goal_distance": task_cfg.green_goal_pos_w,
            "side_goals_blue_goal_distance": task_cfg.blue_goal_pos_w,
        },
    )

    return cfg
