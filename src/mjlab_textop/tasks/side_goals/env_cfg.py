from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.envs.mdp import terminations as base_terminations
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_textop.core.feedback.observation import OnlineTextOpObservationCfg
from mjlab_textop.core.mdp.online_commands import TextOpOnlineSourceMode
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_flat_tracking_env_cfg,
    make_online_textop_onnx_g1_flat_tracking_env_cfg,
)
from mjlab_textop.tasks.side_goals.assets import make_side_goals_spec_fn
from mjlab_textop.tasks.straight import mdp


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
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode = "live",
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineTextOpObservationCfg | None = None,
    task_cfg: SideGoalsTaskCfg = SIDE_GOALS_TASK_CFG,
):
    cfg = make_online_textop_g1_flat_tracking_env_cfg(
        play=play,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
    )
    return _configure_side_goals_cfg(cfg, task_cfg=task_cfg)


def make_side_goals_onnx_g1_env_cfg(
    *,
    play: bool = True,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode = "live",
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineTextOpObservationCfg | None = None,
    task_cfg: SideGoalsTaskCfg = SIDE_GOALS_TASK_CFG,
):
    cfg = make_online_textop_onnx_g1_flat_tracking_env_cfg(
        play=play,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
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
    cfg.episode_length_s = task_cfg.timeout_s
    cfg.rewards = {}
    cfg.metrics.update(
        {
            "side_goals_green_goal_distance": MetricsTermCfg(
                func=mdp.robot_goal_distance,
                params={"goal_pos_w": task_cfg.green_goal_pos_w},
                reduce="last",
            ),
            "side_goals_blue_goal_distance": MetricsTermCfg(
                func=mdp.robot_goal_distance,
                params={"goal_pos_w": task_cfg.blue_goal_pos_w},
                reduce="last",
            ),
            "side_goals_xy_speed": MetricsTermCfg(
                func=mdp.robot_xy_speed,
                reduce="last",
            ),
            "side_goals_stop_trigger": MetricsTermCfg(
                func=mdp.stop_trigger_active,
                params={
                    "goal_pos_w": task_cfg.green_goal_pos_w,
                    "stop_trigger_radius": task_cfg.stop_trigger_radius,
                },
                reduce="last",
            ),
            "side_goals_inside_success_radius": MetricsTermCfg(
                func=mdp.inside_goal_radius,
                params={
                    "goal_pos_w": task_cfg.green_goal_pos_w,
                    "radius": task_cfg.success_radius,
                },
                reduce="last",
            ),
        }
    )
    cfg.terminations = {
        "time_out": TerminationTermCfg(
            func=base_terminations.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=base_terminations.bad_orientation,
            params={
                "limit_angle": 1.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "side_goals_success": TerminationTermCfg(
            func=mdp.success_held,
            params={
                "goal_pos_w": task_cfg.green_goal_pos_w,
                "success_radius": task_cfg.success_radius,
                "speed_threshold": task_cfg.speed_threshold,
                "hold_time_s": task_cfg.hold_time_s,
            },
            time_out=True,
        ),
    }

    return cfg
