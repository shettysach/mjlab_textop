from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.envs.mdp import terminations as base_terminations
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.core.schema import FUTURE_STEPS
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_env_cfg,
)
from mjlab_textop.tasks.straight import mdp
from mjlab_textop.tasks.straight.assets import make_straight_spec_fn


@dataclass(frozen=True)
class StraightTaskCfg:
    goal_pos_w: tuple[float, float, float] = (24.0, 0.0, 0.0)
    goal_size: float = 18.0
    success_radius: float = 0.25
    stop_trigger_radius: float = 0.55
    speed_threshold: float = 0.10
    hold_time_s: float = 1.0
    timeout_s: float = 20.0


STRAIGHT_TASK_CFG = StraightTaskCfg()


def make_straight_g1_env_cfg(
    *,
    play: bool = True,
    future_steps: int = FUTURE_STEPS,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
    policy_format: Literal["pt", "onnx"] = "pt",
    task_cfg: StraightTaskCfg = STRAIGHT_TASK_CFG,
):
    cfg = make_online_textop_g1_env_cfg(
        play=play,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
        policy_format=policy_format,
    )
    return _configure_straight_cfg(cfg, task_cfg=task_cfg)


def _configure_straight_cfg(
    cfg,
    *,
    task_cfg: StraightTaskCfg,
):
    cfg.scene.num_envs = 1
    cfg.scene.spec_fn = make_straight_spec_fn(
        goal_pos_w=task_cfg.goal_pos_w,
        size=task_cfg.goal_size,
    )
    cfg.episode_length_s = task_cfg.timeout_s
    cfg.rewards = {}
    cfg.metrics.update(
        {
            "straight_goal_distance": MetricsTermCfg(
                func=mdp.robot_goal_distance,
                params={"goal_pos_w": task_cfg.goal_pos_w},
                reduce="last",
            ),
            "straight_xy_speed": MetricsTermCfg(
                func=mdp.robot_xy_speed,
                reduce="last",
            ),
            "straight_stop_trigger": MetricsTermCfg(
                func=mdp.stop_trigger_active,
                params={
                    "goal_pos_w": task_cfg.goal_pos_w,
                    "stop_trigger_radius": task_cfg.stop_trigger_radius,
                },
                reduce="last",
            ),
            "straight_inside_success_radius": MetricsTermCfg(
                func=mdp.inside_goal_radius,
                params={
                    "goal_pos_w": task_cfg.goal_pos_w,
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
        "straight_success": TerminationTermCfg(
            func=mdp.success_held,
            params={
                "goal_pos_w": task_cfg.goal_pos_w,
                "success_radius": task_cfg.success_radius,
                "speed_threshold": task_cfg.speed_threshold,
                "hold_time_s": task_cfg.hold_time_s,
            },
            time_out=True,
        ),
        "straight_overshot": TerminationTermCfg(
            func=mdp.overshot_goal,
            params={"goal_pos_w": task_cfg.goal_pos_w, "margin": 1.0},
        ),
    }

    return cfg
