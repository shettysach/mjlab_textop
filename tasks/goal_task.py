from __future__ import annotations

from collections.abc import Mapping

from mjlab.envs.mdp import terminations as base_terminations
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from tasks import goal_mdp as mdp


def configure_goal_task(
    cfg,
    *,
    prefix: str,
    goal_pos_w: tuple[float, float, float],
    success_radius: float,
    stop_trigger_radius: float,
    speed_threshold: float,
    hold_time_s: float,
    timeout_s: float,
    overshoot_margin: float | None = None,
    primary_metric_name: str | None = None,
    extra_distance_metrics: Mapping[str, tuple[float, float, float]] | None = None,
) -> None:
    """Install shared goal metrics and terminations for navigation tasks."""
    cfg.episode_length_s = timeout_s
    cfg.rewards = {}
    metrics = {
        f"{prefix}_xy_speed": MetricsTermCfg(func=mdp.robot_xy_speed, reduce="last"),
        f"{prefix}_stop_trigger": MetricsTermCfg(
            func=mdp.stop_trigger_active,
            params={
                "goal_pos_w": goal_pos_w,
                "stop_trigger_radius": stop_trigger_radius,
            },
            reduce="last",
        ),
        f"{prefix}_inside_success_radius": MetricsTermCfg(
            func=mdp.inside_goal_radius,
            params={"goal_pos_w": goal_pos_w, "radius": success_radius},
            reduce="last",
        ),
    }
    if primary_metric_name is None:
        primary_metric_name = f"{prefix}_goal_distance"
    if primary_metric_name:
        metrics[primary_metric_name] = MetricsTermCfg(
            func=mdp.robot_goal_distance,
            params={"goal_pos_w": goal_pos_w},
            reduce="last",
        )
    if extra_distance_metrics:
        metrics.update(
            {
                name: MetricsTermCfg(
                    func=mdp.robot_goal_distance,
                    params={"goal_pos_w": metric_goal},
                    reduce="last",
                )
                for name, metric_goal in extra_distance_metrics.items()
            }
        )
    cfg.metrics.update(metrics)
    cfg.terminations = {
        "time_out": TerminationTermCfg(
            func=base_terminations.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=base_terminations.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        f"{prefix}_success": TerminationTermCfg(
            func=mdp.success_held,
            params={
                "goal_pos_w": goal_pos_w,
                "success_radius": success_radius,
                "speed_threshold": speed_threshold,
                "hold_time_s": hold_time_s,
            },
            time_out=True,
        ),
    }
    if overshoot_margin is not None:
        cfg.terminations[f"{prefix}_overshot"] = TerminationTermCfg(
            func=mdp.overshot_goal,
            params={"goal_pos_w": goal_pos_w, "margin": overshoot_margin},
        )
