from __future__ import annotations

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
from mjlab_textop.tasks.green_square_stop import mdp
from mjlab_textop.tasks.green_square_stop.assets import make_green_square_spec_fn
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_flat_tracking_env_cfg,
    make_online_textop_onnx_g1_flat_tracking_env_cfg,
)

GREEN_SQUARE_GOAL_POS_W = (8.0, 0.0, 0.0)
GREEN_SQUARE_SIZE = 18.0
GREEN_SQUARE_SUCCESS_RADIUS = 0.25
GREEN_SQUARE_STOP_TRIGGER_RADIUS = 0.55
GREEN_SQUARE_SPEED_THRESHOLD = 0.10
GREEN_SQUARE_HOLD_TIME_S = 1.0
GREEN_SQUARE_TIMEOUT_S = 20.0


def make_green_square_stop_g1_env_cfg(
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
    observation: OnlineTextOpObservationCfg | None = None,
    goal_pos_w: tuple[float, float, float] = GREEN_SQUARE_GOAL_POS_W,
    goal_size: float = GREEN_SQUARE_SIZE,
    success_radius: float = GREEN_SQUARE_SUCCESS_RADIUS,
    stop_trigger_radius: float = GREEN_SQUARE_STOP_TRIGGER_RADIUS,
    speed_threshold: float = GREEN_SQUARE_SPEED_THRESHOLD,
    hold_time_s: float = GREEN_SQUARE_HOLD_TIME_S,
    timeout_s: float = GREEN_SQUARE_TIMEOUT_S,
):
    cfg = make_online_textop_g1_flat_tracking_env_cfg(
        play=play,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation,
    )
    return _configure_green_square_stop_cfg(
        cfg,
        goal_pos_w=goal_pos_w,
        goal_size=goal_size,
        success_radius=success_radius,
        stop_trigger_radius=stop_trigger_radius,
        speed_threshold=speed_threshold,
        hold_time_s=hold_time_s,
        timeout_s=timeout_s,
    )


def make_green_square_stop_onnx_g1_env_cfg(
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
    observation: OnlineTextOpObservationCfg | None = None,
    goal_pos_w: tuple[float, float, float] = GREEN_SQUARE_GOAL_POS_W,
    goal_size: float = GREEN_SQUARE_SIZE,
    success_radius: float = GREEN_SQUARE_SUCCESS_RADIUS,
    stop_trigger_radius: float = GREEN_SQUARE_STOP_TRIGGER_RADIUS,
    speed_threshold: float = GREEN_SQUARE_SPEED_THRESHOLD,
    hold_time_s: float = GREEN_SQUARE_HOLD_TIME_S,
    timeout_s: float = GREEN_SQUARE_TIMEOUT_S,
):
    cfg = make_online_textop_onnx_g1_flat_tracking_env_cfg(
        play=play,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation,
    )
    return _configure_green_square_stop_cfg(
        cfg,
        goal_pos_w=goal_pos_w,
        goal_size=goal_size,
        success_radius=success_radius,
        stop_trigger_radius=stop_trigger_radius,
        speed_threshold=speed_threshold,
        hold_time_s=hold_time_s,
        timeout_s=timeout_s,
    )


def _configure_green_square_stop_cfg(
    cfg,
    *,
    goal_pos_w: tuple[float, float, float],
    goal_size: float,
    success_radius: float,
    stop_trigger_radius: float,
    speed_threshold: float,
    hold_time_s: float,
    timeout_s: float,
):
    cfg.scene.num_envs = 1
    cfg.scene.spec_fn = make_green_square_spec_fn(
        goal_pos_w=goal_pos_w,
        size=goal_size,
    )
    cfg.episode_length_s = timeout_s
    cfg.rewards = {}
    cfg.metrics.update(
        {
            "green_square_goal_distance": MetricsTermCfg(
                func=mdp.robot_goal_distance,
                params={"goal_pos_w": goal_pos_w},
                reduce="last",
            ),
            "green_square_xy_speed": MetricsTermCfg(
                func=mdp.robot_xy_speed,
                reduce="last",
            ),
            "green_square_stop_trigger": MetricsTermCfg(
                func=mdp.stop_trigger_active,
                params={
                    "goal_pos_w": goal_pos_w,
                    "stop_trigger_radius": stop_trigger_radius,
                },
                reduce="last",
            ),
            "green_square_inside_success_radius": MetricsTermCfg(
                func=mdp.inside_goal_radius,
                params={"goal_pos_w": goal_pos_w, "radius": success_radius},
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
        "green_square_success": TerminationTermCfg(
            func=mdp.success_held,
            params={
                "goal_pos_w": goal_pos_w,
                "success_radius": success_radius,
                "speed_threshold": speed_threshold,
                "hold_time_s": hold_time_s,
            },
            time_out=True,
        ),
        "green_square_overshot": TerminationTermCfg(
            func=mdp.overshot_goal,
            params={"goal_pos_w": goal_pos_w, "margin": 1.0},
        ),
    }

    return cfg
