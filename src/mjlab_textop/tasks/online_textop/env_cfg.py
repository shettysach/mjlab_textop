from __future__ import annotations

from typing import Literal

from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.observations import (
    future_anchor_ori_b,
    future_anchor_pos_b,
    future_joint_window_textop_order,
    joint_pos_rel_textop_order,
    joint_vel_rel_textop_order,
    last_action_textop_order,
)
from mjlab_textop.core.mdp.online_cleanup import OnlineTextOpCleanup
from mjlab_textop.core.mdp.online_commands import (
    OnlineSourceMode,
    use_online_textop_motion_command,
)
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.tasks.textop_tracking.env_cfg import (
    configure_textop_actor_observations,
    configure_textop_critic_observations,
)

TEXTOP_DEPLOY_SIM_TIMESTEP = 0.002
TEXTOP_DEPLOY_DECIMATION = 10


def make_online_textop_g1_env_cfg(
    *,
    play: bool = True,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
    policy_format: Literal["pt", "onnx"] = "pt",
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    use_online_textop_motion_command(
        cfg,
        command_name="motion",
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        debug_vis=reference_debug_vis,
        observation=observation,
    )
    cfg.commands["motion"].anchor_body_name = "pelvis"  # ty:ignore[unresolved-attribute]
    cfg.sim.mujoco.timestep = TEXTOP_DEPLOY_SIM_TIMESTEP
    cfg.decimation = TEXTOP_DEPLOY_DECIMATION

    if policy_format == "onnx":
        configure_textop_onnx_actor_observations(cfg)
        cfg.events.pop("push_robot", None)
    else:
        configure_textop_actor_observations(cfg)
        configure_textop_critic_observations(cfg)

    configure_online_textop_tracking_terms(cfg)
    cfg.recorders["online_textop_cleanup"] = RecorderTermCfg(
        func=OnlineTextOpCleanup,
        params={"command_name": "motion"},
    )

    return cfg


def configure_textop_onnx_actor_observations(cfg) -> None:
    old_actor = cfg.observations["actor"]
    terms = {
        "future_joint_window": ObservationTermCfg(
            func=future_joint_window_textop_order,
            params={"command_name": "motion"},
        ),
        "future_anchor_pos_b": ObservationTermCfg(
            func=future_anchor_pos_b,
            params={"command_name": "motion"},
        ),
        "future_anchor_ori_b": ObservationTermCfg(
            func=future_anchor_ori_b,
            params={"command_name": "motion"},
        ),
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "base_lin_vel": old_actor.terms["base_lin_vel"],
        "base_ang_vel": old_actor.terms["base_ang_vel"],
        "joint_pos": ObservationTermCfg(func=joint_pos_rel_textop_order),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel_textop_order),
        "actions": ObservationTermCfg(func=last_action_textop_order),
    }

    cfg.observations["actor"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
    )


def configure_online_textop_tracking_terms(cfg) -> None:
    critic_terms = cfg.observations["critic"].terms
    critic_terms.pop("body_pos", None)
    critic_terms.pop("body_ori", None)

    rewards = cfg.rewards
    rewards.pop("motion_body_pos", None)
    rewards.pop("motion_body_ori", None)
    rewards.pop("motion_body_lin_vel", None)
    rewards.pop("motion_body_ang_vel", None)

    cfg.terminations.pop("ee_body_pos", None)
