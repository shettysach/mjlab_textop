from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_vla.textop import mdp as textop_mdp
from mjlab_vla.textop.contract import TEXTOP_FUTURE_STEPS

TEXTOP_TASK_NAME = "Mjlab-TextOp-Flat-Unitree-G1"


def make_textop_g1_flat_tracking_env_cfg(
    *,
    play: bool = False,
    future_steps: int = TEXTOP_FUTURE_STEPS,
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    textop_mdp.use_textop_motion_command(
        cfg,
        command_name="motion",
        future_steps=future_steps,
    )
    _configure_textop_anchor(cfg)
    _configure_textop_actor_observations(cfg)
    _configure_textop_critic_observations(cfg)

    return cfg


def ensure_textop_task_registered() -> None:
    if TEXTOP_TASK_NAME in list_tasks():
        return

    register_mjlab_task(
        task_id=TEXTOP_TASK_NAME,
        env_cfg=make_textop_g1_flat_tracking_env_cfg(play=False),
        play_env_cfg=make_textop_g1_flat_tracking_env_cfg(play=True),
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )


def _configure_textop_anchor(cfg) -> None:
    motion_cmd = cfg.commands["motion"]

    # Match the current offline TextOp tracker convention. MJLab's base G1
    # tracking task uses torso_link; if pelvis-anchor training is unstable, add a
    # separate torso-anchor TextOp variant rather than changing this one silently.
    motion_cmd.anchor_body_name = "pelvis"


def _configure_textop_actor_observations(cfg) -> None:
    old_actor = cfg.observations["actor"]
    terms = {
        "future_joint_window": ObservationTermCfg(
            func=textop_mdp.future_joint_window,
            params={"command_name": "motion"},
        ),
        "future_anchor_pos_b": ObservationTermCfg(
            func=textop_mdp.future_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        ),
        "future_anchor_ori_b": ObservationTermCfg(
            func=textop_mdp.future_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "projected_gravity": ObservationTermCfg(func=textop_mdp.projected_gravity),
        "base_lin_vel": old_actor.terms["base_lin_vel"],
        "base_ang_vel": old_actor.terms["base_ang_vel"],
        "joint_pos": old_actor.terms["joint_pos"],
        "joint_vel": old_actor.terms["joint_vel"],
        "actions": old_actor.terms["actions"],
    }

    cfg.observations["actor"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=old_actor.enable_corruption,
    )


def _configure_textop_critic_observations(cfg) -> None:
    old_critic = cfg.observations["critic"]
    terms = {
        "future_joint_window": ObservationTermCfg(
            func=textop_mdp.future_joint_window,
            params={"command_name": "motion"},
        ),
        "future_anchor_pos_b": ObservationTermCfg(
            func=textop_mdp.future_anchor_pos_b,
            params={"command_name": "motion"},
        ),
        "future_anchor_ori_b": ObservationTermCfg(
            func=textop_mdp.future_anchor_ori_b,
            params={"command_name": "motion"},
        ),
        "body_pos": old_critic.terms["body_pos"],
        "body_ori": old_critic.terms["body_ori"],
        "base_lin_vel": old_critic.terms["base_lin_vel"],
        "base_ang_vel": old_critic.terms["base_ang_vel"],
        "joint_pos": old_critic.terms["joint_pos"],
        "joint_vel": old_critic.terms["joint_vel"],
        "actions": old_critic.terms["actions"],
    }

    cfg.observations["critic"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
    )


ensure_textop_task_registered()
