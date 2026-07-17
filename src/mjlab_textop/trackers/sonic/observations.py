from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from mjlab_textop.core.mdp.observations import (
    future_anchor_ori_b,
    future_joint_window_isaaclab_order,
    joint_pos_rel_isaaclab_order,
    joint_vel_rel_isaaclab_order,
    last_action_isaaclab_order,
)


def configure_sonic_actor_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """Expose the model-independent values consumed by SonicInputBuilder."""

    old_actor = cfg.observations["actor"]
    terms = {
        "reference_joint_state": ObservationTermCfg(
            func=future_joint_window_isaaclab_order,
            params={"command_name": "motion"},
        ),
        "reference_anchor_orientation": ObservationTermCfg(
            func=future_anchor_ori_b,
            params={"command_name": "motion"},
        ),
        "base_angular_velocity": old_actor.terms["base_ang_vel"],
        "joint_position": ObservationTermCfg(func=joint_pos_rel_isaaclab_order),
        "joint_velocity": ObservationTermCfg(func=joint_vel_rel_isaaclab_order),
        "last_action": ObservationTermCfg(func=last_action_isaaclab_order),
        "gravity": ObservationTermCfg(func=projected_gravity),
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
