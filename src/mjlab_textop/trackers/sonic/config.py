from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab_textop.trackers.sonic.constants import (
    ACTION_SCALE_4010,
    ACTION_SCALE_5020,
    ACTION_SCALE_7520_14,
    ACTION_SCALE_7520_22,
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    EFFORT_4010,
    EFFORT_5020,
    EFFORT_7520_14,
    EFFORT_7520_22,
    SONIC_DECIMATION,
    SONIC_SIM_TIMESTEP,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)
from mjlab_textop.trackers.sonic.observations import (
    configure_sonic_actor_observations,
)

SONIC_ACTION_SCALE = {
    ".*_elbow_joint": ACTION_SCALE_5020,
    ".*_shoulder_pitch_joint": ACTION_SCALE_5020,
    ".*_shoulder_roll_joint": ACTION_SCALE_5020,
    ".*_shoulder_yaw_joint": ACTION_SCALE_5020,
    ".*_wrist_roll_joint": ACTION_SCALE_5020,
    ".*_hip_pitch_joint": ACTION_SCALE_7520_22,
    ".*_hip_roll_joint": ACTION_SCALE_7520_22,
    ".*_knee_joint": ACTION_SCALE_7520_22,
    ".*_hip_yaw_joint": ACTION_SCALE_7520_14,
    "waist_yaw_joint": ACTION_SCALE_7520_14,
    ".*_wrist_pitch_joint": ACTION_SCALE_4010,
    ".*_wrist_yaw_joint": ACTION_SCALE_4010,
    "waist_pitch_joint": ACTION_SCALE_5020,
    "waist_roll_joint": ACTION_SCALE_5020,
    ".*_ankle_pitch_joint": ACTION_SCALE_5020,
    ".*_ankle_roll_joint": ACTION_SCALE_5020,
}


def configure_sonic_tracker(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.sim.mujoco.timestep = SONIC_SIM_TIMESTEP
    cfg.decimation = SONIC_DECIMATION
    configure_sonic_actor_observations(cfg)
    _configure_sonic_g1(cfg)
    cfg.events.pop("push_robot", None)


def _configure_sonic_g1(cfg: ManagerBasedRlEnvCfg) -> None:
    robot = cfg.scene.entities["robot"]
    robot.articulation = make_sonic_g1_articulation()
    if robot.init_state.joint_pos is None:
        robot.init_state.joint_pos = {}
    robot.init_state.joint_pos[".*_wrist_.*_joint"] = 0.0

    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointPositionActionCfg):
        raise TypeError(
            f"SONIC requires a JointPositionActionCfg, got {type(action).__name__}"
        )
    action.scale = SONIC_ACTION_SCALE.copy()
    action.use_default_offset = True


def make_sonic_g1_articulation() -> EntityArticulationInfoCfg:
    """G1 gains and motor assignments used by the released SONIC policy."""

    return EntityArticulationInfoCfg(
        actuators=(
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_elbow_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_roll_joint",
                ),
                stiffness=STIFFNESS_5020,
                damping=DAMPING_5020,
                effort_limit=EFFORT_5020,
                armature=ARMATURE_5020,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_knee_joint",
                ),
                stiffness=STIFFNESS_7520_22,
                damping=DAMPING_7520_22,
                effort_limit=EFFORT_7520_22,
                armature=ARMATURE_7520_22,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
                stiffness=STIFFNESS_7520_14,
                damping=DAMPING_7520_14,
                effort_limit=EFFORT_7520_14,
                armature=ARMATURE_7520_14,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                stiffness=STIFFNESS_4010,
                damping=DAMPING_4010,
                effort_limit=EFFORT_4010,
                armature=ARMATURE_4010,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
                stiffness=2.0 * STIFFNESS_5020,
                damping=2.0 * DAMPING_5020,
                effort_limit=2.0 * EFFORT_5020,
                armature=2.0 * ARMATURE_5020,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                stiffness=2.0 * STIFFNESS_5020,
                damping=2.0 * DAMPING_5020,
                effort_limit=2.0 * EFFORT_5020,
                armature=2.0 * ARMATURE_5020,
            ),
        ),
        soft_joint_pos_limit_factor=0.9,
    )
