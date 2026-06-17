"""Unitree G1 TextOp-style reference-tracking environment config."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp
from .mdp import MotionReferenceCommandCfg


G1_TRACKING_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def g1_textop_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 reference-tracking config."""

  cfg = unitree_g1_flat_tracking_env_cfg(play=play)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.commands["motion"] = MotionReferenceCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
    horizon=32,
    future_steps=5,
    refresh_margin=8,
    reset_to_reference=True,
    auto_refresh=True,
    body_names=G1_TRACKING_BODY_NAMES,
  )

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "motion"},
    ),
    "future_anchor_pos_b": ObservationTermCfg(
      func=mdp.future_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "future_anchor_ori_b": ObservationTermCfg(
      func=mdp.future_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "reference_root_velocity": ObservationTermCfg(
      func=mdp.reference_root_velocity,
      params={"command_name": "motion"},
    ),
    "root_velocity_error": ObservationTermCfg(
      func=mdp.root_velocity_error,
      params={"command_name": "motion"},
    ),
    "joint_pos_error": ObservationTermCfg(
      func=mdp.joint_pos_error,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel_error": ObservationTermCfg(
      func=mdp.joint_vel_error,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "phase": ObservationTermCfg(
      func=mdp.reference_phase,
      params={"command_name": "motion"},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "reference_joint_pos": ObservationTermCfg(
      func=mdp.reference_joint_pos,
      params={"command_name": "motion"},
    ),
    "reference_joint_vel": ObservationTermCfg(
      func=mdp.reference_joint_vel,
      params={"command_name": "motion"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.rewards = {
    "reference_joint_pos": RewardTermCfg(
      func=mdp.reference_joint_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "reference_joint_vel": RewardTermCfg(
      func=mdp.reference_joint_velocity_error_exp,
      weight=0.25,
      params={"command_name": "motion", "std": 2.0},
    ),
    "reference_root_lin_vel": RewardTermCfg(
      func=mdp.reference_root_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "reference_root_ang_vel": RewardTermCfg(
      func=mdp.reference_root_angular_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 1.0},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=0.5,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-2),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-10.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "root_height": TerminationTermCfg(
      func=mdp.root_height_below_reference,
      params={"command_name": "motion", "threshold": 0.35},
    ),
    "root_tilt": TerminationTermCfg(
      func=mdp.root_tilt_too_large,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.8},
    ),
    "reference_exhausted": TerminationTermCfg(
      func=mdp.reference_exhausted,
      params={"command_name": "motion"},
    ),
  }

  if play:
    cfg.episode_length_s = int(1e9)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionReferenceCommandCfg)
    motion_cmd.reset_to_reference = True
    motion_cmd.auto_refresh = True

  return cfg
