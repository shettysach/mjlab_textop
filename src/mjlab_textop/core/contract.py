"""
`T` = number of time frames in the motion clip.
`B` = number of tracked robot bodies/links in the body arrays.

TextOp tracker NPZ struct:
- `joint_pos`, `joint_vel`: `[T, 29]` G1 joints in IsaacLab order.
- `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w`: `[T, B, 3]` world-frame body data.
- `body_quat_w`: `[T, B, 4]` world-frame body quaternions.
- fps is optional when an explicit loader fps is provided.
- body index 0 is the pelvis/root body used as the TextOp anchor.

MJLab normalized motion struct:
- `joint_pos`, `joint_vel`: `[T, 29]` G1 joints in MJLab order.
- `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w`: `[T, B, 3]` MJLab body order.
- `body_quat_w`: `[T, B, 4]` MJLab body order.
- fps is stored in the output NPZ.
"""

from __future__ import annotations

MJLAB_G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# TextOp tracker motions store G1 joints in IsaacLab order.
# fmt: off
TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)
# fmt:on

TEXTOP_G1_JOINT_COUNT = len(MJLAB_G1_JOINT_NAMES)

TEXTOP_REQUIRED_INPUT_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
)
TEXTOP_OPTIONAL_INPUT_KEYS: tuple[str, ...] = (
    "fps",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

TEXTOP_ROOT_BODY_INDEX = 0  # pelvis
TEXTOP_FUTURE_STEPS = 5
