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

# TextOp tracker motions store G1 joints in IsaacLab order. MJLab/MuJoCo uses
# MJLAB_G1_JOINT_NAMES above.
# fmt: off
TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
)
# fmt:on

TEXTOP_G1_JOINT_COUNT = len(MJLAB_G1_JOINT_NAMES)
TEXTOP_ROOT_BODY_INDEX = 0  # G1 pelvis
TEXTOP_REQUIRED_MOTION_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
)
TEXTOP_OPTIONAL_MOTION_KEYS: tuple[str, ...] = (
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def validate_textop_contract() -> None:
    mapping = TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX

    if len(mapping) != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            f"Joint map has length {len(mapping)}, expected {TEXTOP_G1_JOINT_COUNT}"
        )

    if sorted(mapping) != list(range(TEXTOP_G1_JOINT_COUNT)):
        raise ValueError("Joint map must be a permutation of 0..28")
