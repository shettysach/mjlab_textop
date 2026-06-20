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
# MJLab/MuJoCo expects MJLAB_G1_JOINT_NAMES order.
# fmt: off
TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
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

# For now, the TextOp anchor is also the root body: pelvis.
# Torso anchor parity is intentionally deferred until MJLab body ordering is verified.
TEXTOP_ROOT_BODY_INDEX = 0
TEXTOP_FUTURE_STEPS = 5


def validate_textop_contract() -> None:
    if len(MJLAB_G1_JOINT_NAMES) != 29:
        raise ValueError(
            f"Expected 29 MJLab G1 joints, got {len(MJLAB_G1_JOINT_NAMES)}"
        )

    if TEXTOP_G1_JOINT_COUNT != 29:
        raise ValueError(
            f"Expected TextOp G1 joint count 29, got {TEXTOP_G1_JOINT_COUNT}"
        )

    mapping = TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX

    if len(mapping) != TEXTOP_G1_JOINT_COUNT:
        raise ValueError(
            f"Joint map has length {len(mapping)}, expected {TEXTOP_G1_JOINT_COUNT}"
        )

    if sorted(mapping) != list(range(TEXTOP_G1_JOINT_COUNT)):
        raise ValueError("Joint map must be a permutation of 0..28")

    overlap = set(TEXTOP_REQUIRED_INPUT_KEYS) & set(TEXTOP_OPTIONAL_INPUT_KEYS)
    if overlap:
        raise ValueError(f"Required and optional TextOp input keys overlap: {overlap}")

    if TEXTOP_ROOT_BODY_INDEX < 0:
        raise ValueError(f"Invalid root body index: {TEXTOP_ROOT_BODY_INDEX}")

    if TEXTOP_FUTURE_STEPS <= 0:
        raise ValueError(f"Invalid future steps: {TEXTOP_FUTURE_STEPS}")
