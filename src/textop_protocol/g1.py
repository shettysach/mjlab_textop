from __future__ import annotations

import numpy as np

G1_JOINT_NAMES: tuple[str, ...] = (
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

# fmt: off
ISAACLAB_TO_MJLAB_G1_JOINT_INDEX: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)
# fmt: on

MJLAB_TO_ISAACLAB_G1_JOINT_INDEX = np.argsort(ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
G1_JOINT_COUNT = len(G1_JOINT_NAMES)


def isaaclab_to_mjlab_joint_order(values: np.ndarray) -> np.ndarray:
    return values[..., ISAACLAB_TO_MJLAB_G1_JOINT_INDEX]


def mjlab_to_isaaclab_joint_order(values: np.ndarray) -> np.ndarray:
    return values[..., MJLAB_TO_ISAACLAB_G1_JOINT_INDEX]
