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

# TextOp tracker motions store G1 joints in IsaacLab order. The shared stream
# contract owns the conversion to the MJLab/MuJoCo order above.
# fmt: off
TEXTOP_TO_MJLAB_G1_JOINT_INDEX: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)
# fmt: on

MJLAB_TO_TEXTOP_G1_JOINT_INDEX = np.argsort(TEXTOP_TO_MJLAB_G1_JOINT_INDEX)
G1_JOINT_COUNT = len(G1_JOINT_NAMES)
TEXTOP_FPS = 50.0


def textop_to_mjlab_joint_order(values: np.ndarray) -> np.ndarray:
    return values[..., TEXTOP_TO_MJLAB_G1_JOINT_INDEX]


def mjlab_to_textop_joint_order(values: np.ndarray) -> np.ndarray:
    return values[..., MJLAB_TO_TEXTOP_G1_JOINT_INDEX]
