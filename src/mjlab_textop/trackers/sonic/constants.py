from __future__ import annotations

SONIC_CONTROL_HZ = 50.0
SONIC_SIM_TIMESTEP = 0.005
SONIC_DECIMATION = 4

SONIC_JOINT_COUNT = 29
SONIC_REFERENCE_FRAMES = 10
SONIC_HISTORY_FRAMES = 10
SONIC_TOKEN_DIM = 64
SONIC_ENCODER_INPUT_DIM = 1247
SONIC_DECODER_INPUT_DIM = 994
SONIC_RAW_OBSERVATION_DIM = 733

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# Keep the released deployment's literal so gains and action scales match its
# policy_parameters.hpp rather than depending on a platform math constant.
NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def stiffness(armature: float) -> float:
    return armature * NATURAL_FREQUENCY**2


def damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQUENCY


STIFFNESS_5020 = stiffness(ARMATURE_5020)
STIFFNESS_7520_14 = stiffness(ARMATURE_7520_14)
STIFFNESS_7520_22 = stiffness(ARMATURE_7520_22)
STIFFNESS_4010 = stiffness(ARMATURE_4010)

DAMPING_5020 = damping(ARMATURE_5020)
DAMPING_7520_14 = damping(ARMATURE_7520_14)
DAMPING_7520_22 = damping(ARMATURE_7520_22)
DAMPING_4010 = damping(ARMATURE_4010)

EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0


def action_scale(effort: float, position_gain: float) -> float:
    return 0.25 * effort / position_gain


ACTION_SCALE_5020 = action_scale(EFFORT_5020, STIFFNESS_5020)
ACTION_SCALE_7520_14 = action_scale(EFFORT_7520_14, STIFFNESS_7520_14)
ACTION_SCALE_7520_22 = action_scale(EFFORT_7520_22, STIFFNESS_7520_22)
ACTION_SCALE_4010 = action_scale(EFFORT_4010, STIFFNESS_4010)
