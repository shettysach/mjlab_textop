from __future__ import annotations

import pytest
from mjlab.tasks.registry import load_env_cfg, load_runner_cls

from mjlab_textop.core.mdp.online_commands import OnlineMotionCommandCfg
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_env_cfg,
)
from mjlab_textop.tasks.registration import register_task
from mjlab_textop.trackers.sonic import (
    SONIC_LOW_LATENCY_TRACKER,
    SonicOnnxPolicyRunner,
)
from mjlab_textop.trackers.sonic.config import SONIC_ACTION_SCALE
from mjlab_textop.trackers.sonic.constants import (
    ACTION_SCALE_7520_22,
    SONIC_DECIMATION,
    SONIC_RAW_OBSERVATION_DIM,
    SONIC_REFERENCE_FRAMES,
    SONIC_SIM_TIMESTEP,
)


def test_sonic_env_uses_released_timing_observations_and_reference_window() -> None:
    cfg = make_online_textop_g1_env_cfg(
        play=True,
        tracker=SONIC_LOW_LATENCY_TRACKER,
    )

    assert cfg.sim.mujoco.timestep == SONIC_SIM_TIMESTEP
    assert cfg.decimation == SONIC_DECIMATION
    assert list(cfg.observations["actor"].terms) == [
        "reference_joint_state",
        "reference_anchor_orientation",
        "base_angular_velocity",
        "joint_position",
        "joint_velocity",
        "last_action",
        "gravity",
    ]
    assert cfg.observations["actor"].enable_corruption is False
    assert cfg.commands["motion"].reference_window.sample_count == (
        SONIC_REFERENCE_FRAMES
    )
    assert cfg.commands["motion"].reference_window.align_heading is True
    assert "push_robot" not in cfg.events
    assert SONIC_RAW_OBSERVATION_DIM == 733


def test_sonic_env_uses_released_g1_motor_configuration() -> None:
    cfg = make_online_textop_g1_env_cfg(
        play=True,
        tracker=SONIC_LOW_LATENCY_TRACKER,
    )
    action = cfg.actions["joint_pos"]
    articulation = cfg.scene.entities["robot"].articulation

    assert action.use_default_offset is True
    assert action.scale == SONIC_ACTION_SCALE
    assert action.scale[".*_hip_pitch_joint"] == pytest.approx(ACTION_SCALE_7520_22)
    assert cfg.scene.entities["robot"].init_state.joint_pos[
        ".*_wrist_.*_joint"
    ] == 0.0
    assert len(articulation.actuators) == 6


def test_registered_sonic_task_uses_sonic_runner() -> None:
    task_name = register_task(
        "default",
        tracker=SONIC_LOW_LATENCY_TRACKER,
        source_mode="live",
    )

    cfg = load_env_cfg(task_name, play=True)
    motion_cfg = cfg.commands["motion"]

    assert load_runner_cls(task_name) is SonicOnnxPolicyRunner
    assert isinstance(motion_cfg, OnlineMotionCommandCfg)
    assert motion_cfg.reference_window.align_heading is True
