from __future__ import annotations

from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_cleanup import OnlineTextOpCleanup
from mjlab_textop.core.mdp.online_commands import (
    OnlineSourceMode,
    use_online_textop_motion_command,
)
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.trackers.spec import TrackerSpec
from mjlab_textop.trackers.textop.specs import (
    TEXTOP_PYTORCH_TRACKER,
)


def make_online_textop_g1_env_cfg(
    *,
    play: bool = True,
    source: OnlineSource | None = None,
    live_source_cfg: SocketSourceCfg | None = None,
    source_mode: OnlineSourceMode = "live",
    reset_robot_to_reference: bool = True,
    reference_debug_vis: bool | None = None,
    observation: OnlineObservationCfg | None = None,
    tracker: TrackerSpec = TEXTOP_PYTORCH_TRACKER,
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    motion_cfg = use_online_textop_motion_command(
        cfg,
        command_name="motion",
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        debug_vis=reference_debug_vis,
        observation=observation,
        reference_window=tracker.reference_window,
    )
    motion_cfg.anchor_body_name = "pelvis"

    tracker.configure_env(cfg)

    configure_online_textop_tracking_terms(cfg)
    cfg.recorders["online_textop_cleanup"] = RecorderTermCfg(
        func=OnlineTextOpCleanup,
        params={"command_name": "motion"},
    )

    return cfg


def configure_online_textop_tracking_terms(cfg) -> None:
    critic_terms = cfg.observations["critic"].terms
    critic_terms.pop("body_pos", None)
    critic_terms.pop("body_ori", None)

    rewards = cfg.rewards
    rewards.pop("motion_body_pos", None)
    rewards.pop("motion_body_ori", None)
    rewards.pop("motion_body_lin_vel", None)
    rewards.pop("motion_body_ang_vel", None)

    cfg.terminations.pop("ee_body_pos", None)
