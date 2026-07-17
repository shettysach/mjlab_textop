from __future__ import annotations

from mjlab_textop.core.feedback.observation import OnlineObservationCfg
from mjlab_textop.core.mdp.online_commands import OnlineSourceMode
from mjlab_textop.core.online.live import SocketSourceCfg
from mjlab_textop.core.online.source import OnlineSource
from mjlab_textop.tasks.online_textop.env_cfg import make_online_textop_g1_env_cfg
from mjlab_textop.tasks.portrait_corridors.assets import make_portrait_corridors_spec_fn
from mjlab_textop.trackers.spec import TrackerSpec
from mjlab_textop.trackers.textop.specs import TEXTOP_PYTORCH_TRACKER


def make_portrait_corridors_g1_env_cfg(
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
    cfg = make_online_textop_g1_env_cfg(
        play=play,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        reset_robot_to_reference=reset_robot_to_reference,
        reference_debug_vis=reference_debug_vis,
        observation=observation,
        tracker=tracker,
    )
    cfg.scene.num_envs = 1
    cfg.scene.spec_fn = make_portrait_corridors_spec_fn()
    return cfg
