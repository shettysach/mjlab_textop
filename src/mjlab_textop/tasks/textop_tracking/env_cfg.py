from __future__ import annotations

from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from mjlab_textop.core.mdp.offline_commands import use_textop_motion_command
from mjlab_textop.trackers.textop.observations import (
    configure_textop_actor_observations,
    configure_textop_critic_observations,
)


def make_textop_g1_flat_tracking_env_cfg(
    *,
    play: bool = False,
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    use_textop_motion_command(
        cfg,
        command_name="motion",
    )
    configure_textop_actor_observations(cfg)
    configure_textop_critic_observations(cfg)

    return cfg
