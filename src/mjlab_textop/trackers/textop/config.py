from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_textop.trackers.textop.observations import (
    configure_textop_actor_observations,
    configure_textop_critic_observations,
    configure_textop_onnx_actor_observations,
)

TEXTOP_DEPLOY_SIM_TIMESTEP = 0.002
TEXTOP_DEPLOY_DECIMATION = 10


def configure_textop_pytorch_tracker(cfg: ManagerBasedRlEnvCfg) -> None:
    _configure_textop_deploy_timing(cfg)
    configure_textop_actor_observations(cfg)
    configure_textop_critic_observations(cfg)


def configure_textop_onnx_tracker(cfg: ManagerBasedRlEnvCfg) -> None:
    _configure_textop_deploy_timing(cfg)
    configure_textop_onnx_actor_observations(cfg)
    cfg.events.pop("push_robot", None)


def _configure_textop_deploy_timing(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.sim.mujoco.timestep = TEXTOP_DEPLOY_SIM_TIMESTEP
    cfg.decimation = TEXTOP_DEPLOY_DECIMATION
