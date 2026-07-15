from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBaseCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def goal_pos_tensor(
    env: ManagerBasedRlEnv,
    goal_pos_w: tuple[float, float, float],
) -> torch.Tensor:
    return torch.tensor(goal_pos_w, dtype=torch.float32, device=env.device).expand(
        env.num_envs, 3
    )


def robot_goal_distance(
    env: ManagerBasedRlEnv,
    goal_pos_w: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    goal = goal_pos_tensor(env, goal_pos_w)
    return torch.linalg.norm(asset.data.root_link_pos_w[:, :2] - goal[:, :2], dim=-1)


def robot_xy_speed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)


def inside_goal_radius(
    env: ManagerBasedRlEnv,
    goal_pos_w: tuple[float, float, float],
    radius: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    return robot_goal_distance(env, goal_pos_w, asset_cfg) <= radius


def below_speed_threshold(
    env: ManagerBasedRlEnv,
    speed_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    return robot_xy_speed(env, asset_cfg) <= speed_threshold


def stop_trigger_active(
    env: ManagerBasedRlEnv,
    goal_pos_w: tuple[float, float, float],
    stop_trigger_radius: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    return inside_goal_radius(env, goal_pos_w, stop_trigger_radius, asset_cfg)


def overshot_goal(
    env: ManagerBasedRlEnv,
    goal_pos_w: tuple[float, float, float],
    margin: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 0] > goal_pos_w[0] + margin


class success_held:
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self.held_time = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        goal_pos_w: tuple[float, float, float],
        success_radius: float,
        speed_threshold: float,
        hold_time_s: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        close_enough = inside_goal_radius(
            env,
            goal_pos_w,
            success_radius,
            asset_cfg,
        )
        slow_enough = below_speed_threshold(env, speed_threshold, asset_cfg)
        success_now = close_enough & slow_enough
        self.held_time = torch.where(
            success_now,
            self.held_time + env.step_dt,
            torch.zeros_like(self.held_time),
        )
        return self.held_time >= hold_time_s

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.held_time[env_ids] = 0.0
