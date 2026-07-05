from __future__ import annotations

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FEET_ONLY_COLLISION


def use_g1_feet_only_collision(cfg) -> None:
    robot_cfg = cfg.scene.entities.get("robot")
    if robot_cfg is None:
        raise KeyError("Expected scene entity 'robot' to configure G1 collisions")
    robot_cfg.collisions = (FEET_ONLY_COLLISION,)
