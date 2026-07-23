"""Compatibility re-export for the shared navigation task terms."""

from tasks.goal_mdp import (
    below_speed_threshold,
    goal_pos_tensor,
    inside_goal_radius,
    overshot_goal,
    robot_goal_distance,
    robot_xy_speed,
    stop_trigger_active,
    success_held,
)

__all__ = [
    "below_speed_threshold",
    "goal_pos_tensor",
    "inside_goal_radius",
    "overshot_goal",
    "robot_goal_distance",
    "robot_xy_speed",
    "stop_trigger_active",
    "success_held",
]
