from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mjlab_textop.core.mdp.online_types import FutureWindow
from mjlab_textop.core.online.source import MotionBlock


@dataclass
class CollisionRecovery:
    """State machine for collision latching and recovery-block acceptance."""

    active: bool = False
    epoch: int = 0
    contact_active: bool = False
    hold_window: FutureWindow | None = None
    buffering: bool = False

    def collision_edge(self, in_collision: bool) -> bool:
        if not in_collision:
            self.contact_active = False
            return False
        if self.contact_active:
            return False
        self.contact_active = True
        return True

    def activate(self, safe_window: FutureWindow) -> int:
        self.active = True
        self.epoch += 1
        self.hold_window = make_stationary_window(safe_window)
        self.buffering = False
        return self.epoch

    def accepts(self, block: MotionBlock) -> bool:
        return (
            block.prompt is not None
            and block.prompt.strip().lower() == "stand"
            and block.recovery_epoch == self.epoch
        )

    def complete(self) -> None:
        self.active = False
        self.hold_window = None
        self.buffering = False

    def reset(self) -> None:
        self.active = False
        self.contact_active = False
        self.hold_window = None
        self.buffering = False


class CollisionDetector:
    def __init__(
        self,
        model: Any,
        *,
        entity_name: str,
        obstacle_suffix: str | None,
        device: torch.device | str,
    ) -> None:
        self.robot_geom_ids, self.obstacle_geom_ids = find_collision_geom_ids(
            model,
            entity_name=entity_name,
            obstacle_suffix=obstacle_suffix,
            device=device,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.robot_geom_ids.numel() and self.obstacle_geom_ids.numel())

    def has_collision(self, sim_data: Any) -> bool:
        if not self.enabled:
            return False
        contact = getattr(sim_data, "contact", None)
        nacon = getattr(sim_data, "nacon", None)
        if contact is None or nacon is None:
            return False
        contact_geom = getattr(contact, "geom", None)
        if contact_geom is None:
            return False
        return contains_geom_pair(
            contact_geom,
            contact_count=int(nacon[0].item()),
            first_ids=self.robot_geom_ids,
            second_ids=self.obstacle_geom_ids,
        )


def make_stationary_window(window: FutureWindow) -> FutureWindow:
    future_steps = window.joint_pos.shape[0]
    return FutureWindow(
        joint_pos=window.joint_pos[0].repeat(future_steps, 1),
        joint_vel=torch.zeros_like(window.joint_vel),
        anchor_pos_w=window.anchor_pos_w[0].repeat(future_steps, 1),
        anchor_quat_w=window.anchor_quat_w[0].repeat(future_steps, 1),
        stale_steps=0,
    )


def contains_geom_pair(
    contact_geom: torch.Tensor,
    *,
    contact_count: int,
    first_ids: torch.Tensor,
    second_ids: torch.Tensor,
) -> bool:
    if contact_count <= 0:
        return False
    pairs = contact_geom[:contact_count].to(dtype=torch.long)
    matches = torch.isin(pairs[:, 0], first_ids) & torch.isin(pairs[:, 1], second_ids)
    reverse_matches = torch.isin(pairs[:, 1], first_ids) & torch.isin(
        pairs[:, 0], second_ids
    )
    return bool(torch.any(matches | reverse_matches).item())


def find_collision_geom_ids(
    model: Any,
    *,
    entity_name: str,
    obstacle_suffix: str | None,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if obstacle_suffix is None:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    robot_prefix = f"{entity_name}/"
    robot_ids: list[int] = []
    obstacle_ids: list[int] = []
    geom_lookup = getattr(model, "geom", None)
    body_lookup = getattr(model, "body", None)
    geom_bodyid = getattr(model, "geom_bodyid", None)
    if not callable(geom_lookup):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    for geom_id in range(int(model.ngeom)):
        geom_name = geom_lookup(geom_id).name or ""
        body_name = ""
        if callable(body_lookup) and geom_bodyid is not None:
            body_id = int(geom_bodyid[geom_id])
            body_name = body_lookup(body_id).name or ""
        is_robot_geom = geom_name.startswith(robot_prefix) or body_name.startswith(
            robot_prefix
        )
        if is_robot_geom:
            robot_ids.append(geom_id)
        elif geom_name.endswith(obstacle_suffix):
            obstacle_ids.append(geom_id)

    return (
        torch.tensor(robot_ids, dtype=torch.long, device=device),
        torch.tensor(obstacle_ids, dtype=torch.long, device=device),
    )
