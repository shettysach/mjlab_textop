from __future__ import annotations

from typing import Any

import torch


class CollisionDetector:
    """Detect contacts between an entity and configured obstacle geoms."""

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
        return contains_geom_pair(
            sim_data.contact.geom,
            contact_count=int(sim_data.nacon[0].item()),
            first_ids=self.robot_geom_ids,
            second_ids=self.obstacle_geom_ids,
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

    for geom_id in range(int(model.ngeom)):
        geom_name = model.geom(geom_id).name or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name or ""
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
