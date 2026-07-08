from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.debug_visualizer import DebugVisualizer

_REFERENCE_GHOST_COLOR = np.array((0.5, 0.7, 0.5, 0.5), dtype=np.float32)


class OnlineReferenceGhost:
    def __init__(self, env: ManagerBasedRlEnv, robot) -> None:
        self.env = env
        self.robot = robot
        self._ghost_model: Any | None = None

    def draw(
        self,
        visualizer: DebugVisualizer,
        *,
        num_envs: int,
        joint_pos: torch.Tensor,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
    ) -> None:
        env_indices = visualizer.get_env_indices(num_envs)
        if not env_indices:
            return

        ghost_model = self._get_ghost_model()
        for env_id in env_indices:
            visualizer.add_ghost_mesh(
                self._reference_qpos(
                    int(env_id),
                    joint_pos=joint_pos,
                    anchor_pos_w=anchor_pos_w,
                    anchor_quat_w=anchor_quat_w,
                ),
                model=ghost_model,
                alpha=float(_REFERENCE_GHOST_COLOR[3]),
                label=f"online_reference_{env_id}",
            )

    def _get_ghost_model(self) -> Any:
        if self._ghost_model is None:
            ghost_model = copy.deepcopy(self.env.sim.mj_model)
            for geom_id in range(ghost_model.ngeom):
                if (
                    ghost_model.geom_contype[geom_id] != 0
                    or ghost_model.geom_conaffinity[geom_id] != 0
                ):
                    ghost_model.geom_rgba[geom_id, 3] = 0.0
                else:
                    ghost_model.geom_rgba[geom_id] = _REFERENCE_GHOST_COLOR
            self._ghost_model = ghost_model
        return self._ghost_model

    def _reference_qpos(
        self,
        env_id: int,
        *,
        joint_pos: torch.Tensor,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
    ) -> np.ndarray:
        indexing = self.robot.indexing
        free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
        joint_q_adr = indexing.joint_q_adr.cpu().numpy()
        env_joint_pos = joint_pos[env_id]

        if len(free_joint_q_adr) < 7:
            raise ValueError(
                "Online reference ghost requires a floating-base robot with "
                f"at least 7 root qpos addresses, got {len(free_joint_q_adr)}"
            )
        if len(joint_q_adr) != env_joint_pos.numel():
            raise ValueError(
                "Online reference joint count does not match robot qpos indexing: "
                f"{env_joint_pos.numel()} reference joints vs {len(joint_q_adr)} "
                "qpos addresses"
            )

        qpos = np.zeros(self.env.sim.mj_model.nq, dtype=np.float64)
        qpos[free_joint_q_adr[:3]] = anchor_pos_w[env_id].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = anchor_quat_w[env_id].cpu().numpy()
        qpos[joint_q_adr] = env_joint_pos.cpu().numpy()
        return qpos
