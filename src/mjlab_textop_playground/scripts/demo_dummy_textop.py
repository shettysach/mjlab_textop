from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import mjlab_textop_playground.tasks  # noqa: F401


@dataclass(frozen=True)
class DemoConfig:
  num_envs: int = 4
  steps: int = 200
  device: str | None = None
  agent: Literal["zero", "random"] = "zero"


def main() -> None:
  cfg = tyro.cli(DemoConfig)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg("Mjlab-TextOpTracking-Flat-Unitree-G1", play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.terminations = {}

  env = ManagerBasedRlEnv(env_cfg, device=device)
  obs, _ = env.reset()
  del obs

  action_shape = env.action_space.shape
  assert action_shape is not None
  for _ in range(cfg.steps):
    if cfg.agent == "random":
      action = torch.randn(action_shape, device=device)
    else:
      action = torch.zeros(action_shape, device=device)
    obs, rew, terminated, truncated, info = env.step(action)
    del obs, rew, terminated, truncated, info

  command = env.command_manager.get_term("motion")
  print("dummy TextOp rollout complete")
  print(command.metrics)


if __name__ == "__main__":
  main()
