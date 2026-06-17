from __future__ import annotations

import torch

from mjlab_textop_playground.reference import DummyTextReferenceProvider, RobotState


def main() -> None:
  texts = ["stand still", "walk forward", "turn left", "sidestep right"]
  n = len(texts)
  dofs = 29
  state = RobotState(
    root_pos=torch.zeros(n, 3),
    root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
    root_lin_vel=torch.zeros(n, 3),
    root_ang_vel=torch.zeros(n, 3),
    joint_pos=torch.zeros(n, dofs),
    joint_vel=torch.zeros(n, dofs),
  )
  ref = DummyTextReferenceProvider().generate(texts, state, horizon=8, dt=0.02)
  print(f"root_pos: {tuple(ref.root_pos.shape)}")
  print(f"joint_pos: {tuple(ref.joint_pos.shape)}")
  print(f"root_lin_vel[:, 0]: {ref.root_lin_vel[:, 0].tolist()}")
  print(f"root_ang_vel[:, 0]: {ref.root_ang_vel[:, 0].tolist()}")
