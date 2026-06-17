import torch

from mjlab_textop_playground.reference import DummyTextReferenceProvider, RobotState


def test_dummy_provider_maps_texts_to_expected_velocities():
  texts = ["stand still", "walk forward", "turn left", "sidestep right"]
  n = len(texts)
  state = RobotState(
    root_pos=torch.zeros(n, 3),
    root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
    root_lin_vel=torch.zeros(n, 3),
    root_ang_vel=torch.zeros(n, 3),
    joint_pos=torch.zeros(n, 29),
    joint_vel=torch.zeros(n, 29),
  )

  ref = DummyTextReferenceProvider().generate(texts, state, horizon=6, dt=0.02)

  assert torch.allclose(ref.root_lin_vel[0, 0], torch.tensor([0.0, 0.0, 0.0]))
  assert torch.allclose(ref.root_lin_vel[1, 0], torch.tensor([0.4, 0.0, 0.0]))
  assert torch.allclose(ref.root_ang_vel[2, 0], torch.tensor([0.0, 0.0, 0.6]))
  assert torch.allclose(ref.root_lin_vel[3, 0], torch.tensor([0.0, -0.3, 0.0]))
  assert ref.valid is not None
  assert ref.valid.all()
