from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from mjlab_textop_playground.reference import (
  DummyTextReferenceProvider,
  MotionReference,
  RobotState,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class MotionReferenceCommand(CommandTerm):
  """Per-env short-horizon motion-reference command.

  V1 uses a dummy text provider by default, but the command is intentionally
  provider-agnostic: external code can call ``update_reference`` with references
  from TextOp, Kimodo, stored clips, or another Action Expert.
  """

  cfg: MotionReferenceCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionReferenceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.provider = DummyTextReferenceProvider()

    self.cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.needs_update = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.reference_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.texts = list(cfg.default_texts)
    if len(self.texts) == 0:
      self.texts = ["stand still"] * self.num_envs
    if len(self.texts) < self.num_envs:
      repeats = (self.num_envs + len(self.texts) - 1) // len(self.texts)
      self.texts = (self.texts * repeats)[: self.num_envs]
    else:
      self.texts = self.texts[: self.num_envs]

    dofs = self.robot.data.joint_pos.shape[1]
    bodies = len(cfg.body_names)
    h = cfg.horizon
    self.root_pos_buffer = torch.zeros(self.num_envs, h, 3, device=self.device)
    self.root_quat_buffer = torch.zeros(self.num_envs, h, 4, device=self.device)
    self.root_quat_buffer[..., 0] = 1.0
    self.root_lin_vel_buffer = torch.zeros(self.num_envs, h, 3, device=self.device)
    self.root_ang_vel_buffer = torch.zeros(self.num_envs, h, 3, device=self.device)
    self.joint_pos_buffer = torch.zeros(self.num_envs, h, dofs, device=self.device)
    self.joint_vel_buffer = torch.zeros(self.num_envs, h, dofs, device=self.device)
    self.valid_buffer = torch.ones(self.num_envs, h, dtype=torch.bool, device=self.device)
    self.phase_buffer = torch.zeros(self.num_envs, h, device=self.device)

    self.body_pos_buffer = torch.zeros(self.num_envs, h, bodies, 3, device=self.device)
    self.body_quat_buffer = torch.zeros(self.num_envs, h, bodies, 4, device=self.device)
    self.body_quat_buffer[..., 0] = 1.0
    self.body_lin_vel_buffer = torch.zeros(self.num_envs, h, bodies, 3, device=self.device)
    self.body_ang_vel_buffer = torch.zeros(self.num_envs, h, bodies, 3, device=self.device)
    self.has_body_reference = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
      if len(cfg.body_names) > 0
      else [],
      dtype=torch.long,
      device=self.device,
    )
    self.body_pos_relative_w = torch.zeros(self.num_envs, bodies, 3, device=self.device)
    self.body_quat_relative_w = torch.zeros(self.num_envs, bodies, 4, device=self.device)
    self.body_quat_relative_w[..., 0] = 1.0

    for name in [
      "error_root_pos",
      "error_root_rot",
      "error_root_lin_vel",
      "error_root_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "reference_age",
    ]:
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  def compute(self, dt: float) -> None:
    if dt == 0.0:
      self._update_metrics()
      self.update_relative_body_poses()
      return
    super().compute(dt)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self.joint_pos,
        self.joint_vel,
        self.root_lin_vel_w,
        self.root_ang_vel_w,
        self.phase[:, None],
      ],
      dim=-1,
    )

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._at_cursor(self.joint_pos_buffer)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._at_cursor(self.joint_vel_buffer)

  @property
  def root_pos_w(self) -> torch.Tensor:
    return self._at_cursor(self.root_pos_buffer) + self._env.scene.env_origins

  @property
  def root_quat_w(self) -> torch.Tensor:
    return self._at_cursor(self.root_quat_buffer)

  @property
  def root_lin_vel_w(self) -> torch.Tensor:
    return self._at_cursor(self.root_lin_vel_buffer)

  @property
  def root_ang_vel_w(self) -> torch.Tensor:
    return self._at_cursor(self.root_ang_vel_buffer)

  @property
  def phase(self) -> torch.Tensor:
    return self._at_cursor(self.phase_buffer)

  @property
  def valid(self) -> torch.Tensor:
    return self._at_cursor(self.valid_buffer)

  @property
  def reference_exhausted(self) -> torch.Tensor:
    return (self.cursor >= self.cfg.horizon - 1) | ~self.valid

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.root_pos_w

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.root_quat_w

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.root_lin_vel_w

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.root_ang_vel_w

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_root_pos_w(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w

  @property
  def robot_root_quat_w(self) -> torch.Tensor:
    return self.robot.data.root_link_quat_w

  @property
  def robot_root_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.root_link_lin_vel_w

  @property
  def robot_root_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.root_link_ang_vel_w

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot_root_pos_w

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot_root_quat_w

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot_root_lin_vel_w

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot_root_ang_vel_w

  @property
  def body_pos_w(self) -> torch.Tensor:
    if self.body_pos_buffer.shape[2] == 0:
      return self.body_pos_buffer[:, 0]
    return self._at_cursor(self.body_pos_buffer) + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._at_cursor(self.body_quat_buffer)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._at_cursor(self.body_lin_vel_buffer)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._at_cursor(self.body_ang_vel_buffer)

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  def future_root_pos_w(self, steps: int | None = None) -> torch.Tensor:
    future = self._future(self.root_pos_buffer, steps)
    return future + self._env.scene.env_origins[:, None, :]

  def future_root_quat_w(self, steps: int | None = None) -> torch.Tensor:
    return self._future(self.root_quat_buffer, steps)

  def future_joint_pos(self, steps: int | None = None) -> torch.Tensor:
    return self._future(self.joint_pos_buffer, steps)

  def future_joint_vel(self, steps: int | None = None) -> torch.Tensor:
    return self._future(self.joint_vel_buffer, steps)

  def envs_that_need_reference(self) -> torch.Tensor:
    return torch.nonzero(self.needs_update, as_tuple=False).flatten()

  def set_text(self, env_ids: torch.Tensor, texts: Sequence[str]) -> None:
    if len(env_ids) != len(texts):
      raise ValueError("env_ids and texts must have the same length")
    for env_id, text in zip(env_ids.tolist(), texts, strict=True):
      self.texts[int(env_id)] = text
    self.needs_update[env_ids] = True

  def update_reference(self, env_ids: torch.Tensor, reference: MotionReference) -> None:
    if len(env_ids) == 0:
      return
    reference.validate()
    if reference.num_envs != len(env_ids):
      raise ValueError(
        f"reference has {reference.num_envs} envs but env_ids has {len(env_ids)}"
      )
    if reference.horizon < self.cfg.horizon:
      raise ValueError(
        f"reference horizon {reference.horizon} must be >= {self.cfg.horizon}"
      )
    if reference.device != torch.device(self.device):
      raise ValueError(f"reference device {reference.device} != command device {self.device}")

    h = self.cfg.horizon
    self.root_pos_buffer[env_ids] = reference.root_pos[:, :h]
    self.root_quat_buffer[env_ids] = reference.root_quat[:, :h]
    self.root_lin_vel_buffer[env_ids] = reference.root_lin_vel[:, :h]
    self.root_ang_vel_buffer[env_ids] = reference.root_ang_vel[:, :h]
    self.joint_pos_buffer[env_ids] = reference.joint_pos[:, :h]
    self.joint_vel_buffer[env_ids] = reference.joint_vel[:, :h]
    self.valid_buffer[env_ids] = (
      reference.valid[:, :h]
      if reference.valid is not None
      else torch.ones(len(env_ids), h, dtype=torch.bool, device=self.device)
    )
    self.phase_buffer[env_ids] = (
      reference.phase[:, :h]
      if reference.phase is not None
      else torch.linspace(0.0, 1.0, h, device=self.device)[None, :]
    )

    if reference.body_pos is not None and reference.body_quat is not None:
      if reference.body_pos.shape[2] != self.body_pos_buffer.shape[2]:
        raise ValueError("body reference count does not match command body_names")
      self.body_pos_buffer[env_ids] = reference.body_pos[:, :h]
      self.body_quat_buffer[env_ids] = reference.body_quat[:, :h]
      if reference.body_lin_vel is not None:
        self.body_lin_vel_buffer[env_ids] = reference.body_lin_vel[:, :h]
      if reference.body_ang_vel is not None:
        self.body_ang_vel_buffer[env_ids] = reference.body_ang_vel[:, :h]
      self.has_body_reference[env_ids] = True
    else:
      self.has_body_reference[env_ids] = False

    self.cursor[env_ids] = 0
    self.reference_age[env_ids] = 0
    self.needs_update[env_ids] = False
    self.update_relative_body_poses()

  def _update_metrics(self) -> None:
    self.metrics["error_root_pos"] = torch.norm(
      self.root_pos_w - self.robot_root_pos_w, dim=-1
    )
    self.metrics["error_root_rot"] = quat_error_magnitude(
      self.root_quat_w, self.robot_root_quat_w
    )
    self.metrics["error_root_lin_vel"] = torch.norm(
      self.root_lin_vel_w - self.robot_root_lin_vel_w, dim=-1
    )
    self.metrics["error_root_ang_vel"] = torch.norm(
      self.root_ang_vel_w - self.robot_root_ang_vel_w, dim=-1
    )
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )
    self.metrics["reference_age"] = self.reference_age.float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    robot_state = self._robot_state(env_ids)
    reference = self.provider.generate(
      [self.texts[int(i)] for i in env_ids],
      robot_state,
      horizon=self.cfg.horizon,
      dt=self._env.step_dt,
    )
    self.update_reference(env_ids, reference)
    if self.cfg.reset_to_reference:
      self._write_reference_state_to_sim(env_ids)

  def _update_command(self) -> None:
    self.reference_age += 1
    self.cursor = torch.clamp(self.cursor + 1, max=self.cfg.horizon - 1)
    self.needs_update |= self.cursor >= self.cfg.horizon - self.cfg.refresh_margin
    self.needs_update |= ~self.valid

    exhausted = torch.nonzero(self.reference_exhausted, as_tuple=False).flatten()
    if exhausted.numel() > 0 and self.cfg.auto_refresh:
      self._resample_command(exhausted)
    self.update_relative_body_poses()

  def update_relative_body_poses(self) -> None:
    if self.body_pos_buffer.shape[2] == 0:
      return
    anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
    anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
    robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    delta_pos_w = robot_anchor_pos_w
    delta_pos_w[..., 2] = anchor_pos_w[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))
    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w
    )

  def _robot_state(self, env_ids: torch.Tensor) -> RobotState:
    return RobotState(
      root_pos=self.robot.data.root_link_pos_w[env_ids] - self._env.scene.env_origins[env_ids],
      root_quat=self.robot.data.root_link_quat_w[env_ids],
      root_lin_vel=self.robot.data.root_link_lin_vel_w[env_ids],
      root_ang_vel=self.robot.data.root_link_ang_vel_w[env_ids],
      joint_pos=self.robot.data.joint_pos[env_ids],
      joint_vel=self.robot.data.joint_vel[env_ids],
    )

  def _write_reference_state_to_sim(self, env_ids: torch.Tensor) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(self.joint_pos[env_ids], soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, self.joint_vel[env_ids], env_ids=env_ids)
    root_state = torch.cat(
      [
        self.root_pos_w[env_ids],
        self.root_quat_w[env_ids],
        self.root_lin_vel_w[env_ids],
        self.root_ang_vel_w[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _at_cursor(self, buffer: torch.Tensor) -> torch.Tensor:
    return buffer[torch.arange(self.num_envs, device=self.device), self.cursor]

  def _future(self, buffer: torch.Tensor, steps: int | None = None) -> torch.Tensor:
    steps = self.cfg.future_steps if steps is None else steps
    offsets = torch.arange(steps, device=self.device)
    indices = torch.clamp(self.cursor[:, None] + offsets[None, :], max=self.cfg.horizon - 1)
    return buffer[torch.arange(self.num_envs, device=self.device)[:, None], indices]


@dataclass(kw_only=True)
class MotionReferenceCommandCfg(CommandTermCfg):
  entity_name: str
  horizon: int = 32
  future_steps: int = 5
  refresh_margin: int = 8
  reset_to_reference: bool = True
  auto_refresh: bool = True
  default_texts: tuple[str, ...] = (
    "stand still",
    "walk forward",
    "turn left",
    "sidestep right",
  )
  body_names: tuple[str, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionReferenceCommand:
    return MotionReferenceCommand(self, env)
