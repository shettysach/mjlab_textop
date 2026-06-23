from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

from mjlab_vla.textop.task import TEXTOP_TASK_NAME, ensure_textop_task_registered


@dataclass(kw_only=True)
class TrainCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    num_envs: int = 4096
    max_iterations: int = 10000
    logger: Literal["tensorboard", "wandb"] = "wandb"
    experiment_name: str = "textop_tracking"
    run_name: str = "walk_scratch"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


def train_textop_motion(
    cfg: TrainCommand,
    *,
    motion_file: Path,
) -> None:
    ensure_textop_task_registered()
    train_cfg = TrainConfig.from_task(TEXTOP_TASK_NAME)
    motion_cmd = train_cfg.env.commands["motion"]
    if not isinstance(motion_cmd, MotionCommandCfg):
        raise TypeError(
            "Expected env_cfg.commands['motion'] to be a MotionCommandCfg, "
            f"got {type(motion_cmd).__name__}"
        )
    motion_cmd.motion_file = str(motion_file)

    train_cfg.env.scene.num_envs = cfg.num_envs
    train_cfg.agent.max_iterations = cfg.max_iterations
    train_cfg.agent.logger = cfg.logger
    train_cfg.agent.experiment_name = cfg.experiment_name
    train_cfg.agent.run_name = cfg.run_name
    train_cfg.agent.resume = cfg.resume
    train_cfg.agent.load_run = cfg.load_run
    train_cfg.agent.load_checkpoint = cfg.load_checkpoint

    launch_training(TEXTOP_TASK_NAME, train_cfg)
