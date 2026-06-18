from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import tyro

from mjlab_vla.scripts.normalize_textop_npz import normalize_textop_npz

DEFAULT_MOTION_REL = (
    "TextOpTracker/artifacts/Data10k-open/"
    "homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/"
    "motion.npz"
)


def run_textop_motion(
    mode: Literal["train", "play", "normalize"] = "train",
    repo_id: str = "Yochish/TextOp-Data",
    motion_rel: str = DEFAULT_MOTION_REL,
    data_dir: str = "/tmp/textop-data",
    output_file: str = "/tmp/textop_walk_mjlab.npz",
    device: str = "cuda:0",
    extra: Literal["cpu", "cu128"] = "cu128",
    train_num_envs: int = 4096,
    max_iterations: int = 10000,
    logger: Literal["tensorboard", "wandb"] = "tensorboard",
    experiment_name: str = "textop_tracking",
    run_name: str = "walk_scratch",
    resume: bool = False,
    load_run: str = ".*",
    load_checkpoint: str = "model_.*.pt",
    play_num_envs: int = 1,
    viewer: Literal["auto", "native", "viser"] = "viser",
    checkpoint_file: str | None = None,
    skip_download: bool = False,
    skip_normalize: bool = False,
    dry_run: bool = False,
) -> None:
    """Download, normalize, and train/play an MJLab policy on a TextOp motion."""

    data_path = Path(data_dir).expanduser()
    input_file = data_path / motion_rel
    normalized_file = Path(output_file).expanduser()

    if not skip_download:
        _run(
            [
                "uvx",
                "hf",
                "download",
                repo_id,
                "--repo-type",
                "dataset",
                "--include",
                motion_rel,
                "--local-dir",
                str(data_path),
            ],
            dry_run=dry_run,
        )

    if not skip_normalize:
        if dry_run:
            print(
                "DRY RUN:",
                "normalize_textop_npz",
                f"--input-file={input_file}",
                f"--output-file={normalized_file}",
                f"--device={device}",
            )
        else:
            normalize_textop_npz(
                input_file=str(input_file),
                output_file=str(normalized_file),
                device=device,
            )

    if mode == "normalize":
        return

    if mode == "train":
        command = [
            "uv",
            "run",
            "--extra",
            extra,
            "train",
            "Mjlab-Tracking-Flat-Unitree-G1",
            "--env.commands.motion.motion-file",
            str(normalized_file),
            "--env.scene.num-envs",
            str(train_num_envs),
            "--agent.max-iterations",
            str(max_iterations),
            "--agent.logger",
            logger,
            "--agent.experiment-name",
            experiment_name,
            "--agent.run-name",
            run_name,
        ]
        if resume:
            command.extend(
                [
                    "--agent.resume",
                    "True",
                    "--agent.load-run",
                    load_run,
                    "--agent.load-checkpoint",
                    load_checkpoint,
                ]
            )
        _run(command, dry_run=dry_run)
        return

    if checkpoint_file is None:
        raise ValueError("`checkpoint_file` is required when mode='play'")
    _run(
        [
            "uv",
            "run",
            "--extra",
            extra,
            "play",
            "Mjlab-Tracking-Flat-Unitree-G1",
            "--agent",
            "trained",
            "--checkpoint-file",
            checkpoint_file,
            "--motion-file",
            str(normalized_file),
            "--num-envs",
            str(play_num_envs),
            "--device",
            device,
            "--viewer",
            viewer,
        ],
        dry_run=dry_run,
    )


def _run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    tyro.cli(run_textop_motion)


if __name__ == "__main__":
    main()
