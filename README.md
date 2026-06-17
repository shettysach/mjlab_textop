# MJLab TextOp Playground

External MJLab task package for TextOp-style short-horizon motion-reference
tracking on Unitree G1.

The first milestone is deliberately narrow: prove that MJLab can consume
per-environment `MotionReference` buffers from a dummy text provider and run a
G1 tracking rollout. Real TextOp, Kimodo, Gemma, vLLM, residual PPO, and final
motion quality are later milestones.

## Architecture

```text
text / action expert
  -> MotionReferenceProvider
  -> MotionReference
  -> MotionReferenceCommand
  -> MJLab G1 tracking task
  -> rollout / render / policy
```

## Dependencies

This package uses upstream MJLab pinned to the latest verified `main` commit:

```text
0cdc56246999409b83622764f5b38edb660cf16e
```

It does not modify or depend on `../repos/mjlab`.

Dependency selection follows MJLab's upstream uv extras pattern:

```text
cpu   -> mjlab + torch from pytorch-cpu
cu128 -> mjlab + torch from pytorch-cu128
```

Use exactly one extra at a time. For local CPU verification:

```bash
uv sync --extra cpu
```

For the GPU machine:

```bash
uv sync --extra cu128
```

`pyproject.toml` declares the extras as conflicting, so uv rejects selecting
both CPU and CUDA dependencies in the same environment.

The extras depend on plain `mjlab`, not `mjlab[cpu]` or `mjlab[cu128]`. This
repo is the top-level uv project, so it owns the torch wheel selection through
`tool.uv.sources`. Pulling MJLab's own extras transitively causes uv to merge
CPU and CUDA torch indexes during lock resolution.

## Registered Task

The package registers:

```text
Mjlab-TextOpTracking-Flat-Unitree-G1
```

## Local shell

Use the Nix shell through `direnv` before running MJLab commands:

```bash
direnv reload
```

The shell defaults MuJoCo to EGL with Mesa `llvmpipe` so local verification can
run without CUDA:

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

On a GPU machine, keep the same Python code and use `--extra cu128`. If the
vendor EGL stack is available, override the shell variables or remove the Mesa
software-driver hints.

## Commands

List the registered task locally:

```bash
uv run --extra cpu list-envs --keyword TextOp
```

Expected table entry:

```text
Mjlab-TextOpTracking-Flat-Unitree-G1
```

MJLab's `list-envs` command returns the number of matched tasks through
`tyro`, so one match exits with status `1`. For registration checking, the
table entry is the signal.

Inspect dummy references without creating an MJLab environment:

```bash
uv run --extra cpu inspect-reference --text "walk forward"
```

Run the dummy MJLab rollout:

```bash
uv run --extra cpu demo-dummy-textop --steps 200 --num-envs 4
```

The dummy provider maps:

```text
stand still
walk forward
turn left
sidestep right
```

to simple short-horizon root/joint references.

On the GPU machine, use the same commands with `--extra cu128`.
