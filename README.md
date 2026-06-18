# MJLab VLA

Utilities for running low-level TextOp tracker motions through MJLab's native
Unitree G1 tracking stack.

The current integration boundary is deliberately narrow: convert a canonical
TextOp tracker NPZ into MJLab's native tracking NPZ format, then use MJLab's
existing `MotionCommand` task, rewards, metrics, and play/train/evaluate flow.

## Architecture

```text
TextOp tracker NPZ
  -> normalize-textop-npz
  -> MJLab-native motion.npz
  -> Mjlab-Tracking-Flat-Unitree-G1
  -> MJLab MotionCommand
```

## Dependencies

This package uses upstream MJLab pinned to the latest verified `main` commit.
Dependency selection follows MJLab's upstream uv extras pattern:

```text
cpu   -> mjlab + torch from pytorch-cpu
cu128 -> mjlab + torch from pytorch-cu128
```

Use exactly one extra at a time. For local CPU verification. `pyproject.toml` declares the extras as conflicting, so uv rejects selecting
both CPU and CUDA dependencies in the same environment.

The extras depend on plain `mjlab`, not `mjlab[cpu]` or `mjlab[cu128]`. This
repo is the top-level uv project, so it owns the torch wheel selection through
`tool.uv.sources`. Pulling MJLab's own extras transitively causes uv to merge
CPU and CUDA torch indexes during lock resolution.

## Commands

For low-level TextOp tracker motions, normalize the TextOp NPZ into MJLab's
native tracking format first:

```bash
uv run --extra cpu normalize-textop-npz \
  --input-file /path/to/textop_motion.npz \
  --output-file /tmp/textop_mjlab_motion.npz \
  --device cpu
```

Then use MJLab's built-in G1 tracking task and `MotionCommand`:

```bash
uv run --extra cpu play Mjlab-Tracking-Flat-Unitree-G1 \
  --agent zero \
  --motion-file /tmp/textop_mjlab_motion.npz \
  --num-envs 1 \
  --no-terminations True
  --viewer viser \
```

The normalizer expects TextOp's canonical tracker NPZ fields. It reorders
TextOp IsaacLab G1 joints into MJLab/MuJoCo order and replays root plus joints
through MJLab so body references are written in MJLab's own body order.

For the downloaded TextOp walking motion on a GPU machine, train an MJLab
tracking policy from scratch:

```bash
uv run --extra cu128 train-textop-motion
```

Useful overrides:

```bash
uv run --extra cu128 train-textop-motion \
  --train-num-envs 8192 \
  --max-iterations 30000 \
  --run-name walk_scratch_long
```

To finetune from a previous MJLab run:

```bash
uv run --extra cu128 train-textop-motion \
  --resume \
  --load-run '.*walk_scratch.*' \
  --load-checkpoint 'model_.*.pt' \
  --run-name walk_finetune
```

To view a trained MJLab checkpoint:

```bash
uv run --extra cu128 train-textop-motion \
  --mode play \
  --checkpoint-file /path/to/model.pt
```

To only download and normalize:

```bash
uv run --extra cu128 train-textop-motion --mode normalize
```

To print the MJLab command without running it:

```bash
uv run --extra cu128 train-textop-motion --skip-download --skip-normalize --dry-run
```
