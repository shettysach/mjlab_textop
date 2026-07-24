# MJLab TextOp

MJLab TextOp connects TextOp/RobotMDAR motion generation to MJLab for motion
normalization, policy training, offline replay, and live text-to-motion control.

## Source layout

- `src/textop_live_protocol` owns the shared motion and observation contracts.
- `src/robotmdar_textop` owns RobotMDAR generation and prompt planning.
- `src/mjlab_textop` owns MJLab policy execution and simulation integration.
- `src/mjlab_scout` owns the MCP task-inspection service.
- `tasks` stays at the repository root as the experiment and environment layer.

The protocol and RobotMDAR namespaces do not import MJLab. The live workflow
crosses the boundary only through motion blocks and observation messages.

## Workflow

| Stage | Environment | Command | Output |
| --- | --- | --- | --- |
| Record | TextOp/RobotMDAR | `python -m robotmdar_textop.record` | Raw RobotMDAR NPZ |
| Normalize | MJLab TextOp | `mjlab-textop normalize` | MJLab train-ready NPZ |
| Train | MJLab TextOp | `train` | RSL-RL checkpoint |
| Replay | MJLab TextOp | `play` or `mjlab-textop play-online` | Offline simulation |
| Run live | Both environments | `python -m robotmdar_textop.produce` and `mjlab-textop play-live` | Live simulation |

## Environments

The project uses two Python environments:

- The MJLab TextOp environment handles normalization, training, replay, and
  simulation.
- The TextOp/RobotMDAR environment generates raw motion records and live motion
  blocks.

### MJLab TextOp dependencies

MJLab is pinned in `pyproject.toml` to the latest verified upstream `main`
commit. Select exactly one hardware extra when running MJLab commands:

```text
cpu   -> mjlab + torch from pytorch-cpu
cu128 -> mjlab + torch from pytorch-cu128
```

The extras are mutually exclusive, so `uv` rejects selecting CPU and CUDA
dependencies together.

Both extras depend on plain `mjlab`, rather than `mjlab[cpu]` or
`mjlab[cu128]`. This repository is the top-level `uv` project and owns the
Torch wheel selection through `tool.uv.sources`. Pulling MJLab's extras
transitively would cause `uv` to merge CPU and CUDA Torch indexes during lock
resolution.

### TextOp/RobotMDAR environment

Create a dedicated environment outside the `mjlab_textop` repository. TextOp
does not need a persistent checkout when its source will not be modified:

```bash
mkdir -p ../textop-runtime
cd ../textop-runtime

uv venv --python 3.10
uv pip install torch
uv pip install git+https://github.com/openai/CLIP.git
uv pip install "git+https://github.com/TeleHuman/TextOp.git@ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22#subdirectory=deps/isaac_utils"
uv pip install "git+https://github.com/TeleHuman/TextOp.git@ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22#subdirectory=TextOpRobotMDAR"

# Point this Python 3.10 environment at the shared protocol and producer code.
export PYTHONPATH=/absolute/path/to/mjlab_textop/src

uvx hf download Yochish/TextOp-Data \
  --repo-type dataset \
  --local-dir /tmp/textop-data \
  --include 'TextOpRobotMDAR/logs/**' \
  --include 'TextOpRobotMDAR/dataset/**' \
  --include 'TextOpRobotMDAR/description/**'
```

RobotMDAR commands below run in this environment and can be launched from any
directory. `PYTHONPATH` must contain the `src` directory—not the repository
root—so Python can import both `robotmdar_textop` and
`textop_live_protocol`. The MJLab environment does not need this setting;
`uv run` installs the project there.

### ONNX policy

Download TextOp's released `latest.onnx` policy before using an
`--onnx-file` option:

```bash
uvx hf download Yochish/TextOp-Data \
  TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx \
  --repo-type dataset \
  --local-dir /tmp

export ONNX_PATH=/tmp/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx
```

`--checkpoint-file` and `--onnx-file` are mutually exclusive.

## Offline workflow

### 1. Record raw RobotMDAR motion

Generate a raw reference record without starting an MJLab socket consumer:

```bash
# Run from the TextOp/RobotMDAR environment.
uv run python -m robotmdar_textop.record \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --prompt "walk" \
  --num-blocks 200 \
  --output /tmp/walk_forward.npz
```

The raw record stores `joint_pos`, `joint_vel`, `anchor_pos_w`, and
`anchor_quat_w`. Joint arrays remain in TextOp/IsaacLab G1 order.

### 2. Normalize the motion

Convert the raw record into a complete MJLab train-ready NPZ:

```bash
# Run from the mjlab_textop directory.
uv run --extra cu128 mjlab-textop normalize \
  --input-motion-file /tmp/walk_forward.npz \
  --output-motion-file ./outputs/walk_forward.npz
```

Normalization reindexes the joints from TextOp/IsaacLab order into MJLab order
exactly once, uses the raw anchor trajectory as the robot root, runs MJLab
forward kinematics, and saves full MJLab body positions, orientations, and
velocities.

### 3. Train a tracking policy

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/walk_forward.npz \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.experiment-name textop_tracking \
  --agent.run-name robotmdar_walk_forward \
  --env.commands.motion.anchor-body-name pelvis
```

Checkpoints are written under
`logs/rsl_rl/textop_tracking/<timestamp>_<run-name>/`. Set the checkpoint used
by the replay commands:

```bash
export CHECKPOINT=logs/rsl_rl/textop_tracking/2026-06-25_00-20-00_robotmdar_walk_forward/model_5000.pt
```

To fine-tune from an earlier run:

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/stand_still.npz \
  --agent.resume True \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.experiment-name textop_tracking \
  --agent.load-run 2026-06-25_00-20-00_robotmdar_pelvis_scratch \
  --agent.load-checkpoint model_5000.pt \
  --agent.run-name stand_still \
  --env.commands.motion.anchor-body-name pelvis
```

### 4. Replay a checkpoint

Use MJLab's standard player:

```bash
uv run --extra cu128 play Mjlab-TextOp-Flat-Unitree-G1 \
  --checkpoint-file "${CHECKPOINT}" \
  --motion-file ./outputs/walk_forward.npz
```

To exercise the online reference buffer with a recorded motion:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --checkpoint-file "${CHECKPOINT}" \
  --motion-file ./outputs/walk_forward.npz
```

The same replay can use the released ONNX policy:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --onnx-file "${ONNX_PATH}" \
  --motion-file ./outputs/walk_forward.npz
```

ONNX inference uses the CPU provider by default. To run the actor on the MJLab
CUDA device with direct GPU input/output binding:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --onnx-file "${ONNX_PATH}" \
  --onnx-provider cuda \
  --motion-file ./outputs/walk_forward.npz
```

## Live workflow

Live control uses a RobotMDAR producer and an MJLab consumer connected over
localhost NDJSON. When VLM prompt selection is enabled, the producer also
receives MJLab observations over HTTP and queries an OpenAI-compatible chat
server.

### 1. Start a chat server

This component is only required for the VLM planner. With LiteRT-LM:

```bash
uvx litert-lm serve --host 127.0.0.1 --port 9379
```

### 2. Start the RobotMDAR producer

For the default planner:

```bash
# Run from the TextOp/RobotMDAR environment.
uv run python -m robotmdar_textop.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1
```

For VLM prompt selection:

```bash
# Run from the TextOp/RobotMDAR environment.
uv run python -m robotmdar_textop.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --planner vlm \
  --prompt "walk" \
  --observation-listen-port 8766 \
  --vlm-base-url http://127.0.0.1:9379 \
  --vlm-model gemma-4-E4B-it \
  --vlm-history-length 5
```

The invariant controller prompt comes from
[`src/robotmdar_textop/prompt/INVARIANT.md`](src/robotmdar_textop/prompt/INVARIANT.md).
The generated `TASK.md` is appended to it by default, while the per-turn command
list comes from
[`src/robotmdar_textop/prompt/USER.md`](src/robotmdar_textop/prompt/USER.md). Override
the task and user prompt files with `--vlm-system-prompt` and
`--vlm-user-prompt`.

`--vlm-history-length` bounds the number of user-image turns in a request,
including the current turn. Its default of `5` sends four completed
user-image/assistant pairs before the current user-image turn. Use `1` for
stateless requests. Assistant reasoning is preserved when the server returns
it, as required by thinking models such as Gemma 4. Add `--vlm-reasoning` to
print returned reasoning.

### 3. Start the MJLab consumer

Run with a trained checkpoint:

```bash
# Run from the mjlab_textop directory.
uv run --extra cu128 mjlab-textop play-live \
  --checkpoint-file "${CHECKPOINT}" \
  --host 127.0.0.1 \
  --port 8765 \
  observation:observation-params \
  --observation.url http://127.0.0.1:8766/observation \
  --observation.every-frames 20 \
  --observation.image-width 320 \
  --observation.image-height 240
```

Or run with the released ONNX policy:

```bash
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "${ONNX_PATH}" \
  --onnx-provider cuda \
  --host 127.0.0.1 \
  --port 8765 \
  observation:observation-params \
  --observation.url http://127.0.0.1:8766/observation \
  --observation.every-frames 20 \
  --observation.image-width 320 \
  --observation.image-height 240
```

Live observations are disabled by default. The
`observation:observation-params` subcommand enables the HTTP observation
publisher.

### Live scheduling and feedback

The producer sends motion chunks indexed at 50 Hz. MJLab consumes them at the
online command rate, clamps stale future frames during underruns, and exposes
buffer and source diagnostics through command metrics.

The first VLM query starts after the initial motion block is sent. Later queries
run only when a new image is available, and the same image is never queried
twice. Only one request can be in flight; images arriving during inference are
coalesced so the next request uses the newest one. The last selected prompt
remains active between queries.

The `play-live` publisher controls the image cadence with
`--observation.every-frames`. A value of `20` at the 50 Hz reference rate
sends at most 2.5 images per second. There is no separate producer-side
every-N-blocks query option.

MJLab observations are HTTP JSON posts containing a base64 JPEG render. Safety
updates can carry collision-stop state and a recovery epoch without an image.
Collision-only observations do not trigger VLM queries.

Enable `--reference-debug-vis` to render the live RobotMDAR reference as
a translucent ghost beside the simulated robot.

### ONNX runtime behavior

The ONNX path uses the online source and ONNX actor directly, without a `.pt`
checkpoint. The provider is selected with `--onnx-provider`:

- CPU inference copies inputs and outputs through NumPy.
- The `cuda` provider requires a CUDA `--device` and uses strict ONNX Runtime
  I/O binding. Actor observations must already be float32, contiguous, detached
  CUDA tensors on the same device as `--device`.

## Live navigation tasks

The live command supports fixed navigation tasks through `--task`. The
`straight` task uses `Mjlab-VLA-Straight-G1`. The `blocked-straight`
variant uses `Mjlab-VLA-BlockedStraight-G1` and adds a centered obstacle that
requires a left or right bypass.

```bash
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "${ONNX_PATH}" \
  --host 127.0.0.1 \
  --port 8765 \
  --task straight \
  observation:observation-params \
  --observation.url http://127.0.0.1:8766/observation \
  --observation.every-frames 20 \
  --observation.image-width 320 \
  --observation.image-height 240
```

The environment owns success and termination logic while the producer supplies
the motion prompt stream. Start with `stand` or `walk`, depending on the
prompt policy being tested. In the blocked variant, the obstacle is centered
and wide enough that ordinary walking drift does not count as successful
avoidance.

The producer can use task-specific VLM prompts:

```bash
# Run from the TextOp/RobotMDAR environment.
uv run python -m robotmdar_textop.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --planner vlm \
  --prompt "stand" \
  --observation-listen-port 8766 \
  --vlm-base-url http://127.0.0.1:9379 \
  --vlm-model gemma-4-E4B-it \
  --vlm-system-prompt ./sys.md \
  --vlm-user-prompt ./user.md
```
