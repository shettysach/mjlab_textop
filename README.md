# mjlab-textop

Connects TextOp/RobotMDAR motion generation to MJLab for motion
normalization, policy training, replay, and live text-to-motion control.

The project uses two Python environments:

- **MJLab TextOp** runs normalization, training, replay, and simulation.
- **TextOp/RobotMDAR** generates motion records and live motion blocks.

## Env Setups

### MJLab TextOp

Run MJLab commands from this repository. Select either the `cu128` or `cpu`
extra, but not both:

```bash
uv sync --extra cu128
```

### TextOp/RobotMDAR

Create a separate Python 3.10 environment next to this repository:

```bash
mkdir -p ../textop-runtime
cd ../textop-runtime

git clone --no-recurse-submodules https://github.com/TeleHuman/TextOp.git TextOp
git -C TextOp checkout --detach ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22

uv venv --python 3.10
uv pip install git+https://github.com/openai/CLIP.git
uv pip install -e ./TextOp/deps/isaac_utils
uv pip install -e ./TextOp/TextOpRobotMDAR
uv pip install torch torchvision

export PYTHONPATH=/absolute/path/to/mjlab_textop/src
```

Download the RobotMDAR checkpoint, dataset, and G1 assets:

```bash
uvx hf download Yochish/TextOp-Data \
  --repo-type dataset \
  --local-dir /tmp/textop-data \
  --include 'TextOpRobotMDAR/logs/**' \
  --include 'TextOpRobotMDAR/dataset/**' \
  --include 'TextOpRobotMDAR/description/**'
```

Optionally download TextOp's released ONNX policy:

```bash
uvx hf download Yochish/TextOp-Data \
  TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx \
  --repo-type dataset \
  --local-dir /tmp

export ONNX_PATH=/tmp/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx
```

## Offline workflow

### 1. Record RobotMDAR motion

Run this from the TextOp/RobotMDAR environment:

```bash
uv run python -m robotmdar_textop.record \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --prompt "walk" \
  --num-blocks 200 \
  --output /tmp/walk_forward.npz
```

### 2. Normalize the motion

Run this from the MJLab TextOp repository:

```bash
uv run --extra cu128 mjlab-textop normalize \
  --input-motion-file /tmp/walk_forward.npz \
  --output-motion-file ./outputs/walk_forward.npz
```

### 3. Train a tracking policy (needed only for offline)

```bash
uv run --extra cu128 train Mjlab-TextOp-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ./outputs/walk_forward.npz \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.experiment-name textop_tracking \
  --agent.run-name robotmdar_walk_forward \
  --env.commands.motion.anchor-body-name pelvis
```

```bash
export CHECKPOINT=logs/rsl_rl/textop_tracking/YOUR_RUN/model_5000.pt
```

### 4. Replay the motion (offline)

Use MJLab's standard player:

```bash
uv run --extra cu128 play Mjlab-TextOp-Flat-Unitree-G1 \
  --checkpoint-file "${CHECKPOINT}" \
  --motion-file ./outputs/walk_forward.npz
```

### 4. Replay the motion (online)

Use the online reference buffer with a checkpoint or ONNX policy:

```bash
uv run --extra cu128 mjlab-textop play-online \
  --onnx-file "${ONNX_PATH}" \
  --motion-file ./outputs/walk_forward.npz
```

`--checkpoint-file` and `--onnx-file` are mutually exclusive.

## Live workflow

Live control runs a RobotMDAR producer and an MJLab consumer. The VLM planner
also needs an OpenAI-compatible chat server and MJLab observations.

### 1. Start the chat server

Skip this step when using the manual planner.

- OpenAI chat compatible server with serving a single model, eg - llama.cpp or vllm server 

### 2. Start the RobotMDAR producer

Run this from the TextOp/RobotMDAR environment:

```bash
uv run python -m robotmdar_textop.produce \
  --ckpt /tmp/textop-data/TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth \
  --datadir /tmp/textop-data/TextOpRobotMDAR/dataset/PRIVATE-DATA \
  --skeleton-asset-root /tmp/textop-data/TextOpRobotMDAR/description/robots/g1 \
  --planner vlm \
  --prompt "stand" \
  --observation-listen-port 8766 \
  --vlm-base-url http://127.0.0.1:9379 \
  --vlm-model gemma-4-E4B-it
```

For manual control, omit the VLM options and use `--planner manual`.
Task-specific VLM instructions default to `TASK.md`; the invariant and user
prompts live under `src/robotmdar_textop/prompt`.

VLM control uses requested observations. After each bounded command and its
generated `stand` transition, the producer inserts a stationary observation
request and pauses. MJLab responds when it reaches that exact reference frame;
only then does the producer query the VLM and resume. Periodic mode is available
for best-effort monitoring but does not drive this VLM planner. The producer and
consumer must use the same motion and observation protocol version.

### 3. Start MJLab

Run this from the MJLab TextOp repository:

```bash
uv run --extra cu128 mjlab-textop play-live \
  --onnx-file "${ONNX_PATH}" \
  --task straight \
  obs \
  --obs.url http://127.0.0.1:8766/observation \
  --obs.mode requested \
  --obs.image-size 320 240
```

Use `--obs.mode periodic --obs.every-frames 20` for best-effort observations
instead of motion-synchronized requests.

Use `--checkpoint-file "${CHECKPOINT}"` instead of the ONNX options to run a
trained checkpoint. Omit `obs` and its options when observations are not
needed. Add `--ref-vis` before `obs` to display the generated reference motion.

Available live tasks are `straight`, `blocked-straight`, `side-goals`, `turn`,
and `portrait-corridors`. Omit `--task` for the default live environment.

## Command help

```bash
uv run --extra cu128 mjlab-textop --help
uv run --extra cu128 mjlab-textop play-live --help
uv run python -m robotmdar_textop.produce --help
```

See [`notes/COMMANDS.md`](notes/COMMANDS.md) for detailed runtime and tuning
notes.
