Yes, this correction is important. Your earlier plan should be revised around the repo’s **actual online source contract**.

The corrected phase-1 goal is:

```text
Use a scripted square planner to append correctly indexed TextOpMotionBlocks into
the existing live queue source while MJLab play runs normally.
```

Not:

```text
Manually reset the source with a new full motion every phase.
```

That distinction matters.

---

# Updated intention

The long-term intention is still:

```text
MJLab state/image
    ↓
VLM planner
    ↓
high-level prompt: "walk forward" / "turn left" / "stand still"
    ↓
TextOpRobotMDAR or motion provider
    ↓
TextOpMotionBlock stream
    ↓
TextOpTracker/MJLab online command
    ↓
MJLab rollout
```

But phase 1 should validate only this smaller loop:

```text
temporary scripted square planner
    ↓
prompt sequence
    ↓
pre-normalized motion files
    ↓
sliced TextOpMotionBlocks with continuous indices
    ↓
QueueTextOpOnlineSource.append(...)
    ↓
existing MJLab play path
```

So the temporary placeholder is **not a fake VLM**. It is a simple producer that behaves like a future VLM+RobotMDAR producer would behave: it emits prompt-selected motion chunks into the live source.

---

# Big correction: do not invent `online_source.reset(motion)`

The repo already has the right shape:

```text
QueueTextOpOnlineSource
TextOpMotionBlock
append(...)
poll(...)
```

So `play_square.py` should not push a whole “MJLab motion” object into a reset method. Instead, it should append one or more `TextOpMotionBlock`s:

```python
source.append(block)
```

or:

```python
for block in blocks:
    source.append(block)
```

The phase transition should create blocks like:

```text
TextOpMotionBlock(index=global_start_frame, ...)
TextOpMotionBlock(index=global_start_frame + block_size, ...)
...
```

The block index is the timeline identity. That means every new phase must continue from the previous phase’s frame index.

---

# Updated phase-1 architecture

Use two moving pieces:

```text
1. Existing MJLab play runner
   - runs the env
   - consumes live online source
   - tracks whatever blocks are available

2. Square producer
   - owns phase sequence
   - loads/slices normalized motions
   - converts them to TextOpMotionBlock
   - appends blocks to QueueTextOpOnlineSource
```

So instead of writing your own `env.step(action)` loop, phase 1 should try to keep using the existing play infrastructure.

That means the square demo is closer to:

```text
create live QueueTextOpOnlineSource
start/launch existing play with source_mode="live"
producer appends square phases into that source
```

If your current `run_play` entrypoint does not expose the source object directly, then the minimal work is to add a small hook or wrapper that lets `play_square.py` construct and pass the live source.

---

# Updated implementation plan

## Step 1: keep `play_square.py`

Add:

```text
src/mjlab_textop/scripts/play_square.py
```

This script is the temporary phase-1 demo.

It should not become a framework. It should be a concrete demo script with a few helpers.

---

## Step 2: use safer phases

Start with stand gaps, because generated clips may not transition smoothly.

```python
PHASES = [
    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 30),

    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 30),

    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 30),

    ("walk forward", 150),
    ("stand still", 30),
    ("turn left", 90),
    ("stand still", 120),
]
```

These frame counts are temporary tuning knobs.

RobotMDAR does not need to understand distance or turn angle. The square script controls execution length by slicing frames.

---

## Step 3: use normalized MJLab NPZs as inputs

Use already-normalized files:

```text
walk_forward_mjlab.npz
turn_left_mjlab.npz
stand_still_mjlab.npz
```

The phase-1 script should not redo the whole TextOp-to-MJLab normalization path.

It should consume normalized files and convert slices into `TextOpMotionBlock`s using existing online/replay conversion logic.

So the CLI should probably look like:

```python
@dataclass
class PlaySquareCfg:
    walk_motion_file: str
    turn_motion_file: str
    stand_motion_file: str
    block_size: int = 30
    num_envs: int = 1
    device: str = "cuda:0"
```

---

## Step 4: add a small helper around existing replay logic

Do not invent `normalize_motion_for_mjlab`.

Add a helper near the online/replay code, maybe:

```text
src/mjlab_textop/core/online/replay.py
```

or a new small utility if you prefer:

```text
src/mjlab_textop/core/online/blocks.py
```

Function:

```python
def load_sliced_mjlab_npz_blocks(
    path: Path,
    *,
    frames: int,
    start_index: int,
    block_size: int,
) -> list[TextOpMotionBlock]:
    ...
```

Its job:

```text
1. Load normalized MJLab NPZ.
2. Take first `frames` frames.
3. Split into blocks of size `block_size`.
4. Assign continuous global TextOpMotionBlock.index values.
5. Return blocks ready for QueueTextOpOnlineSource.append(...).
```

Conceptually:

```python
blocks = load_sliced_mjlab_npz_blocks(
    path=motion_files[prompt],
    frames=phase_frames,
    start_index=next_frame_index,
    block_size=block_size,
)
```

Then:

```python
for block in blocks:
    source.append(block)

next_frame_index += phase_frames
```

This is the most important correction.

---

# Step 5: maintain `next_frame_index`

The square script should own:

```python
next_frame_index = 0
```

Each phase uses the current value:

```python
blocks = load_sliced_mjlab_npz_blocks(
    path=path,
    frames=frames,
    start_index=next_frame_index,
    block_size=cfg.block_size,
)
```

Then after appending:

```python
next_frame_index += frames
```

Do not restart phase indices at zero.

Wrong:

```text
walk phase index=0
turn phase index=0
walk phase index=0
```

Correct:

```text
walk phase index=0
turn phase index=150
stand phase index=240
walk phase index=270
...
```

This keeps the rolling buffer timeline coherent.

---

# Step 6: do not rely on `remaining_frames()`

Since the live source does not know what the tracker has consumed, do not design around:

```python
online_source.remaining_frames()
```

For phase 1, use a producer-side timer.

Simple approach:

```python
phase_remaining_frames = 0
```

When you append a phase:

```python
phase_remaining_frames = frames
```

Each control tick or simulated step:

```python
phase_remaining_frames -= 1
```

When it reaches a margin:

```python
if phase_remaining_frames <= prefetch_margin:
    append_next_phase()
```

However, if `run_play` owns the stepping loop, then `play_square.py` may not naturally receive one callback per env step. In that case, there are two options.

---

## Option A: append the full square upfront

This is the simplest phase-1 implementation.

The producer does:

```python
next_frame_index = 0

for prompt, frames in PHASES:
    blocks = load_sliced_mjlab_npz_blocks(
        motion_files[prompt],
        frames=frames,
        start_index=next_frame_index,
        block_size=cfg.block_size,
    )
    for block in blocks:
        source.append(block)

    next_frame_index += frames
```

Then `run_play` consumes the full queued square sequence.

This weakens the future “state → choose prompt” boundary, but it is the least invasive way to validate:

```text
multi-phase motion streaming
continuous indices
live source consumption
clip switching
```

For the first smoke test, this is probably acceptable.

You can still call it a temporary scripted producer.

---

## Option B: background producer thread with local timing

A slightly closer version to the VLM loop:

```text
run_play runs normally
background producer appends phases over time
producer sleeps or waits approximately phase duration
```

Pseudo-shape:

```python
def square_producer(source, motion_files, phases, fps, block_size):
    next_frame_index = 0

    for prompt, frames in phases:
        blocks = load_sliced_mjlab_npz_blocks(
            path=motion_files[prompt],
            frames=frames,
            start_index=next_frame_index,
            block_size=block_size,
        )

        for block in blocks:
            source.append(block)

        print(f"[square] appended {prompt!r}, frames={frames}, start={next_frame_index}")

        next_frame_index += frames

        time.sleep(frames / fps)
```

This gives you a live-ish producer without touching the MJLab step loop.

But it is less deterministic than owning env steps, and you need to be careful about sim speed versus wall-clock speed.

---

## Option C: later, add a callback/hook from play loop

This is the best long-term shape:

```text
on every env step:
  build state
  maybe choose next prompt
  append blocks
```

But this requires touching the play runner. I would not start there unless the existing play script already has a clean hook.

---

# Recommended phase-1 choice

For now, I would choose **Option A first**:

```text
append the full square upfront using continuous indices
then run existing play
```

Because it validates the most important technical issue:

```text
Can the existing online/live source consume a planned sequence of TextOpMotionBlocks
that switches between walk/turn/stand clips?
```

Then immediately after that works, move to **Option B**:

```text
background producer appends phases over time
```

Then later, when integrating VLM, move to **Option C** or a proper runner callback.

---

# Step 7: implement the square producer

The concrete helper can be:

```python
def append_square_sequence(
    source: QueueTextOpOnlineSource,
    motion_files: dict[str, Path],
    phases: list[tuple[str, int]],
    *,
    block_size: int,
) -> int:
    next_frame_index = 0

    for phase_index, (prompt, frames) in enumerate(phases):
        blocks = load_sliced_mjlab_npz_blocks(
            path=motion_files[prompt],
            frames=frames,
            start_index=next_frame_index,
            block_size=block_size,
        )

        for block in blocks:
            source.append(block)

        print(
            f"[square] phase={phase_index} "
            f"prompt={prompt!r} "
            f"frames={frames} "
            f"start_index={next_frame_index} "
            f"blocks={len(blocks)}"
        )

        next_frame_index += frames

    return next_frame_index
```

This is the phase-1 heart.

---

# Step 8: wire it into existing play

The script should roughly do:

```python
def play_square(cfg: PlaySquareCfg) -> None:
    source = QueueTextOpOnlineSource(...)

    motion_files = {
        "walk forward": Path(cfg.walk_motion_file),
        "turn left": Path(cfg.turn_motion_file),
        "stand still": Path(cfg.stand_motion_file),
    }

    total_frames = append_square_sequence(
        source=source,
        motion_files=motion_files,
        phases=PHASES,
        block_size=cfg.block_size,
    )

    print(f"[square] queued total_frames={total_frames}")

    run_play_with_live_source(
        source=source,
        source_mode="live",
        ...
    )
```

The exact function name depends on how your current `commands.py`/`play_online.py` is structured. The important repo-level change is:

```text
Make the existing play path accept a QueueTextOpOnlineSource instance,
or add a square-specific source mode that constructs one and preloads it.
```

Avoid writing a separate manual low-level env loop in phase 1.

---

# Step 9: temporary state/image placeholder

Because Option A prequeues the whole square, phase 1 will not yet use MJLab state/images.

That is okay, but be explicit:

```text
Phase 1 does not yet close the visual feedback loop.
It validates the streaming contract and prompt-motion sequencing.
```

You can still leave a placeholder function:

```python
def build_planner_state_placeholder() -> dict:
    return {
        "task": "walk in a square",
        "image": None,
    }
```

But do not overdo it.

The real image feedback comes in the next phase when the producer/planner is called during rollout.

---

# Corrected phased roadmap

## Phase 1A: prequeued square sequence

Goal:

```text
Prove multi-phase TextOpMotionBlock streaming into MJLab.
```

Implementation:

```text
- Load normalized walk/turn/stand NPZs.
- Slice them according to PHASES.
- Convert slices to TextOpMotionBlocks.
- Assign continuous global indices.
- Append all blocks to QueueTextOpOnlineSource.
- Run existing play with source_mode="live".
```

No VLM. No images. No state feedback.

This is the minimal working square demo.

---

## Phase 1B: timed square producer

Goal:

```text
Move from prequeued sequence to event-ish phase appending.
```

Implementation:

```text
- Same PHASES.
- Same block helper.
- Background producer appends one phase at a time.
- Uses local timing or source lag stats.
- Still no VLM.
```

This gets closer to the future online setup.

---

## Phase 2: planner boundary

Goal:

```text
Replace hardcoded sequence with choose_next_prompt(state).
```

Implementation:

```text
- Add a simple state dict.
- State can initially include phase index, current prompt, completed prompts.
- Later add tracking metrics and image.
- Scripted planner still returns square sequence.
```

This is where the code starts to look like the VLM loop.

---

## Phase 3: MJLab image capture

Goal:

```text
Attach camera image to planner state.
```

Implementation:

```text
- Find MJLab/IsaacLab render API.
- Capture one env camera frame.
- Downsample to 224x224 or 336x336.
- Store image in planner state.
- Scripted planner ignores it.
```

This validates image extraction separately from VLM inference.

---

## Phase 4: VLM planner

Goal:

```text
Replace scripted next-prompt function with VLM.
```

Implementation:

```text
- Send state + image to VLM.
- Restrict output to allowed prompts:
  ["walk forward", "turn left", "stand still"].
- Map prompt to frame duration.
- Use same motion-block append path as phase 1.
```

---

## Phase 5: live RobotMDAR

Goal:

```text
Replace offline motion files with RobotMDAR generation.
```

Implementation:

```text
- prompt → RobotMDARClient.generate(prompt)
- generated motion → normalized/sliced block stream
- append to live source
```

This should come after the block streaming path is stable.

---

# Final corrected phase-1 description

The phase-1 implementation should be described like this:

```text
We add a temporary scripted square producer that validates the live TextOpMotionBlock streaming path. It uses a hardcoded sequence of RobotMDAR-supported prompts, loads pre-normalized MJLab motion files for those prompts, slices each prompt to a fixed number of frames, assigns continuous global TextOpMotionBlock indices, and appends the resulting blocks to QueueTextOpOnlineSource. MJLab then consumes the source through the existing play path. This does not yet use the VLM or camera feedback; it is a placeholder to validate the prompt-to-motion-to-tracker loop before replacing the scripted producer with a VLM planner.
```

That is the right, repo-aligned version.
