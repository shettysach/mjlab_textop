Yes. Before adding `play-online`, I would first rearrange the repo so the boundaries are clean.

Right now the repo has the right ingredients, but the layout is slightly misleading:

```text
textop/script/motion.py      # actually reusable TextOp motion IO
textop/script/normalize.py   # actually normalization implementation + CLI config
textop/online.py             # online block/source/buffer all in one file
textop/mdp/offline_commands.py
textop/mdp/online_commands.py
textop/task.py
```

The issue is that `script/` contains core logic, while `online.py` contains several concepts that will grow once you add replay sources, ROS2, and RobotMDAR adapters. The refactor should make this distinction explicit:

```text
core reusable logic
offline TextOp tracker path
online TextOp stream path
CLI wrappers
task registration
```

---

# Target layout

I would aim for this structure:

```text
src/mjlab_vla/
  __init__.py

  tracking.py                         # maybe later rename/offline-only

  textop/
    __init__.py
    contract.py                       # shared constants: joint names, indices, future_steps

    io/
      __init__.py
      motion.py                       # TextOpMotion, load_textop_motion, reindex helpers
      normalized.py                   # validate/load MJLab-normalized motion npz, optional

    offline/
      __init__.py
      normalize.py                    # normalize_textop_npz implementation
      command.py                      # optional move from mdp/offline_commands.py later

    online/
      __init__.py
      block.py                        # TextOpMotionBlock
      source.py                       # TextOpOnlineSource, Queue/List sources
      buffer.py                       # TextOpRollingMotionBuffer
      errors.py                       # OnlineTextOpWarmupError, OnlineTextOpUnderrunError
      replay.py                       # later: NPZ/block replay source helpers

    mdp/
      __init__.py
      future_reference.py             # TextOpFutureReferenceCommand Protocol
      observations.py                 # shared obs terms
      offline_commands.py             # keep initially, or move to offline/command.py
      online_commands.py              # keep initially, or move to online/command.py

    script/
      __init__.py
      cli.py                          # tyro command union + path resolution only
      normalize.py                    # NormalizeCommand wrapper only
      train.py
      play.py
      eval.py
      play_online.py                  # later

    task.py                           # task config/registration
```

This is not a huge rewrite. It is mostly moving files and updating imports.

---

# Main rule for the rearrangement

Use this rule:

```text
If it is reusable by tests, online, offline, or ROS later:
  it should not live in script/.

If it is only a tyro command dataclass or launch wrapper:
  it can live in script/.
```

So `script/cli.py`, `script/train.py`, `script/play.py`, and `script/eval.py` are fine as scripts. But `script/motion.py` and the core part of `script/normalize.py` should move.

---

# Phase 1: move TextOp motion IO out of `script/`

## Current

You currently have reusable motion loading logic in:

```text
src/mjlab_vla/textop/script/motion.py
```

It defines `TextOpMotion`, `load_textop_motion`, `reindex_textop_g1_joints_to_mjlab`, FPS validation, joint/body validation, quaternion normalization, and finite-difference fallback. That is core library code, not script code.

## Move to

```text
src/mjlab_vla/textop/io/motion.py
```

Create:

```text
src/mjlab_vla/textop/io/__init__.py
```

with exports:

```python
from mjlab_vla.textop.io.motion import (
    TextOpMotion,
    load_textop_motion,
    reindex_textop_g1_joints_to_mjlab,
)

__all__ = [
    "TextOpMotion",
    "load_textop_motion",
    "reindex_textop_g1_joints_to_mjlab",
]
```

## Update imports

Change:

```python
from mjlab_vla.textop.script.motion import load_textop_motion
```

to:

```python
from mjlab_vla.textop.io import load_textop_motion
```

The main file affected is `script/normalize.py`, which currently imports `load_textop_motion` from `script.motion`.

## After this phase

Delete:

```text
src/mjlab_vla/textop/script/motion.py
```

or leave a temporary compatibility import if you want a low-risk transition:

```python
from mjlab_vla.textop.io.motion import *  # temporary compatibility shim
```

But I would delete it if nothing external imports it.

---

# Phase 2: split normalize command wrapper from normalize implementation

## Current

`script/normalize.py` contains both:

```text
NormalizeCommand dataclass
normalize_textop_npz implementation
_validate_normalized_output
_append_frame
```

The implementation is substantial: it creates an MJLab scene, writes root/joint state into sim, forwards the sim, logs body/joint arrays, saves NPZ, and validates the output.

That should be offline implementation code, not script code.

## Move implementation to

```text
src/mjlab_vla/textop/offline/normalize.py
```

Move these functions there:

```text
normalize_textop_npz
_append_frame
_validate_normalized_output
```

Keep `DEFAULT_MOTION_REL` either in `script/normalize.py` or move it to offline normalize. I prefer keeping command defaults in `script/normalize.py`, because it is a CLI default, not a core algorithm default.

Create:

```text
src/mjlab_vla/textop/offline/__init__.py
```

with:

```python
from mjlab_vla.textop.offline.normalize import normalize_textop_npz

__all__ = ["normalize_textop_npz"]
```

## Keep wrapper in `script/normalize.py`

After refactor, `script/normalize.py` should be tiny:

```python
from __future__ import annotations

from dataclasses import dataclass

from mjlab_vla.textop.offline import normalize_textop_npz

DEFAULT_MOTION_REL = (
    "TextOpTracker/artifacts/Data10k-open/"
    "homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/"
    "motion.npz"
)


@dataclass(kw_only=True)
class NormalizeCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    motion_rel: str = DEFAULT_MOTION_REL
    data_dir: str = "/tmp/textop-data"
    device: str = "cuda:0"


__all__ = ["NormalizeCommand", "normalize_textop_npz"]
```

This preserves current CLI behavior because `cli.py` imports both `NormalizeCommand` and `normalize_textop_npz` from `script.normalize`.

The only difference is that `script.normalize.normalize_textop_npz` is now re-exported from `offline.normalize`.

---

# Phase 3: make online a package, not one file

## Current

You have one file:

```text
src/mjlab_vla/textop/online.py
```

It contains:

```text
TextOpMotionBlock
TextOpOnlineSource
QueueTextOpOnlineSource
TextOpRollingMotionBuffer
validation helpers
joint reindex helper
tensor conversion helper
```

That is okay at 213 lines, but it will become messy once you add replay sources and ROS adapters.

## Target

Replace the file with a package:

```text
src/mjlab_vla/textop/online/
  __init__.py
  block.py
  source.py
  buffer.py
  errors.py
```

### `online/block.py`

Move:

```python
@dataclass(frozen=True)
class TextOpMotionBlock:
    ...
```

from current `online.py`.

Optionally add a method:

```python
def validate(self) -> None:
    ...
```

or keep validation in the buffer for now.

### `online/source.py`

Move:

```python
class TextOpOnlineSource(Protocol): ...
class QueueTextOpOnlineSource: ...
```

from current `online.py`.

Add a `ListTextOpOnlineSource` if you want clearer test/replay semantics:

```python
class ListTextOpOnlineSource(QueueTextOpOnlineSource):
    pass
```

or just keep `QueueTextOpOnlineSource`.

### `online/buffer.py`

Move:

```python
class TextOpRollingMotionBuffer:
    ...
```

and its validation helpers:

```text
_validate_joint_array
_validate_anchor_array
_normalize_quat
_reindex_textop_joints_to_mjlab
_to_tensor
```

from current `online.py`.

### `online/errors.py`

Add explicit errors:

```python
class OnlineTextOpError(RuntimeError):
    pass

class OnlineTextOpWarmupError(OnlineTextOpError):
    pass

class OnlineTextOpUnderrunError(OnlineTextOpError):
    pass
```

Then later `online_commands.py` can raise these instead of generic `RuntimeError`.

### `online/__init__.py`

Re-export the public API so existing imports keep working:

```python
from mjlab_vla.textop.online.block import TextOpMotionBlock
from mjlab_vla.textop.online.buffer import TextOpRollingMotionBuffer
from mjlab_vla.textop.online.source import QueueTextOpOnlineSource, TextOpOnlineSource
from mjlab_vla.textop.online.errors import (
    OnlineTextOpError,
    OnlineTextOpWarmupError,
    OnlineTextOpUnderrunError,
)

__all__ = [
    "TextOpMotionBlock",
    "TextOpRollingMotionBuffer",
    "TextOpOnlineSource",
    "QueueTextOpOnlineSource",
    "OnlineTextOpError",
    "OnlineTextOpWarmupError",
    "OnlineTextOpUnderrunError",
]
```

This is important because current tests and `online_commands.py` import from `mjlab_vla.textop.online`.

With the re-exporting `__init__.py`, those imports do not need to change.

## Important git detail

You cannot have both:

```text
textop/online.py
textop/online/
```

as import targets in a clean way. So the refactor is:

```text
delete/rename online.py
create online/ package
```

In Git terms:

```bash
mkdir -p src/mjlab_vla/textop/online
git mv src/mjlab_vla/textop/online.py src/mjlab_vla/textop/online/__init__.py
```

Then split content into submodules and leave only re-exports in `__init__.py`.

---

# Phase 4: decide where offline command lives

You currently have:

```text
src/mjlab_vla/textop/mdp/offline_commands.py
src/mjlab_vla/textop/mdp/online_commands.py
```

This is acceptable. In fact, I would **not** move these immediately, because they are MJLab command terms and belong close to MDP/observations.

But since you asked about making offline its own module, here are the two reasonable options.

---

## Option A: conservative, recommended now

Keep:

```text
textop/mdp/
  future_reference.py
  observations.py
  offline_commands.py
  online_commands.py
```

Add:

```text
textop/offline/
  normalize.py
```

This means:

```text
offline/ = offline data pipeline
mdp/offline_commands.py = MJLab command term for offline path
```

This is least disruptive and probably best right now.

---

## Option B: stronger offline/online separation

Move command terms too:

```text
textop/offline/
  normalize.py
  command.py              # TextOpMotionCommand, TextOpMotionCommandCfg

textop/online/
  block.py
  source.py
  buffer.py
  command.py              # OnlineTextOpMotionCommand, OnlineTextOpMotionCommandCfg
```

Then `mdp/` only contains shared MDP terms:

```text
textop/mdp/
  future_reference.py
  observations.py
```

This is conceptually clean:

```text
offline path owns offline command
online path owns online command
mdp owns shared observations/interface
```

But it requires more import churn:

```python
from mjlab_vla.textop.mdp.offline_commands import use_textop_motion_command
```

becomes:

```python
from mjlab_vla.textop.offline.command import use_textop_motion_command
```

and:

```python
from mjlab_vla.textop.mdp.online_commands import use_online_textop_motion_command
```

becomes:

```python
from mjlab_vla.textop.online.command import use_online_textop_motion_command
```

Your `task.py` currently imports both command helpers from `mdp`.

### My recommendation

Do **Option A now**, Option B later if it still feels useful.

Reason: your immediate pain is `script/` and `online.py`, not command file names.

---

# Phase 5: clean `task.py` after the moves

`task.py` is okay structurally, but after rearrangement it should import:

```python
from mjlab_vla.textop.mdp.offline_commands import use_textop_motion_command
from mjlab_vla.textop.mdp.online_commands import use_online_textop_motion_command
from mjlab_vla.textop.online import TextOpOnlineSource
```

if you choose Option A. That is basically what it does already.

If you choose Option B later, update those imports to:

```python
from mjlab_vla.textop.offline.command import use_textop_motion_command
from mjlab_vla.textop.online.command import use_online_textop_motion_command
```

## One suggested task.py cleanup

Extract shared TextOp observation config into a helper so online/offline config remains identical except for command source and online-specific reward pruning.

Current:

```python
make_textop_g1_flat_tracking_env_cfg(...)
make_online_textop_g1_flat_tracking_env_cfg(...)
```

Both call:

```python
_configure_textop_anchor
_configure_textop_actor_observations
_configure_textop_critic_observations
```

That is fine. You can leave it.

Later, maybe introduce:

```python
def _configure_textop_tracking_observations(cfg) -> None:
    _configure_textop_actor_observations(cfg)
    _configure_textop_critic_observations(cfg)
```

Not necessary now.

---

# Phase 6: clarify `tracking.py`

Current `tracking.py` assumes `MotionCommandCfg`:

```python
def get_motion_command_cfg(commands) -> MotionCommandCfg:
    ...
def set_motion_file(env_cfg, motion_file: Path) -> None:
    ...
```

That is offline-specific. Online commands will not have `motion_file`.

## Option A: rename functions only

Change names to:

```python
get_offline_motion_command_cfg
set_offline_motion_file
```

Then update imports in train/eval.

## Option B: move file

Move:

```text
src/mjlab_vla/tracking.py
```

to:

```text
src/mjlab_vla/textop/offline/motion_cfg.py
```

with:

```python
get_motion_command_cfg
set_motion_file
```

Then update:

```python
from mjlab_vla.tracking import set_motion_file
```

to:

```python
from mjlab_vla.textop.offline.motion_cfg import set_motion_file
```

`train.py` and `eval.py` currently import from `mjlab_vla.tracking`.

### My recommendation

Move it to:

```text
textop/offline/motion_cfg.py
```

because it is not generic tracking anymore; it is specifically setting the offline MJLab motion file for TextOp tracking.

---

# Phase 7: keep CLI as dispatcher only

`cli.py` is already good:

```text
resolve paths
verify files
dispatch to normalize/train/play/eval
```

Keep it that way.

After moving normalize implementation, `cli.py` should still look nearly identical.

Eventually, when adding `play-online`, add only:

```python
from mjlab_vla.textop.script.play_online import PlayOnlineCommand, play_online_textop_motion

TextOpCommand = NormalizeCommand | TrainCommand | PlayCommand | EvalCommand | PlayOnlineCommand

"play-online": PlayOnlineCommand()
```

and a match case.

Do not put block-source construction directly in `cli.py`; put it in `script/play_online.py`.

---

# Phase 8: update tests after moves

Current tests import:

```python
from mjlab_vla.textop.mdp.online_commands import ...
from mjlab_vla.textop.online import ...
```

If you create `textop/online/__init__.py` with re-exports, these tests should mostly survive.

Add or update tests for moved modules:

```text
tests/textop/test_io.py
tests/textop/test_offline_normalize.py     # optional, maybe heavy
tests/textop/test_online.py                # existing
```

At minimum, after the rearrangement:

```bash
uv run --extra cpu pytest tests/textop/test_online.py
uv run --extra cpu ruff check src tests
```

---

# Exact move list

Here is the concrete move plan.

## Move 1

```text
FROM:
  src/mjlab_vla/textop/script/motion.py

TO:
  src/mjlab_vla/textop/io/motion.py
```

Add:

```text
src/mjlab_vla/textop/io/__init__.py
```

Update imports.

Delete old file.

---

## Move 2

```text
FROM:
  normalize_textop_npz
  _append_frame
  _validate_normalized_output
  inside src/mjlab_vla/textop/script/normalize.py

TO:
  src/mjlab_vla/textop/offline/normalize.py
```

Add:

```text
src/mjlab_vla/textop/offline/__init__.py
```

Keep `NormalizeCommand` in:

```text
src/mjlab_vla/textop/script/normalize.py
```

and re-export/import `normalize_textop_npz` from offline.

---

## Move 3

```text
FROM:
  src/mjlab_vla/textop/online.py

TO PACKAGE:
  src/mjlab_vla/textop/online/
    __init__.py
    block.py
    source.py
    buffer.py
    errors.py
```

Keep public imports working from:

```python
from mjlab_vla.textop.online import TextOpMotionBlock
```

by re-exporting in `online/__init__.py`.

---

## Move 4

```text
FROM:
  src/mjlab_vla/tracking.py

TO:
  src/mjlab_vla/textop/offline/motion_cfg.py
```

Update imports in:

```text
src/mjlab_vla/textop/script/train.py
src/mjlab_vla/textop/script/eval.py
```

Optional: delete old `tracking.py`, or leave a temporary compatibility shim.

---

## Move 5, optional later

```text
FROM:
  src/mjlab_vla/textop/mdp/offline_commands.py

TO:
  src/mjlab_vla/textop/offline/command.py
```

and:

```text
FROM:
  src/mjlab_vla/textop/mdp/online_commands.py

TO:
  src/mjlab_vla/textop/online/command.py
```

But I would **not** do this in the first rearrangement unless you are comfortable with the churn.

---

# Suggested final structure after conservative refactor

This is the version I recommend now:

```text
src/mjlab_vla/
  __init__.py

  textop/
    __init__.py
    contract.py
    task.py

    io/
      __init__.py
      motion.py

    offline/
      __init__.py
      normalize.py
      motion_cfg.py

    online/
      __init__.py
      block.py
      source.py
      buffer.py
      errors.py

    mdp/
      __init__.py
      future_reference.py
      observations.py
      offline_commands.py
      online_commands.py

    script/
      __init__.py
      cli.py
      normalize.py
      train.py
      play.py
      eval.py
```

Then later, when adding runnable online demo:

```text
    online/
      replay.py

    script/
      play_online.py
```

Then later, when adding RobotMDAR/ROS:

```text
    ros/
      __init__.py
      source.py
```

---

# What not to do during this rearrangement

Do **not** change behavior yet.

Specifically, do not simultaneously:

```text
change command update semantics
add play-online
add ROS2
change future_steps
change reward config
change online stale behavior
```

This rearrangement should be a pure import/layout refactor.

The goal is:

```text
Before refactor:
  tests pass
  offline normalize/train/play/eval paths work
  online unit tests pass

After refactor:
  exact same behavior
  same CLI commands
  same task names
  same tests pass
```

Then add online demo in the next patch.

---

# Best checkpoint boundary

I would tell the agent:

```text
Checkpoint 1:
  Move reusable TextOp motion IO out of script/.
  Move normalize implementation into textop/offline/.
  Convert textop/online.py into textop/online/ package.
  Move tracking.py offline motion-file helpers into textop/offline/motion_cfg.py.
  Do not add play-online yet.
  Do not change behavior.
  Update imports and tests only.
```

That gives you a clean base for the next checkpoint:

```text
Checkpoint 2:
  Add play-online replay source and online demo.
```

This is the cleanest sequence.
