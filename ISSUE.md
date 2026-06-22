# The main issues

## Issue 1: online exists, but there is no usable online CLI/demo path yet

Your CLI still only exposes:

```text id="uv6mnd"
normalize
train
play
eval
```

No `play-online`.

So the repo has online internals, but no clear command like:

```bash id="lv4bsx"
uv run --extra cpu textop-tracking play-online ...
```

This matters because the weekly task wants a TextOp-like demo. Right now, a reviewer can see online code and tests, but not easily run an online TextOp-like flow.

### Fix

Add:

```text id="pn23aj"
src/mjlab_vla/textop/script/play_online.py
```

and add `PlayOnlineCommand` to `cli.py`.

For v1, it can use a fake/replay source from an NPZ or synthetic block list. It does not need ROS2 yet.

Suggested command shape:

```python id="1pl4uf"
@dataclass(kw_only=True)
class PlayOnlineCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    checkpoint_file: str = tyro.MISSING
    device: str = "cuda:0"
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 8
    max_stale_steps: int = 25
```

Then CLI adds:

```python id="m7r71d"
"play-online": PlayOnlineCommand()
```

This gives you the actual demo path.

---

## Issue 2: the registered online task has no real source

`ensure_textop_task_registered()` registers:

```text id="ea4ii9"
Mjlab-OnlineTextOp-Flat-Unitree-G1
```

using `make_online_textop_g1_flat_tracking_env_cfg(play=True)` with no source argument.

Because `source=None`, the cfg uses the default `QueueTextOpOnlineSource`, which is empty.

So if someone directly runs the registered online task, the online command will just wait for startup frames and eventually hit the startup timeout. The startup logic explicitly waits until the buffer has enough contiguous future frames.

### Fix

Do not rely on static registration for the real online demo.

Use one of these:

### Option A, best for now

Create `play-online` that constructs the env cfg directly and injects a source:

```text id="2sl79d"
source = make_replay_source_from_npz(...)
cfg = make_online_textop_g1_flat_tracking_env_cfg(source=source)
run play/inference with that cfg
```

### Option B, later

Add a source registry/factory:

```python id="rp9ckk"
set_default_online_textop_source(source)
```

Then registered online task can pick it up. But that is uglier.

For now, use Option A.

---

## Issue 3: no anchor/world-frame alignment yet

The online buffer stores and returns raw `anchor_pos_w`. The online command returns it directly through `future_anchor_pos_w`.

That is risky.

Offline references go through MJLab-normalized motion and add env origins for future anchor positions.  Online RobotMDAR/TextOpDeploy positions may be in the generator’s root/world frame, not necessarily MJLab’s current robot spawn frame.

So the robot may see a future anchor target that is spatially offset from where the MJLab robot actually is.

### Fix

Add an alignment option to `OnlineTextOpMotionCommandCfg`:

```python id="c476yc"
anchor_alignment: Literal["direct_world", "align_to_robot_start"] = "align_to_robot_start"
```

On startup, when the buffer first becomes ready:

```text id="kzhzra"
reference_start = buffer anchor_pos_w[start_frame]
robot_start = robot_anchor_pos_w[0]
offset = robot_start - reference_start
```

Then future anchor positions become:

```text id="p8kbxe"
aligned_anchor_pos_w = raw_anchor_pos_w + offset
```

Do position alignment first. Yaw/orientation alignment can come later.

This is important for the live text-to-motion demo.

---

## Issue 4: current timing assumes one MJLab command update equals one TextOp frame

The online command increments:

```python id="mcd1tk"
self.current_frame += 1
```

on every `_update_command()` after startup.

That is fine only if:

```text id="ucw52t"
MJLab policy/control step rate == TextOp source FPS
```

RobotMDAR/TextOpDeploy’s config uses `control_dt: 0.02`, i.e. 50 Hz, and block size 8.

If your MJLab command update/policy step is also 50 Hz, fine. If not, the online command will drift in time.

### Fix

For v1, add an explicit comment/config assertion:

```text id="th38hr"
v1 assumes one TextOp frame per MJLab policy/control step.
```

Then later add resampling:

```python id="2r57jy"
source_fps: float = 50.0
control_dt: float | None = None
```

and compute frame index from elapsed simulation time instead of blindly incrementing.

Do not overbuild this immediately, but do make the assumption explicit.

---

## Issue 5: online task reward/termination pruning may be incomplete

You correctly remove several tracking terms for online:

```python id="lh5qde"
motion_body_pos
motion_body_ori
motion_body_lin_vel
motion_body_ang_vel
ee_body_pos termination
```

That is probably necessary because the online command is not a full MJLab `MotionCommand` and does not expose all body reference fields.

But you should verify the complete reward/termination set after replacing the command. If any remaining MJLab tracking reward expects `MotionCommand` fields like `body_pos_w`, `body_quat_w`, `body_pos_relative_w`, or `robot_body_pos_w`, the online env may crash at runtime.

### Fix

Add an integration smoke test that builds the online env config and steps once with a fake source.

Minimum test:

```text id="m3hpnx"
make_online_textop_g1_flat_tracking_env_cfg(source=QueueTextOpOnlineSource([...]))
ManagerBasedRlEnv(cfg, device="cpu")
env.reset()
env.step(zero_action)
```

If full `ManagerBasedRlEnv` is too heavy for CI, at least test that all reward/termination terms left in the config do not reference missing motion-command properties.

---

## Issue 6: `TextOpRollingMotionBuffer._resolve_frame()` is permissive before earliest frame

If a requested frame is missing, `_resolve_frame()` does:

```python id="4ed0yf"
available = [idx for idx in self._joint_pos if idx <= frame]
if available:
    return max(available)

return min(self._joint_pos)
```

The final `return min(self._joint_pos)` means: if the requested frame is before the earliest available frame, use the earliest frame.

That is okay only after startup if you intentionally allow clamping both directions. But your intended semantics were:

```text id="0un2mh"
before startup:
  require contiguous window

after startup:
  short underrun means hold/repeat latest valid frame
```

“Hold latest valid frame” means clamp backward to the latest frame at or before the requested frame, not jump forward to the earliest available future frame.

### Fix

Change `_resolve_frame()` to either:

```python id="uxx7uv"
raise RuntimeError(f"No available frame at or before requested frame {frame}")
```

when no previous frame exists, or make this behavior conditional.

In normal post-start underrun, you want:

```text id="pb19vo"
requested 12, latest available 10 -> use 10
```

not:

```text id="w6odko"
requested 0, earliest available 100 -> use 100
```

That second case should probably be startup failure or reset failure.

---

## Issue 7: reset/resample semantics are not quite right

`_resample_command()` resets:

```text id="8og49d"
current_frame = start_frame
_started = False
startup counters reset
```

but it does not clear the buffer.

That means after an env reset, the command may try to restart at frame 0 while the buffer contains only later live-stream frames, or old frames from a previous run.

### Fix

For v1, choose one explicit behavior:

### Option A: reset clears buffer

```python id="n85drr"
self.buffer.clear()
self.current_frame = self.cfg.start_frame
```

This is clean for replay/demo sources.

### Option B: reset attaches to latest live stream

```python id="py8jf3"
self.current_frame = max(self.cfg.start_frame, self.buffer.latest_index - self.cfg.future_steps + 1)
```

This is more live-stream-like.

For single-env play, I would start with Option A unless you are explicitly attaching to an always-running RobotMDAR stream.

---
