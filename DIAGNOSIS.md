Yes—this is a **reference-origin discontinuity**, not “RobotMDAR decided to turn left.”

The observation that proves it is:

> **At the discontinuity, the ghost starts exactly from the robot’s anchor.**

That is precisely what `_align_reference_anchor()` is designed to do.

## What is happening

During ordinary playback, the reference remains anchored to its initial relationship with the robot:

```python
robot_start_anchor
+ current_raw_reference
- initial_raw_reference
```

So the small robot–ghost offset remains stable.

But `_can_advance_live_frame()` contains a mid-run resync:

```python
if not self.buffer.can_start(self.current_frame, self.cfg.future_steps):
    if latest_start_frame > self.current_frame:
        self.current_frame = latest_start_frame
        self._align_reference_anchor()
    return False
```

There is another equivalent branch immediately afterward.

When that executes, `_align_reference_anchor()` records:

```python
self._reference_start_anchor_pos_w = raw_reference_at_new_frame
self._robot_start_anchor_pos_w = current_robot_position
```

Consequently, the next `_fixed_start_reference_pos()` calculation guarantees:

```text
aligned reference position at resync frame
= current robot position
```

That is exactly why the ghost suddenly appears to originate from the robot.

## The actual sequence

The likely runtime sequence is:

```text
Robot follows reference with a small stable tracking offset
        ↓
Consumer loses the current contiguous future window
        ↓
current_frame jumps forward to latest_start_frame
        ↓
_align_reference_anchor() is called
        ↓
New reference frame is translated onto the robot's current anchor
        ↓
Ghost–robot offset suddenly becomes zero
        ↓
Later trajectory segment may have a different tangent/yaw/pose
        ↓
Ghost sharply departs left and tracker cannot follow the discontinuity
```

So the “left turn” is only what the newly selected later segment happens to do after it has been rebased onto the robot.

## Why the tracker fails

The resync only makes **position** appear continuous.

It does not restore continuity in:

* joint positions,
* joint velocities,
* root orientation,
* root angular velocity,
* gait phase,
* future trajectory tangent.

`_future_window()` translates `anchor_pos_w`, but passes `anchor_quat_w`, joints, and velocities through unchanged.

Therefore the visual root position starts at the robot, while the rest of the reference state may come from a substantially later motion frame.

## Fix the resync behavior

Remove all automatic jump-to-latest behavior from `_can_advance_live_frame()`.

Use:

```python
def _can_advance_live_frame(self) -> bool:
    """Advance only through contiguous live reference frames."""

    if not self.buffer.can_start(
        self.current_frame,
        self.cfg.future_steps,
    ):
        raise RuntimeError(
            "Lost the active live reference window: "
            f"current={self.current_frame}, "
            f"earliest={self.buffer.earliest_index}, "
            f"latest={self.buffer.latest_index}, "
            f"future_steps={self.cfg.future_steps}"
        )

    next_frame = self.current_frame + 1

    # New frames have not arrived yet. Hold the current frame rather than
    # skipping forward and rebasing the reference.
    return self.buffer.can_start(
        next_frame,
        self.cfg.future_steps,
    )
```

The resulting behavior becomes:

```text
Next future window unavailable
→ hold current reference temporarily

Current reference window disappeared
→ explicit error
```

Instead of:

```text
Current window unavailable
→ jump forward
→ re-anchor onto robot
→ discontinuity
```

## `_align_reference_anchor()` should only run in two cases

Keep it for:

1. initial startup;
2. an intentional environment reset.

It should **never run during ordinary live stream advancement**.

I would even rename it to make that constraint obvious:

```python
def _initialize_reference_origin(self) -> None:
    ...
```

Then no streaming recovery function should call it.

## Find why the active window disappeared

After making the strict change, the error should expose the underlying source. The likely causes are:

### Rolling-buffer eviction

The live buffer retains only 512 frames by default.

Eviction is based on the newest received index:

```python
first_kept = latest_index - max_frames + 1
```

If MJLab runs slower than the producer for long enough, the frame currently being consumed can be evicted.

For diagnosis:

```python
max_buffer_frames=None
```

### Socket queue drops

The socket queue drops the oldest block when it reaches 32 blocks:

```python
if len(self._queue) >= self.cfg.max_queue_blocks:
    self._queue.popleft()
    self.diagnostics.blocks_dropped += 1
```

Temporarily use:

```python
SocketSourceCfg(
    host=...,
    port=...,
    fps=50.0,
    max_queue_blocks=256,
)
```

### A genuine index gap

Add strict continuity checking when polling:

```python
def _poll_source(self) -> None:
    for _ in range(self.cfg.max_poll_blocks):
        block = self.source.poll()
        if block is None:
            return

        latest = self.buffer.latest_index
        if latest is not None:
            expected = latest + 1
            if block.index != expected:
                raise RuntimeError(
                    "Non-contiguous RobotMDAR stream: "
                    f"expected block start {expected}, "
                    f"received {block.index}, "
                    f"delta={block.index - expected}"
                )

        self.buffer.append_block(block)
        self._clear_future_cache()
```

Your producer increments every block index by the exact number of emitted frames, so live blocks should be contiguous.

## Add one definitive log

Before changing `current_frame`:

```python
previous_frame = self.current_frame
can_advance = self._can_advance_live_frame()

if self.current_frame != previous_frame:
    raise RuntimeError(
        "Live reference frame changed inside availability check: "
        f"{previous_frame} -> {self.current_frame}"
    )
```

With the corrected function, `_can_advance_live_frame()` must be a pure availability check. It should never mutate the reference timeline.

## Conclusion

Your visual observation identifies the bug very specifically:

> **The consumer loses continuity, jumps to a later reference frame, and calls `_align_reference_anchor()`, which rebases that later frame onto the robot.**

That is why:

* the prior offset disappears;
* the ghost begins exactly at the robot;
* the subsequent motion sharply diverges;
* the tracker cannot adapt.

The primary fix is to **remove mid-stream resynchronization and mid-stream re-anchoring**, then fail or hold when the live stream is not contiguous.
