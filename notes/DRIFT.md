# Drift Investigation

Issue is noted here = https://github.com/TeleHuman/TextOp/blob/main/USAGE.md#L405-L409

**Problem**

In `play-live`, the TextOp ghost drifts forward even while standing/squatting. 
The MJLab robot stays put, then walks forward to catch up and eventually loses balance.

**Findings**

- `observation_computer.cpp` is the relevant live-deploy reference. `deploy_mujoco.py` is offline replay and resets to the first motion pose.
- TextOp's C++ deploy has a debug block that zeros future anchor position; its Python deploy keeps the displacement. MJLab needs the real displacement for walking.
- RobotMDAR's producer and MJLab `produce.py` agree on anchor position and quaternion conventions.
- TextOp's fixed-start global reference transform is unsuitable for live RobotMDAR data: drifting anchor offsets make the ghost move away, so the policy chases it.

**Experiments**

- Fixed-start global XY alignment: brought the forward drift back.
- Moving-window XY alignment fixed the drift: subtract the first reference anchor and add the current robot anchor. Z remains startup-relative: robot start Z plus reference Z delta. Quaternions are unchanged.

- Only zeroing `future_anchor_pos_b`: robot walked slowly while the ghost moved away.
- Restoring real anchor displacement: better.
- Strict no-skip frame clock: worse. Default catch-up behavior is preferable here.
- Action scales/default offsets and static observation layout: matched; not the primary issue.

**Current state**

The anchor-position fix and Z-position fix are both required. Stored start anchors are tensors, not `Tensor | None`, so nullable-indexing diagnostics are avoided. The command currently supports one environment; `_aligned_reference_pos` still uses `[0]`. True batched references would require per-environment vectorization.

Legs cross while walking. Walking is clunky and slow.

**`deploy_mujoco.py`**

- Offline replay. 
- Shows the same problems. Stand keeps stepping forward.

**`observation_computer.cpp`**


TextOp’s actual online C++ deployment does this:

1. It initially transforms reference poses into the robot’s initial yaw/position frame.
2. But in motion_anchor_pos_b_future() it then has an unconditional debug block:

```cpp
if (true) {
    pos_b[i * 3 + 0] = 0;
    pos_b[i * 3 + 1] = 0;
    pos_b[i * 3 + 2] = 0;
}
```

---

1. Your moving-window alignment is not equivalent to zeroing all anchor positions.

Your code makes each current reference frame coincide with the robot:

```text
aligned[t] = robot_now
```

but preserves displacement within the 5-frame future window:

```text
aligned[t+i] - robot_now = ref[t+i] - ref[t]
```

So its anchor-position observation is approximately:

```text
frame 0:      [0, 0, reference-Z-offset]
frames 1–4:   local reference deltas over the next 20–80 ms
```

TextOp’s `if (true)` override produces:

```text
frames 0–4: [0, 0, 0]
```

It discards even those local forward/lateral/upward deltas. Neither method changes the joint-position or joint-velocity reference; they only alter the anchor-position part of the policy observation.

2. TextOp has two related mechanisms:

- A normal fixed-start reference-to-robot transform: it aligns the reference’s initial pose to the robot when tracking starts, then would retain the reference’s global displacement.
- A `LockXY` path that adjusts the stored alignment when lock mode is enabled.

But in the committed C++ online controller, the unconditional `if (true)` zeroing happens after those calculations. Therefore its actual live behavior is: compute transforms, then discard all future anchor-position offsets.

The standalone `deploy_mujoco.py` does not do this override. It uses the reference displacement normally.

---
