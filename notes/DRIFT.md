# Drift Investigation

**Problem**

In `play-live`, the TextOp ghost drifts forward while standing/squatting. The MJLab robot stays put, then walks forward to catch up and eventually loses balance.

**Findings**

- `observation_computer.cpp` is the relevant live-deploy reference. `deploy_mujoco.py` is offline replay and resets to the first motion pose.
- TextOp's C++ deploy has a debug block that zeros future anchor position; its Python deploy keeps the displacement. MJLab needs the real displacement for walking.
- RobotMDAR's producer and MJLab `produce.py` agree on anchor position and quaternion conventions.
- TextOp's fixed-start global reference transform is unsuitable for live RobotMDAR data: drifting anchor offsets make the ghost move away, so the policy chases it.

**Experiments**

- Only zeroing `future_anchor_pos_b`: robot walked slowly while the ghost moved away. Wrong.
- Restoring real anchor displacement: better. Keep it.
- Strict no-skip frame clock: worse. Default catch-up behavior is preferable here.
- Action scales/default offsets and static observation layout: matched; not the primary issue.
- `direct_world`: removed; unnecessary for TextOp compatibility.
- Fixed-start global XY alignment: brought the forward drift back.
- Moving-window XY alignment fixed the drift: subtract the first reference anchor and add the current robot anchor. Z remains startup-relative: robot start Z plus reference Z delta. Quaternions are unchanged.

**Current state**

The anchor-position fix and Z-position fix are both required. Stored start anchors are tensors, not `Tensor | None`, so nullable-indexing diagnostics are avoided. The command currently supports one environment; `_aligned_reference_pos` still uses `[0]`. True batched references would require per-environment vectorization.

Legs cross while walking

**Validation**

Focused online tests: 42 passed. Ruff and `ty` passed. The full suite previously passed: 147 tests.
