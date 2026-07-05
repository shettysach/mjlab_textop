**Problem**

`play-live` with TextOp ONNX policy shows unstable/chasing behavior. The reference ghost drifts smoothly and can move much farther than the robot. The robot sometimes looks slow, catch-up-ish, or worse depending on bridge changes.

**Observations**

- Current static ONNX contract mostly matches TextOp Python `deploy_mujoco.py`:
  - Observation order matches.
  - `future_anchor_pos_b` should be real, not zeroed.
  - Anchor body is `pelvis`.
  - Control rate is 50 Hz.
  - Action scale/default offset appear matched.
  - Reset-to-reference is enabled and writes joint pos/vel plus root pos/quat.

- TextOp has conflicting deploy paths:
  - C++ deploy manually zeros future anchor position.
  - Python MuJoCo ONNX deploy does not zero it.
  - Your released ONNX behavior matches Python deploy better than C++ deploy.

- Remaining likely suspects:
  - RobotMDAR root trajectory magnitude or vertical drift.
  - Live block continuity across block boundaries.
  - Yaw/reference-frame convention mismatch.
  - Startup/reset root velocity being zero.
  - Live buffering behavior may matter, but strict no-skip timing was not better.

**Experiments and Results**

- **Zero `future_anchor_pos_b` for ONNX**
  - Result: robot walked very slowly; ghost drifted farther away.
  - Conclusion: wrong default. ONNX policy needs translational anchor displacement for walking.

- **Restore true `future_anchor_pos_b`**
  - Result: better than zeroing.
  - Conclusion: keep true future anchor position.

- **Strict live clock: no catch-up jumps**
  - Change: only advance from `current_frame` to `current_frame + 1`; never jump to latest buffered frame.
  - Result: worse.
  - Conclusion: strict clock is not the fix; default catch-up behavior is preferable in current live setup.

- **Action scale suspicion**
  - Checked TextOp and MJLab values.
  - Result: values appear intentionally matched.
  - Conclusion: not the top suspect.

- **Static TextOp compatibility check**
  - Result: repo already matches most Python deploy assumptions.
  - Conclusion: issue is likely dynamic/reference-data related, not observation layout.

Stop trying to exactly copy Python deploy at the frame-clock level. Python deploy is offline replay. Your system is online generated rolling-reference control.
