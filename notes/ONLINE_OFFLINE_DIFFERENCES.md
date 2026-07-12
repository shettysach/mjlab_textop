# TextOp Online vs. Offline Control Differences

This note compares the committed TextOp online C++ controller
(`TextOpDeploy/src/textop_ctrl`) with the standalone Python/MuJoCo harness
(`TextOpTracker/scripts/deploy_mujoco.py`).  It deliberately excludes ROS/DDS
transport, wall-clock jitter, simulator/robot-model differences, contact
parameters, and sensor noise.

## Shared behavior

Both paths use the released ONNX tracker with:

- 29-joint reference position and velocity in TextOp/IsaacLab order;
- a five-frame future reference window;
- a 50 Hz policy period (`0.02 s`); and
- the same broad observation structure: future joints, future anchor pose,
  gravity, base velocity, joints, and previous action.

## Behaviorally relevant differences

| Area | Offline `deploy_mujoco.py` | Online C++ controller | Consequence |
| --- | --- | --- | --- |
| Future anchor position | Computes and passes the five reference anchor offsets relative to the current robot anchor. | Computes the offsets, then unconditionally overwrites all 15 values with zero in `motion_anchor_pos_b_future()`. | Online removes both global catch-up error and local 20--80 ms anchor translation; offline does not. |
| Reference frame at startup | Initializes the simulated robot directly from reference frame 0. | Transforms the reference into the robot's initial position/yaw frame. | Relevant when robot and reference begin at different XY position or heading. |
| Future anchor orientation | Uses the recorded reference quaternion directly when forming the robot-relative observation. | Applies the startup yaw alignment to the reference quaternion before forming the observation. | Orientation targets can differ if initial headings differ. |
| Reference-clock startup | Starts at frame 0 of the supplied NPZ. | Starts from a controller-owned initial buffer and is activated by the toggle/joystick workflow. | The first frames presented to the policy need not be the same unless the online stream is constructed to match the NPZ exactly. |
| Reference-clock advancement | Advances one reference frame for each 50 Hz policy update. | Derives the reference index from elapsed time after activation. | With ideal 50 Hz timing they agree; the indexing logic is nevertheless different. |
| Optional `LockXY` | No equivalent mode. | Has a `LockXY` adjustment of the stored reference-to-robot XY alignment. | With the current unconditional zero-anchor-position block, it cannot restore positional anchor information; it may still affect other frame-alignment state. |

## Important clarification about the online zeroing block

The C++ code contains:

```cpp
if (true) {
    pos_b[i * 3 + 0] = 0;
    pos_b[i * 3 + 1] = 0;
    pos_b[i * 3 + 2] = 0;
}
```

This is unconditional.  It is not controlled by `LockXY`, a launch argument,
or a configuration option.  Consequently, the live C++ controller always
supplies an all-zero future anchor-position observation, while retaining the
joint reference and future anchor-orientation observation.

## What this means for the standing-drift experiment

If a frozen `stand` record steps forward in both paths, the shared joint/root
reference is sufficient to explain the behavior; it is not introduced by
MJLab's socket or reference-buffer code.  The zero-anchor online mode may
change the severity or gait quality, but it cannot make a forward-stepping
joint reference into a stationary one.

For a focused comparison, run the same NPZ in three modes:

1. Native offline anchors.
2. MJLab moving-window anchors.
3. Offline script patched to return all-zero future anchor positions, matching
   the committed online C++ controller.
