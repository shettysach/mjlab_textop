## 1. Can you check how “walk forward” behaves on TextOp’s own stack?

Yes. There are two useful levels of comparison.

### A. Check RobotMDAR’s generated walk only

Run the standalone online generator:

```bash
robotmdar --config-name=loop_dar \
    ckpt.dar=/path/to/ckpt_200000.pth \
    guidance_scale=5.0 \
    ${DATAFLAGS}
```

Enter:

```text
walk forward
```

This tells you whether the **reference motion itself** has:

* narrow or crossing feet,
* slow root translation,
* forward drift during `stand`,
* odd global-position accumulation.

It does **not** tell you whether TextOp’s tracker follows that motion cleanly.

### B. Check the native TextOp tracker with MuJoCo, without the online ROS loop

This is probably the best first diagnostic for you.

Use TextOpTracker’s simple MuJoCo deployment:

```bash
cd TextOpTracker

python scripts/deploy_mujoco.py \
    --motion_path=/path/to/walk_motion.npz \
    --policy_path=logs/rsl_rl/Pretrained/checkpoints/latest.onnx
```

TextOp explicitly provides this as a ROS-free MuJoCo evaluation path.

The only missing piece is a walk NPZ. You can:

1. Record a `walk forward` output from your current RobotMDAR producer.
2. Save it in TextOp’s expected NPZ format.
3. Feed that exact file to both:

   * TextOp’s `deploy_mujoco.py`
   * your MJLab replay path.

That is the cleanest apples-to-apples experiment:

```text
Same RobotMDAR walk reference
            ├── TextOp native MuJoCo tracker
            └── Your MJLab tracker
```

Interpretation:

* Bad in both: RobotMDAR/reference problem.
* Good in TextOp, bad in yours: integration, observations, timing, gains, joint order, or frame handling.
* Different ghost but similar robot behavior: likely visualization/alignment only.
* Good replay but bad online: queue/index/history/timing issue.

### C. Check the exact native online stack

For the exact upstream chain:

```text
rmdar.py → /dar/motion → textop_onnx_controller → unitree_mujoco
```

the repository implementation does require ROS 2 unless you port the transport/controller code. The online producer publishes `MotionBlock` messages, while the controller consumes them and drives Unitree MuJoCo.

I would not start there. The prerecorded identical-walk comparison isolates the problem much better and avoids adding ROS, DDS, joystick state, and Unitree communication as extra variables.

---

## 2. Are these settings really necessary?

```python
cfg.sim.mujoco.timestep = TEXTOP_DEPLOY_SIM_TIMESTEP
cfg.decimation = TEXTOP_DEPLOY_DECIMATION
```

With:

```python
TEXTOP_DEPLOY_SIM_TIMESTEP = 0.002
TEXTOP_DEPLOY_DECIMATION = 10
```

they produce:

```python
policy_dt = 0.002 * 10
          = 0.02 seconds
```

So the policy runs at:

```text
50 Hz
```

That exactly matches TextOpDeploy’s controller configuration:

```yaml
control_dt: 0.02
```

### The important distinction

The **individual values** are not sacred:

```text
physics timestep = 0.002
decimation       = 10
```

The most important quantity for the tracker is:

```python
control_dt = sim_timestep * decimation
```

For the pretrained TextOp tracker, that should remain `0.02`.

These alternatives have the same policy rate:

```python
0.001 * 20 = 0.02
0.002 * 10 = 0.02
0.004 * 5  = 0.02
```

But they are not physically identical because contact integration and PD control are evaluated at different physics rates.

## Is the 0.02-second policy period necessary?

For a faithful comparison: **yes**.

The pretrained policy was trained and deployed assuming observations and actions advance every 20 ms. The TextOp deployment controller explicitly uses `control_dt: 0.02`.

Changing the policy period changes the effective meaning of:

* joint velocity observations,
* previous action history,
* future reference offsets,
* action holding duration,
* feedback bandwidth,
* generated motion frame advancement.

For example, suppose the motion reference is 50 fps.

At `control_dt = 0.02`:

```text
one policy step = one reference frame
```

At `control_dt = 0.01`:

```text
two policy steps occur per reference frame
```

unless you interpolate or modify reference indexing.

At `control_dt = 0.04`:

```text
one policy step skips two reference frames
```

That can absolutely produce slow motion, aggressive catch-up, poor foot placement, or phase mismatch.

## Is `timestep = 0.002` specifically necessary?

Not strictly, but it is strongly recommended while diagnosing.

A 2 ms physics step gives 10 MuJoCo integration steps per policy action. That matters for:

* foot-ground contacts,
* ankle behavior,
* self-collision,
* PD actuator response,
* numerical stability.

Your current configuration deliberately matches the native deployment ratio:

```text
physics: 500 Hz
policy:   50 Hz
```

Your repo applies those settings directly in the online environment config.

### My verdict

Keep these settings for now:

```python
cfg.sim.mujoco.timestep = 0.002
cfg.decimation = 10
```

They are not likely to be the cause of your newly narrow/crossing gait. They make your deployment **more faithful**, not less.

However, verify one crucial point:

```python
motion_frame_dt == cfg.sim.mujoco.timestep * cfg.decimation == 0.02
```

If your command manager advances the reference using a different timestep, then matching the simulator alone is not enough.

For example, this would be wrong:

```python
policy_dt = 0.02
motion_dt = 0.04
```

because the tracker executes twice as quickly as the reference advances, making walking appear slow.

Likewise:

```python
policy_dt = 0.02
motion_dt = 0.01
```

makes the reference run ahead, producing continual catch-up.

## The diagnostic I would run

Record one continuous `walk forward` sequence from RobotMDAR and freeze it.

Then run:

```text
Test 1: TextOp deploy_mujoco.py + that NPZ
Test 2: your MJLab replay + that NPZ, dt=0.002, decimation=10
Test 3: your MJLab online loop using the same generated sequence
```

Do not involve anchor realignment changes between these tests.

That will identify the failing layer much faster than installing the complete ROS stack.
