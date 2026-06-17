Here’s the short context summary you can feed your coding agent:

We are building a small MJLab playground repo for a TextOp-style text-to-motion tracking demo. The immediate goal is **not** to run Gemma, vLLM, residual PPO, or full TextOp yet. The first milestone is to prepare MJLab to accept **TextOp-style motion references** and track them with a Unitree G1 task.

The core idea is:

```text
text command
  -> command/reference generator
  -> MotionReference buffer
  -> MJLab G1 tracking task
  -> controller/policy tracks the reference
  -> physics rollout / render
```

The repo can be structured like `mjlab_playground`: task-oriented, with reusable MJLab task configs, MDP terms, commands, and demo scripts.

The first actual task should be:

```text
G1 reference-pose / reference-motion tracking in MJLab
```

Start by running or copying/inheriting from the existing MJLab G1 velocity tracking task. Then replace the velocity command with a `MotionReferenceCommand` / `command_multi.py` style interface that stores per-env references.

The important interface is:

```text
MotionReference:
  root pose / orientation
  root linear + angular velocity
  joint positions
  joint velocities
  optional body poses / contacts
  valid mask / phase
```

MJLab should not care whether this reference came from dummy code, TextOp, Kimodo, Gemma, keyboard input, or a stored motion clip. It should only consume `MotionReference`.

The coding direction is:

```text
1. Create repo skeleton
2. Run existing MJLab G1 velocity tracking task
3. Locate obs/reward/command/termination/action configs
4. Create/port a command_multi.py-style command manager
5. Add observation terms for target pose, target velocity, pose error, phase
6. Add simple tracking rewards: joint pose, root velocity, uprightness, smooth action
7. Make a dummy text-to-reference demo with 4 envs:
   stand still / walk forward / turn left / sidestep right
8. Later replace dummy generator with real TextOp
```

The key design rule:

```text
TextOp/Kimodo/Gemma side produces MotionReference.
MJLab side tracks MotionReference.
```

So the first deliverable is not “full text-to-motion model works.” It is:

```text
MJLab G1 can consume per-env TextOp-style references and produce a tracking rollout.
```

Reference repositories -

../repos/mjlab/
../repos/mjlab_playground/
../repos/kimolab/
../repos/TextOp/
