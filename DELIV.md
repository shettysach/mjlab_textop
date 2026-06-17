To prepare MJLab for TextOp-style references, think of MJLab as needing a **reference-tracking interface**.

TextOp’s high level does this:

```text
text command
  → short-horizon kinematic trajectory
```

TextOp’s low level does this:

```text
current robot state + reference trajectory
  → actions / joint targets
  → physics rollout
```

MJLab should implement the second part. Since MJLab uses a manager-based environment style with modular observations, rewards, commands, and terminations, the clean move is to make a new **command/reference manager** and a **tracking task** that consume TextOp-like reference data. MJLab is explicitly built around composable environment design and manager-based terms, so this is exactly the right abstraction. ([GitHub][1])

## 1. Define the reference format first

Before touching rewards or obs terms, define what a TextOp reference means inside MJLab.

For a first version, use something like this:

```python
@dataclass
class MotionReference:
    # shape: [num_envs, horizon, ...]
    root_pos: torch.Tensor        # [N, H, 3]
    root_quat: torch.Tensor       # [N, H, 4]
    root_lin_vel: torch.Tensor    # [N, H, 3]
    root_ang_vel: torch.Tensor    # [N, H, 3]

    joint_pos: torch.Tensor       # [N, H, num_dofs]
    joint_vel: torch.Tensor       # [N, H, num_dofs]

    body_pos: torch.Tensor | None = None   # [N, H, num_bodies, 3]
    body_quat: torch.Tensor | None = None  # [N, H, num_bodies, 4]

    mask: torch.Tensor | None = None       # [N, H], valid timesteps
```

This is the contract between TextOp and MJLab.

TextOp-style generation is short-horizon and online: the high-level model continuously generates kinematic trajectories from the current text command, and the low-level tracking policy executes them. ([TextOp][2])

So your MJLab env should not expect one full episode-long motion clip. It should expect something more like:

```text
every K simulation steps:
    receive/update next short reference window
```

For the fake first version, the reference can be produced by a dummy generator.

## 2. Add a `TextOpCommand` / `MotionReferenceCommand` class

This is probably where `command_multi.py` comes in.

The command class should own per-env reference buffers:

```text
command_manager
  stores:
    text_command[N]
    reference_motion[N, H, ...]
    reference_time_index[N]
    command_age[N]
    needs_update[N]
```

It should expose methods like:

```python
class MotionReferenceCommand:
    def reset(self, env_ids):
        ...

    def update_text(self, env_ids, texts):
        ...

    def update_reference(self, env_ids, reference: MotionReference):
        ...

    def advance(self):
        # move reference cursor forward by one env step
        ...

    def current(self):
        # return reference at current tracking timestep
        ...

    def future_window(self):
        # return short horizon for observations
        ...
```

The key is that MJLab should not care whether the reference came from:

```text
hardcoded velocity command
TextOp
Kimodo
Gemma
human keyboard
stored mocap file
```

It should only consume the normalized `MotionReference`.

## 3. Add observation terms that expose reference error

A tracking policy needs to see both the robot state and the target reference.

Minimum useful observation terms:

```text
robot base angular velocity
robot projected gravity
joint positions
joint velocities
previous action

reference joint positions
reference joint velocities
reference root velocity
reference root angular velocity

joint position error
joint velocity error
root velocity error
phase / time-to-go
```

For TextOp-style control, I would start with these observation terms:

```python
def ref_joint_pos(env):
    return env.command_manager.get_command("motion_ref").current().joint_pos

def ref_joint_vel(env):
    return env.command_manager.get_command("motion_ref").current().joint_vel

def joint_pos_error(env):
    ref = env.command_manager.get_command("motion_ref").current().joint_pos
    q = env.robot.data.joint_pos
    return ref - q

def root_lin_vel_error(env):
    ref = env.command_manager.get_command("motion_ref").current().root_lin_vel
    v = env.robot.data.root_lin_vel_b
    return ref - v
```

Your supervisor’s “port some obs terms” probably means TextOpTracker has extra terms like reference pose, reference future trajectory, phase, body pose error, or selected key body targets that MJLab’s existing tracking task does not expose yet.

## 4. Add reward terms that track the reference

Start with simple tracking rewards.

```text
joint position tracking
joint velocity tracking
root linear velocity tracking
root angular velocity tracking
base height tracking
orientation tracking
feet contact / no-slip penalties
action smoothness
torque penalty
alive reward
```

A first reward set could be:

```python
reward =
    + joint_pos_tracking
    + root_vel_tracking
    + root_ang_vel_tracking
    + base_orientation_tracking
    - action_rate_penalty
    - torque_penalty
    - foot_slip_penalty
```

Do not overcomplicate this in week one. The goal is to prove the reference interface works.

## 5. Add termination terms

Use normal humanoid tracking terminations:

```text
base height too low
base orientation too tilted
large joint limit violation
NaN / unstable sim
episode timeout
```

For TextOp-style references, add one more:

```text
reference exhausted and no replacement available
```

But in the intended online system, the reference should keep getting refreshed.

## 6. Implement a dummy reference generator first

Before plugging real TextOp, create:

```python
class DummyTextOpGenerator:
    def generate(self, texts, obs, horizon):
        ...
```

Map text to simple references:

```text
"stand still"     → stationary pose
"walk forward"    → nominal gait / velocity ref
"turn left"       → yaw velocity ref
"sidestep right"  → lateral velocity ref
```

At the very beginning, this does **not** need full-body generated motion. It can return velocity-level references and default standing joint poses.

That means stage one is:

```text
text → dummy reference → MJLab tracking
```

not:

```text
text → real TextOp diffusion → MJLab tracking
```

This lets you debug the env without model complexity.

## 7. Then plug real TextOp behind the same interface

Once the dummy path works, TextOp becomes just another backend:

```python
class TextOpGenerator:
    def generate(self, texts, robot_state, history, horizon) -> MotionReference:
        # call TextOp high-level model
        # retarget/normalize output to G1 joint/body format
        # return MotionReference
```

Then the runner loop is:

```python
obs = env.reset()

while training_or_playing:
    env_ids = command_manager.envs_that_need_new_reference()

    if len(env_ids) > 0:
        texts = command_manager.texts[env_ids]
        robot_state = extract_state_for_textop(obs, env_ids)
        ref = textop.generate(texts, robot_state, horizon=H)
        command_manager.update_reference(env_ids, ref)

    action = policy(obs)
    obs, rew, done, info = env.step(action)
    command_manager.advance()
```

## 8. The minimal demo target

Your first demo should be:

```text
4 vectorized G1 envs

env 0: "stand still"
env 1: "walk forward"
env 2: "turn left"
env 3: "sidestep right"

dummy TextOp command source
  → MotionReferenceCommand
  → MJLab G1 tracking task
  → render rollout
```

That proves MJLab can accept TextOp-style references.

Then your second demo is:

```text
same MJLab task
same command interface
replace dummy generator with actual TextOp generator
```

## The important design rule

Do **not** let TextOp-specific code leak everywhere.

Keep this boundary:

```text
TextOp / Kimodo / Gemma side:
    produces MotionReference

MJLab side:
    tracks MotionReference
```

So the architecture should look like this:

```text
Text command
   ↓
Command source
   ├── DummyTextOpGenerator
   ├── RealTextOpGenerator
   ├── KimodoGenerator
   └── Gemma-chosen command
   ↓
MotionReference
   ↓
MJLab MotionReferenceCommand
   ↓
Observation terms: ref pose, ref velocity, tracking error
   ↓
Reward terms: pose tracking, velocity tracking, stability
   ↓
G1 policy/controller
   ↓
physics rollout
```

The one-week version is mostly:

```text
MotionReference format
+ command_multi.py style command manager
+ obs/reward terms
+ dummy text-to-reference demo
```

That is what “prepare MJLab to accept TextOp-style references” means.

[1]: https://github.com/mujocolab/mjlab?utm_source=chatgpt.com "mujocolab/mjlab: Isaac Lab API, powered by MuJoCo- ..."
[2]: https://text-op.github.io/?utm_source=chatgpt.com "TextOp: Real-time Interactive Text-Driven Humanoid Robot ..."
