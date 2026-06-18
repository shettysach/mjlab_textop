# MJLab TextOp-Style Reference Tracking Implementation Plan

## Summary

Create this repo as an external MJLab task package that prepares MJLab to
consume TextOp-style short-horizon motion references for Unitree G1 tracking.

Use upstream `mujocolab/mjlab`, not `../repos/mjlab`. Dependency selection
follows MJLab's upstream extras pattern:

```toml
[project.optional-dependencies]
cpu = ["mjlab", "torch>=2.7.0"]
cu128 = ["mjlab", "torch>=2.7.0"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu128" }]]

[tool.uv.sources]
mjlab = { git = "https://github.com/mujocolab/mjlab", rev = "0cdc56246999409b83622764f5b38edb660cf16e" }
torch = [
  { index = "pytorch-cpu", extra = "cpu", marker = "sys_platform != 'darwin'" },
  { index = "pytorch-cu128", extra = "cu128", marker = "sys_platform != 'darwin'" },
]
```

The local Nix shell uses MuJoCo EGL with Mesa `llvmpipe` for CPU-friendly
verification. Run local commands with `--extra cpu`; the GPU machine should use
`--extra cu128`. The Python code should not depend on that backend choice.

The extras intentionally depend on plain `mjlab`, not `mjlab[cpu]` or
`mjlab[cu128]`. This repo is the top-level uv project and owns torch wheel
selection through `tool.uv.sources`; selecting MJLab's extras transitively makes
uv merge CPU and CUDA torch indexes during lock resolution.

Primary commands:

```bash
uv sync --extra cpu
uv run --extra cpu inspect-reference --text "walk forward"
uv run --extra cpu list-envs --keyword TextOp
uv run --extra cpu demo-dummy-textop --steps 200 --num-envs 4
```

On the GPU machine, replace `--extra cpu` with `--extra cu128`.

Core boundary:

```text
Action Expert / TextOp / Kimodo / dummy provider
  -> MotionReference
  -> MJLab MotionReferenceCommand
  -> G1 reference tracking task
  -> rollout / render / policy training
```

## Project Structure

```text
.
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── src/mjlab_vla/
│   ├── __init__.py
│   ├── reference/
│   │   ├── __init__.py
│   │   ├── dummy_provider.py
│   │   ├── providers.py
│   │   └── types.py
│   ├── tasks/g1_textop_tracking/
│   │   ├── __init__.py
│   │   ├── env_cfg.py
│   │   ├── rl_cfg.py
│   │   └── mdp/
│   │       ├── __init__.py
│   │       ├── commands.py
│   │       ├── observations.py
│   │       ├── rewards.py
│   │       └── terminations.py
│   └── scripts/
│       ├── demo_dummy_textop.py
│       └── inspect_reference.py
└── tests/
```

This V1 dummy-task package structure has been superseded by the current
TextOp-motion normalization package. The current project does not register an
MJLab task entry point; it exposes the `normalize-textop-npz` console script.

```toml
[project.scripts]
normalize-textop-npz = "mjlab_vla.scripts.normalize_textop_npz:main"
```

## V1 Implementation

V1 proves MJLab can consume per-env short-horizon references and run a G1
tracking rollout. It does not target high-quality walking.

Implement `MotionReference`:

```python
@dataclass
class MotionReference:
  root_pos: torch.Tensor
  root_quat: torch.Tensor
  root_lin_vel: torch.Tensor
  root_ang_vel: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos: torch.Tensor | None = None
  body_quat: torch.Tensor | None = None
  body_lin_vel: torch.Tensor | None = None
  body_ang_vel: torch.Tensor | None = None
  valid: torch.Tensor | None = None
  phase: torch.Tensor | None = None
```

Expected shapes:

```text
root_pos:      [N, H, 3]
root_quat:     [N, H, 4]
root_lin_vel:  [N, H, 3]
root_ang_vel:  [N, H, 3]
joint_pos:     [N, H, D]
joint_vel:     [N, H, D]
body_pos:      [N, H, B, 3] optional
body_quat:     [N, H, B, 4] optional
valid:         [N, H] optional
phase:         [N, H] optional
```

Implement `MotionReferenceCommand` as an MJLab `CommandTerm` with per-env
reference buffers, text labels, cursors, refresh flags, current-reference
properties, and short future lookahead.

Implement `DummyTextReferenceProvider`:

```text
stand still     -> zero root velocity, default standing joints
walk forward    -> positive root x velocity
turn left       -> positive yaw velocity
sidestep right  -> lateral root velocity
```

Register:

```text
Mjlab-TextOpTracking-Flat-Unitree-G1
```

V1 observations:

```text
base angular velocity
projected gravity
joint position / velocity
last action
reference joint position / velocity
joint position / velocity error
reference root linear/angular velocity
root velocity error
phase
3-5 step future anchor pose lookahead
```

V1 rewards:

```text
joint position tracking
joint velocity tracking
root linear/angular velocity tracking
uprightness
action rate penalty
joint limit penalty
self-collision penalty if available
```

V1 demo:

```text
4 vectorized G1 envs
stand still / walk forward / turn left / sidestep right
dummy provider -> MotionReferenceCommand -> rollout
```

## V2 Implementation

V2 adds trusted body references while preserving the same provider interface.

Add:

```text
G1 FK utility
stored-reference provider
NPZ import/export utility
body pose observation terms
body pose/orientation rewards
key-body tracking metrics
reference quality checks
```

Enable dense body rewards only when `body_pos` and `body_quat` are present and
trusted:

```text
anchor/root pose tracking
relative body position tracking
relative body orientation tracking
body linear/angular velocity tracking
selected feet/hands/torso tracking
```

Acceptance target:

```text
A stored or FK-expanded reference drives the same G1 tracking task without task-code changes.
```

## V3 Implementation

V3 plugs real Action Experts behind `MotionReferenceProvider`.

Providers:

```text
TextOpProvider
KimodoProvider
KeyboardProvider
StoredClipProvider
```

TextOp path:

```text
text command
  -> TextOp high-level generator
  -> short-horizon kinematic trajectory
  -> retarget/normalize to G1
  -> optional FK body fields
  -> MotionReference
```

V3 additions:

```text
online reference refresh every K env steps
provider latency measurement
fallback to previous valid reference on provider failure
provider-agnostic metrics
optional observation/action delay experiments inspired by Kimolab
tracking policy training config
```

Acceptance target:

```text
The MJLab task remains unchanged while swapping dummy, stored-clip, TextOp, or Kimodo providers.
```

## Tests

V1:

```text
MotionReference shape/device validation
dummy provider text mapping
MotionReferenceCommand reset/update/advance
task entry-point registration
G1 task config load
4-env smoke rollout
```

V2:

```text
FK body shape checks
stored NPZ provider slicing
dense rewards disabled without body fields
dense rewards enabled with body fields
```

V3:

```text
provider adapters return valid MotionReference
online refresh preserves continuity
provider failure uses fallback
same task runs with dummy and non-dummy providers
```

## Assumptions

- Do not modify MJLab.
- Do not depend on Kimolab.
- V1 body pose fields are optional.
- Dense full-body rewards start in V2.
- V1 proves interface correctness and rollout stability, not final motion quality.
