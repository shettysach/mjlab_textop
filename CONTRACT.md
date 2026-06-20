You already added **most of the contract**, just not as a clearly named contract module.

This file:

```text
src/mjlab_vla/textop_motion.py
```

already contains:

```text
MJLab G1 joint names
TextOp -> MJLab joint reindex
TextOp NPZ required keys
fps reading
root = body index 0 assumption
quaternion normalization
shape validation
```

So you do **not** need to add a big new subsystem. The remaining work is mostly cleanup and correctness hardening.

## 1. Add/rename a small `contract.py`

Right now `textop_motion.py` is doing two jobs:

```text
1. defining the TextOp/MJLab contract
2. loading a TextOp motion file
```

I would split the static assumptions into:

```text
src/mjlab_vla/textop_contract.py
```

or:

```text
src/mjlab_vla/textop/contract.py
```

Move these there:

```python
MJLAB_G1_JOINT_NAMES
TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
TEXTOP_ROOT_BODY_INDEX = 0
TEXTOP_REQUIRED_KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
TEXTOP_OPTIONAL_KEYS = ("body_lin_vel_w", "body_ang_vel_w")
```

Then `textop_motion.py` imports from it.

This is not functionally necessary, but it makes the repo easier to explain:

```text
contract.py:
  what TextOp means

textop_motion.py:
  how we load and validate TextOp motions

normalize_textop_motion.py:
  how we replay through MJLab and save MJLab-native motion
```

## 2. Make body velocities optional

Your current loader requires:

```python
"body_lin_vel_w",
"body_ang_vel_w",
```

But earlier we noted TextOp’s `deploy_mujoco.py` treats these as not always essential. So this is the biggest practical fix.

Change this:

```python
_require_keys(
    data,
    (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ),
)
```

to:

```python
_require_keys(
    data,
    (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
    ),
)
```

Then handle missing root velocities:

```python
root_pos_w = body_pos_w[:, 0].astype(np.float32)
root_quat_w = body_quat_w[:, 0].astype(np.float32)

if "body_lin_vel_w" in data:
    root_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, 0]
else:
    root_lin_vel_w = _finite_difference_linear_velocity(root_pos_w, resolved_fps)

if "body_ang_vel_w" in data:
    root_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, 0]
else:
    root_ang_vel_w = np.zeros_like(root_lin_vel_w, dtype=np.float32)
```

For now, setting missing angular velocity to zero is acceptable for a first robust loader. Later you can compute it from quaternions.

Add:

```python
def _finite_difference_linear_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[0] > 1:
        vel[:-1] = (pos[1:] - pos[:-1]) * fps
        vel[-1] = vel[-2]
    return vel
```

## 3. Validate joint arrays explicitly

Right now `reindex_textop_g1_joints_to_mjlab()` checks only the last dimension. Add validation that `joint_pos` and `joint_vel` are shaped like `[T, 29]`.

Something like:

```python
def _validate_joint_array(name: str, value: np.ndarray) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be shaped [T, J], got {value.shape}")
    if value.shape[-1] != len(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX):
        raise ValueError(
            f"{name} must have {len(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)} joints, "
            f"got {value.shape[-1]}"
        )
```

Then in `load_textop_motion()`:

```python
joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
_validate_joint_array("joint_pos", joint_pos)
_validate_joint_array("joint_vel", joint_vel)
```

Then use:

```python
joint_pos=reindex_textop_g1_joints_to_mjlab(joint_pos),
joint_vel=reindex_textop_g1_joints_to_mjlab(joint_vel),
```

## 4. Validate the reindex map once

Add this near the constants or in a test:

```python
def validate_textop_contract() -> None:
    mapping = TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
    expected = len(MJLAB_G1_JOINT_NAMES)

    if len(mapping) != expected:
        raise ValueError(f"Joint map has length {len(mapping)}, expected {expected}")

    if sorted(mapping) != list(range(expected)):
        raise ValueError("Joint map must be a permutation of 0..28")
```

Then call it from `load_textop_motion()` or, better, test it.

This catches silent bugs like duplicate joint indices or missing joints.

## 5. Add an output sanity check in the normalizer

After this line:

```python
np.savez(output_file, **log)
```

Optionally reload and check shapes:

```python
def _validate_normalized_output(path: Path) -> None:
    data = np.load(path)
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Normalized MJLab motion is missing keys: {missing}")

    num_frames = data["joint_pos"].shape[0]
    for key in required:
        if key == "fps":
            continue
        if data[key].shape[0] != num_frames:
            raise ValueError(
                f"Normalized output key {key} has inconsistent frame count: "
                f"{data[key].shape[0]} vs {num_frames}"
            )
```

Then:

```python
np.savez(output_file, **log)
_validate_normalized_output(output_file)
```

## 6. Add tests

This is probably the most important thing to add now.

Minimal tests:

```text
tests/test_textop_motion.py
```

Test these:

```text
joint reindex accepts [T, 29]
joint reindex rejects wrong joint count
contract mapping is a permutation
loader rejects missing required keys
loader accepts missing body velocities
loader normalizes quaternions
frame-count mismatch is rejected
```

You do not need a full MJLab sim test for every unit test. Keep most tests pure NumPy.

## 7. Do not add high-level TextOp yet

Your current result proves:

```text
TextOp reference file -> MJLab native tracking -> train/eval works
```

The next thing is not MDAR/text generation. The next thing is making this low-level integration **auditable and robust**.

So the priority order is:

```text
1. optional body velocity handling
2. contract constants split or at least documented
3. joint/body/fps validations
4. tests
5. only then TextOp-style command/obs terms
```

## One small naming issue

Your script file is named:

```text
normalize_textop_motion.py
```

but the function is:

```python
normalize_textop_npz(...)
```

That is fine, but for clarity I would pick one naming style:

```text
normalize_textop_npz.py
normalize_textop_npz()
```

or:

```text
normalize_textop_motion.py
normalize_textop_motion()
```

I prefer:

```text
normalize_textop_npz.py
```

because the input is specifically a TextOp NPZ file.

## Bottom line

You have already added the core contract. What you still need is:

```text
Make body velocities optional.
Validate joint/body/fps assumptions harder.
Move static constants into a contract module.
Add tests proving the TextOp -> MJLab mapping is correct.
```

You do **not** need to port `deploy_mujoco.py` wholesale, and you do **not** need to add high-level TextOp yet.
