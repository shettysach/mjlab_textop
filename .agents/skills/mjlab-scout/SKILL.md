---
name: mjlab-scout
description: Inspect an MJLab TextOp task with the mjlab-scout MCP tools and write a visually grounded, task-specific TASK.md prompt for a separate VLM execution context. Use when preparing or regenerating a task prompt for any MJLab TextOp environment.
---

# MJLab Scout

Inspect one task and create `TASK.md` for a later live navigation run. Read the
complete [TASK contract](references/task-contract.md) before writing it.

## The two phases

**Phase 1 is scouting.** Use the `mjlab-scout` MCP tools and all returned camera
preset views to understand the static task, then write `TASK.md`. Inspect the
actual images to learn the robot's starting perspective, the target's appearance,
the general environment, landmarks, obstacles, distractors, and observable
success condition. Do not navigate, send motion commands, advance the simulation,
or inspect source code. Phase 1's only deliverable is `TASK.md`.

The normal `agent` view resembles what the robot can see during live control.
Views such as `overview`, `overhead`, and task-specific presets are privileged
Phase 1 views. Use them to understand the environment, but never mention their
names, existence, or images in `TASK.md`.

**Phase 2 is live navigation.** It starts with a clean context and combines a
VLM, TextOp/RobotMDAR, and an MJLab simulation. The VLM receives the invariant
controller prompt, `TASK.md`, a live image from the simulated robot's normal
forward-facing camera, and the allowed TextOp motion commands. It selects one
command, TextOp/RobotMDAR generates the motion, and MJLab executes it.

Phase 2 does not receive the Phase 1 conversation or reasoning, Scout tools,
overview or overhead images, task-specific preset views, view names, or any way
to request those views. Therefore, `TASK.md` must turn the privileged Phase 1
inspection into a self-contained visual description usable from the live robot
camera.

Describe the robot's initial surroundings, general layout, openings, obstacles, landmarks, 
target appearance, distractors, and observable success condition. 
Spatial detail is useful for the environment, but never bind the target 
to a privileged location or tell Phase 2 where to go.

## Procedure

1. Call `list_tasks` only if the task name is unknown, then call `load_task` once.

2. Inspect **ALL camera preset views** returned by `load_task`, including `agent`,
   `overview`, `overhead`, and every task-specific preset. Only then compare the
   images and infer the environment, target, obstacles, distractors, and success
   condition. Reconcile them as separate perspectives of one scene; do not treat
   a privileged preset composition as the robot's current view.

3. Build a compact scene model containing only useful visual and structural
   facts. Use exact counts only when clear and relevant, and omit uncertain
   details. Keep target appearance separate from target location or route.

4. Write or replace `TASK.md` in the workspace root exactly according to the
   contract. Describe what Phase 2 needs to recognize, not a privileged target
   location, route, action plan, or the views and tools used to discover it.

5. Call `close_task` after writing. If scouting or writing fails, still call
   `close_task` before reporting the failure.
