---
name: mjlab-scout
description: Inspect an MJLab TextOp task with the mjlab-scout MCP tools and write a visually grounded, task-specific TASK.md prompt for a separate VLM execution context. Use when preparing or regenerating a task prompt for any MJLab TextOp environment.
---

# MJLab Scout

Inspect one task and create a concise, visually grounded `TASK.md` for a later
live navigation run. Phase 1's only deliverable is `TASK.md`.

## The two phases

- **Phase 1: scouting.** Use the `mjlab-scout` MCP tools to inspect the static
  task through every returned camera preset. Use these views to understand the
  initial surroundings, layout, openings, obstacles, landmarks, target,
  distractors, and observable success condition. Do not navigate, send motion
  commands, advance the simulation, or inspect source code.
- **Phase 2: live navigation.** A clean-context VLM controls an MJLab simulation
  through TextOp/RobotMDAR. It receives the invariant controller prompt,
  `TASK.md`, a live image from the robot's normal forward-facing camera, and the
  allowed motion commands.
- **Phase 2 does not have** the Phase 1 conversation or reasoning, Scout tools,
  camera presets, overview or overhead images, view names, or any way to request
  those views. `TASK.md` must therefore carry only the visual knowledge Phase 2
  needs while leaving it to discover the target's location and route live.

## Procedure

1. Call `list_tasks` only if the task name is unknown, then call `load_task` once.

2. Inspect **ALL camera preset views** returned by `load_task` with
   `capture_view`. Only then begin interpreting the scene or drafting `TASK.md`.
   Treat the images as separate perspectives of one environment, not as the
   robot's current live view.

3. Match the target to the supplied objective across all views. A centered,
   nearby, or prominent object is not the target unless it matches the objective.
   Treat visible non-matches as distractors.

4. Build a compact description using only supported visual facts. Describe the
   initial surroundings and useful scene structure, but keep the target's
   appearance separate from its location or route.

5. Write or replace `TASK.md` in the workspace root using the mandatory format
   below.

6. Call `close_task` after writing. If scouting or writing fails, still call
   `close_task` before reporting the failure.

## Mandatory TASK.md format

Write raw Markdown with exactly four first-level headings, each appearing once
and in this order. Do not add an outer title or code fence.

1. `# Objective` — One sentence preserving the supplied required end state.
2. `# Environment` — Brief prose describing the initial surroundings, useful
   layout, openings, obstacles, landmarks, and distractors.
3. `# Target` — Brief prose describing the target's visible appearance and
   distinguishing cues without revealing its location.
4. `# Success` — Brief prose stating evidence of completion observable from the
   live forward-facing camera and required final pose.

## TASK.md rules

- Ground every detail in the inspected images. Omit uncertain details instead of
  guessing; use exact counts only when unambiguous and useful.
- Preserve the objective. Use the images to identify and describe its target,
  not to broaden, narrow, or reinterpret the task.
- Describe navigation-relevant environment structure, but never bind the target
  to a side, direction, branch, room, corridor, ordering, adjacency, landmark,
  turn sequence, or other privileged location clue.
- Describe what Phase 2 should visually recognize, not where it should go. Do
  not include a route, search order, action plan, or motion commands.
- Do not present a privileged camera composition as the robot's current view.
- Make success stricter than merely seeing the target unless the objective says
  otherwise. Do not invent distances, tolerances, or hidden reward conditions.
- Do not mention scouting, phases, MCP, tools, cameras, presets, view names,
  unavailable context, or prompt-generation instructions.
- Do not add personas, execution instructions, generic navigation advice,
  simulator or robot make/model details, coordinates, dimensions, source-code
  names, implementation details, uncertainty commentary, or extra sections.
