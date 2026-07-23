---
name: mjlab-scout
description: Inspect an MJLab TextOp task with the mjlab-scout MCP tools and write the task-specific TASK.md prompt for a fresh VLM execution context. Use when preparing or regenerating a task prompt from an unknown straight, blocked-straight, side-goals, turn, or portrait-corridors environment.
---

# MJLab Scout

Explore one task without changing or executing it, then write `TASK.md` in the
current working directory. Read [TASK contract](references/task-contract.md)
before writing the file.

## Workflow

1. Call `list_tasks` if the requested task name is unknown.
2. Call `load_task` once with the selected task.
3. Call `get_scene_summary` to learn the objective and structured geometry.
4. Call `capture_view` with `agent` first.
5. Call `capture_view` with `overview` when the route, goal, or choices are not
   clear. Use `overhead` only when spatial layout remains ambiguous.
6. Infer the useful visual landmarks, obstacles, route constraints, and stopping
   condition. Prefer what the images show; use geometry only for spatial context.
7. Write `TASK.md` according to the contract. Describe the environment in visual,
   qualitative terms and do not include coordinates, hidden implementation details,
   an oracle route, or a sequence of mandatory actions.
8. Call `close_task` after writing the file, including when exploration fails.

Normally two images are sufficient. Request a third only when it resolves a
specific ambiguity. Do not modify the simulation, advance it, or inspect source
code to solve the task.
