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
3. Read the view names returned by `load_task`. Capture each `inspection_*` view
   when present; these show important task features from inside the environment.
4. Capture `overview` to understand the surrounding layout. Use `overhead` only
   when routes or spatial relationships remain ambiguous.
5. Infer the useful visual landmarks, obstacles, route constraints, and stopping
   condition from the images.
6. Write `TASK.md` according to the contract. Describe the environment in visual,
   qualitative terms and do not include coordinates, hidden implementation details,
   an oracle route, or a sequence of mandatory actions.
7. Call `close_task` after writing the file, including when exploration fails.

Normally two images are sufficient. Request a third only when it resolves a
specific ambiguity. Do not modify the simulation, advance it, or inspect source
code to solve the task.
