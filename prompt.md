Use the mjlab-scout skill and the mjlab-scout MCP tools to inspect the
`<TASK_NAME>` task and create TASK.md in the current working directory. Replace
`<TASK_NAME>` with the task to inspect before starting Phase 1.

This is Phase 1: environment scouting and prompt generation. Your job is not to
execute the navigation task. Your only deliverable is TASK.md, which will become
the task-specific system prompt for a separate Phase 2 VLM run.

Phase 2 will start with a completely clean context. It will receive TASK.md and
live images from the agent’s normal observation camera. It will control the
agent by selecting motion commands. It will not have access to the Scout MCP,
the overview or overhead cameras, any task-specific camera presets, your Phase 1
reasoning, or this request. Therefore, TASK.md must contain all task and
environment information that Phase 2 needs.

Load the requested task and read the available view names returned by
`load_task`. Inspect them as follows:

- Use `agent`, when available, to understand the agent’s starting perspective
  and forward direction.
- Use `overview`, when available, to understand the overall layout.
- Capture every additional task-specific preset view returned by `load_task`.
  Use these views to inspect important regions and visual landmarks that may not
  be visible from the initial agent view.
- Use `overhead`, when available, only if a route or spatial relationship
  remains unclear.

Compare the views carefully. Any view other than the normal agent view is a
privileged Phase 1 inspection view and will not exist during Phase 2. Use the
privileged views to produce an accurate qualitative description, but do not
mention Scout, MCP, camera names, or privileged views in TASK.md.

TASK.md must be concise and self-contained, with exactly these sections:

# Objective

State what the agent must accomplish.

# Environment

Describe the environment as Phase 2 will encounter it: the initial surroundings,
important visible landmarks, environment structure, route choices, obstacles, and
the visual characteristics needed to recognize the intended destination. Use
qualitative spatial language such as ahead, left, right, near, and far.

# Success

Describe the observable final condition that means the task is complete,
including where the agent should stop relative to the intended target.

Only include information grounded in the rendered views and the task objective.
Do not include coordinates, dimensions, simulator names, source-code details,
camera names, MCP instructions, Phase 1 analysis, uncertainty commentary, an
exact action sequence, or a mandatory oracle route.

Do not attempt to navigate or advance the simulation. Write TASK.md, verify that
it is sufficient for a fresh Phase 2 VLM with only its live agent-camera images,
and then call close_task.
