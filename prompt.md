Use the mjlab-scout skill and the mjlab-scout MCP tools to inspect the
`portrait-corridors` task and create TASK.md in the current working directory.

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

TASK.md must be concise and self-contained. Follow the format.
Clearly analyze anad explain the environment.

Do not attempt to navigate or advance the simulation. Write TASK.md, verify that
it is sufficient for a fresh Phase 2 VLM with only its live agent-camera images,
and then call close_task.
