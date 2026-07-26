# TASK.md contract

`TASK.md` is the task-specific portion of the system prompt for a fresh Phase 2
VLM, not a Phase 1 scouting report. It must be concise, self-contained, and
grounded in the rendered images inspected during scouting. Phase 2 uses it while
controlling a robot through live forward-facing renders from an MJLab simulation
and allowed TextOp motion commands.

Keep each section brief and use prose, not an exhaustive or numbered inventory.

Use exactly these four sections and no others:

```markdown
# Objective

<one sentence preserving the required end state>

# Environment

<initial surroundings and useful scene structure, openings, obstacles,
landmarks, and distractors>

# Target

<visual appearance and distinguishing cues without location information>

# Success

<observable condition or final pose that means the task is complete>
```

## Content rules

- Preserve the supplied objective exactly. Use the images to ground it, not to
  broaden, narrow, or reinterpret it.
- Describe the particular scene, including the robot's initial surroundings and
  navigation-relevant topology. Spatial detail is allowed for walls, openings,
  turns, obstacles, and landmarks, but not for locating the target.
- Keep target appearance separate from target location. Give only useful,
  supported cues that distinguish it from distractors.
- Treat the images as separate viewpoints of one scene. Do not present a
  privileged camera composition as the robot's current view.
- Use exact counts only when unambiguous and relevant. Omit uncertain details
  rather than guessing.
- State success using evidence available from the live forward-facing camera and
  the requested final pose. Do not invent numerical distances, tolerances, or
  hidden reward conditions. Merely seeing the target is not completion unless
  the objective says so.

## Prohibited content

Do not include:

- Scout, MCP, tools, camera/view names, view availability, overview/overhead
  images, presets, or instructions to inspect any of them;
- Phase 1, Phase 2, scouting analysis, uncertainty commentary, or references to
  another prompt or execution context;
- instructions to begin scouting, plan with privileged views, issue commands, or
  execute the task during prompt generation;
- an agent persona, generic capabilities, procedural boilerplate, or sections
  such as `Views Available`, `Instructions`, or `Constraints and Rules`;
- world coordinates, exact dimensions, body/geometry identifiers, source-code
  names, or simulator implementation details;
- a route, search order, action plan, or sequence of motion commands; or
- the target's privileged location, directly or indirectly, including its side,
  branch, room, corridor, direction, ordering, adjacency, nearby landmark, or
  required turns.

Do not restate generic command-selection, output-format, collision-avoidance, or
corridor-following behavior. The invariant Phase 2 controller prompt already
provides those instructions.

Describe what Phase 2 should visually recognize, not where it should go. Phase 2
must discover the goal's location and choose its route from live observations.
