# TASK.md contract

`TASK.md` is the task-specific portion of the system prompt for a fresh Phase 2
VLM, not a Phase 1 scouting report. It must be concise, self-contained, and
grounded in the rendered images inspected during scouting. Phase 2 uses it while
controlling a robot through live forward-facing renders from an MJLab simulation
and allowed TextOp motion commands.

Keep each section brief. Write `Environment` as one short paragraph containing
only task-relevant recognition cues, not an exhaustive scene report or numbered
inventory.

Use exactly these three sections and no others:

```markdown
# Objective

<one sentence stating what the robot must accomplish>

# Environment

<concise visual description of the surroundings, target, distractors, and
obstacles without revealing the target's privileged location>

# Success

<observable condition or final pose that means the task is complete>
```

## Required content

- Describe the **particular environment actually observed**, not merely its task
  type or a generic statement that it contains corridors, goals, or obstacles.
- Describe the goal's visible appearance and distinguish it from distractors when
  the images support doing so.
- Describe the general structure, visible object types, obstacles, and
  distractors without mapping the goal to a particular location.
- State a success condition that Phase 2 can recognize from its live normal-camera
  images or from the intended final pose.
- Include only claims supported by the task information and rendered images. Do
  not invent missing visual or spatial details.

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
  names, or simulator implementation details; or
- an exact route, oracle action sequence, or sequence of mandatory motion
  commands; or
- the goal's privileged location, including which side, branch, room, alcove,
  station, corridor, or direction contains it.

Do not restate generic command-selection, output-format, collision-avoidance, or
corridor-following behavior. The invariant Phase 2 controller prompt already
provides those instructions.

Describe what Phase 2 should visually recognize, not where it should go. Phase 2
must discover the goal's location and choose its route from live observations.
