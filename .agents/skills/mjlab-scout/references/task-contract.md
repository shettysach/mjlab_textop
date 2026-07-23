# TASK.md contract

Write a concise, self-contained Markdown prompt for the Phase 2 VLM. It must
contain these sections:

```markdown
# Objective

<one sentence>

# Environment

<visual description of the robot's surroundings and important landmarks>

# Success

<observable condition that means the task is complete>
```

Include only information available to an agent operating from rendered images.
Use qualitative relationships such as left, right, ahead, behind, near, and far.
Mention obstacles or route choices when they materially affect navigation.

Do not include:

- world coordinates, dimensions, body or geometry identifiers;
- source-code names or simulator implementation details;
- an exact route or oracle action sequence;
- tool instructions for Scout;
- Phase 1 analysis or uncertainty commentary.
