## A faster evaluation design

### 1. Create a frozen scouting benchmark

Run MJLab once per task and save:

- Every preset camera image.
- The task objective.
- The list of expected view names.
- Target and distractor annotations.
- A few normal forward-camera images representing target-visible, distractor-visible, and success-pose situations.

Then expose these through a mock Scout MCP server that behaves exactly like the real one but only serves cached data and records calls.

After creating this dataset, GEPA evaluations require no MJLab simulation.

### 2. Mechanically grade the Scout trace

These checks are fully verifiable:

- `load_task` called exactly once.
- Every view returned by `load_task` captured before writing.
- No unknown or repeated views.
- No movement or source-inspection tools.
- `close_task` always called.
- `TASK.md` created.
- Required headings appear exactly once.
- Word-count limit satisfied.
- No literal view names, phase terminology, tool names, or forbidden sections.

This verifies procedural compliance without an LLM judge.

It cannot prove that the model mentally considered every image, but it can prove that every image was placed in its context before writing. Output grounding tests cover the remaining part.

### 3. Use a multimodal grader for semantic quality

Give a separate VLM:

- All cached scouting images.
- The task objective.
- The generated `TASK.md`.
- A structured rubric.

Have it return JSON fields such as:

- `claims_supported`
- `unsupported_claims`
- `target_recognizable`
- `distractors_distinguished`
- `environment_grounded`
- `goal_location_leaked`
- `route_prescribed`
- `success_observable`
- `unnecessarily_detailed`

Crucially, ask it to identify offending sentences, not merely return a score. Those diagnostics become GEPA’s Actionable Side Information. GEPA evaluators officially support returning a numeric score plus structured diagnostic feedback. [GEPA evaluator API](https://gepa-ai.github.io/gepa/api/optimize_anything/Evaluator/)

### 4. Add a cheap Phase 2 proxy

Instead of navigating, test whether `TASK.md` contains the right information for navigation:

- Show the grader shuffled target and distractor crops and ask it to select the target using only `TASK.md`.
- Show normal forward-camera frames and ask whether the goal is visible.
- Show near-goal and non-goal frames and ask which satisfies the stated success condition.
- Give only `TASK.md` to a relation extractor and check whether it reveals a target-to-location relation.

These are short, independent VLM calls. They directly test recognition and leakage—the aspects of Phase 2 that `TASK.md` controls—without running TextOp or motion generation.

### 5. Reserve real Phase 2 for final validation

After GEPA produces its best few candidates:

- Manually inspect them.
- Run perhaps one real Phase 2 episode per task for the final candidate.
- Keep those results out of the optimization loop.

That avoids hundreds of slow navigation episodes while still providing an end-to-end sanity check before adoption.

## Guarding against evaluator gaming

An LLM grader alone can be exploited, so I would combine it with hard checks:

- Treat oracle location leakage as a hard failure, even if other quality scores are high.
- Use a grader model different from the Scout model when possible.
- Calibrate the grader against a small set of human-rated good and bad `TASK.md` outputs.
- Keep a held-out task or use leave-one-task-out evaluation.
- Do not optimize the skill and grader simultaneously.
- Freeze the skill frontmatter and required contract structure.
- Prevent GEPA from copying task-specific examples into the generic skill.

GEPA’s own guidance recommends calibrating LLM judges with human-labelled examples and returning explicit anti-overfitting constraints in evaluator feedback. [GEPA evaluator guidance](https://gepa-ai.github.io/gepa/guides/faq/)

So the practical setup is:

```text
Cached views + mock MCP
        ↓
Actual skill-driven Scout rollout
        ↓
Mechanical trace/format checks
        ↓
Multimodal grounding and leakage grader
        ↓
Cheap forward-camera recognition tests
        ↓
GEPA revision
```

This evaluates the artifact we actually care about, keeps the deployed skill pure Markdown, and makes Phase 2 optional until final validation.
