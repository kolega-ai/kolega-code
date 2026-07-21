---
name: new-code-loop
description: Structured feature-building with parallel generation and independent verification — Goal → Generate → Verify → Select → Report
---

# New Code Loop — Structured Feature-Building Methodology

## Workflow Overview

```
GOAL → GENERATE → VERIFY → SELECT → REPORT
```

Maximum **3 generate attempts**. Workspace is reverted between failed attempts.

## Loop State Tools

The following tools are available to manage loop state deterministically:

- `loop_state_init(task_id, loop_type, max_attempts=3)` — initialize a new loop
- `loop_state_attempt(task_id)` — increment attempt counter, check limit
- `loop_state_backup(task_id)` — snapshot current state as revert point
- `loop_state_revert(task_id)` — return command to revert to last good state
- `loop_state_log(task_id, status, summary, phase="")` — record attempt history
- `loop_state_status(task_id)` — full work-log state

Call `loop_state_init` at the start of the workflow with `loop_type="new-code"`
and `max_attempts=3`. Call `loop_state_attempt` before each generate cycle —
if it returns `{"exceeded": true}`, stop immediately.

---

## Phase 0 — GOAL (The Blueprint)

1. Determine the feature specification from the user's request.
   If unclear, ask before proceeding.
2. Create a CONTRACT.md in the project root with specific, measurable criteria:
   - **Goal**: one sentence describing what to build
   - **Boundaries**: exactly what IS and IS NOT in scope
   - **Success Criteria**: numbered, testable statements (3-7 items)
   - **Acceptance**: what tests must pass
3. Call `loop_state_backup(task_id)` to snapshot the current workspace.

---

## Phase 1 — GENERATE (Parallel)

1. Call `loop_state_attempt(task_id)`. If `{"exceeded": true}`, stop and report.
2. Dispatch 2-3 coding sub-agents in parallel, each on a separate
   git branch (`loop/<task-id>-v<attempt>-<letter>`).
3. Each Generator must:
   - Implement the feature described in CONTRACT.md
   - Write automated tests covering the success criteria
   - Run tests to confirm they pass
   - Return: branch name, files changed, test count/pass count, summary

Use `dispatch_coding_agent` for Generators.

---

## Phase 2 — VERIFY (Parallel)

For each Generator branch, dispatch one investigation agent to grade
the implementation against CONTRACT.md:

- **All tests pass**: run the full test suite on the branch
- **Coverage >= 80%**: measure and report exact percentage
- **Code quality**: modular, legible, no God objects
- **Contract criteria**: verify each success criterion is met

All four must pass for a PASS. Verifiers return **numbers**, not
subjective language. "Coverage is 87%" is good. "Looks decent" is not.

---

## Phase 3 — SELECT & KEEP/REVERT

**If at least one branch passes**: Select the best candidate (highest
coverage, fewest issues). Merge into main. Clean up other branches.
Call `loop_state_log(task_id, "kept", summary, phase="select")`.

**If ALL branches fail**: Call `loop_state_revert(task_id)` and execute
the returned command to revert to the pre-attempt snapshot. Return to
Phase 1 for the next attempt (max 3 total).

---

## Phase 4 — REPORT

Print a comprehensive report:
- Task ID
- Attempts made
- Status (success/failure)
- Artifacts created
- Test results (total/passed/coverage)
- Selected branch and rationale
- Summary
- Recommended next step

Call `loop_state_status(task_id)` and include the full work-log in
the report.

## Key Principles

- **Parallel generation** — multiple independent implementations
  increase the chance of a correct, high-quality result
- **Independent verification** — the agent that writes the code is
  never the agent that grades it
- **Numbers over opinions** — verifiers must report metrics, not
  subjective assessments
- **Clean slate on failure** — revert to known-good state between
  attempts; never accumulate broken state
- **Maximum 3 attempts** — if three full generate-verify cycles all
  fail, escalate to the user
