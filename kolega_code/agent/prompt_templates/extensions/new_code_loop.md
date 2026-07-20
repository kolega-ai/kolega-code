## New Code Loop — Structured Feature-Building Methodology

When the user requests a new feature (keywords: build, create, add,
implement, feature, new, make, develop, write, generate, scaffold),
follow this structured four-phase methodology with parallel code
generation and independent verification.

### Workflow Overview

```
GOAL → GENERATE → VERIFY → SELECT → REPORT
```

Maximum 3 generate attempts. Workspace is reverted between failed attempts.

### Phase 0 — GOAL (The Blueprint)

1. Determine the feature specification from the user's request.
   If unclear, ask before proceeding.
2. Create a CONTRACT.md with specific, measurable criteria:
   - **Goal**: one sentence
   - **Boundaries**: exactly what IS and IS NOT in scope
   - **Success Criteria**: numbered, testable statements
   - **Acceptance**: what tests must pass
3. Snapshot the current workspace as a revert point.

### Phase 1 — GENERATE (Parallel)

1. Dispatch 2-3 coding sub-agents in parallel, each on a separate
   git branch (`loop/<task-id>-v<attempt>-<letter>`).
2. Each Generator must:
   - Implement the feature described in CONTRACT.md
   - Write automated tests covering the success criteria
   - Run tests to confirm they pass
   - Return: branch name, files changed, test count/pass count, summary

Use `dispatch_coding_agent` for Generators.

### Phase 2 — VERIFY (Parallel)

For each Generator branch, dispatch one investigation agent to grade
the implementation against CONTRACT.md:

- **All tests pass**: run the full test suite on the branch
- **Coverage >= 80%**: measure and report exact percentage
- **Code quality**: modular, legible, no God objects
- **Contract criteria**: verify each success criterion is met

All four must pass for a PASS. Verifiers return **numbers**, not
subjective language. "Coverage is 87%" is good. "Looks decent" is not.

### Phase 3 — SELECT & KEEP/REVERT

**If at least one branch passes**: Select the best candidate (highest
coverage, fewest issues). Merge into main. Clean up other branches.
Record success.

**If ALL branches fail**: Revert the workspace to the pre-attempt
snapshot. Return to Phase 1 for the next attempt (max 3 total).

### Phase 4 — REPORT

Print a comprehensive report: task ID, attempts, status, artifacts
created, test results (total/passed/coverage), summary, and recommended
next step.

### Key Principles

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
