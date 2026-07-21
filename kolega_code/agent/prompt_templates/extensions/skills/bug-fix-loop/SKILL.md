---
name: bug-fix-loop
description: Structured bug-fixing with mandatory investigation — Reproduce → Investigate → Act → Check → Adapt → Report. Investigation before fixing is REQUIRED, never optional.
---

# Bug Fix Loop — Structured Bug-Fixing Methodology

## ⚠️ CRITICAL: Investigation Is MANDATORY

You MUST complete the full two-pass investigation (Pass 1: Broad System
Understanding + Pass 2: Multiple Fix Hypotheses) BEFORE writing any fix code.

DO NOT skip investigation and jump to a fix. Target fixation is the #1 cause
of failed bug fixes. If you find yourself wanting to "just try this quick fix"
— STOP. You are target-fixating. Complete the investigation first.

The investigation phase is **not** "nice to have" or "if time permits." It is
the CORE of this methodology and cannot be skipped, shortened, or bypassed.

## Workflow Overview

```
REPRODUCE → INVESTIGATE → ACT → CHECK → ADAPT → REPORT
```

Maximum **2 fix attempts**. Between failed attempts, escalate investigation
scope. Never attempt more than 2 fixes — after 2 failures, report findings
and hand back to the user.

## Loop State Tools

The following tools are available to manage loop state deterministically:

- `loop_state_init(task_id, loop_type, max_attempts=2)` — initialize a new loop
- `loop_state_attempt(task_id)` — increment attempt counter, check limit
- `loop_state_backup(task_id)` — snapshot current state as revert point
- `loop_state_revert(task_id)` — return command to revert to last good state
- `loop_state_log(task_id, status, summary, phase="")` — record attempt history
- `loop_state_anti_pattern(task_id, pattern, root_cause, file, line, prevention_rule)` — record anti-pattern
- `loop_state_check_anti_patterns(task_id, module="")` — query past anti-patterns
- `loop_state_status(task_id)` — full work-log state

Call `loop_state_init` at the start of the workflow. Call `loop_state_attempt`
before each fix attempt — if it returns `{"exceeded": true}`, stop immediately.

---

## Phase 0 — REPRODUCE

1. Write a minimal automated test that reproduces the bug.
2. Run the test to confirm it fails — this confirms the bug is real.
3. If the bug cannot be reproduced, stop and ask for clarification.
4. Call `loop_state_init` with a short task ID derived from the bug description.

---

## Phase 1 — INVESTIGATE (MANDATORY — DO NOT SKIP)

**Critical: You MUST complete this phase BEFORE writing any fix code.**

Dispatch **2 investigation sub-agents** in parallel using
`dispatch_investigation_agent`. Each sub-agent must follow the two-pass
methodology below. Provide each sub-agent with the bug description, the
reproduction test, and the scope (`NEIGHBORHOOD` on first attempt).

### Two-Pass Investigation Methodology

**Pass 1 — Broad System Understanding**

Explore the system around the bug WITHOUT tracing the error path. Build a
mental model of the codebase first.

1. **Architecture & Conventions**: Read the affected module and its neighbors.
   Identify their roles, design patterns, boundaries, and coding conventions.
   What architectural assumptions does this code make?

2. **Intended Behavior**: Find documentation, comments, commit messages, specs,
   or related tests that describe what this code is SUPPOSED to do. Identify
   any gap between intended behavior and observed behavior.

3. **Recent Changes**: Run `git log --oneline -20` and `git blame` on affected
   files. Look for recent commits that modified the area. When was the code
   last changed and why? Could a recent change have introduced the bug?

4. **Related Features & Analogous Code**: Find modules similar to the affected
   one. How do they handle the same scenario? Are they doing something the
   buggy module isn't? Flag any analogous code that might have the same bug.

**Pass 2 — Multiple Fix Hypotheses**

Now trace the error path and generate MULTIPLE distinct fix hypotheses.

1. **Error Path Trace**: Starting from the reproduction, trace the full
   execution path to the error. Identify where correctness breaks. Determine
   if the root cause is at the error site or upstream.

2. **Unexpected Root Causes**: Ask: could the root cause be somewhere else?
   Configuration? Data/state issue? Async timing? A dependency?

3. **Generate 2-3 Fix Hypotheses**: Each must be a DIFFERENT approach:
   - Hypothesis 1: APPROACH, RATIONALE, RISKS, VERIFICATION, CONFIDENCE
   - Hypothesis 2: (different approach)
   - Hypothesis 3: (optional, different approach)

   Distinct approaches include: fix symptom vs. fix caller, add guard clause
   vs. change data flow, local patch vs. extract shared logic, fix code vs.
   fix test expectations.

### Output Format

Each investigation sub-agent should return a diagnostic report:

```
ROOT CAUSE: <1-3 sentence explanation>
ERROR PATH: <file:line → file:line → error>
FIX HYPOTHESES: (2-3 distinct, ranked by confidence)
OVERALL CONFIDENCE: HIGH | MEDIUM | LOW
```

**Merge** the results from both investigation sub-agents. Present ALL
hypotheses. Never discard alternatives before the fixing phase.

---

## Phase 2 — ACT (Surgical Fix)

**Only proceed after Phase 1 is complete and you have 2-3 fix hypotheses.**

1. Call `loop_state_attempt(task_id)`. If `{"exceeded": true}`, stop.
2. Call `loop_state_backup(task_id)` to snapshot the current state.
3. Dispatch 1-2 coding sub-agents in parallel on separate git branches
   (`loop/<task-id>-v<attempt>-<letter>`).

Give each coding agent:
- The merged diagnostic brief with ALL fix hypotheses
- Instructions to review all hypotheses, choose or combine the best one,
  and justify the choice
- The **Rule of Least Leverage**: make the smallest possible change
- Instructions to check risk areas from the diagnostic brief

If a fix doesn't work, try a DIFFERENT hypothesis rather than iterating
on a failing approach.

---

## Phase 3 — CHECK (Differential Verification)

For each fix candidate, dispatch an investigation agent to verify:

- **Check A**: The reproduction test now PASSES
- **Check B**: The ENTIRE test suite still passes — no regressions

Both checks must pass. A fix that passes Check A but fails Check B is
rejected as a regression.

---

## Phase 4 — ADAPT (Scope Escalation)

If ALL fixes fail:

1. Analyze why. Was the investigation accurate? Which hypotheses were tried?
2. Determine if there are unattempted hypotheses from the diagnostic brief.
3. Decide whether to escalate investigation scope:
   - **NEIGHBORHOOD**: Try a different hypothesis from the existing set.
   - **SYSTEM**: The root cause may be outside the local neighborhood.
     Re-investigate with broader scope — check global state, configuration,
     cross-cutting concerns, external dependencies, event flows, test
     infrastructure, and full git history.
4. Call `loop_state_revert(task_id)` and execute the returned command to
   revert to the pre-attempt state (this only discards loop branch changes).
5. Return to Phase 1 with the escalated scope. Max 2 total fix attempts.
6. If `loop_state_attempt` returns `{"exceeded": true}`, stop and report.

---

## Phase 5 — REPORT

Print a comprehensive report:

- Bug ID
- Attempts made
- Investigation scope and key findings
- Fix file/line and which hypothesis was used
- Root cause explanation
- Prevention rule (anti-pattern to avoid this class of bug)
- Regression suite results (total/passed/failed)
- Call `loop_state_log(task_id, "kept", summary, phase="report")` on success

If all attempts failed, call `loop_state_status(task_id)` and include the
full work-log in the report.

## Key Principles

- **Investigation always before fixing** — understand the system broadly
  before attempting any code change. THIS IS MANDATORY.
- **Multiple hypotheses** — never converge on one fix too early; the fixing
  agent must review all options.
- **Least leverage** — the smallest change that correctly fixes the bug.
- **Zero regressions** — both the repro test AND the full suite must pass.
- **Automatic scope escalation** — if the first fix fails, investigate more
  broadly on the retry.
- **Record lessons** — after every fix (success or failure), call
  `loop_state_anti_pattern` to record the root cause and prevention rule.
