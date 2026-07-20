## Bug Fix Loop — Structured Bug-Fixing Methodology

When the user reports a bug (keywords: fix, bug, crash, error, broken,
regression, not working, debug, repair, patch, resolve, incorrect, wrong,
fails), follow this structured five-phase methodology to avoid target
fixation and fix bugs correctly on the first attempt.

### Workflow Overview

```
REPRODUCE → INVESTIGATE → ACT → CHECK → ADAPT → REPORT
```

### Phase 0 — REPRODUCE

1. Write a minimal automated test that reproduces the bug.
2. Run the test to confirm it fails — this confirms the bug is real.
3. If the bug cannot be reproduced, stop and ask for clarification.

### Phase 1 — INVESTIGATE (Broad Exploration)

**Do NOT attempt to fix yet.** Dispatch 2 investigation sub-agents in
parallel using `dispatch_investigation_agent`. Each should follow the
two-pass investigation methodology:

- **Pass 1 (Broad)**: Explore architecture, conventions, intended behavior,
  recent git changes, and analogous code WITHOUT tracing the error path.
- **Pass 2 (Narrow)**: Trace the error path, explore unexpected root causes,
  and generate 2-3 distinct fix hypotheses with confidence ratings.

Merge both diagnostic reports. Present ALL hypotheses to the fixing agent —
never discard alternatives.

**Scope**: NEIGHBORHOOD on first attempt. If a retry is needed, the Adapt
phase may escalate to SYSTEM scope for deeper investigation covering global
state, cross-cutting concerns, configuration, external dependencies, event
flows, test infrastructure, and full git history.

### Phase 2 — ACT (Surgical Fix)

Dispatch 1-2 coding sub-agents in parallel on separate git branches.
Give each agent:

- The merged diagnostic brief with ALL fix hypotheses
- Instructions to review all hypotheses, choose or combine the best one,
  and justify the choice
- The Rule of Least Leverage: make the smallest possible change
- Instructions to check risk areas from the diagnostic brief

Use `dispatch_coding_agent` for these agents. If a fix doesn't work,
try a DIFFERENT hypothesis rather than iterating on a failing approach.

### Phase 3 — CHECK (Differential Verification)

For each fix candidate, dispatch an investigation agent to verify:

- **Check A**: The reproduction test now PASSES
- **Check B**: The ENTIRE test suite still passes — no regressions

Both checks must pass. A fix that passes Check A but fails Check B is
rejected as a regression.

### Phase 4 — ADAPT (Scope Escalation)

If ALL fixes fail:

1. Analyze why. Was the investigation accurate? Which hypotheses were tried?
2. Determine if there are unattempted hypotheses from the diagnostic brief.
3. Decide whether to escalate investigation scope:
   - **NEIGHBORHOOD**: Try a different hypothesis from the existing set.
   - **SYSTEM**: The root cause may be outside the local neighborhood.
     Re-investigate with broader scope — check global state, configuration,
     cross-cutting concerns, external dependencies, event flows, test
     infrastructure, and full git history.
4. Revert changes (git checkout main, delete fix branches) and retry.
   Default: escalate to SYSTEM if all fixes failed for reasons unrelated
   to fix quality (e.g., "fix was correct but bug persisted").
5. Maximum 2 total fix attempts.

### Phase 5 — REPORT

Print a comprehensive report: bug ID, attempts, investigation scope and
findings, fix file/line, hypothesis used, root cause, prevention rule,
and regression suite results (total/passed/failed).

### Key Principles

- **Investigation always before fixing** — understand the system broadly
  before attempting any code change
- **Multiple hypotheses** — never converge on one fix too early; the fixing
  agent must review all options
- **Least leverage** — the smallest change that correctly fixes the bug
- **Zero regressions** — both the repro test AND the full suite must pass
- **Automatic scope escalation** — if the first fix fails, investigate more
  broadly on the retry
