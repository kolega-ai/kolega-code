---
title: Goal-Conditioned Work
description: Set an autonomous completion goal the agent works toward until it is verifiably met.
---

Set a verifiable completion condition and the agent works autonomously toward it,
verifying its own progress after every turn, until the goal is met, a turn cap is
hit, or you stop it. It's available as the `/goal` slash command in the
[TUI](../tui/interface/) and as `kolega-code ask --goal` from the
[CLI](../cli/ask/).

## How it works

The goal loop runs after each completed work turn:

1. **Work turn** — the agent uses its full toolset (read, edit, run commands, run
   tests, dispatch sub-agents) to make progress toward the goal.
2. **Verification** — a **read-only investigation sub-agent** inspects the current
   codebase state and decides whether the goal is met, ending its reply with a
   JSON verdict: `{"ok": true}` or `{"ok": false, "reason": "<remaining gap>"}`.
3. **Met** — the goal completes, a confirmation message is shown, and the
   active-goal prompt extension is dropped so subsequent turns are normal again.
4. **Not met** — the agent is nudged to continue. The nudge includes the
   verifier's stated remaining gap and the number of turns left before the cap, so
   the agent can prioritize accordingly. The loop repeats from step 1.

A **turn cap** (default **50**) is a safety backstop. If the goal isn't met within
that many evaluation turns, the loop pauses and tells you to refine the goal or
clear it.

## Safety model

:::note
The verifier is a **read-only investigation sub-agent**. It can read files, search
the code, and run commands or tests, but it **cannot edit** — so it can never game
an autonomous goal by falsifying its own completion.
:::

A few guarantees that keep the loop safe:

- The verifier inspects the codebase **fresh each evaluation** (stateless across
  turns), so it judges the real current state, not a stale snapshot.
- A malformed, unparseable, or failed verdict is **always treated as not-met**. A
  broken evaluator can never falsely complete a goal — the loop keeps running or
  hits the cap.
- The verifier runs on the configured long-context model, separate from the
  working agent's turn.

## In the TUI: `/goal`

Type `/goal` in the [composer](../tui/composer/) to set, check, or clear a goal.

| Form | Effect |
| --- | --- |
| `/goal <condition>` | Set a goal and start working toward it |
| `/goal` | Show goal status (condition, runtime, turns evaluated, tokens spent, verifier's latest reason) |
| `/goal clear` | Remove the goal (aliases: `stop`, `off`, `reset`, `none`, `cancel`) |
| `/goal -p <condition>` | Run-to-completion mode (alias: `--print`) — no pauses until the goal is met or capped |

A few things to know:

- **Esc pauses** the goal loop after the current turn finishes. Sending any
  message **resumes** it from where it left off.
- You can't set or clear a goal while a turn is active — stop the turn first.
- Setting a new goal replaces an active unmet one.
- When the turn cap is reached, the goal is **paused** with a note. Refine it with
  `/goal <condition>` or remove it with `/goal clear`.
- The **status dashboard** (Status side-panel tab) shows a `Goal` line with the
  condition (truncated if long) and a state label — `active`, `paused`, or `met`.
- **Persistence**: goal state is saved with the session and restored on resume, so
  the loop picks up where it left off. See
  [Sessions & Resuming](../tui/sessions-and-resume/).

```text
/goal all tests pass and the linter is clean
```

## From the CLI: `ask --goal`

```bash
kolega-code ask --goal "<condition>" [--goal-max-turns N] [options]
```

The positional `prompt` is **optional** when `--goal` is given — the CLI
synthesizes the first work-turn message from the condition. You can still pass a
prompt to give the agent a head start:

```bash
# No prompt needed — the condition drives the loop
kolega-code ask --goal "all tests pass and ruff is clean" --project .

# Lower the turn cap for a bounded task
kolega-code ask --goal "the failing test in test_parser.py passes" --goal-max-turns 10 --json

# Give the agent a starting point, then let the goal loop take over
kolega-code ask "start by fixing the parser" --goal "all tests pass" --project .
```

| Option | Description |
| --- | --- |
| `--goal <condition>` | Set an autonomous completion goal and loop until it is met or capped (no prompt required) |
| `--goal-max-turns <N>` | Maximum evaluation turns before an unmet goal gives up (default 50) |

### Plain output

In plain (non-JSON) mode, after each evaluation the CLI prints a line to
**stderr**:

```text
[goal] turn 1: not met — test_parser.py still fails on test_parse_empty
[goal] turn 2: not met — two tests still failing in test_parser.py
```

When the loop ends, a final summary line is printed:

```text
[goal] met after 3 turn(s)
# or
[goal] not met (turn cap reached) after 10 turn(s)
```

The agent's response text is written to **stdout** as usual, so piping stdout still
gives you just the answer.

### Exit code

With `--goal`, the exit code reflects the outcome:

| Code | Meaning |
| --- | --- |
| `0` | The goal was met |
| `1` | The turn cap was reached without meeting the goal |

## JSON output

With `--json`, the command streams newline-delimited JSON objects as usual, plus
two goal-specific `kind` values alongside the existing `chunk`, `event`, and
`summary` kinds:

| `kind` | Meaning |
| --- | --- |
| `goal_eval` | Emitted after each evaluation: `{met, turns, reason}` |
| `goal_result` | Final outcome: `{met, turns, reason}` |

Example stream (abbreviated):

```json
{"kind": "chunk", "data": {"type": "response", "content": "Fixing the parser…"}}
{"kind": "goal_eval", "data": {"met": false, "turns": 1, "reason": "test_parse_empty still fails"}}
{"kind": "chunk", "data": {"type": "response", "content": "Added the empty-input guard…"}}
{"kind": "goal_eval", "data": {"met": true, "turns": 2, "reason": ""}}
{"kind": "goal_result", "data": {"met": true, "turns": 2, "reason": ""}}
{"kind": "summary", "chunks": 4, "session_id": "abc123"}
```

## Tips

- Write conditions that are **verifiable** — "all tests pass", "the file exists",
  "the command succeeds" — rather than subjective ones like "make the code better".
- Keep the turn cap in mind for open-ended goals. Lower it with `--goal-max-turns`
  for bounded tasks where you don't want the agent to spin for 50 turns.
- The verifier is strict. "Mostly done" is not met — the goal is met only when
  there is concrete evidence (described files/code exist, relevant tests or
  commands pass).
- A good condition names a **checkable outcome**, not a process. "Refactor the
  parser" is a process; "the parser module has 100% test coverage and all tests
  pass" is a checkable outcome.
