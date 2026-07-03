---
title: ask
description: Run a single prompt non-interactively and print the answer.
---

`kolega-code ask` runs one prompt against the agent and prints the response. It's
the scriptable counterpart to the TUI — useful for automation, quick questions, and
piping output into other tools.

```bash
kolega-code ask "<prompt>" [options]
```

## Arguments & options

| Argument / option | Description |
| --- | --- |
| `prompt` | The prompt to send (optional when `--goal` is given; otherwise required) |
| `--project <PATH>` | Project directory to work in (default `.`) |
| `--goal <condition>` | Set an autonomous completion goal and loop until it is met or capped (no prompt required) |
| `--goal-max-turns <N>` | Maximum evaluation turns before an unmet goal gives up (default 50) |
| `--save` | Persist the session after the prompt completes |
| `--json` | Emit response chunks and events as JSON |
| `--browser-visible` | Launch visible Playwright browser windows |
| `--permission-mode <auto\|ask>` | Shell/edit permission mode (default `auto`) |
| `--session <ID>` | Resume or create a specific session |
| `--state-dir <PATH>` | Directory for CLI session state |

All the [global model options](../overview/#global-model-options)
(`--provider`, `--model`, `--fast-model`, …) are also accepted.
`ask` requires a provider/model from those options, environment variables, or
saved Settings. API key variables alone are not enough.

## Examples

Ask a question about the current project:

```bash
kolega-code ask "summarize this repository" --project .
```

Pick a provider and model just for this run:

```bash
kolega-code ask "summarize this repository" --project . \
  --provider deepseek --model deepseek-v4-pro
```

Save the result as a resumable session:

```bash
kolega-code ask "add unit tests for the parser" --project . --save
```

## File mentions

Just like the TUI composer, `ask` understands `@` file mentions. Referenced files
are attached to the prompt:

```bash
kolega-code ask "explain @src/main.py and suggest improvements" --project .
```

If a mention can't be resolved, the CLI notes it on stderr and sends the text
as-is:

```text
Note: @missing/file.py not found, sent as plain text
```

## Skills

If your prompt is a skill command (e.g. `/skills` or `/my-skill`), `ask` resolves
it against the project's [Agent Skills](../../skills/):

- `kolega-code ask "/skills"` prints the available-skills catalog.
- `kolega-code ask "/my-skill"` (with no extra text and no `--save`/`--session`)
  prints the skill's activation content.
- `kolega-code ask "/my-skill do the thing"` activates the skill and runs the
  remaining prompt.

## Goal mode

Pass `--goal "<condition>"` to set an autonomous completion goal. The agent works
toward the goal, and after each turn a read-only verifier checks whether it's met.
The loop continues until the goal is met or the turn cap (default 50, override
with `--goal-max-turns`) is reached. The positional `prompt` is optional with
`--goal` — the CLI synthesizes the first work-turn message from the condition:

```bash
kolega-code ask --goal "all tests pass and ruff is clean" --project .
kolega-code ask "start by fixing the parser" --goal "all tests pass" --project .
```

See [Goal-Conditioned Work](../../goal/) for the loop behavior, safety model, and
JSON event details.

## JSON output

With `--json`, the command streams newline-delimited JSON objects, each tagged with
a `kind`, so you can parse them programmatically:

```bash
kolega-code ask "count the Python files" --project . --json
```

The stream includes:

| `kind` | Meaning |
| --- | --- |
| `chunk` | A streamed piece of the response |
| `event` | An agent event (sub-agent activity, tool calls, terminal output) |
| `skill` | Skill activation metadata |
| `goal_eval` | Emitted after each goal evaluation (with `--goal`): `met`, `turns`, `reason` |
| `goal_result` | Final goal outcome (with `--goal`): `met`, `turns`, `reason` |
| `summary` | A final object with the chunk count and `session_id` |

In plain (non-JSON) mode, the answer is written to **stdout** while sub-agent and
tool activity is reported on **stderr** — so piping stdout gives you just the
answer.

## Permissions

`ask` defaults to `--permission-mode auto` so scripts do not stop for
confirmations. If you pass `--permission-mode ask`, shell commands and file edits
prompt on stderr when stdin is interactive. Persisted allow rules are stored in
the project at `.kolega/permissions.json`.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success (or: the goal was met when using `--goal`) |
| `1` | With `--goal`: the turn cap was reached without meeting the goal |
| `2` | Configuration / usage error (e.g. invalid provider, missing API key) |
| `130` | Interrupted (`Ctrl+C`) |
