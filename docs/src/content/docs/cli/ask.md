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
| `--json` | Emit complete messages and events as JSON |
| `--browser-visible` | Launch visible Playwright browser windows |
| `--permission-mode <auto\|ask>` | Shell/edit permission mode (default `auto`) |
| `--web-search <auto\|hosted\|client\|off>` | Web tool mode: hosted server-side search, client tools, or none (default `auto`) |
| `--lsp <on\|off>` | Force LSP tools on or off for this session (overrides settings; not persisted) |
| `--subagents <on\|off>` | Force sub-agent dispatch (`dispatch_agent`) on or off for this session (overrides settings; not persisted; default on) |
| `--skills <on\|off>` | Force [Agent Skills](../../skills/) on or off for this session (overrides settings; not persisted; default on). When off, skills are not discovered and the `skill` tool is not exposed |
| `--compression-threshold PERCENT` | Context-window usage that triggers automatic history compression (`10`–`100`, default `95`; `100` disables proactive compression; over-limit recovery and explicit context caps remain enforced). Overrides settings for the session; not persisted |
| `--session <ID>` | Resume or create a specific session |
| `--state-dir <PATH>` | Directory for CLI session state |
| `--extension <MODULE:FACTORY>` | Load one installed Python extension for this run (see [Extensions](../../concepts/extensions/)) |
| `--extension-config <PATH>` | Opaque config file path passed to the `--extension` factory (requires `--extension`) |
| `--worktree <PATH_OR_BRANCH>` | Start in an existing registered worktree |
| `--create-worktree <BRANCH>` | Create a worktree and start a new session there |
| `--from <REF>` | Commit start point for a new branch (requires `--create-worktree`) |
| `--worktree-path <PATH>` | Explicit checkout destination (requires `--create-worktree`) |

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

When Agent Skills are disabled (`--skills off`, `KOLEGA_CODE_SKILLS=off`, or the
saved `skills_enabled` setting), `/skills` reports that Agent Skills are disabled
and `/skill-name` prompts are sent to the model as plain text instead of being
activated.

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

## Scheduled loops

Pass `--loop <interval>` or `--loop-cron "<expr>"` to re-run the prompt on a
schedule until a cap or expiry is reached:

```bash
kolega-code ask "check the deploy and report" --loop 10m --loop-max-iterations 12
kolega-code ask "summarize new PRs" --loop-cron "0 9 * * 1-5" --project .
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--loop <interval>` | — | Fixed interval: `30s`, `5m`, `2h`, `1d`, `every 2 hours` |
| `--loop-cron "<expr>"` | — | 5-field cron schedule (mutually exclusive with `--loop`) |
| `--loop-max-iterations <n>` | `100` | Stop after this many iterations |
| `--loop-expires <duration>` | `7d` | Stop this long after the loop starts |
| `--loop-fresh` | off | Clear conversation history before each iteration after the first |

Interval loops run the first iteration immediately and then wait; cron loops wait
for the first matching local time. The positional `prompt` is optional — without
one the CLI reads `.kolega/loop.md`. Progress lines go to stderr so piped stdout
stays the answers, and the command exits `0` when the loop ends at its cap or
expiry. `--loop` cannot be combined with `--goal`.

See [Scheduled Loops](../../loop/) for the full interval and cron syntax.

## JSON output

With `--json`, the command streams the **semantic event protocol**: UTF-8
newline-delimited JSON where every stdout line is one complete, flushed event
envelope. The same records — with the same ids, sequence numbers, and
timestamps — are returned later by
`kolega-code sessions export <id> --format events-jsonl`, because both are the
public projection of the session journal.

> **Breaking change:** the former `{"kind": "message"|"event"|"summary"}`
> record family was removed with no compatibility flag. Machine consumers must
> parse the envelope below.

```bash
kolega-code ask "count the Python files" --project . --json
```

Every line has this shape:

```json
{
  "schema": "kolega.session.event",
  "version": 2,
  "id": "event-uuid",
  "session_id": "session-id",
  "seq": 42,
  "epoch_id": "epoch-uuid",
  "turn_id": "turn-uuid-or-null",
  "timestamp": "2026-08-11T09:54:09.372000+00:00",
  "actor": "assistant",
  "agent_id": "agent-uuid",
  "agent_name": "main",
  "parent_agent_id": null,
  "parent_tool_call_id": null,
  "depth": 0,
  "type": "assistant.message",
  "payload": {},
  "artifacts": []
}
```

`seq` is a gapless session-wide sequence. Subagent events share it, carrying
their own `agent_id` with `parent_agent_id`/`parent_tool_call_id`/`depth`
lineage. Unknown future `type`s must be ignored, not treated as errors.

Common event types:

| `type` | Payload |
| --- | --- |
| `session.created` / `context.epoch_started` | Session metadata and context boundaries |
| `turn.started` | The complete user message |
| `context.system` | The rendered system context (recorded when it changes) |
| `context.tools` | Public schemas of the tools available to the agent (recorded when they change) |
| `assistant.message` | One completed LLM response (`origin_type: "llm"`, `llm_call_id`, `provider`, `model`, `llm_call_count: 1`) or one deterministic notice (`origin_type: "synthetic"`, `llm_call_count: 0`, `notice_code`). The `message` carries content blocks with readable reasoning kept and provider-opaque state removed; tool calls expose the canonical `tool_call_id` plus `provider_call_id`, `input`, `input_kind`, and normalized `arguments` |
| `tool.results` | Durable results, correlated to calls by the same `tool_call_id`; oversized content carries a `content_artifact` reference |
| `turn.completed` / `turn.failed` / `turn.cancelled` | Turn outcome with counts and timing |
| `agent.started` / `agent.completed` / `agent.failed` | Subagent lifecycle |
| `context.compacted` / `context.rewound` | Context management, with provenance |
| `skill.activated`, `goal.evaluated`, `goal.completed`, `loop.iteration_started`, `loop.sleeping`, `loop.completed` | Skill/goal/loop progress |
| `run.completed` / `run.failed` / `run.cancelled` | The final record: status, timestamps, token totals, provider/model, and (on failure) a safe `error` code/message |

Configured secrets and home-directory paths are scrubbed; provider-opaque
replay state (thinking signatures, encrypted reasoning) never appears. Runs
without `--save` emit the identical protocol from an in-memory journal and
leave no session state behind.

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
| `0` | Success (or: the goal was met with `--goal`; a loop reached its cap or expiry with `--loop`) |
| `1` | With `--goal`: the turn cap was reached without meeting the goal |
| `2` | Configuration / usage error (e.g. invalid provider, missing API key, invalid loop schedule) |
| `130` | Interrupted (`Ctrl+C`) |
