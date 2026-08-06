---
title: CLI Overview
description: Invocation forms, global options, and the command surface of kolega-code.
---

The `kolega-code` command has two shapes:

- **No subcommand** → launches the interactive [Terminal UI](../../tui/interface/).
- **A subcommand** (`ask`, `sessions`, `share`, `doctor`, `agents`, `update`) → runs a
  specific non-interactive task.

```bash
kolega-code [PROJECT_PATH] [options]      # interactive TUI
kolega-code ask "<prompt>" [options]       # one-shot prompt
kolega-code sessions <list|delete|export> [options]
kolega-code share export <session-id> [options]
kolega-code doctor [options]
kolega-code agents <list|validate> [options]
kolega-code update
```

## Commands

| Command | What it does |
| --- | --- |
| `kolega-code .` | Launch the Textual TUI in the given project directory (default `.`) |
| [`ask`](../ask/) | Run a single prompt and print the answer |
| [`sessions`](../sessions/) | List, delete, or export saved sessions |
| [`share`](../share/) | Export a session as a self-contained browser replay |
| [`doctor`](../doctor/) | Check local configuration and API-key status |
| [`agents`](../../custom-agents/#list-validate-and-reload) | List or validate user and project custom-agent definitions |
| `update` | Update Kolega Code to the latest released version |

## Launching the TUI

```bash
kolega-code [PROJECT_PATH]
```

| Argument / option | Description |
| --- | --- |
| `PROJECT_PATH` | Project directory to work in (default `.`) |
| `--new` | Start a new session (this is the default) |
| `--resume [SESSION_ID]` | Resume the latest saved session, or a specific Resume ID from `sessions list`. Legacy thread IDs are also accepted |
| `--browser-visible` | Launch visible Playwright browser windows instead of headless |
| `--show-logs` | Show the optional diagnostic Logs side-panel tab. Hidden by default to avoid unnecessary log rendering work |
| `--permission-mode <auto\|ask>` | Shell/edit permission mode. TUI sessions default to `ask` |
| `--lsp <on\|off>` | Force LSP tools on or off for this session (overrides settings; not persisted) |
| `--session <ID>` | Legacy alias for `--resume SESSION_ID` |

See [Sessions & Resuming](../../tui/sessions-and-resume/) for the full session
workflow.

## Starting in a Git worktree

A registered worktree can be used directly as `PROJECT_PATH`, or selected from
another checkout in the same repository:

```bash
kolega-code /repo --worktree fix/image-history
kolega-code /repo --worktree .kolega/worktrees/fix-image-history
```

Create a new managed worktree and start a new session there with:

```bash
kolega-code /repo --create-worktree fix/image-history
kolega-code /repo --create-worktree fix/image-history --from origin/main
kolega-code /repo --create-worktree fix/image-history --worktree-path /other/destination
```

`--worktree` accepts an exact registered branch name or a path (including a
nested path) inside a registered worktree. `--create-worktree` creates or checks
out the named local branch without force-moving, deleting, or reusing an
occupied destination. `--from` and `--worktree-path` are valid only with
`--create-worktree`; creation cannot be combined with `--resume` or `--session`.
The same options are available on `kolega-code ask`.

During a TUI session, the top-level **build-mode** agent can call
`switch_worktree`, and only when you have explicitly asked it to move the
session to another worktree — it never creates a worktree on its own
initiative, and plan mode has no workspace-switching tool at all. Every
agent-initiated switch asks you to confirm first; declining, or leaving the
prompt unanswered, leaves the active workspace unchanged.

Plan mode still knows about the build-only handoff so it can produce an
implementable worktree plan. When you explicitly request a new worktree, the
plan should create or register it first, make `switch_worktree` the Build
agent's next action, and defer dependency setup, file access, tests, and
delegated work until the session continues in the switched checkout. The
planner cannot execute either worktree creation or the switch itself.

Once approved, the switch is committed to the session immediately and
applied when the agent's turn ends: the workspace is rebuilt in the selected
checkout — filesystem, terminal, search/edit, LSP, snapshots, skills,
custom agents, and future sub-agent work all follow — and the agent is then
prompted to continue there. Trust, hooks, configuration, private memory, and
saved permission rules stay scoped to the launch checkout, so a rule saved
while a worktree is active survives that worktree and applies to the whole
session. A fresh **Workspace switched** Changes/Rewind
baseline starts, so checkpoints from the previous workspace are no longer
displayed. The switch is durable session state and is not undone by
conversation rewind. It must run alone, and it is refused while background
terminal sessions are active.

## Global model options

These options are accepted by the TUI launch, `ask`, and `doctor`. They override
[environment variables and saved settings](../../configuration/providers-and-models/)
for the run.

| Option | Description |
| --- | --- |
| `--provider` | Provider for the main coding model |
| `--model` | Main coding model |
| `--fast-provider` | Provider for fast utility calls |
| `--fast-model` | Fast utility model |
| `--thinking-effort` | Model-specific thinking effort, such as `auto`, `medium`, `high`, or `max` |
| `--edit-protocol` | Override the model-facing edit language with `search_replace`, `codex_apply_patch`, or `claude_code` |
| `--environment` | Environment label for tracing/metadata |
| `--compression-threshold PERCENT` | Context-window usage that triggers automatic history compression (`10`–`100`, default `80`; `100` effectively disables it). Overrides settings for the session; not persisted |

## Session-state options

| Option | Description |
| --- | --- |
| `--state-dir <PATH>` | Directory for CLI session state (defaults to the platform state directory) |
| `--session <ID>` | Session ID to resume or create |

The default state directory and the `KOLEGA_CODE_STATE_DIR` override are described
in [Settings & API Keys](../../configuration/settings-and-api-keys/).

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Goal not met: `ask --goal` reached the turn cap without completing the goal |
| `2` | Configuration / usage error (e.g. invalid provider, missing API key, project path not found, Textual not installed) |
| `130` | Interrupted (`Ctrl+C`) |
