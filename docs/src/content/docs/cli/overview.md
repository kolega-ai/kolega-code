---
title: CLI Overview
description: Invocation forms, global options, and the command surface of kolega-code.
---

The `kolega-code` command has two shapes:

- **No subcommand** → launches the interactive [Terminal UI](../../tui/interface/).
- **A subcommand** (`ask`, `sessions`, `doctor`, `agents`, `update`) → runs a
  specific non-interactive task.

```bash
kolega-code [PROJECT_PATH] [options]      # interactive TUI
kolega-code ask "<prompt>" [options]       # one-shot prompt
kolega-code sessions <list|delete|export> [options]
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
| `--resume [THREAD_ID]` | Resume the latest saved thread, or a specific thread/session ID |
| `--browser-visible` | Launch visible Playwright browser windows instead of headless |
| `--show-logs` | Show the optional diagnostic Logs side-panel tab. Hidden by default to avoid unnecessary log rendering work |
| `--permission-mode <auto\|ask>` | Shell/edit permission mode. TUI sessions default to `ask` |
| `--session <ID>` | Legacy alias for `--resume THREAD_ID` |

See [Sessions & Resuming](../../tui/sessions-and-resume/) for the full session
workflow.

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
| `--thinking-provider` | Provider for think-hard operations |
| `--thinking-model` | Model for think-hard operations |
| `--thinking-effort` | Model-specific thinking effort, such as `auto`, `medium`, `high`, or `max` |
| `--edit-protocol` | Override the model-facing edit language with `search_replace`, `codex_apply_patch`, or `claude_code` |
| `--environment` | Environment label for tracing/metadata |

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
