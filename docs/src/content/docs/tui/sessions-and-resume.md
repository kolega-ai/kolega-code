---
title: Sessions & Resuming
description: How Kolega Code saves your work and how to pick it back up.
---

A **session** captures a conversation: the project it belongs to, the message
history, the model configuration, and — for the TUI — the latest plan and shared
task list. Sessions let you stop and resume work later.

## How sessions are created

- **In the TUI**, launching `kolega-code .` starts a **fresh session by default**.
  Your work is saved when you quit (`Ctrl+Q`, `/quit`, or `/exit`).
- **With `ask`**, a session is only persisted if you pass `--save` or `--session`.
  See [`kolega-code ask`](../../cli/ask/).

Sessions are stored as JSON, one file per session, in your state directory. See
[Settings & API Keys](../../configuration/settings-and-api-keys/) for the exact
location and the `KOLEGA_CODE_STATE_DIR` override.

## Resuming in the TUI

```bash
# Resume the most recent session for this project
kolega-code . --resume

# Resume a specific session using the Resume ID from `sessions list`
kolega-code . --resume <session-id>
```

A few rules:

- `--resume` with no ID resumes the **latest** session for the project. If there
  are none, the CLI reports that.
- `sessions list` labels the session ID to copy as **Resume ID**. Previously
  saved commands that use a thread ID remain supported for compatibility.
- A session belongs to the project it was created in — resuming it from a
  different project directory is rejected.
- `--new` forces a fresh session (this is the default behavior).
- `--session <ID>` is a **legacy alias** for `--resume <ID>`; you can't combine
  `--resume` and `--session` in the same command.

## Managing sessions

Use the [`sessions`](../../cli/sessions/) subcommand to list, delete, or export
saved sessions:

```bash
kolega-code sessions list --project .
kolega-code sessions export <session_id> --output run.json
kolega-code sessions delete <session_id>
```

:::note
All CLI sessions use the CLI-specific coding-agent prompt, including resumed ones.
Resuming restores the conversation history so the agent picks up with full context.
An active [goal](../../goal/) is also saved with the session and restored on resume,
so the autonomous loop continues where it left off.
:::
