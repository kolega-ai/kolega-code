---
title: sessions
description: List, delete, and export saved Kolega Code sessions.
---

`kolega-code sessions` manages the local session records that the TUI and
`ask --save` create. Sessions hold the conversation history, the project they
belong to, and the model configuration in use.

```bash
kolega-code sessions <list|delete|export> [options]
```

See [Settings & API Keys](../../configuration/settings-and-api-keys/) for where
session files are stored, and [Sessions & Resuming](../../tui/sessions-and-resume/)
for how resuming works in the TUI.

## `sessions list`

List saved sessions, optionally filtered to one project.

```bash
kolega-code sessions list --project .
```

| Option | Description |
| --- | --- |
| `--project <PATH>` | Only show sessions for this project |
| `--state-dir <PATH>` | Directory for CLI session state |

Each row is tab-separated:

```text
<session_id>  <thread_id>  <updated_at>  <mode>  <project_path>  <title>
```

## `sessions delete`

Delete a session by ID.

```bash
kolega-code sessions delete <session_id>
```

| Argument / option | Description |
| --- | --- |
| `session_id` | The session to delete (required) |
| `--state-dir <PATH>` | Directory for CLI session state |

## `sessions export`

Print a session as JSON, or write it to a file.

```bash
kolega-code sessions export <session_id>                 # to stdout
kolega-code sessions export <session_id> --output run.json
```

| Argument / option | Description |
| --- | --- |
| `session_id` | The session to export (required) |
| `--output <PATH>` | Write JSON to a file instead of stdout |
| `--state-dir <PATH>` | Directory for CLI session state |

The exported JSON includes the session metadata, model configuration summary, and
full message history — handy for archiving, debugging, or analysis.
