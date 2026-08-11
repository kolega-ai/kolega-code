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

Each saved session is shown as a labeled block:

```text
Updated:   2026-07-19T10:00:00+00:00
Title:     Improve session listing
Mode:      code
Project:   /path/to/project
Resume ID: 0123456789abcdef0123456789abcdef
```

Sessions are ordered from oldest to newest, so the most recently updated
session and its Resume ID appear nearest the shell prompt. Pass that Resume ID
to `--resume`, `sessions delete`, or `sessions export`.

## `sessions delete`

Delete a session by ID.

```bash
kolega-code sessions delete <session_id>
```

| Argument / option | Description |
| --- | --- |
| `session_id` | The Resume ID shown by `sessions list` (required) |
| `--state-dir <PATH>` | Directory for CLI session state |

## `sessions export`

Export a session to stdout or a file, in one of two formats.

```bash
kolega-code sessions export <session_id>                          # replay JSON (default)
kolega-code sessions export <session_id> --output run.json
kolega-code sessions export <session_id> --format events-jsonl    # semantic event log
```

| Argument / option | Description |
| --- | --- |
| `session_id` | The Resume ID shown by `sessions list` (required) |
| `--format <json\|events-jsonl>` | `json` (default): effective-history replay snapshot. `events-jsonl`: the canonical public semantic event log |
| `--output <PATH>` | Write the export to a file instead of stdout |
| `--state-dir <PATH>` | Directory for CLI session state |

The default `json` format includes the session metadata, model configuration
summary, and full **effective** message history (superseded context epochs and
rewound turns are omitted) — handy for archiving, debugging, or analysis.

`events-jsonl` emits one v2 event envelope per line — the same records, ids,
and sequence numbers `ask --json` streams live (see
[JSON output](../ask/#json-output)). It is the complete auditable trajectory:
turns, LLM responses with call ids, tool calls/results correlated by canonical
id, subagent lineage, compactions, rewinds, and terminal records. Sessions
recorded by older Kolega versions export with deterministic fallbacks (derived
root agent identity); facts those sessions never captured are not invented.
Secrets are scrubbed and provider-opaque replay state is excluded in both
formats' event output.
