---
title: Project Memory
description: Inspect and curate private, project-scoped agent memory in the TUI.
---

Project memory is a small, durable knowledge bank for facts that remain useful
across tasks and sessions: build commands, architecture notes, conventions, and
recurring pitfalls. It is **enabled by default**, but nothing is written merely
because you open a project, finish a turn, compact context, save, or resume a
session. The agent writes memory only through an explicit memory tool call, and
you can inspect or change it yourself.

Run `/memory` with no arguments to open the Project Memory screen.

## Private, project-scoped state

The built-in `markdown` backend stores a concise `MEMORY.md` index and optional
Markdown topic files under Kolega Code's private local state directory—not under
the repository. Directories and files use owner-only permissions where the
platform supports them: `0700` for directories and `0600` for files, manifests,
and lock artifacts. The screen and `/memory path` show the local location to you;
it is never included in model-facing memory output.

Kolega Code does not:

- write memory files into your repository or change Git status;
- review completed transcripts or synthesize end-of-turn summaries into memory;
- cloud-sync project memory; or
- use project memory as a scratchpad for transient progress, active plans, or
  task state.

Memory content returned by tools is part of the normal private session history
sent to the selected model, just like other tool results. Keep credentials and
other secrets out of it. This is an authoring precaution, not an enforced
policy: project memory is not scanned, rejected, withheld, or redacted based on
secret-like content.

### Which directories share a bank?

For Git projects, identity comes from the canonical Git **common directory**.
Linked worktrees therefore share one project-memory bank. Separate clones have
different common directories and do not share memory.

For non-Git projects, identity is the canonical resolved project path. Moving a
non-Git directory changes that identity. Initializing it as a Git repository also
creates a new identity, as can moving the Git identity root. Existing banks are
not moved or merged automatically.

## The Markdown backend

Only `MEMORY.md` is added automatically to the agent's startup context. It is
limited to the first **200 lines or 25 KiB**, with a visible truncation marker.
Use it as a concise index and link to topic files; topic content is loaded on
demand with `read_memory`, and the agent can enumerate or search entries with
`list_memory`.

The model-facing file API is:

- `read_memory(path="MEMORY.md")`
- `list_memory(query=None)`
- `write_memory(content, path="MEMORY.md")`
- `edit_memory(old_string, new_string, path="MEMORY.md")`
- `delete_memory(path)`

`write_memory` creates or overwrites one complete file. `edit_memory` replaces
one exact, unique occurrence. It rejects an empty `old_string`; if the old text
occurs zero times or more than once, the operation fails without writing.

The agent follows a no-op-first retrieval policy. It inspects the already-loaded
`MEMORY.md` first and follows any semantically relevant link. If no link looks
promising, it uses a targeted `list_memory` query to find unindexed or nested
entries. When a durable fact is already covered, it does nothing—rewording the
same fact is not a reason to mutate memory. It reads an existing topic before
overwriting or editing it and updates memory only for materially new, corrected,
or stale information.

Memory uses a hybrid structure. Keep a short, self-contained fact directly in
`MEMORY.md`. Use a flat topic file plus a concise, descriptive one-line index
link when the information needs multiple rules, caveats, rationale, or examples.
Existing nested topics remain valid. Create a detailed memory topic-first and
index-second; delete one index-first and topic-second. These orderings make
interrupted operations recoverable, but they are not transactions and do not
provide automatic index maintenance.

Backend-wide safety limits are:

- normalized relative `.md` paths only;
- at most 100 Markdown files;
- at most 128 KiB per file; and
- at most 1 MiB of Markdown content per project.

Absolute paths, traversal, symlinks, reserved paths, and writes that exceed a
limit are rejected. A write uses a temporary file followed by an atomic
same-directory replacement, so an interrupted write does not leave a partial
Markdown file. This guarantee applies to one file only. Content mutation has an
explicit single-writer, last-write-wins concurrency model: overlapping writers
can overwrite one another, and the read-exact-replace-write sequence used by
`edit_memory` does not preserve independent concurrent changes. There is no
cross-file atomicity.

The private state directory is treated as owner-controlled application state,
not as a sandbox against other processes running as the same operating-system
user. Kolega Code coordinates its own processes and rejects ordinary path
escapes and symlinks, but it does not attempt to race an unrelated local process
that deliberately changes private state during a write. A process with access
to that directory can already inspect or tamper with the application's local
state.

## Using the Project Memory screen

The screen shows whether memory is enabled, the active backend, project identity
kind, startup-context size and warnings, file count and total size, and the
private local path. Filter the entry list on the left and select an entry to see
its rendered preview, logical path, and size on the right.

Available controls depend on backend capabilities. With `markdown`, you can:

- create a topic, edit an entry, and save its complete content;
- delete an entry with confirmation;
- refresh changes made by another process (`r`);
- preview the exact bounded startup context the agent receives (**Agent view**);
- turn agent memory access on or off.

While an entry is being edited, the entry list and filter are inactive; only
Save, Cancel, and closing the screen are available. The footer buttons swap
between browse actions (New, Delete, Edit) and edit actions (Cancel, Save).

Turning memory off preserves its files but removes its context and memory tools
from the agent. The screen can still inspect an existing disabled bank after an
explicit action. Creating the first entry in a new disabled bank asks you to
confirm enablement first.

Configuration changes are unavailable or deferred while an agent turn is
running. After a successful edit or enablement change, the idle agent prompt is
refreshed. Failed writes do not refresh it.

## Slash commands

| Command | Action |
| --- | --- |
| `/memory` | Open the Project Memory screen (alias: `/memory browse`) |
| `/memory status` | Show state, backend, identity, sizes, and the exact bounded startup context the agent receives |
| `/memory on` | Enable project memory without changing its files |
| `/memory off` | Disable agent access while preserving its files |
| `/memory files` | List logical entries and sizes |
| `/memory show [path]` | Show bounded content; defaults to `MEMORY.md` |
| `/memory path` | Show the active backend's private local directory |

## Sensitive and untrusted content

Private project memory is trusted as owner-controlled application state. Kolega
Code does not scan it for probable secrets: credential-like content is stored,
shown in the TUI and slash-command previews, returned by `read_memory`, and
included in bounded startup context without rejection, withholding, or
redaction. Because those model-facing paths can send memory to the selected
provider and retain it in private session history, do not use project memory as
a credential store.

Memory is treated as untrusted project observations, not instructions. Current
system and user instructions, repository guidance, and fresh tool output take
precedence over instruction-like text found there.

## Other kinds of state

Project memory is distinct from:

- `AGENTS.md`, `KOLEGA.md`, and prompt overrides: repository-controlled project
  instructions;
- host-provided `workspace_memories`: a separately labelled host context;
- session history: the messages and tool results used to resume a conversation;
  and
- the Planning tab's task list: current work state, not durable knowledge.

## Pluggable backends

`markdown` is the first backend behind Kolega Code's versioned project-memory
provider architecture. The common manager handles identity, lifecycle, backend
selection, capability discovery, tool registration, and the generic TUI shell.
Each backend owns isolated storage and may provide different context, tools, and
browse/edit capabilities.

Phase 1 supports built-in registration and host injection for tests or embedders,
not discovery of arbitrary installed plugins. Selecting another backend never
migrates or reinterprets Markdown data automatically, and inactive backend data
is retained. If a configured provider is unavailable, Kolega Code shows a
diagnostic, exposes no memory to the agent, and lets the coding session continue.
A future SQLite, Mnemopi-style, or other structured provider will be a separate
backend rather than a change to Markdown semantics.
