---
title: Interface Tour
description: A tour of the Kolega Code terminal UI and its panels.
---

Launching `kolega-code .` opens a full terminal UI built with
[Textual](https://textual.textualize.io/). This page is a map of what you're
looking at.

## Starting up

Once the terminal UI opens, a themed block-letter **KOLEGA CODE** splash
appears while the workspace, skills, tools, and any saved conversation are
prepared. The stage line reports the work being performed—not a simulated
percentage or countdown—and the project path and version appear below it.

The wordmark shares one gentle shimmer sweep in truecolor terminals.
256-color output, `NO_COLOR`, and reduced animation settings
(`TEXTUAL_ANIMATIONS=basic` or `TEXTUAL_ANIMATIONS=none`) use a static logo.
Small terminals use compact text instead of clipped block letters.

There is no minimum display time: the splash gives way as soon as startup
is ready, or earlier if a question or permission prompt needs an answer.
First-run setup, discovery tips, and startup errors keep their normal behavior.
The composer stays disabled until it can accept input; `Ctrl+Q` quits during
startup, and `Esc` or `Ctrl+C` cancels startup when no prompt is active.

## Layout

The screen is split into two columns:

- **Metadata strip** — the project, cached Git branch, short session ID, mode,
  and permissions fit on one line. Build and Plan use distinct theme-aware
  colors; their names remain visible when color is disabled.
  On narrow terminals, session and branch yield
  space to the project and permission state; Auto permissions use a warning
  color. Hover for unabbreviated context. Full session titles and configuration
  remain in the startup card and Status tab.
- **Conversation panel** (left, larger) — your chat with the agent. Responses
  stream in live, tool calls and sub-agent activity appear inline, and detailed
  tool results are collapsed by default so you can expand only what you need.
- **Side panel** (right) — a set of tabs for status, the terminal, planning,
  and settings. Toggle it with `Ctrl+O` or `/sidebar`. The diagnostic Logs tab is
  opt-in; launch with `--show-logs` when you want it.

At the bottom sits the **composer** — the text box where you type prompts. When
you submit follow-ups while the agent is working, a small queued-messages panel
appears above the composer until those prompts are sent. See
[Chat Composer](../composer/) for everything it can do.

Run `/memory` to open the full-screen [Project Memory](../project-memory/)
browser/editor. It manages private durable project knowledge separately from the
conversation and the Status tab's task list.

## Side-panel tabs

| Tab | What it shows |
| --- | --- |
| **Status** | The active provider/model and thinking effort, the current interaction mode (Build/Plan), permission mode, the agent's turn state (idle, generating, thinking, running a tool, running sub-agents, waiting for input, …), token usage (including cache hit rate), context warnings, the shared **Task List**, the active [goal](../../goal/) status when a goal is set, and the active [scheduled loop](../../loop/) with its schedule, countdown to the next iteration, and iteration count when a loop is running. |
| **Logs** | Optional. Launch with `--show-logs` to show a timestamped, color-coded diagnostic activity log. New entries preserve manual scrollback and an indicator flags unseen entries when you're on another tab. |
| **Terminal** | Live output from commands the agent runs. |
| **Plan** | The current plan document (Markdown from the planning agent). Both modes can update the shared Task List in **Status**. |
| **Settings** | A compact summary of the model, credential source, agent overrides, tools, and theme. Select **Open Settings** for the categorized full-screen editor, or **Continue Setup** when disconnected. See [Settings & API Keys](../../configuration/settings-and-api-keys/). |

In **Status**, a fresh session keeps empty sections quiet: context reads
**Context · Not measured**, and Usage and Task List are muted single-line rows.
Their full cards appear automatically when there are recorded requests, usage,
or tasks. Failed requests and incomplete historical usage still show their
details, even when token totals are zero. Clearing the task list returns it to
the compact row; resetting the thread does not erase lifetime session usage.

Populated task lists use accented `[ ]` markers for pending work and muted
`[x]` markers with static strikethrough for completed work. Nested tasks retain
their own completion state, and long descriptions wrap under their text rather
than under the checkbox. The list stays in its original order and is selectable
for copying; clicking a checkbox does not toggle it. This is display-only
styling: saved Markdown and the shared task-list tools are unchanged, with no
completion animation or added counters.

When no valid model configuration exists, a separate first-run wizard opens over
the interface. It handles only the initial account/provider connection and model
choice; the full editor remains available for every advanced setting.

## Conversation details

- **Startup card** — a static header shows the project beside the version,
  followed by compact model/effort and mode/permissions/credential-status rows.
  Language-server readiness appears alongside the status when detected. Expand
  **Session & configuration** for full paths, credential sources, language-server
  details, session details, and diagnostics.
  The card folds after your first message and starts folded when restoring a
  conversation. Click its title (or focus it and press `Enter`) to reopen it;
  later messages respect that choice. Resetting the thread restores the card.
- **Recent sessions** — on a fresh launch, the startup card lists up to three
  recent sessions from the same project, between its status rows and
  **Session & configuration**. Click a title, or focus it with `Tab` and press
  `Enter`, to resume that conversation. The session is locked before the current
  app closes; if another instance owns it, a warning appears and you stay where
  you are. Close the other instance and click again to retry. Unsent drafts or
  a saved worktree change require confirmation. The list is absent when there
  are no other saved sessions, when launching with `--resume`, or after starting
  a conversation; clearing the thread does not bring the choices back.
- **Discovery tips** — optional, off by default. Enable **Discovery tips** in
  **Settings → Appearance** and Apply to show a compact **QUICK TIP** modal
  once per fresh thread, after setup or Settings closes. Shortcuts are highlighted
  and a counter shows your place in the tip list. Click **Next →** (or focus it
  and press `Enter`) to cycle; tips never rotate automatically. **Got it** or
  `Esc` closes the modal and returns focus to the conversation. Dismissed tips
  stay dismissed until the thread is reset, and do not appear on resumed
  conversations. The preference is saved, but cycling and dismissal are local;
  tips are not sent to the agent. The catalogue covers planning, sub-agents,
  file mentions, scheduled loops, rewind, Git worktrees, project memory,
  handoff, skills, permissions, copying responses, and language-server status.
- **Your messages** — a subtle background distinguishes your prompts from agent
  replies. The blank separator below each message stays unshaded.
- **Streaming** — the view stays anchored to the bottom while the agent is
  responding. When you scroll up, a **jump-to-bottom** affordance appears so you
  can return to the live edge.
- **Working indicator** — a smooth spinner, current activity, elapsed time, and
  `Esc to interrupt` hint appear just above the composer during a turn. Press
  `Esc` or `Ctrl+C` to interrupt. When the turn ends, the line shows its outcome
  and duration instead. In truecolor terminals, a soft theme-colored shimmer
  sweeps across the activity text; the time and interrupt hint stay steady.
  The shimmer uses the existing status timer and stops when the turn ends.
  It falls back to flat text with 256-color output, `NO_COLOR`, or reduced
  animation settings (`TEXTUAL_ANIMATIONS=basic` or `TEXTUAL_ANIMATIONS=none`).
- **Queued follow-ups** — prompts submitted while the agent is still running show
  as `Queued` transcript entries. When the active turn finishes, they are sent
  automatically in FIFO order and become normal user messages.
- **Tool results** — shown as collapsible blocks with a state indicator
  (running / done / failed). A short file path, command, search query, or URL
  identifies the subject: for example, `read · src/app.py · done`. Subjects
  shorten to fit one line while keeping the state visible, including in the
  sub-agent inspector and restored conversations. If no safe subject is
  available, the row keeps its tool name and state. Expand to see the full result.
  Long paths shorten in the middle, retaining the filename and nearby parent
  directories; filenames are emphasized and directories muted. Expand to select
  the complete display path. Single-file edits put the filename in the title
  only, above their diff. Multi-file patches label each file above its own diff
  instead of repeating a path list in the title.
- **Commands** — short commands stay in the tool title. Longer or multiline
  commands have an indented preview of at most two lines; expanding replaces
  that preview with the complete safe command before the output. Wrapping does
  not rewrite shell syntax, and expanded commands retain their newlines.
  Embedded code, heredoc bodies, and sensitive payloads remain suppressed;
  oversized or ambiguous display metadata is omitted rather than exposed.
  These display rules also apply in the sub-agent inspector and restored sessions.
- **Display safety** — subject and detail text is plain, not a hyperlink; known
  credentials are redacted, and
  URL credentials, query strings, and fragments are omitted. This sanitizes the
  display metadata only, not the tool's full output.
- **Sub-agents** — when the main agent dispatches a sub-agent, a live card tracks
  it inline: the agent name, elapsed time, tool count, token usage, what it's doing
  right now, and a tail of its latest output. Press `Ctrl+G` (or click the card) to
  open the full [sub-agent inspector](#sub-agent-inspector).
- **Option lists** — when the agent asks you to choose between options (including
  plan decisions and tool approvals), they render as a **vertical,
  arrow-key-selectable list**.
  Use the arrow keys to highlight, number keys (`1`–`9`) to jump, and `Enter` to
  confirm.

## Sub-agent inspector

The inline cards summarize sub-agent activity; the **inspector** shows the whole
story. Press `Ctrl+G` (or click any sub-agent card) to open a full-screen
"mission control" view:

- **Roster** (left) — every sub-agent dispatched this turn, running or finished,
  each with a live spinner, status, elapsed time, tool count, and token usage.
  Nested sub-agents are indented by depth.
- **Trajectory** (right) — the selected agent's full run: its thinking, each tool
  call, the tool results (expandable, just like the main transcript), and its
  responses, streaming live as it works.

| Keys | Action |
| --- | --- |
| `Ctrl+G` | Open the inspector (on the most recently active sub-agent) |
| `↑` / `↓` | Switch between sub-agents |
| `Tab` then `Enter` | Focus a tool call and expand it (or click it) |
| `o` | Toggle follow — auto-scroll to the newest activity |
| `y` | Copy the selected agent's trajectory to the clipboard |
| `Esc` / `q` | Close the inspector |

The inspector is read-only: opening or closing it never interrupts the agent, so
the turn keeps running while you look around.

## Key bindings at a glance

The bottom shortcut bar follows focus: **send** while idle, **queue** while
working, **answer** for question text, and **choose / select / dismiss** during
completion. Option lists show their own selection keys. Hints fit as whole
items rather than clipped labels; omitted shortcuts still work. Press `F1` or
click **…** for the complete local reference without replacing your draft.
`/help` remains an agent command. Inspector and settings shortcuts are unchanged.

| Keys | Action |
| --- | --- |
| `F1` / click `…` | Open the local keyboard shortcut reference |
| `Shift+Tab` | Toggle Build ⇄ Plan mode (`/plan` / `/build` if Shift is unavailable) |
| `Ctrl+P` | Toggle shell/edit permissions between Ask ⇄ Auto |
| `Ctrl+O` | Show or hide the side panel |
| `Ctrl+G` | Open the sub-agent inspector |
| `Enter` | Send the prompt |
| `Shift+Enter` / `Ctrl+J` | Insert a newline |
| `Ctrl+Shift+V` / `Alt+V` | Paste an image from the system clipboard (`/attach` also works) |
| `Ctrl+C` / `Escape` | Cancel the current generation |
| `Ctrl+Q` | Save the session and quit |

A complete composer-and-completion key reference is in
[Chat Composer](../composer/). If Shift chords fail inside tmux, see
[Terminal & tmux shortcuts](../../troubleshooting/terminal-tmux/).
