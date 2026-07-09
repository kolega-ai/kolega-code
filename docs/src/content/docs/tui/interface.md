---
title: Interface Tour
description: A tour of the Kolega Code terminal UI and its panels.
---

Launching `kolega-code .` opens a full terminal UI built with
[Textual](https://textual.textualize.io/). This page is a map of what you're
looking at.

## Layout

The screen is split into two columns:

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

## Side-panel tabs

| Tab | What it shows |
| --- | --- |
| **Status** | The active provider/model and thinking effort, the current interaction mode (Build/Plan), permission mode, the agent's turn state (idle, generating, thinking, running a tool, running sub-agents, waiting for input, …), token usage, context warnings, and the active [goal](../../goal/) status when a goal is set. |
| **Logs** | Optional. Launch with `--show-logs` to show a timestamped, color-coded diagnostic activity log. New entries preserve manual scrollback and an indicator flags unseen entries when you're on another tab. |
| **Terminal** | Live output from commands the agent runs. |
| **Planning** | The current **Plan** (markdown from the planning agent) and the shared **Task List** that both modes can edit. |
| **Settings** | Provider, model, thinking effort, and API-key configuration. See [Settings & API Keys](../../configuration/settings-and-api-keys/). |

## Conversation details

- **Streaming** — the view stays anchored to the bottom while the agent is
  responding. When you scroll up, a **jump-to-bottom** affordance appears so you
  can return to the live edge.
- **Queued follow-ups** — prompts submitted while the agent is still running show
  as `Queued` transcript entries. When the active turn finishes, they are sent
  automatically in FIFO order and become normal user messages.
- **Tool results** — shown as collapsible blocks with a state indicator
  (running / done / failed). Expand to see the full result.
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

| Keys | Action |
| --- | --- |
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
