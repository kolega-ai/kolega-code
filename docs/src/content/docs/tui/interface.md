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
- **Side panel** (right) — a set of tabs for status, logs, the terminal, planning,
  and settings.

At the bottom sits the **composer** — the text box where you type prompts. See
[Chat Composer](../composer/) for everything it can do.

## Side-panel tabs

| Tab | What it shows |
| --- | --- |
| **Status** | The active provider/model and thinking effort, the current interaction mode (Build/Plan), the agent's turn state (idle, generating, thinking, running a tool, running sub-agents, waiting for input, …), token usage, and context warnings. |
| **Logs** | A timestamped, color-coded activity log: sub-agent lifecycle, tool calls, configuration changes. An indicator flags new entries when you're on another tab. |
| **Terminal** | Live output from commands the agent runs. |
| **Planning** | The current **Plan** (markdown from the planning agent) and the shared **Task List** that both modes can edit. |
| **Settings** | Provider, model, thinking effort, and API-key configuration. See [Settings & API Keys](../../configuration/settings-and-api-keys/). |

## Conversation details

- **Streaming** — the view stays anchored to the bottom while the agent is
  responding. When you scroll up, a **jump-to-bottom** affordance appears so you
  can return to the live edge.
- **Tool results** — shown as collapsible blocks with a state indicator
  (running / done / failed). Expand to see the full result.
- **Sub-agents** — when the main agent dispatches a sub-agent, its activity is
  tracked live with status updates.
- **Option lists** — when the agent asks you to choose between options (including
  plan decisions), they render as a **vertical, arrow-key-selectable list**.
  Use the arrow keys to highlight, number keys (`1`–`9`) to jump, and `Enter` to
  confirm.

## Key bindings at a glance

| Keys | Action |
| --- | --- |
| `Shift+Tab` | Toggle Build ⇄ Plan mode |
| `Enter` | Send the prompt |
| `Shift+Enter` / `Ctrl+J` | Insert a newline |
| `Ctrl+C` / `Escape` | Cancel the current generation |
| `Ctrl+Q` | Save the session and quit |

A complete composer-and-completion key reference is in
[Chat Composer](../composer/).
