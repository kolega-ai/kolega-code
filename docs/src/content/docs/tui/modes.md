---
title: Build & Plan Modes
description: How Kolega Code separates planning from building, and how to switch.
---

Kolega Code has two interaction modes. Press `Shift+Tab` to toggle between them at
any time.

## Build mode

The default. The agent has its full toolset — it can read and edit files, run
commands, browse the web, and dispatch sub-agents — and it works toward
implementing whatever you ask.

Use Build mode for hands-on work: making changes, running tests, fixing bugs.

Build mode is still subject to the current **permission mode**. TUI sessions
default to `ask`, so shell commands and file edits prompt before they run. Press
`Ctrl+P` to toggle between `ask` and `auto`, or use `/permissions ask`,
`/permissions auto`, and `/permissions toggle`.

When you choose an “always allow” approval, Kolega Code stores the local rule in
`.kolega/permissions.json` for that project.

If [gigacode](../../gigacode/) is enabled, the sub-agents inside a workflow run in
`auto` permission mode regardless of this setting — see
[Behavior and safety](../../gigacode/#behavior-and-safety).

## Plan mode

A **read-only** planning pass. Plan mode uses a standalone planning agent that does
not edit your code. Instead it investigates, thinks through an approach, and
produces a plan.

While planning, the agent can:

- Read and search the codebase.
- Run shell commands to investigate (e.g. `git log`, `grep`, running tests) —
  subject to the current permission mode, so commands prompt for approval in `ask`.
- Maintain a shared **task list** (visible in the Planning tab).
- Write a structured plan, which appears in the **Planning** tab.

When the planning agent submits a **complete plan**, you're prompted to decide what
happens next — typically to **implement the plan** or to **keep discussing** it.
The decision is presented as a vertical, arrow-key-selectable option list.

Use Plan mode when a change is non-trivial and you want to agree on the approach
before any files are touched. With [gigacode](../../gigacode/) on, a planning agent
can fan out parallel research across the codebase; those workflow sub-agents stay
read-only too.

## A typical loop

1. Start in **Build mode** and describe the goal.
2. Press `Shift+Tab` to switch to **Plan mode** for anything substantial.
3. Review the plan and shared task list in the **Planning** tab.
4. Choose to implement it.
5. Press `Shift+Tab` back to **Build mode**; the agent works through the task list,
   marking items complete as it goes.

## Switching with slash commands

You don't have to use the keyboard shortcut — the
[slash commands](../slash-commands/) `/plan` and `/build` switch modes too.

:::note
All CLI sessions — including resumed ones — use the CLI-specific coding-agent
prompt. The mode you're in controls which tools are available and whether the agent
edits code, not which prompt template is used.
:::
