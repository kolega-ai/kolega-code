---
title: Agents
description: The agent types Kolega Code uses and how they divide up work.
---

Kolega Code is built around several **agent types**. They share the same core loop
but differ in which tools they can use and what they're specialized for. The main
agent can hand off to the others when a task benefits from focus.

## Agent types

| Agent | Specialty | Can edit files? |
| --- | --- | --- |
| **Coder** | The main, general-purpose coding agent. Full toolset. | Yes |
| **Investigation** | Read-only exploration of the codebase. | No |
| **Browser** | Web navigation and interaction via Playwright. | No (browser tools only) |
| **General** | A flexible agent that can read and dispatch other agents. | Limited |
| **Planning** | Drives [Plan mode](../../tui/modes/): investigates and writes a plan, read-only. | No |

## Dispatching sub-agents

For larger jobs, the main agent can **dispatch** a sub-agent and incorporate its
findings. The available dispatch targets include:

- `dispatch_investigation_agent` — explore and report back, without making changes.
- `dispatch_browser_agent` — perform a web task.
- `dispatch_coding_agent` — hand off a self-contained coding task.
- `dispatch_general_agent` — a general-purpose helper.

When [gigacode](../../gigacode/) is enabled, the main agent can also orchestrate
**many** of these sub-agents at once through a workflow — running them in parallel
or in pipelines and collecting the results.

Sub-agent activity is reported live. In the TUI each sub-agent gets a live card in
the conversation, and pressing `Ctrl+G` opens the
[sub-agent inspector](../../tui/interface/#sub-agent-inspector) — a full-screen view
of every sub-agent's complete trajectory: its thinking, tool calls, and results.
Lifecycle events also appear in the **Logs** tab. With `ask`, sub-agent activity
surfaces as events (on stderr in plain mode, or as `event` objects with `--json`).

## Modes vs. agents

It's worth separating two ideas:

- **Interaction mode** (Build / Plan) is what *you* toggle with `Shift+Tab`. It
  determines whether the agent is editing (Build, the Coder agent) or planning
  read-only (Plan, the Planning agent).
- **Agent type** is which specialized agent is doing a piece of work — including
  sub-agents the main agent dispatches under the hood.

See [Build & Plan Modes](../../tui/modes/) for the user-facing workflow.
