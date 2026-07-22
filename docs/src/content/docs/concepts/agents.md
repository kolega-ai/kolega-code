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
| **Investigation** | Read-only exploration of the codebase. Can run investigative commands. | No |
| **Browser** | Web navigation and interaction via Playwright. | No (browser tools only) |
| **General** | A flexible agent for self-contained workspace tasks. | Yes |
| **Planning** | Drives [Plan mode](../../tui/modes/): investigates and writes a plan. Can run commands but cannot edit files. | No |

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

### Per-dispatch model routing

Sub-agent dispatches normally inherit the CLI/host model configuration for the
target role. Custom agents continue to use their Markdown model settings and then
their normal General-role inheritance. Omit `model_override` to keep this behavior.

When a particular task needs another model, the agent can call the read-only
`list_subagent_models` tool. It reports configured providers, provider-qualified
model IDs, exact effort options, vision support, and current role defaults without
exposing credentials. The agent should use it before choosing a non-default route
rather than guessing model or effort names.

An ordinary dispatch accepts one complete object under `model_override`:

```json
{
  "task": "Review the authentication design for subtle security flaws.",
  "model_override": {
    "provider": "anthropic",
    "model": "claude-opus-4-8",
    "thinking_effort": "high"
  }
}
```

All three fields are mandatory when the object is present; Kolega Code never
inherits only the missing provider, model, or effort. A model with effort controls
requires one of the exact `thinking_effort` strings returned by
`list_subagent_models`. A model without effort controls requires an explicit
`"thinking_effort": null`—`null` is valid only in that case.

Selections are revalidated at dispatch time. Unsupported provider/model pairs,
unconfigured providers, incompatible Browser models, invalid efforts, and
incomplete objects fail the dispatch instead of falling back to another model.
A Browser dispatch must select a catalog entry with `supports_vision: true`.
A complete dispatch override replaces a custom agent's Markdown model and effort
as a unit.

[Gigacode](../../gigacode/#per-worker-model-overrides) uses the same atomic rule,
but names the third field `effort` inside `model_override`. Its override applies
only to the direct workflow worker; descendants keep their role defaults unless
their own dispatch provides a complete override. In Plan mode, model routing never
changes the forced read-only Investigation agent or its permissions.

Sub-agent activity is reported live. In the TUI each sub-agent gets a live card in
the conversation, and pressing `Ctrl+G` opens the
[sub-agent inspector](../../tui/interface/#sub-agent-inspector) — a full-screen view
of every sub-agent's complete trajectory: its thinking, tool calls, and results.
Lifecycle events also appear in the **Logs** tab. With `ask`, sub-agent activity
surfaces as events (on stderr in plain mode, or as `event` objects with `--json`).

## Custom agents

You can also define reusable, named specialists as Markdown files. A custom agent
runs in a fresh sub-agent context with its own prompt and optional mode, tool,
model, effort, and iteration limits, while remaining inside the invoking agent's
permission boundary.

See [Custom Agents](../../custom-agents/) to create, validate, and invoke project
or user definitions.

## Modes vs. agents

It's worth separating two ideas:

- **Interaction mode** (Build / Plan) is what *you* toggle with `Shift+Tab`. It
  determines whether the agent is editing (Build, the Coder agent) or planning
  read-only (Plan, the Planning agent).
- **Agent type** is which specialized agent is doing a piece of work — including
  sub-agents the main agent dispatches under the hood.

See [Build & Plan Modes](../../tui/modes/) for the user-facing workflow.
