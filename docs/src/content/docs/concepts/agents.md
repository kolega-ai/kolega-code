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

Sub-agent activity is reported live. In the TUI each sub-agent gets a live card in
the conversation, and pressing `Ctrl+G` opens the
[sub-agent inspector](../../tui/interface/#sub-agent-inspector) — a full-screen view
of every sub-agent's complete trajectory: its thinking, tool calls, and results.
Lifecycle events also appear in the **Logs** tab. With `ask`, sub-agent activity
surfaces as events (on stderr in plain mode, or as `event` objects with `--json`).

## Custom agents

Custom agents are named sub-agents defined in Markdown. Put project definitions
under `.kolega/agents/`, or user definitions under the `agents/` directory in
Kolega Code's state directory. The latter follows `KOLEGA_CODE_STATE_DIR` and
`--state-dir`; otherwise Kolega uses its normal platform-specific state location.
Subdirectories are scanned recursively.

For example, `.kolega/agents/code-reviewer.md`:

```markdown
---
name: code-reviewer
description: Reviews changes for correctness, regressions, and missing tests.
mode: build
tools:
  - read_entire_file
  - read_file_section
  - search_codebase
  - find_files_by_pattern
  - exec_command
model: anthropic/claude-opus-4-8
thinking_effort: high
max_iterations: 20
---

You are a rigorous code reviewer. Report findings in severity order, cite the
relevant files and lines, and identify missing test coverage. Do not edit files.
```

The YAML frontmatter supports:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Lowercase kebab-case identifier, up to 64 characters. |
| `description` | Yes | Routing guidance that tells the parent when to delegate. |
| `mode` | No | `build`, `plan`, or `all`; defaults to `build`. |
| `tools` | No | Exact tool allowlist. Omit it to inherit the caller's eligible tools; use `[]` for no tools. |
| `model` | No | Main model in `<provider>/<model-id>` form. Omit it to inherit the General-agent model. |
| `thinking_effort` | No | Reasoning effort supported by the effective model. |
| `max_iterations` | No | Positive limit for the custom agent's tool loop. |

The Markdown body is the agent's base system prompt. Kolega appends the same
dynamic project context used by its built-in agents, including `AGENTS.md`, agent
memory, workspace memories, and propagatable Agent Skills or host extensions.

Project definitions override user definitions with the same `name`. Invalid
files are skipped with diagnostics rather than preventing Kolega from starting.
Inspect the effective registry with:

```sh
kolega-code agents list --project .
kolega-code agents validate --project .
```

The Build or Plan agent sees the names and descriptions enabled for its mode and
can select one through `dispatch_custom_agent`. Build is the safe default; an
agent must explicitly declare `mode: plan` or `mode: all` before Plan can use it.
You can request a particular agent in plain
language, for example: “Use the code-reviewer custom agent to review this change.”
Each call receives a fresh context and reports its result back to the parent.

Custom agents cannot elevate authority. Their tools are always a subset of the
invoking agent's effective tools, they inherit the session's permission mode and
approval callback, and they cannot dispatch other agents or gigacode workflows.
Consequently, even an agent explicitly enabled for Plan mode cannot acquire editing tools.
Per-agent hooks, MCP configuration, structured output, and primary-session custom
agents are not supported in this version.

## Modes vs. agents

It's worth separating two ideas:

- **Interaction mode** (Build / Plan) is what *you* toggle with `Shift+Tab`. It
  determines whether the agent is editing (Build, the Coder agent) or planning
  read-only (Plan, the Planning agent).
- **Agent type** is which specialized agent is doing a piece of work — including
  sub-agents the main agent dispatches under the hood.

See [Build & Plan Modes](../../tui/modes/) for the user-facing workflow.
