---
title: Architecture
description: A high-level look at how Kolega Code works under the hood.
---

A terminal agent loop with tools is the baseline. Kolega Code keeps that loop, then
adds the pieces that matter for wider work: mode boundaries, evented local
execution, specialized sub-agent dispatch, role-specific models, and
[Gigacode](../../gigacode/) workflow orchestration.

You don't need the internals for everyday use, but this mental model helps when
you're debugging behavior, choosing models, or deciding when a task should fan out.

## The agent loop

At the center is an **agent**: a loop that sends your conversation to an LLM,
receives a response, and — when the model asks to use a tool — runs that tool and
feeds the result back. It repeats until the model produces a final answer.

The agent owns three things worth knowing about:

- **Conversation** — the running message history.
- **History compression** — when the conversation grows large, older context is
  compressed to stay within the model's budget. You can trigger this manually with
  [`/compress`](../../tui/slash-commands/) and inspect the current size with
  `/context`.
- **Events** — the agent emits a stream of typed events (chat messages, tool
  activity, terminal output, status updates, sub-agent lifecycle). The TUI renders
  these live; `ask --json` prints them.

## Tools

Everything the agent *does* — reading a file, editing code, running a command,
fetching a web page — happens through a **tool**. Which tools an agent can use
depends on its type and mode (for example, Plan mode is restricted to read-only
tools). See [Tools](../tools/) for the categories.

## Agents and sub-agents

There isn't just one agent. The main agent can **dispatch sub-agents** for focused
work — investigating the codebase, driving a browser, or handling a self-contained
coding task — and track their progress. Different agent types expose different
toolsets. See [Agents](../agents/).

Sub-agents are useful one at a time, but they become more powerful when a task can
be split across independent workstreams. That's what Gigacode automates.

## Gigacode workflows

With [Gigacode](../../gigacode/) enabled, Kolega Code can write a small workflow that
launches many sub-agents, organizes them into phases, and synthesizes their
results. Workflows can run broad audits, migration checks, implementation batches,
or review panels without forcing one model to do every step serially.

Workflow runs are evented like normal agent work, so the TUI can show phase
progress and sub-agent activity. They also save artifacts — result files,
transcripts, raw JSONL, and a resume journal — under Kolega Code's state directory.

## Models per role

A single turn may use more than one model. The main reasoning runs on the
**long-context** model, small utility calls use the **fast** model, and extended
reasoning uses the **thinking** model. You control each independently — see
[Providers & Models](../../configuration/providers-and-models/).

Kolega Code can also override models per agent role: planning, building, investigation,
general, and browser. This lets wide workflows put cheaper or faster models on
routine investigation while reserving stronger models for implementation or
synthesis.

## Local execution

Kolega Code runs against your real environment:

- **Filesystem** — reads and writes files in your project directory.
- **Terminal** — runs shell commands and streams their output.
- **Browser** — automates a real browser (via Playwright) for web tasks.

Sessions, settings, permissions, and credentials are persisted locally. That local
state is what lets Kolega Code operate on a real development workspace while preserving
resumable sessions and project-specific controls.

Private [Project Memory](../../tui/project-memory/) is another local state
service, but it is not session history: an agent or user must explicitly curate
it, and linked Git worktrees share it through their common Git identity. Its
provider interface keeps the built-in Markdown backend separate from possible
future structured backends; switching providers never migrates data
automatically.
