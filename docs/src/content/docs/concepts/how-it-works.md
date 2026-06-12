---
title: Architecture
description: A high-level look at how Kolega Code works under the hood.
---

You don't need to know the internals to use Kolega Code, but a mental model helps
when you're debugging behavior or extending it. This is a light overview — not an
API reference.

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

## Models per role

A single turn may use more than one model. The main reasoning runs on the
**long-context** model, small utility calls use the **fast** model, file edits use
the **edit** model, and extended reasoning uses the **thinking** model. You control
each independently — see [Providers & Models](../../configuration/providers-and-models/).

## Local execution

Kolega Code runs against your real environment:

- **Filesystem** — reads and writes files in your project directory.
- **Terminal** — runs shell commands and streams their output.
- **Browser** — automates a real browser (via Playwright) for web tasks.

Sessions and settings are persisted locally. This local-first design is what makes
the agent useful for real development work rather than a sandboxed demo.
