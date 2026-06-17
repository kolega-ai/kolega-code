---
title: Tools
description: The categories of tools the Kolega Code agent can use.
---

A **tool** is any concrete action the agent can take — reading a file, running a
command, taking a screenshot. The model decides which tools to call; the agent runs
them and feeds the results back into the conversation. Which tools are available
depends on the [agent type](../agents/) and the current [mode](../../tui/modes/).

## Tool categories

### File & code

Read, search, and edit your project:

- `list_directory` — list files in a directory.
- `read_entire_file`, `read_file_section` — read file contents.
- `find_files_by_pattern` — glob-based file search.
- `search_codebase` — search across the codebase.
- Create and edit files — create new files and apply precise edits.

### Terminal

Run shell commands in the project and stream their output to the **Terminal** tab.

### Browser

Drive a real browser (Playwright) for web tasks:

- `launch_browser`, `list_browsers`, `close_browser`
- `get_browser_interactive_elements`, `get_browser_console_logs`
- `take_browser_screenshot`
- `interact_with_browser`, `set_browser_select_value`

Launch visible browser windows (instead of headless) with `--browser-visible` on
the [TUI](../../cli/overview/) or [`ask`](../../cli/ask/).

### Web

- `web_fetch` — fetch and parse a web page without a full browser.

### Reasoning & memory

- `think_hard` — an extended-reasoning step that uses the
  [thinking model](../../configuration/providers-and-models/) and its token budget.
- `read_memory`, `write_memory` — read and write workspace memories.

### Sub-agent dispatch

The agent can spawn focused sub-agents — see [Agents](../agents/) for
`dispatch_investigation_agent`, `dispatch_browser_agent`, `dispatch_coding_agent`,
and `dispatch_general_agent`.

## Read-only vs. full access

Tools are gated by mode. In a read-only context — like [Plan mode](../../tui/modes/)
or an investigation sub-agent — only non-mutating tools are available
(`list_directory`, `read_entire_file`, `read_file_section`, `search_codebase`,
`find_files_by_pattern`, `web_fetch`, `think_hard`, and reading memory). Editing
files and running commands require Build mode's full toolset.

This separation is what makes Plan mode safe to run against any codebase: the
planning agent can look but not touch.

In the Textual TUI, Build mode defaults to `ask` permission mode. Shell commands
and file edits must be approved before they run unless you switch to `auto` or
save a matching allow rule in `.kolega/permissions.json`.
