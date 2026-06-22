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

- `web_search` — search the web for relevant pages and return ranked results.
- `web_fetch` — fetch and parse a web page without a full browser.

Use `web_search` when you do not already know the right URL, then follow up with
`web_fetch` to read a result in depth. The default backend is keyless DuckDuckGo;
Firecrawl, Tavily, and self-hosted SearXNG can be selected in Settings or with
environment variables.

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
or an investigation sub-agent — the agent can read and search the codebase
(`list_directory`, `read_entire_file`, `read_file_section`, `search_codebase`,
`find_files_by_pattern`, `web_search`, `web_fetch`, `think_hard`, and reading
memory) **and** run shell commands to investigate. Editing files still requires
Build mode's full toolset.

This separation is what keeps Plan mode safe to run against any codebase: the
planning agent can look and run investigative commands, but it has no file-edit
tools. Shell commands are further gated by the active permission mode — in `ask`
they prompt before running.

In the Textual TUI, Build mode defaults to `ask` permission mode. Shell commands
and file edits must be approved before they run unless you switch to `auto` or
save a matching allow rule in `.kolega/permissions.json`.
