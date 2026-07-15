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
- `search_codebase` — search the codebase by regular expression (ripgrep/grep), e.g. `foo|bar`.
- `lsp` — read diagnostics, symbols, definitions,
  references, hover text, call hierarchy, and code action metadata from
  [configured language servers](../../configuration/lsp/).
- Create and edit files — create new files and apply precise edits.

### Terminal

Run shell commands in the project and stream their output to the **Terminal** tab.

### Browser

Drive a real browser (Playwright) for web tasks:

- `browser_navigate`, `browser_snapshot`, `browser_find`, `browser_close`
- `browser_click`, `browser_type`, `browser_fill_form`, `browser_select_option`
- `browser_tabs`, `browser_wait_for`, `browser_handle_dialog`, `browser_file_upload`
- `browser_console_messages`, `browser_network_requests`, `browser_take_screenshot`

The browser agent uses accessibility snapshots with element refs such as `e12`.
Actions return an updated snapshot, so the agent can interact deterministically
without inventing CSS selectors or relying on screenshots.

Launch visible browser windows (instead of headless) with `--browser-visible` on
the [TUI](../../cli/overview/) or [`ask`](../../cli/ask/).

### Web

- `web_search` — search the web for relevant pages and return ranked results.
- `web_fetch` — retrieve a URL, extract local-readable content, and answer an
  instruction with source evidence. It handles HTML, plain text, Markdown,
  JSON/XML/feeds, PDF, DOCX, PPTX, XLSX, and XLS. HTML extraction uses a
  quality-gated local chain (Trafilatura, Readability, semantic DOM, then full
  visible text) and automatically selects the best result.

Use `web_search` when you do not already know the right URL, then follow up with
`web_fetch` to read a result in depth. The default backend is keyless DuckDuckGo;
Firecrawl, Tavily, and self-hosted SearXNG can be selected in Settings or with
environment variables.

`web_fetch` never sends the URL or page content to a third-party reader service
and does not launch a browser. If a page appears to be a JavaScript-rendered SPA,
the result says that its content may be incomplete so the agent can use the
browser tools instead. Scanned/image-only PDFs and legacy DOC/PPT files require
OCR/conversion outside this tool.

### Reasoning & memory

- `think_hard` — an extended-reasoning step that uses the
  [thinking model](../../configuration/providers-and-models/) and its token budget.
- `read_memory(path="MEMORY.md")` — read a private project-memory index or topic,
  including its logical path, SHA-256 revision, byte count, and bounded content.
- `write_memory(memory_content, path="MEMORY.md", mode="append",
  expected_sha256=None)` — append an exact Markdown fragment, or replace a whole
  entry using its current revision. Creation by replacement uses
  `expected_sha256="missing"`.
- `delete_memory(path, expected_sha256)` — delete an entry only if its current
  revision matches.

These are the model tools supplied by the built-in `markdown` project-memory
backend. They write to owner-private Kolega Code state, never the repository.
Read before replacing or deleting: compare-and-swap conflicts reject stale
changes instead of overwriting them. Paths and content are bounded, but content
is not secret-scanned or redacted. See
[Project Memory](../../tui/project-memory/) for storage identity, limits, model
exposure, and the `/memory` browser.

Memory tool registration is capability- and provider-driven. Enabled top-level
coder, general, and planning agents can read and explicitly curate memory;
private memory mutation is an intentional exception to Plan mode's ban on
repository edits. Built-in sub-agents get read-only access. Exact custom-agent
tool allowlists remain the final gate. When memory is disabled or its configured
backend is unavailable, no memory tools or context are exposed.

### Sub-agent dispatch

The agent can spawn focused sub-agents — see [Agents](../agents/) for
`dispatch_investigation_agent`, `dispatch_browser_agent`, `dispatch_coding_agent`,
and `dispatch_general_agent`. Named [custom agents](../../custom-agents/) are
available through `dispatch_custom_agent` when matching definitions are discovered.

## Read-only vs. full access

Tools are gated by mode. In a read-only context — like [Plan mode](../../tui/modes/)
or an investigation sub-agent — the agent can read and search the codebase
(`list_directory`, `read_entire_file`, `read_file_section`, `search_codebase`,
`find_files_by_pattern`, `web_search`, `web_fetch`, `think_hard`, and reading
memory) **and** run shell commands to investigate. Editing files still requires
Build mode's full toolset.

This separation is what keeps Plan mode safe to run against any codebase: the
planning agent can look and run investigative commands, but it has no file-edit
tools. Explicit writes to private project memory are the one exception; they
never change repository files. Shell commands are further gated by the active
permission mode — in `ask` they prompt before running.

In the Textual TUI, Build mode defaults to `ask` permission mode. Shell commands
and file edits must be approved before they run unless you switch to `auto` or
save a matching allow rule in `.kolega/permissions.json`.
