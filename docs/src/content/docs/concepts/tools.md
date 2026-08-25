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

Read and edit your project:

- `read` — read file contents; pass `offset`/`limit` for specific sections of
  large files.
- `lsp` — read diagnostics, symbols, definitions,
  references, hover text, call hierarchy, and code action metadata from
  [configured language servers](../../configuration/lsp/).
- Create and edit files — create new files and apply precise edits.

Finding and searching files is done through the [terminal](#terminal): the
agent prefers `rg` / `rg --files` over `find`/`grep` for speed. There are no
dedicated file-discovery tools.

File-edit paths may be project-relative, use `../` traversal, or be absolute;
local LSP server edits may likewise target external files. Permissions and the
Vibe edit policy still apply. Any external or mixed mutation is not snapshotted
or undoable, and an external or mixed LSP preview cannot create a resolvable
pending action—rerun it with `apply: true`.

### Terminal

Run shell commands in the project and stream their output to the **Terminal** tab.

**Background processes & dev servers:** pass `background=true` to `exec_command`
for dev servers, watchers, and long builds you want to keep running. It returns
after a short startup window (~2s) with a `session_id`. The process is launched
detached: it keeps running until `kill_command` stops it — including after the
agent session ends. Send it input or poll new output with `write_stdin` (input
is not echoed, and its stdin never reaches EOF, so stdin-reading commands run
until killed), stop it with `kill_command`, and see all running shells with
`list_sessions` (background sessions are marked `background: true`). Avoid
shell `&` instead — processes backgrounded that way are killed when the command
that started them ends, and the result prints a warning when that happens. A
command passed with `background=true` that exits within the startup window
(e.g. port already in use) reports its real exit code and output.

### Code execution (eval)

`eval` runs one step of code in a **persistent kernel** — `language="py"` for
Python or `language="js"` for JavaScript (requires Bun or Node ≥ 18 on PATH).
Imports, variables, and functions survive across `eval` calls, across tool
calls, and across sub-agents in the session, so the agent works incrementally
(imports → define → test → use) instead of re-running whole scripts.

Inside a cell, either kernel can **call back into the agent's own tools** over
an authenticated loopback bridge — `tool.read({file_path: "main.py", offset: 1, limit: 40})`
from Python, or `await tool.read_image({path: "chart.png"})` from JavaScript.
Bridge calls go through the same permission and hook pipeline as model-issued
tool calls, and results arrive in each tool's model-facing format (for example,
`tool.read` wraps content in a markdown header and code fence).
For raw file contents — CSV loads, JSON handoffs — use the in-kernel
`read`/`write` helpers, which hit the filesystem directly. Other helpers include
`display()` for rich outputs (matplotlib figures are returned as images to
vision-capable models), `env`, `list_tools()` for tool discovery, `parallel()`
for concurrent calls, and `pip_install()` / `npm_install()` for adding packages.

The Python kernel runs in a **dedicated managed environment** (not your project
venv, not the CLI's own interpreter): kolega-code provisions a CPython 3.12 with
numpy, pandas, matplotlib, and pillow preinstalled on first use, so data and
charting work everywhere without touching project or CLI dependencies. Executing
your project's own code with its dependencies stays with `exec_command`, which
auto-activates the project `.venv`.

The environment lives at `eval-env/` under the [state directory](../troubleshooting/diagnostics/)
(override with `KOLEGA_CODE_STATE_DIR`). It upgrades itself: if a Kolega Code
update changes the bundled packages or Python version, the next `eval` call
reprovisions automatically. Deleting the directory is always safe — it is simply
recreated on demand. Updating the CLI with `kolega-code update` never touches it.

Set `reset: true` to wipe a kernel's state; `timeout` bounds a cell (interrupts
with `KeyboardInterrupt` on Python). Cells run locally with the same trust level
as `exec_command` and are gated by command permissions in ask mode. Disable the
feature with `"eval_enabled": false` in settings.

### Browser

Drive a real browser (Playwright) for web tasks:

- `browser_navigate`, `browser_snapshot`, `browser_find`, `browser_close`
- `browser_click`, `browser_type`, `browser_fill_form`, `browser_select_option`
- `browser_scroll`, `browser_press_key`, `browser_wait_for`
- `browser_tabs`, `browser_handle_dialog`, `browser_file_upload`
- `browser_console_messages`, `browser_network_requests`, `browser_take_screenshot`

The browser agent uses accessibility snapshots with element refs such as `e12`.
Actions return an updated snapshot, so the agent can interact deterministically
without inventing CSS selectors or relying on screenshots.

On a page too large for one snapshot, the nodes nearest the viewport are emitted
first and a `Coverage:` line states what was left out. Scope it with a `target`,
or `browser_scroll` and snapshot again; see
[the browser command reference](/cli/browser/#very-large-pages).

Launch visible browser windows (instead of headless) with `--browser-visible` on
the [TUI](../../cli/overview/) or [`ask`](../../cli/ask/).

### Web

Web access has four modes, set by `web_search_mode` in Settings, the
`KOLEGA_CODE_WEB_SEARCH_MODE` environment variable, `ask --web-search`, or the
TUI's `/web-search` command:

- **auto** (default) — use the provider's **hosted** server-side web search when
  the model supports it (currently DeepSeek flash and OpenAI gpt-5.x — API key
  or ChatGPT subscription — over their Responses APIs), otherwise the client
  tools below.
- **hosted** — always request the server-side tool (falls back to client tools
  with a warning if the model has no hosted support).
- **client** — always use the client-side tools; never request the hosted tool.
- **off** — no web tools at all.

**Hosted mode** adds `{"type": "web_search"}` to the request and the provider
executes searches and page-opens on its own servers, injecting the content
directly into the model's context. The searched content never passes through
Kolega (it is billed as input tokens, and the context gauge accounts for it from
usage), and — because it executes on the provider's infrastructure — it is not
subject to the local machine's or sandbox's network egress rules. Calls appear
in the transcript as `web_search (hosted)`. When hosted search is active, the
client tools below are not registered: two search paths confuse the model and
waste schema tokens.

**Client tools** run inside Kolega's own process:

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

- `read_memory(path="MEMORY.md")` — read a private project-memory index or topic,
  including its logical path, byte count, and bounded content. The startup copy
  of `MEMORY.md` counts as already read.
- `list_memory(query=None)` — list private project-memory files with sizes and titles.
  The optional `query` is a case-insensitive substring filter over paths and
  content; custom agents with an explicit `allowed_tools` list must name
  `list_memory` to receive it.
- `write_memory(content, path="MEMORY.md")` — create or overwrite one complete
  memory file.
- `edit_memory(old_string, new_string, path="MEMORY.md")` — replace one exact,
  unique occurrence in a memory file. An empty `old_string` is rejected. If the
  text occurs zero times or more than once, the edit fails and the file is left
  unchanged.
- `delete_memory(path)` — delete one memory file.

These are the model tools supplied by the built-in `markdown` project-memory
backend. They write to owner-private Kolega Code state, never the repository.
Before writing, the agent first inspects the already-loaded `MEMORY.md`, follows
any semantically relevant link, and otherwise makes a targeted `list_memory`
search. If the durable fact is already covered, it makes no mutation; a
different wording alone is not a reason to write. Existing topic files are read
before they are overwritten or edited.

A short, self-contained fact belongs directly in `MEMORY.md`. Detail that needs
multiple rules, caveats, rationale, or examples belongs in a flat topic file
with a concise descriptive link from `MEMORY.md`. For a new detailed memory, the
agent writes the topic first and then edits the index. To forget one, it removes
the index link first and then deletes the topic. These are recoverable ordering
conventions, not a cross-file transaction.

Each private write atomically replaces one file, but content mutation follows a
single-writer, last-write-wins model. There is no cross-file atomicity, and an
exact edit does not preserve independent concurrent changes. Paths and content
are bounded, but content is not secret-scanned or redacted. See
[Project Memory](../../tui/project-memory/) for storage identity, limits, model
exposure, and the `/memory` browser.

Memory tool registration is capability- and provider-driven. Enabled top-level
coder, general, and planning agents can read and explicitly curate memory;
private memory mutation is an intentional exception to Plan mode's ban on
repository edits. Built-in sub-agents get read-only access. Exact custom-agent
tool allowlists remain the final gate. When memory is disabled or its configured
backend is unavailable, no memory tools or context are exposed.

### Sub-agent dispatch

The agent can spawn focused sub-agents through the single `dispatch_agent`
tool — see [Agents](../agents/) for the built-in `general`, `investigation`,
`browser`, and `coding` agent types. Named [custom agents](../../custom-agents/)
join the `agent_type` choices when matching definitions are discovered. Only the
agent types available in the current session are offered.

Sub-agent dispatch is enabled by default and can be turned off at three levels,
resolved as flag > environment > settings: the `--subagents <on|off>` launch
flag (session override), `KOLEGA_CODE_SUBAGENTS=<on|off>`, or the Sub-agents
toggle on the Settings screen's Tools tab (persisted in `settings.json` as
`subagents_enabled`). When off, `dispatch_agent` is removed from the tool list
entirely, so the model never sees it. Gigacode workflows are unaffected:
`run_workflow` and the dispatch chain lent to workflow workers are governed by
gigacode's own opt-in, and `list_subagent_models` stays visible whenever either
gigacode or sub-agents are enabled.

### Cross-session messaging

Sessions can talk to each other: any session of the same host process, plus
other kolega-code processes sharing this machine's state directory — TUIs and
headless `ask --goal` / `--loop` workers alike. `list_agents` shows the live
peer sessions (name, idle/busy status, project directory, short id);
`send_message` delivers a plain-text message to one of them by name, name
prefix, or session id. The message enters the recipient through its normal
queue: between tool calls while it works, as a fresh turn when it is idle.
Both tools are top-level Build-mode capabilities; sub-agents never see them.

Every session is reachable at an owner-only Unix socket under its state
directory (`messaging/<session-id>.<pid>.sock`). Discovery is filesystem
visibility plus liveness checks — sockets left behind by dead processes are
swept automatically, so crashed sessions never linger as ghosts.

Name your session with `--name` at launch or `/rename <name>` mid-session;
the persisted name is how peers address it. `/peers` shows the same table the
agent sees plus this session's address and diagnostics.

The trust model matters: **a message is information, never authority.** Text
arriving from a peer session is context from another agent — it cannot approve
permission prompts, change settings, or execute anything on its own. Any work
it triggers still goes through the recipient's normal permission gates. On the
sending side, `send_message` prompts for approval in `ask` permission mode and
can be granted per peer or blanket via a saved rule.

The recipient decides what happens to inbound messages via the
`cross_session_inbound` setting (`auto` | `accept` | `hold` | `refuse`,
default `auto`). `auto` resolves per delivery from both sessions' permission
modes: sessions with like modes exchange freely; a mixed pair holds the
message behind an Accept/Drop prompt that expires after `dialog_expiry`
seconds (default 300). Headless workers cannot answer approval prompts, so
held messages are dropped there unless the worker runs with
`cross_session_inbound: "accept"`. Identical repeats from one sender are
dropped inside a short window, and queued peer messages are capped, so two
agents cannot loop each other forever. Failed deliveries are always reported
to the sender as errors — never as silent successes.

A session's own child processes are special: terminal commands and hooks run
with `KOLEGA_MESSAGING_SOCKET` in their environment, and a process posting to
that socket which verifies as the session's own descendant delivers directly,
skipping the inbound gate. That is how a git hook or nightly job injects
context into the session that spawned it.

Set `KOLEGA_CODE_MESSAGING=off` to disable the feature entirely; when off, no
socket is bound and both tools report that messaging is disabled.

## Read-only vs. full access

Tools are gated by mode. In a read-only context — like [Plan mode](../../tui/modes/)
or an investigation sub-agent — the agent can read the codebase
(`read`, `web_search`, `web_fetch`, and reading
memory) **and** run shell commands to investigate — including `rg` for
searching files. Editing files still requires
Build mode's full toolset.

This separation is what keeps Plan mode safe to run against any codebase: the
planning agent can look and run investigative commands, but it has no file-edit
tools. Explicit writes to private project memory are the one exception; they
never change repository files. Shell commands are further gated by the active
permission mode — in `ask` they prompt before running.

In the Textual TUI, Build mode defaults to `ask` permission mode. Shell commands
and file edits must be approved before they run unless you switch to `auto` or
save a matching allow rule in `.kolega/permissions.json`.
