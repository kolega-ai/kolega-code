# Changelog

All notable changes to Kolega Code are documented here.

This project uses GitHub Releases for detailed generated release notes. This file provides a concise, human-maintained summary of user-visible changes.

## Unreleased

### Added

- DeepSeek V4-Pro now routes to the Responses API by default (joining V4-Flash),
  removing the model-name carve-out. Both DeepSeek models support `none`, `low`,
  `high`, and `max` reasoning effort (`low` is new; `none` disables reasoning).
- Added support for xAI Grok 4.6 (`grok-4.6`): 500K-token context, vision input,
  and `low`/`medium`/`high`/`xhigh` reasoning effort. Selectable in the Settings
  UI, `/model` picker, CLI flags, and env vars.

## 0.29.0 - 2026-08-13

### Added

- Custom endpoints: use models from any OpenAI Chat Completions, OpenAI
  Responses, or Anthropic Messages-compatible server (LM Studio, Ollama, vLLM,
  ...). Define endpoints under `custom_endpoints` in settings.json or via the
  `--custom-endpoints` / `--endpoint-url` flags and `KOLEGA_CODE_CUSTOM_ENDPOINTS`
  / `KOLEGA_CODE_ENDPOINT_*` environment variables (never persisted), then select
  them as `custom:<id>` anywhere a provider is chosen. Supports thinking-effort
  modes per wire dialect and native reasoning replay.

- Custom endpoints accept a `temperature` field (per endpoint, or per model in
  its `models` map; default 1.0), exposed in the TUI endpoint editor, via
  `--endpoint-temperature`, and via `KOLEGA_CODE_ENDPOINT_TEMPERATURE`. Sent on
  Chat Completions and Anthropic requests; ignored by Responses-style endpoints.

### Fixed

- Together models (e.g. Kimi K2.7-Code) now replay prior reasoning through the
  native `reasoning` field instead of visible `*Thinking:*` text, avoiding a full
  re-derivation of the chain-of-thought on every turn. Reasoning models without a
  declared thinking-effort spec now resolve their provider's replay field.

## 0.28.4 - 2026-08-12

### Fixed

- Conversation compaction now keeps the summarization request within the run's
  input token budget, so auto-compaction no longer fails on long conversations
  and lets the run continue instead of aborting.

## 0.28.3 - 2026-08-12

### Changed

- Claude-style `edit` and `write` tools are now the default model-facing edit
  protocol. Explicit overrides and model-catalog preferences still take
  precedence, and the legacy search/replace protocol remains available.

## 0.28.2 - 2026-08-12

### Fixed

- Native Tinker sessions now keep repeated provider tool-call IDs paired with
  their distinct executions after persistence, avoiding false interrupted-tool
  placeholders and preserving every result in subsequent model history.

## 0.28.1 - 2026-08-12

### Fixed

- Restored native Tinker tool dispatch so generated tool calls execute through
  the agent loop instead of being returned as plain assistant output.

## 0.28.0 - 2026-08-11

### Added

- ATIF v1.7 trajectory export: `sessions export --format atif` converts any
  session (including pre-upgrade v1 sessions, with explicit conversion
  warnings) into a validated [ATIF](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
  document, and `ask --atif-output FILE` writes one directly when a run ends —
  completed, failed, or cancelled, with or without `--save`. Embedded subagent
  trajectories, per-step token metrics, hydrated oversized tool results, and
  atomic file+assets output; image trajectories require `--output`/a file
  destination for portable relative asset paths.

- The interactive CLI and `ask` can now load one installed Python extension at
  launch with `--extension MODULE:FACTORY` and an optional opaque
  `--extension-config PATH`. An extension contributes ordinary prompt and tool
  extensions, binds to each top-level agent generation before its first model
  request, may observe structured LLM trace records (forwarded to every LLM
  client like the usage ledger), and is cleaned up exactly once per generation,
  including on interactive agent rebuilds. Extension contracts are exported
  from the package root, alongside `llm_call_origin`/`helper_origin` and
  `TinkerTraceRecord`; a minimal example package lives in `examples/extension`.

- Agent Skills can now be enabled or disabled from Settings,
  `settings.json`, `KOLEGA_CODE_SKILLS`, or `--skills on|off`.

- The TUI status usage panel now shows the provider cache hit percentage
  alongside the other usage figures.

- After quitting, the CLI prints the exact
  `kolega-code . --resume <id>` command to resume the session that was just
  saved.

- Hosts can now continue a restored conversation without inserting a user
  message: `BaseAgent.continue_from_history_stream()` runs the ordinary agent
  loop (compaction, tool execution, usage accounting, and cancellation
  included) and yields the same chunk format. Sessions containing such a
  continuation turn are not loadable by older kolega-code versions.

- Oversized terminal command output is now spilled to a file instead of being
  lost: the tool result keeps the bounded head/tail preview plus a path to
  the full output, and the PTY is drained before detaching so a killed
  command keeps its final output.

### Changed

- Automatic compaction now gets one recovery attempt when the ordinary pass —
  which keeps the six most recent messages verbatim — still leaves a request
  above the model's resolved input budget: a second pass with no retained tail
  (`keep_recent=0`, journaled with trigger `auto_tail_fallback`) folds
  everything before the safe boundary into the summary, still without touching
  the raw history or splitting a tool call from its result. A pass that failed
  at the model or hit the minimum-history guard is not retried. When the paired
  `--context-window-tokens` + `--max-output-tokens` cap is active and the
  request still does not fit, the run now fails closed with
  `LLMContextWindowExceededError` before the primary model is called (recorded
  as the `run.failed` error code) instead of dispatching the oversized request;
  uncapped runs keep the legacy provider-accept-or-reject behavior.

- **Breaking:** `kolega-code ask --json` now streams the semantic session-event
  protocol — one v2 event envelope per line (`schema: "kolega.session.event"`,
  stable `id`/`seq`/`timestamp`, agent lineage, typed `payload`) — replacing
  the former `{"kind": "message"|"event"|"summary"}` records with no
  compatibility flag. Live output and the new
  `sessions export --format events-jsonl` return the same records. Every run
  records the full journal (LLM call ids, subagent turns and tool results,
  `context.system`, compaction/rewind provenance, and `run.completed`/
  `run.failed`/`run.cancelled` terminals on every exit path); unsaved runs use
  an in-memory journal and leave no session state. Plain (non-JSON) `ask`
  output is unchanged. Session journals now write schema v2 events (v1
  sessions remain readable; resumed v1 sessions append v2 lines — older
  kolega-code versions will refuse to read those journals rather than
  mis-replaying them).

- Tool definitions are now explicit, checked-in artifacts. Every built-in
  tool's model-visible description lives as data beside the prompt assets
  (`kolega_code/agent/prompt_templates/tools/`) and its input schema as a
  literal dict in `kolega_code/agent/tool_definitions.py`; nothing about what
  the model sees is derived from signatures or docstrings anymore, and the
  migration is byte-identical on the wire. `ToolExtension` tools (including
  CLI extensions loaded with `--extension` and MCP tools) now declare an
  explicit `tool_descriptions` entry alongside `tool_schemas`; a tool missing
  either fails at registration with a clear error before any model request.

- TUI responsiveness for long sessions is improved: the transcript scrollback
  window is capped at 120 mounted entries (scroll-up still walks the full
  history), flush cadences are unified and paced by event-loop lateness,
  modal inspectors stay on the unpaced base cadence, and the native Tinker
  stack no longer loads at package import time — faster startup and fewer
  event-loop stalls.

### Removed

- The `tool_definition_from_callable` helper (and the `Args:`-block
  strip/keep rules behind it) is gone, including its `kolega_code` package
  root export: definitions are declared, not derived.

### Fixed

- A session could be opened in two kolega-code instances at once, interleaving
  duplicate event sequence numbers into the session journal until it became
  unreadable (and `switch_worktree`/`save` failed with "Session event sequence
  gap"). Resuming a session that is already open in another instance is now
  refused with a clear error, journal appends allocate sequence numbers under
  a cross-process lock, and `kolega-code sessions repair <id>` renumbers a
  corrupted journal back into a contiguous sequence.

- File tools (`read`, `write`, `edit`, `multi_edit`, `read_image`) now expand
  a `$KOLEGA_SCRATCHPAD` (or `${KOLEGA_SCRATCHPAD}`) reference in path
  arguments to the session scratchpad at the tool-dispatch choke point, so
  the shell spelling no longer creates a literal `$KOLEGA_SCRATCHPAD`
  directory in the workspace.

- Runtime model-catalog caches created before input-budget conventions were
  introduced are now ignored safely instead of injecting incomplete model
  specs that fail later with `KeyError: 'input_budget'`. Refreshing a catalog
  rewrites it in the current schema.

- Kimi Coding Plan model IDs now match the provider's real wire IDs:
  `k3-256k` (fixed 256K K3 route) and `kimi-for-coding-highspeed` are
  recognized, and the fabricated `k3[1m]` ID is gone (Kimi's 1M context is
  the same `k3` ID, tier-gated). Kimi and Moonshot input budgets treat output
  as a separate allowance (verified by live probes), so the full context
  window is usable as input instead of being cut down by the output cap.

- OpenAI and ChatGPT gpt-5.x requests that reproduced reserved Harmony
  control-token spellings as data (for example after a tool result read a
  file containing `<|channel|>`-style tokens) were rejected with
  `APIError: Request blocked`, permanently poisoning the session. The
  spellings are now escaped on the transport copy only — the persisted
  transcript is unchanged — and already-poisoned sessions heal on the next
  request.

- Responses-based providers (OpenAI, ChatGPT, DeepSeek) now return multiple
  tool calls per model round instead of one at a time.

- After history compaction (automatic or `/compact`) the approved plan and
  the shared task list are re-injected into the model's context as system
  reminders, so the model no longer loses track of them mid-run.

- Terminal sessions that end while a backgrounded `&` job still holds the PTY
  are now reported as exited instead of spinning through the whole yield
  window, and naturally exited sessions are hidden from the session listing
  instead of lingering as stale entries.

- The shared terminal-tools guidance now warns that `rg -r` means
  `--replace` (recursion is the default), which previously made `rg -rn`
  output look like a redaction layer.

## 0.27.1 - 2026-08-10

### Added

- Thinking Machines models can now be used through either the hosted API or
  the native Tinker `SamplingClient`, with model discovery, settings, usage
  tracking, and connection testing integrated across the CLI and TUI.
- History compression thresholds can now be configured from Settings,
  `settings.json`, `KOLEGA_CODE_COMPRESSION_THRESHOLD`, or
  `--compression-threshold`.
- Sub-agent dispatch can now be enabled or disabled for a session from
  Settings, `settings.json`, `KOLEGA_CODE_SUBAGENTS`, or `--subagents on|off`.

### Changed

- Compression summaries can use up to 8,192 tokens, reducing the risk that
  important context is lost in longer sessions.
- The bundled OpenRouter catalogue was refreshed from the 2026-08-10 model
  list and weekly rankings. It now includes 260 tool-capable models, including
  featured `deepseek/deepseek-v4-flash-0731`.

### Fixed

- Provider connection tests now send a system message, improving compatibility
  with models that reject probes containing only a user message.

### Security

- Updated `h2` to `4.4.1` to address `GHSA-6hr6-w5qg-qmwg`.

## 0.27.0 - 2026-08-06

### Changed

- The `eval`, `exec_command`, and `lsp` tool descriptions shipped to model
  providers are slimmer (about 450 cl100k tokens less across the three,
  roughly 8% of the whole tool payload): prose now states decision rules and
  contracts instead of restating parameter schema text, `eval`'s JavaScript
  prelude folds to "same helpers as Python, camelCase" with only the
  JS-specific pitfalls spelled out, `lsp`'s operation list groups the seven
  operations that share the `(path, line, symbol)` arguments onto one line,
  and `exec_command`'s `timeout`-alias note is gone (the alias layer absorbs
  it silently). No parameter names, types, or presence change, and no
  behavior changes ride along.

- Tool definitions sent to model providers no longer duplicate parameter
  documentation. Every tool docstring's `Args:` section is still parsed into
  per-parameter schema descriptions, but the wire description no longer also
  carries the whole section verbatim (and the schema descriptions no longer
  include the raw docstring continuation-line indentation). Tools whose
  parameters are not fully documented in their schema — freeform tools and
  tools with enum-only schema properties such as the browser tools' `button`,
  `action`, `level`, `part`, `image_type`, and `scale` — keep their `Args:`
  block, since it is the only place those parameters are documented.

- The per-type agent dispatch tools (`dispatch_general_agent`,
  `dispatch_investigation_agent`, `dispatch_browser_agent`,
  `dispatch_coding_agent`, and `dispatch_custom_agent`) are consolidated into a
  single `dispatch_agent(agent_type, task, model_override?, browser_target?)`
  tool. `agent_type` is an enum computed per session from the same conditions
  that previously gated the individual tools: `general` and `investigation`
  whenever dispatch is available, `browser` only when the browser-agent model
  supports vision, `coding` where the coding dispatch was previously offered
  (never for the coder itself), and custom agent names when definitions are
  discovered. The description documents the shared dispatch mechanics once plus
  one line per available agent type, and an unavailable `agent_type` returns an
  error listing the valid values. Old sessions that used the removed tool names
  still restore and display normally.

- All terminal-capable agents (coder, planning, investigation, general) now
  share a terminal-tools prompt section with codex-style guidance: prefer
  `rg` / `rg --files` over `grep`/`find` when searching for text or files (it is
  much faster; use alternatives if `rg` is not found), and do not use Python
  scripts (or `eval`) to print chunks of a file — use the file-reading tools
  instead.
- The `resolve` tool — which applies or discards a pending `lsp_edit(apply: false)`
  preview — is now gated like the other LSP tools. With LSP disabled (via
  settings, `KOLEGA_CODE_LSP`, or `--lsp off`) it is removed from the model's
  toolset entirely, since pending actions can only be created while LSP is on.
  Applying a pending action with `resolve` now also goes through the same
  edit-permission prompt as `edit`, `lsp_edit`, and `write`.
- The two file-read tools are merged into one `read(file_path, offset?, limit?)`
  tool. `offset` is the 1-indexed first line (default 1) and `limit` bounds the
  number of lines; omitted parameters read from the top. Output is truncated to
  2000 lines or 50KB — whichever binds first — snapped to line boundaries, with
  actionable notices such as "[Showing lines a-b of N (50KB limit). Use
  offset=b+1 to continue.]" instead of the old line-only warning and
  100,000-character backstop. A single line larger than the 50KB budget gets an
  explicit message suggesting `rg` / `exec_command` for targeted extraction.
  `path` is accepted at the dispatch boundary as an alias for the canonical
  `file_path`.

- `write(path=…)` now also succeeds under the claude_code edit protocol, where
  `write` binds to `claude_write(file_path, content)`: `path` is accepted at
  the dispatch boundary as an alias for `file_path`, matching every other file
  tool. Under the search/replace protocol `path` stays the canonical
  parameter and passes through untouched — alias resolution is signature-aware
  and never rewrites an argument name the bound tool itself accepts.

- A successful whole-file `write` (including `hashline_write` and `apply_patch`'s
  Add File) now records the just-written contents as read, so the read-before-edit
  guard accepts a subsequent edit without a fresh read — exactly as it would
  right after a read. Staleness tracking is unchanged: a file modified after the
  write still requires a fresh read before editing.
- The three skill host tools (`list_skills`, `activate_skill`, and
  `read_skill_resource`) are consolidated into a single `skill` tool with the
  same contract as the old `activate_skill`: it takes the skill name (without a
  leading slash) and returns the activation envelope with the skill's
  instructions, its absolute directory, and the resource file list. Activating
  an already-active skill short-circuits as before, and an unknown name now
  lists close matches (prefix first, then fuzzy) from the catalog so a
  mistyped activation self-corrects in one turn. The in-prompt skill roster
  always lists every skill by name: description text is shed first when the
  metadata budget binds, and skills are never omitted wholesale.

### Removed

- The `read_entire_file` and `read_file_section` tools are gone, replaced by the
  merged `read` tool above. Deleted: the `ReadFileTool.read_entire_file` /
  `read_file_section` methods and their `read_only_tools` registrations, the
  truncation notices that named `read_file_section`, and the stale
  `dispatch_investigation_agent` docstring tool list. Existing session
  histories keep the old tool names as recorded data and render as-is (restore
  is name-agnostic).

- The file-discovery tools are gone from the agent toolset: `glob` (which had
  replaced `list_directory` and `find_files_by_pattern`), and `search_codebase`.
  File and text discovery is now done through the terminal with `rg` /
  `rg --files` (falling back to `find`/`grep`). Deleted: the `GlobTool` and
  `SearchCodebaseTool` backends (the underlying `workspace_scan` service stays),
  the `ToolCollection.glob` / `ToolCollection.search_codebase` methods and their
  `read_only_tools` registrations, the stale `dispatch_investigation_agent`
  docstring tool list (now naming the real surface, including terminal access),
  and the `search_codebase` name from hashline history adaptation.
  `@`-mention autocomplete (`cli/file_index.py`) keeps its exclusions by
  inlining the constants it borrowed. Permission rules and custom-agent
  allowlists naming the removed tools must be updated to the terminal tools
  (`exec_command`, `write_stdin`, `kill_command`, `list_sessions`). Existing
  session histories keep the old tool names as recorded data and render as-is.

- The `sleep` tool is gone from the agent toolset: its blind wall-clock wait is
  strictly dominated by `write_stdin`'s empty-input poll, which waits for more
  output or process exit and returns early when the process finishes. The
  `yield_time_ms` clamp semantics (250–30000 ms when writing, 5000–300000 ms
  when polling) moved from the `write_stdin` docstring body into its parameter
  description so the poll path stays discoverable. Existing session histories
  keep the old tool name as recorded data and render as-is.

- The `think_hard` tool is gone from the agent toolset, along with the thinking
  model slot that powered it. Deleted: the `ThinkHardTool` backend and its
  `auxiliary/tools/think_hard.system.md` prompt template, the
  `ToolCollection.think_hard` method and its `read_only_tools` registration, the
  `THINK_HARD_PROMPT` constant, and the `thinking_config` model slot itself.
  `prompt` hooks now select only `"fast"` or `"long"`; the
  `--thinking-provider`/`--thinking-model` flags, the
  `KOLEGA_CODE_THINKING_PROVIDER`/`KOLEGA_CODE_THINKING_MODEL` env vars, and the
  Settings Thinking slot row were removed, and a saved `thinking` slot entry in
  `settings.json` is ignored on load. The active model's own thinking effort
  (`--thinking-effort`, `KOLEGA_CODE_THINKING_EFFORT`) is unaffected.

- The `get_host` tool is gone from the local toolset. It needs a sandbox host
  provider, so it is now offered only when a terminal manager carrying a
  `sandbox` attribute is injected (cloud sandbox deployments), and the
  localhost fallback branch in the handler is deleted — a registered
  `get_host` always has a sandbox. The gate duck-types on the attribute rather
  than `isinstance`, so kolega-code-e2b's injected `SandboxTerminalManager`
  (possibly subclassed from a pinned older version) keeps qualifying. Existing
  session histories that recorded `get_host` calls keep them as data and
  render as-is (restore is name-agnostic).

- The `list_skills`, `activate_skill`, and `read_skill_resource` host tools are
  gone, replaced by the single `skill` tool above. Deleted:
  `SkillCatalog.read_resource` and its truncation cap and traversal guard
  (skill resources are read with the ordinary `read` tool, whose `file_path`
  is relative to the absolute skill directory in the activation output), the
  model-facing list query surface (`format_model_catalog`), and the
  omitted-skills budget pathway in the prompt roster. Existing session
  histories keep the old tool names as recorded data and render as-is
  (restore is name-agnostic).

### Fixed

- Kimi-family usage is no longer dropped when a request is served entirely from
  the provider's prompt cache. `kimi-for-coding` / `moonshot` report
  `input_tokens: -1` as a sentinel for "no uncached input tokens" on a full
  cache hit; the strict usage validator previously classified that as malformed
  and nulled the whole record, silently removing both input and output tokens
  from session accounting for those turns. The sentinel is now interpreted as
  zero, and the inclusive input total is reconstructed from the cache fields
  (verified live: `-1` + 33 cached = 33 total, matching the same prompt's
  partial-cache report of 27 uncached + 6 cached).

## 0.26.10 - 2026-08-05

### Added

- Hosted (server-side) web search for Responses-API providers. On DeepSeek
  flash and OpenAI gpt-5.x (API key or ChatGPT subscription), kolega can
  request the provider's own
  `{"type": "web_search"}` tool: searches and page-opens execute on the
  provider's servers and the content is injected directly into the model's
  context (billed as input tokens; the context gauge accounts for it from
  usage, since the content never reaches the client). Calls render in the
  transcript as `web_search (hosted)`. Controlled by a new web tool mode —
  `auto` (default: hosted when the model supports it, else the client tools) /
  `hosted` / `client` / `off` — via Settings, `KOLEGA_CODE_WEB_SEARCH_MODE`,
  `ask --web-search`, or the TUI's `/web-search`. When hosted search is active
  the client-side `web_search`/`web_fetch` tools are not registered; `off`
  removes all web tools (for sandboxed/benchmark runs). Notes: provider
  per-search fees, if any, are not modeled in cost reporting (injected tokens
  are); sessions that used hosted search store a new `web_search_call` history
  block that older kolega builds cannot load.
- LSP tools can now be forced on or off for a single launch: `--lsp on|off`
  on both the TUI and `kolega-code ask`, plus a `KOLEGA_CODE_LSP` env var,
  resolved flag > env > settings > default-enabled. `--lsp on` overrides a
  persisted `lsp_enabled=false` without writing back to settings. Turning LSP
  off now removes the `lsp`/`lsp_edit` tools from the model entirely —
  previously they stayed registered and could only answer "LSP is not
  available" — and the Settings LSP status notes when the launch flag is
  forcing the mode.
- The per-session scratchpad directory is now exported as `KOLEGA_SCRATCHPAD`
  to every shell command and eval kernel, so scripts and kernels can reach the
  session's throwaway workspace by environment variable instead of hardcoding
  a temp path.

### Changed

- Thinking now renders as a collapsed row in the TUI transcript — `● Thinking
  · … · 1.2k words` while streaming (a live word counter, never the text),
  flipping to `● Thought · 12s · 1.2k words` on completion; expand the row to
  read the full reasoning. Long reasoning turns (DeepSeek flash especially) no
  longer dominate the transcript with full-height thinking walls.
- Resumed sessions now restore reasoning as the same collapsed thinking rows,
  in true interleaved order with tool and hosted web-search rows: Anthropic
  thinking text, DeepSeek flash's raw chain-of-thought, and OpenAI/ChatGPT
  reasoning summaries all render (encrypted-only reasoning items have no
  renderable text and are skipped). Previously reasoning vanished entirely
  from restored transcripts.
- The coder prompt now follows a Codex-style final-message contract: the
  closing report leads with what was delivered and its verified state, stays
  brief by default with structure scaled to complexity, references files by
  path instead of re-pasting contents, and never restates the plan step by
  step. Preambles before tool calls are tightened to a single short line
  naming the purpose of the next action, and verification effort is bounded —
  tool results are trusted (no re-reading a file to confirm an edit applied)
  and verification stops once no remaining check could realistically fail.

### Fixed

- Hosted web-search calls now render in their true position in the TUI
  transcript. Previously the rows mounted at the bottom and stayed pinned
  there while post-search reasoning kept streaming into the thinking bubble
  above them; the stream now closes the open thinking/response segment at
  each hosted call, so reasoning after a search opens a new bubble below its
  rows (think → search → think → answer). Resumed sessions now render hosted
  `web_search (hosted)` rows from history too — previously they were dropped
  from the restored transcript.
- `ask --json` no longer emits spurious synthetic message fragments when
  hosted web search flushes response text mid-stream: the lost-text fallback
  now applies once at end of turn, and only when the turn produced no
  assistant message.
- The context gauge no longer under-reads on `deepseek-v4-flash`. DeepSeek's
  Responses API silently re-attaches every prior round's reasoning server-side and
  bills it as input, but the client history held none of it, so the gauge (and the
  auto-compaction trigger) could run over 100k tokens behind the real billed
  context on reasoning-heavy sessions. Flash's plain-text reasoning is now retained
  in history and resent each turn — the same shape Codex replays, deduped by the
  server so billing is unchanged — which keeps the gauge on the real number and
  preserves chain-of-thought continuity across session restores. The retained
  reasoning text also now appears in `ask --json` message payloads (parity with
  Anthropic thinking text); encrypted-only reasoning from OpenAI/ChatGPT stays
  stripped as before.

### Security

- Dependency bumps for advisories published 2026-08-03/04: `cryptography`
  48.0.1 → 50.0.0 (PKCS#7 Bleichenbacher oracle CVE-2026-69247, a name-constraint
  wildcard bypass CVE-2026-69248, and a path-building DoS CVE-2026-69249) and
  `aiohttp` 3.14.1 → 3.14.3 (three parser advisories, including an out-of-bounds
  heap read in the C response parser). Exception: Intel Macs stay on
  `cryptography` 48.0.1 — every fixed release dropped Intel-mac wheels, and
  upgrading would break the curl installer there with a Rust+OpenSSL source
  build — so those three advisories are knowingly accepted on that legacy
  platform only.

## 0.26.9 - 2026-08-03

### Added

- Settings → Model & Account gains a **Model Slots** section for pinning the fast
  and thinking models, each with its own provider. Previously these slots were
  reachable only through CLI flags and environment variables, so in the TUI they
  silently inherited the active model — every `web_fetch` answering call ran on
  the main coding model. Slots still inherit by default, and each row now shows
  which model it currently resolves to.
- The README now includes a direct invitation to the Kolega Code Discord
  community.

### Changed

- Settings gains a **Providers** category, first in the rail, holding every model
  provider's API key and the ChatGPT sign-in, each row labelled with its live
  credential status. **Model & Account** is renamed **Models** and keeps only what
  runs. Keys now stage per provider, so credentials for a provider you are not
  currently using — needed by a fast/thinking slot or an agent-role override — can be
  entered without switching your main model and switching back. **Test Connection**
  moves with them and probes the selected provider rather than the active model, so a
  credential can be checked without running on that provider.
- **Breaking:** provider and model are now always specified together. `--provider`
  without `--model` (and the per-role/per-slot equivalents such as
  `KOLEGA_CODE_FAST_PROVIDER`) is an error instead of silently selecting that
  provider's default model, and a model named without a provider is rejected rather
  than assumed to be Anthropic's. Scripts passing only one of the pair need updating;
  the error names the missing flag. Saved settings are unaffected — the Settings UI
  always writes both — and a saved model that later leaves the catalog still degrades
  to the provider's default rather than blocking startup.

### Fixed

- `deepseek-v4-flash` no longer has its reasoning truncated mid-derivation. The
  64000-token output cap added in 0.26.8 was measured on the chat API for
  `deepseek-v4-pro` and applied to flash by assumption; flash actually runs to
  at least 112990 output tokens in a single Responses call. Hard tasks that
  reason for a long time before acting were cut off at the cap and produced no
  tool call at all. flash now uses its full published output budget, while pro
  and the fireworks/openrouter DeepSeek routes keep the 64000 cap where their
  measured ceilings apply.

### Removed

- The unreachable `DEFAULT_LONG_MODEL`, `DEFAULT_FAST_MODEL`, `DEFAULT_THINKING_MODEL`
  and their provider constants. No model was ever resolved through them — an unset
  slot inherits the active model — but they were documented as defaults, which is
  where "what is my fast model?" became unanswerable.

## 0.26.8 - 2026-08-03

### Added

- Completed model messages now carry normalized token usage across supported
  providers. Session usage is persisted across resume, exported with provider
  and model breakdowns, and shown in a dedicated Usage card on the Status tab.
- `kolega-code ask --json` now emits complete, replay-safe messages instead of
  streaming prose chunks, including normalized usage and an explicit final
  summary.
- Detached background terminal sessions now keep stdin open, so callers can
  send multiple inputs to long-running processes with `write_stdin`.

### Changed

- Build and ask agents now share stronger act-first and independent-verification
  guidance. Silent reasoning-only turns are retried with escalating prompts,
  and visible responses cut off at the output limit are asked to continue.
- Tool dispatch accepts common parameter aliases for `exec_command`, `eval`,
  and `find_files_by_pattern`, while canonical arguments still take precedence.
- Turn status now appears immediately while checkpoints are prepared, and
  session-diff baseline/checkpoint work is asynchronous and caches unchanged
  files for faster turns.
- Switching between Plan and Build with existing history now gives the model a
  hidden toolset-change notice so it does not attempt tools from the old mode.
- Web connection failures now distinguish unreachable hosts, TLS failures,
  rate limits, and retryable errors, with guidance that reflects whether
  outbound access is available.

### Fixed

- Installing on an Intel Mac no longer fails with a `cryptography` source-build
  error (`Failed to build cryptography==50.0.0` / missing OpenSSL). cryptography
  49+ stopped publishing Intel-mac wheels, so the installer tried to compile it
  with Rust and OpenSSL; Intel Macs now install the last release that ships a
  prebuilt universal2 wheel. Other platforms are unaffected.
- DeepSeek requests now always carry an explicit output-token cap (64000, on
  every provider that serves DeepSeek models: first-party, Fireworks, Ollama
  Cloud, OpenRouter). DeepSeek's real per-response ceiling is ~64k — far below
  its published 384000 — and a server-enforced cutoff is reported as a clean
  finish, so long reasoning runs were being truncated silently mid-word. An
  explicit client cap is enforced and reported honestly, so truncation now
  surfaces as a real `max_tokens` stop the agent loop can react to.
- The Responses API stream wrapper now captures the terminal
  `response.incomplete` event, so a response truncated at `max_output_tokens`
  reports stop reason `max_tokens` (and keeps its usage/billing metadata)
  instead of looking like a clean `end_turn` with no usage.
- A reply cut off at the output-token limit mid-message is no longer delivered
  as-is: the agent asks the model to continue (up to 3 escalating continuation
  prompts, mirroring the silent-turn guard), and only finalizes with the
  partial output if the reply is still truncated after that.
- The non-streaming chat `generate()` path now maps `finish_reason` (it lives
  on the choice, not the message), so its responses carry a real stop reason
  instead of `None`.
- Quitting while a question or approval is pending no longer raises a Textual
  teardown traceback.
- JavaScript eval kernels no longer wait for orphaned cell timers during
  shutdown.

### Removed

- The `KOLEGA_CODE_OPENROUTER_MAX_TOKENS` environment variable. OpenRouter
  requests still send no `max_tokens` by default (it acts as a routing filter
  on the gateway), except DeepSeek models, which now always send the honest
  clamped cap.

## 0.26.7 - 2026-08-03

### Fixed

- Local token counting no longer crashes on content that contains a tokenizer
  special-token literal (for example `<|endoftext|>` or `<|im_start|>`, common in
  ML/tokenizer files and chat-format prompts); those strings are now counted as
  ordinary text instead of aborting the run.
- The Private Project Memory screen no longer crashes with an internal
  "NoMatches" error when it is closed while still loading.

## 0.26.6 - 2026-08-03

### Changed

- `kolega-code ask` now runs with a system prompt built for autonomous,
  non-interactive use: it carries a task through to a verified result on its own
  instead of pausing to check in.
- A background process started with `exec_command background=true` (for example a
  dev server) now keeps running after kolega-code exits, instead of being stopped
  when the session ends. Stop it explicitly with `kill_command`.

## 0.26.5 - 2026-08-02

### Fixed

- DeepSeek `deepseek-v4-flash` on the Responses API no longer crashes mid-task
  with a 400 "No tool output found for tool call ..." when the model returns tool
  calls together with assistant text. The text was serialized between the tool
  calls and their outputs, which DeepSeek rejects; it is now emitted before the
  calls so each call's output stays adjacent. Tool calls are also defensively
  guaranteed a matching output on replay.

## 0.26.4 - 2026-08-02

### Changed

- `deepseek-v4-flash` now connects over DeepSeek's **Responses API**
  (`https://api.deepseek.com`) for its native tool-calling format and graded
  reasoning effort. Every other DeepSeek model (including `deepseek-v4-pro`)
  continues to use the OpenAI-compatible Chat Completions endpoint. This is a
  per-model routing decision; no configuration change is required.

## 0.26.3 - 2026-08-02

### Added

- `kolega-code ask --no-memory-tools` disables the persistent project-memory
  tools (read/list/write/edit/delete memory) and their prompt guidance for a
  run — useful for headless or benchmark use where there is no durable memory
  store.
- `search_codebase` accepts `path` to restrict the search to a subdirectory and
  `max_results` to cap how many files are returned.

### Changed

- `eval` now defaults `language` to `py`, so a call that omits the language runs
  Python instead of failing.
- A tool call made with the wrong arguments now returns a clear message naming
  the expected signature (for example `Usage: eval(code, language='py', …)`) and
  no longer exposes internal class or method names.
- "Memory off" now has a single, consistent representation: a non-local host,
  the new `--no-memory-tools` flag, and the memory toggle all behave the same
  way, hiding both the memory tools and the memory prompt section.
- The browser agent's dispatch tool is now hidden when the model configured for
  the browser agent cannot accept images, instead of failing only when invoked.

## 0.26.2 - 2026-07-31

### Added

- `kolega-code ask --gigacode` enables gigacode workflow orchestration in
  headless runs, matching the TUI's `/gigacode` toggle: the `run_workflow` and
  `list_workflow_runs` tools are exposed and the authoring guide is injected.
  The setting persists with the session, so `ask --session <id>` resumes with
  gigacode still on.

### Changed

- Gigacode's concurrent-worker cap is now a flat 8 instead of being derived from
  the host's core count. Workers wait on provider responses rather than CPU, so
  the old formula only made fan-out width vary with the machine — and collapse to
  a single serial worker on a one-CPU host or container.

## 0.26.1 - 2026-07-31

### Fixed

- Gigacode resume now matches cached `agent()` results by call content instead
  of call position, so completed work no longer re-runs after `pipeline()`
  timing drift or script edits that shift call positions (one production resume
  had salvaged only 6 of 91 completed calls). Resumed runs also journal what
  they replayed, making a resume of a resume exact, and a duplicate `agent()`
  label now logs a one-time authoring hint. Existing journals work unchanged.
- Workflow scripts now have a designated drafting location. The authoring
  guide and `script_path` docs direct models to draft long scripts in the
  session scratchpad instead of the project working tree — generated scripts
  in the repo pollute `git status` and go stale for later sessions. The
  executed script is persisted under the run directory either way.
- Workflow token budgets are harder to undersize by accident. The
  `token_budget` schema, tool docstrings, and authoring guide now carry
  evidence-based sizing anchors (measured across real runs: review calls
  median ~6k/p90 ~24k output tokens, coder calls median ~20k/p90 ~80k,
  reasoning included — size as review_calls × 15k + coder_calls × 50k,
  doubled, or omit), the run warns in the progress log at 75% and 90% spend,
  and a run
  that dies on budget exhaustion tells the model in its tool result that it is
  resumable — with the exact `resume_from_run_id` call and the count of
  journaled calls that will replay free.
- Workflow fan-out failures are no longer silent. A `pipeline()` stage or
  `parallel()` thunk that raises still drops its item to `None`, but the drop
  is now reported live in the progress log and recorded in the transcript with
  the exception and the offending script line; the `run_workflow` result warns
  with an aggregate (e.g. `AttributeError at script line 299 (x6)`) and
  `run.json` records `script_exception_drops`. A fan-out that comes back all
  `None` with drops is called out as a likely script bug. Budget and agent-cap
  exhaustion now propagate out of fan-outs and fail the run (cancelling and
  draining the remaining chains) instead of shredding it into `None`s —
  previously a run could report `completed` after a script bug silently
  discarded every result.
- An interrupted gigacode workflow's run id is now recoverable. Runs are
  stamped with their owning session, cancelling a turn marks the run
  `interrupted` in `run.json`, and the new session-scoped `list_workflow_runs`
  tool lists this session's runs (status, tokens, journaled calls, artifact
  paths) so the agent can resume an interrupted run with `resume_from_run_id`
  instead of re-running it from scratch — previously the run id was only
  returned on normal completion, so an interrupted run's journal was orphaned.

## 0.26.0 - 2026-07-31

### Added

- **New bundled research and writing skills.** `deep-research` orchestrates
  evidence-backed research and materializes cited reports, while `humanizer`
  helps revise generated prose into more natural writing.
- **OpenRouter provider.** `openrouter` (`OPENROUTER_API_KEY`) reaches the
  OpenRouter gateway, shipping a generated catalog of every tool-capable
  OpenRouter model with per-model context, output, vision, reasoning-effort and
  edit-protocol metadata. Because the catalog runs to hundreds of models, the
  Settings picker, `/model` and the sub-agent model catalog list OpenRouter's 20
  most-used models — in the order of OpenRouter's own LLM Leaderboard — while
  every other catalogued model stays selectable by exact ID via `--model`,
  `/model`, `KOLEGA_CODE_MODEL` or `settings.json`. Reasoning effort maps to
  OpenRouter's nested `reasoning` control, prompt caching is enabled
  automatically, and `max_tokens` is deliberately not sent so an output cap
  cannot bias upstream routing (override with
  `KOLEGA_CODE_OPENROUTER_MAX_TOKENS`). Models default to the Claude Code-style
  edit tool, except OpenAI models which use Codex `apply_patch`.
- **`kolega-code models` command.** `models list` inspects the model catalog
  (`--provider`, `--featured`, `--sort`, `--json`), and `models refresh` caches
  OpenRouter's current model list locally so models published after a release
  can be used without upgrading. Startup merges the cache without any network
  access; `KOLEGA_CODE_OPENROUTER_CATALOG` and
  `KOLEGA_CODE_DISABLE_OPENROUTER_CATALOG` control it.
- **Settings "Other…" model picker.** The main Model dropdown and every Agent
  Models role row lead with OpenRouter's most-used models and then offer an
  "Other…" entry backed by a free-text field, so any catalogued model id
  (bundled or added by `models refresh`) can be selected from Settings instead
  of only by exact ID on the command line. Typed ids are validated against the
  catalog before saving, and the Browser role keeps its vision-capability gate.
- **Worktree-aware startup and workspace switching.** Interactive and headless
  launches can select a registered checkout with `--worktree`, or create one
  safely with `--create-worktree` plus optional `--from` and
  `--worktree-path`. In the TUI, the top-level build-mode agent can switch its
  complete active workspace between registered sibling worktrees, but only when
  the user has explicitly asked it to: the agent never creates a worktree or
  moves the session on its own initiative, plan mode has no workspace-switching
  tool at all, and every agent-initiated switch is confirmed by the user first,
  with a declined or unanswered prompt leaving the workspace unchanged. Once
  approved, the switch commits immediately and applies at the end of the agent's
  turn by rebuilding the workspace — terminal, filesystem, LSP, snapshots,
  skills, autocomplete, and Changes/Rewind scope move together — after which the
  agent is prompted to continue in the new checkout. Session identity and
  sibling-checkout isolation remain unchanged, and active workspace selection
  persists across resume.
- The public `ToolExtension` API now accepts additive `exclusive_tools`
  metadata for tools that must run alone in a model tool-call batch.
- `browser_scroll` on both browser backends, taking exactly one of a target, a
  count of viewports, or an absolute offset. On a page too large to snapshot in
  one call, scrolling is how the rest is reached, and until now nothing could
  move the viewport on the Chrome backend at all.

### Changed

- The TUI `set_goal` tool is now build-mode only. Plan mode's tool inventory is
  meant to be read-only plus investigation, and a planning turn should not be
  able to start an autonomous loop on its own; `/goal <condition>` still works in
  either mode, and a goal set in build mode keeps driving turns after switching
  to plan mode.
- Project memory, repository guidance (`AGENTS.md`/`KOLEGA.md`), and the current
  date are no longer rendered into the system prompt. They now reach the agent as
  `<system-reminder>` context updates, sent once each time they change. Because
  providers cache on a prefix match and the system prompt is processed ahead of
  every message, a single `edit_memory` call previously invalidated the cached
  prefix for the whole conversation and re-billed it on the next turn — on the
  order of a dollar per write in a long session, and worse on providers whose
  prefix caching is automatic and offers no way to compensate. The system prompt
  is now byte-stable for the life of a session.
- Prompt-cache breakpoints are placed more deliberately: the tool list and the
  system prompt each get one with a 1-hour TTL, and the conversation carries two
  rolling breakpoints a turn apart rather than one. A single trailing breakpoint
  could fail to match after a turn with many parallel tool calls, which discarded
  the whole conversation's cache; the second one bounds that loss to one turn.

- The Chrome browser backend now works on large real-world pages. Selector
  resolution, snapshots, text search and focus handling previously failed
  outright on any page over about a thousand DOM nodes, so an ordinary
  application page returned an empty snapshot and refused every selector. Work
  is now bounded by a wall-clock budget with the browser's own APIs doing the
  filtering, and a bound that is reached degrades with a report instead of
  failing.
- Snapshots emit the nodes nearest the viewport first and attach a `Coverage:`
  line naming what was omitted, why, and where the viewport sits in the page, so
  a partial snapshot reads as "narrow the scope" rather than "the page is
  unreadable".
- `browser_find` distinguishes three outcomes: absent from the page, present but
  outside the region the snapshot covered, and undetermined because the search
  was truncated. Only the first is a reliable absence.
- Text waits that could not read the whole page now fail immediately with a
  `search_truncated` explanation instead of polling to a timeout and implying the
  text was not there.
- `browser_press_key` scrolls for `PageDown`, `PageUp`, `Home`, `End`, arrow keys
  and `Space`, matching the Playwright backend, unless the page handles the key
  itself or focus is in a text field or select.
- Chrome snapshot `depth` now counts emitted nodes rather than raw DOM nesting,
  so it means roughly what it means on the Playwright backend.
- Tall full-page screenshots on Chrome are clipped from the current scroll
  position and report the omission, instead of being downscaled until unreadable.
  Native Messaging envelopes are now bounded per direction, following Chrome's
  own asymmetric limits.
- Errors from the Chrome extension keep their error code, and codes that mean
  "too large to cover in one call" carry the concrete next step.
- The responsiveness watchdog now captures stacks for event-loop stalls from ~1
  second (was 5) and counts shorter gaps into a periodic `loop_gap_histogram`
  diagnostics entry, so choppy-but-not-frozen UI leaves evidence instead of
  nothing. Stack dumps are capped per session so a pathological session cannot
  fill the timeline.
- `read_image` now refuses files above 20 MB (the limit image encoding already
  applied elsewhere) instead of reading them and having the provider reject the
  request.

### Fixed

- Standalone generated `<system-reminder>` context no longer appears in the TUI
  transcript after resuming a session or rebuilding the agent for model and
  settings changes; the context remains unchanged in model and session history.
- Continuing after cancelling an in-flight tool call no longer duplicates the
  next prompt or makes Anthropic reject the request for exceeding its maximum
  of four `cache_control` blocks.
- Plan mode now carries requested Build-mode session handoffs into approved
  plans: worktree plans switch the complete session before setup or edits in
  the linked checkout, and authorized autonomous-goal plans place `set_goal` at
  the intended transition after any required prerequisites.
- Stopped image-heavy sessions from freezing the TUI. Building request history
  re-encoded every screenshot in history on every LLM request, on the event
  loop — dozens of multi-second stalls per session, multiplied by each running
  sub-agent. Resized copies are now memoized per image and dimension cap
  (surviving the small byte-budget shifts a newly captured screenshot causes),
  and the repair/adapt pass runs in a worker thread.
- Moved LLM request preparation off the event loop. Building a request body is
  proportional to conversation size, and on the Anthropic streaming path it
  blocked the UI for 0.4 s per request with an image-heavy history — multiplied
  by every concurrently running sub-agent. The worst event-loop gap per request
  drops from 111 ms to 19 ms with 6 MiB of images in history, and from 425 ms to
  69 ms at 24 MiB.
- Stopped `web_fetch` from freezing the UI on large pages. Scoring an extraction
  re-parsed the entire raw document on the event loop, once per extractor
  attempt plus once more for single-page-app detection; the document is now
  measured once per fetch in a worker thread. On an 11.6 MB page the worst
  event-loop gap drops from 774 ms to 219 ms, and on a pathological 13.8 MB page
  from 48 s to 0.5 s.
- Moved `read_image`'s file read and base64 encode off the event loop.

## 0.25.2 - 2026-07-28

### Fixed

- Downscaled outbound image copies to satisfy Anthropic's per-image dimensions
  and aggregate request-size limit, preventing image-heavy conversations from
  failing later turns while preserving full-resolution history and artifacts.

## 0.25.1 - 2026-07-28

### Fixed

- Resized oversized images only in outbound Anthropic request history while
  preserving full originals in session artifacts and replay, preventing one
  large image from breaking every later turn in a resumed conversation.

## 0.25.0 - 2026-07-28

### Added

- **Scheduled loops.** `/loop 5m check whether CI went green` re-runs a prompt on
  a fixed interval or a 5-field cron schedule (`/loop --cron "0 9 * * 1-5" ...`)
  inside the current session. Iterations run only between turns while the session
  is idle, messages you type always take priority, and missed windows never pile
  up — one iteration runs when the agent goes idle, not one per skipped window.
  Loops are capped (100 iterations, 7 days by default), shown live in the status
  dashboard, saved with the session and restored on resume, and stopped by `Esc`,
  `/loop stop`, `/clear`, or a rewind. `--fresh` starts each iteration from a
  clean thread for long-running watchdogs, and `.kolega/loop.md` can commit a
  project's standard loop prompt and schedule. Also available headless via
  `kolega-code ask --loop`/`--loop-cron`.
- **Shareable session replay.** `kolega-code share export <session-id>` writes a
  self-contained bundle that plays a session back in any browser with pause,
  seek, turn-by-turn jumping, and 1x/2x/4x/max playback. Long idle gaps are
  compressed so a session with a break in it stays watchable. Secrets are
  scrubbed, local filesystem paths are stripped, and only tool output and images
  are included — opaque provider payloads such as reasoning signatures are never
  shared. The command prints exactly what it removed.
- **Built-in session server.** `kolega-code serve` exposes recorded sessions over
  HTTP and WebSocket, including a replay player, a generated OpenAPI reference at
  `/docs`, and a stream endpoint that replays a session's backlog and then follows
  it live, so a second client can attach to a running session and lose nothing on
  reconnect. Bound to loopback by default, with `--token` and `--bind` for use
  behind a private network or tunnel. Read-only.
- **Durable session event stream.** Sessions now record an ordered event stream
  alongside their conversation history, sharing one sequence space so UI activity
  and provider messages are correlatable. High-volume output is coalesced, large
  payloads are offloaded to content-addressed artifacts, and retention ceilings
  record an explicit truncation marker rather than losing data silently.
- Assistant responses, reasoning, and turn boundaries are now carried on the event
  stream. They previously reached only the terminal UI, which meant no other
  frontend could render a conversation.
- Added Anthropic Claude Opus 5 (`claude-opus-5`) with a 1M-token context
  window, 128K max output, vision input, and adaptive thinking with
  `low`/`medium`/`high`/`xhigh`/`max` effort levels.
- Added Fireworks Kimi K3 (`accounts/fireworks/models/kimi-k3`) with a 1M-token
  context window, 131K configured output, vision and tool-calling support, and
  max reasoning effort.

### Changed

- `AgentEvent` gained `session_id`, `workspace_id`, `thread_id`, `seq`,
  `elapsed_ms`, `artifacts`, and `schema_version`; `event_type` is now an open
  string rather than a closed literal, and `timestamp` is timezone-aware UTC.
  `fastapi`, `uvicorn`, `websockets`, and `pygments` are now direct dependencies.
  See `MIGRATION.md` for host applications embedding this package.

- Made Claude Opus 5 the default Anthropic long-context and thinking model.

## 0.24.1 - 2026-07-26

### Changed

- Updated provider SDK dependencies (`google-genai` and `openai`) for improved wire compatibility.

### Fixed

- Serialized Google Gemini tool execution results as user turns to comply with wire protocol requirements.

## 0.24.0 - 2026-07-24

### Added

- Added persistent Python and JavaScript eval kernels whose state survives
  between calls, with a loopback bridge for invoking agent tools from eval code.
- Added managed background shell sessions for development servers and other
  long-running processes.
- Added a private per-session scratchpad for temporary agent files.

### Changed

- Kept agent-created Git worktrees in the project's `.kolega/worktrees/`
  directory instead of using temporary system paths.

### Fixed

- Enabled mouse text selection and copying in the sidebar Terminal and Logs tabs.
- Announced eval environment provisioning only when provisioning actually starts.
- Corrected shell guidance for background processes so it no longer recommends
  unsupported ways to outlive the agent session.

### Security

- Updated and constrained eval bundle dependencies to resolve known
  vulnerabilities.

## 0.23.0 - 2026-07-23

### Added

- Added stable Google Gemini 3.6 Flash and Gemini 3.5 Flash-Lite model support,
  including model-specific thinking levels and omission of their deprecated
  sampling temperature parameter.
- Displayed each sub-agent's provider, model, and thinking effort in live status,
  completed transcript summaries, and the sub-agent inspector.

## 0.22.1 - 2026-07-22

### Fixed

- Eliminated post-edit LSP diagnostics delays by invalidating cached diagnostics
  immediately and avoiding redundant waits after successful edits.
- Hardened subagent dispatch guidance and schemas so optional model overrides are
  omitted by default and explicit overrides cannot contain blank route fields.
- Clarified that contingent future work remains ordinary task execution and does
  not authorize autonomous goal mode.

## 0.22.0 - 2026-07-22

### Added

- Bundled the `docx`, `pdf`, `pptx`, `review`, `skill-authoring`, and `xlsx`
  workflows from `kolega-skills` v0.2.1 so they are available out of the box.
- Added a top-level TUI `set_goal` tool so explicit user, Agent Skill, or host
  workflow instructions can enter the existing persistent autonomous goal loop
  without synthesizing a `/goal` command.
- Added atomic per-dispatch model overrides for built-in, custom, and Gigacode
  subagents, with credential-free model discovery and strict route validation.
- Allowed file-editing and LSP tools to operate on explicitly requested absolute
  and parent-relative paths outside the project while preserving permission and
  policy gates.

### Changed

- Exposed the active model's vision capability to every bundled agent prompt and
  prompt override.
- Made saved-session listings easier to resume with labeled IDs, chronological
  ordering, and a clear empty state.

### Fixed

- Bounded Gigacode delegation depth so direct workflow workers are leaves by
  default, with an explicit one-hop nested-agent opt-in. Built-in nested workers
  share the workflow lifetime-agent count and output-token budget; concurrency
  caps active execution chains and serializes nested dispatch within each direct
  worker. Completed spend blocks later launches while in-flight calls may produce
  bounded overshoot. Depth-2 recursion supports only built-in dispatch tools until
  opaque host `ToolExtension` callbacks have a workflow-aware accounting and depth
  protocol. Run totals and failed-agent artifacts now retain finalized nested or
  pre-failure output usage. Ordinary non-workflow extension and delegation behavior
  is unchanged, and no supported API, metadata schema, cache-key field, or artifact
  field was removed or renamed.
- Resolved relative shell-command working directories against the target project
  instead of the directory where Kolega Code was launched.
- Prevented model-facing shells from inheriting Kolega Code's own `uv run`
  virtualenv while preserving user-activated and project-local virtualenvs.

### Security

- Updated the documentation toolchain and `pyasn1` dependency to patched
  releases, clearing the known dependency alerts.

## 0.21.1 - 2026-07-20

### Changed

- Kept the terminal UI responsive in large sessions by bounding mounted
  transcript and subagent rows while preserving scrollback and navigation.

### Fixed

- Preserved the active model and reasoning effort when opening settings instead
  of allowing stale selector events to reset them.

## 0.21.0 - 2026-07-17

### Added

- Added `/rewind [turns-back]` to restore workspace files and conversation to
  their state before an earlier turn, with a pre-rewind snapshot captured for
  recovery.

### Fixed

- Replaced the leftover lime text-selection color in the docs with a
  theme-aware cyan.
- Hardened diagnostics file writes to avoid a CodeQL clear-text-storage false
  positive, creating log and sidecar files owner-only from the outset.

## 0.20.0 - 2026-07-17

### Added

- Added Kimi K3 as the default Moonshot model, with native vision, a 1M-token
  context window, fixed sampling parameters, and `max` reasoning effort.
- Added Kimi Coding Plan's tier-specific `k3` and `k3[1m]` model IDs.

## 0.19.0 - 2026-07-16

### Added

- Added private, project-scoped memory outside repositories, with bounded
  Markdown storage, model tools, startup context, linked-worktree sharing,
  `/memory` commands, settings controls, and a TUI browser and editor.

### Changed

- Pinned the MCP SDK to stable `1.28.1` and stopped globally enabling
  prerelease dependency resolution.
- Added the stable-lock `filelock` dependency required by project memory.

### Removed

- Stopped loading or modifying legacy repository `AGENT_MEMORY.md` files.
  Existing files are not automatically migrated to private project memory.

### Fixed

- Preserved separate OpenAI and ChatGPT Responses reasoning-summary parts so
  thinking summaries render as clean, separated lines.
- Fixed MCP 1.x stream, timeout, and camelCase result compatibility, and now
  report nested transport failures as credential-safe tool errors instead of
  opaque `TaskGroup` messages.

### Security

- Sanitized terminal-rendered control sequences and unsafe hyperlink metadata
  while preserving raw model, tool, and session content internally.

## 0.18.0 - 2026-07-14

### Added

- Added custom Markdown subagents with project and user discovery, YAML
  frontmatter configuration, safe tool and mode restrictions, and CLI/TUI
  commands for listing and validating definitions.
- Added GPT-5.6 Sol, Terra, and Luna to the OpenAI API-key and ChatGPT-subscription
  model catalogs, with vision and `none` through `max` reasoning effort.
- Added model-specific editing surfaces, including Codex-style `apply_patch`,
  Claude-style editing, and a reproducible multi-language benchmark corpus.
- Replaced mutable session snapshots with an append-only event journal that
  preserves more context across crashes, repairs incomplete tails, and lazily
  migrates existing saved sessions.
- Added guided first-run onboarding and a full-screen settings editor with
  staged validation, connection testing, and safe apply/discard controls.
- Queued messages can now steer an active agent turn at the next tool boundary
  instead of always waiting to start a separate turn.

### Changed

- GPT-5.6 Sol is now the default for the OpenAI API-key and
  ChatGPT-subscription providers.
- OpenAI and ChatGPT models now prefer `apply_patch`, while direct DeepSeek
  models prefer Claude-style edits; explicit edit-protocol choices remain
  respected.
- Corrected the reasoning-effort choices exposed for GPT-5.4 Mini.
- Refreshed the TUI settings, onboarding, and transcript presentation with
  clearer summaries, compact controls, and inline role glyphs.

### Fixed

- Prevented empty completed responses from creating blank subagent trajectory
  steps rendered as orphan role glyphs.

## 0.17.0 - 2026-07-10

### Changed

- Rebuilt the browser agent on a Playwright MCP-style accessibility-snapshot
  toolset (`browser_navigate`, `browser_snapshot`, `browser_find`,
  `browser_click`, `browser_type`, and more). Actions use stable element refs
  (e.g. `e12`) and return an updated snapshot, so the agent interacts
  deterministically without inventing CSS selectors or relying on screenshots.
- The browser agent now requires a vision-capable model. Models without image
  input support are rejected with guidance to choose a vision-capable model or
  inherit the default.

### Fixed

- Prevented `glob`, `search_codebase`, and LSP language detection from hanging
  on large or broad project directories by adding bounded, cancellable,
  timeout-limited workspace traversal.
- Normalized TUI transcript indentation for consistent rendering.

### Security

- Hardened browser URL validation to address CodeQL static-analysis findings.

## 0.16.1 - 2026-07-10

### Changed

- Rebuilt `web_fetch` as a fully local, content-type-aware pipeline with bounded
  HTTP retries, automatic quality-gated Trafilatura/Readability/DOM extraction,
  JSON/text/feed handling, and local PDF/Office conversion.
- `web_fetch` now returns grounded fast-model answers with verified page excerpts,
  long-content chunking, and a bounded extracted-content fallback when answering fails.

### Fixed

- `web_fetch` now reports HTTP, extraction, unsupported-content, scanned-document,
  and JavaScript-rendered SPA failures precisely instead of collapsing them into
  empty Trafilatura output, and UI progress failures no longer discard results.
- Removed the arbitrary 512-character answer clipping and first-100,000-character
  page truncation that could omit valid answers near the end of a resource.

## 0.16.0 - 2026-07-09

### Added

- Added support for xAI Grok 4.5 (`grok-4.5`): 500K-token context, vision input,
  and `low`/`medium`/`high` reasoning effort. Selectable in the Settings UI,
  `/model` picker, CLI flags, and env vars; now the default `xai` model.
- Send a stable `x-grok-conv-id` header on xAI Chat Completions requests so
  multi-turn agent loops pin to a cache-warm server (xAI prompt-caching guidance).
- Added session-scoped workspace snapshots for agent file mutations: `snapshot`
  (list/show/create/restore, including `snapshot_id='latest'` undo) and `resolve`
  for applying or discarding pending `lsp_edit` preview actions. Restore refuses
  on post-snapshot drift unless `force=true`.
- Improved TUI usability inside tmux/screen when Shift-modified keys never reach
  the app: `/attach` with no path pastes an image from the system clipboard,
  `Alt+V` is a portable image-paste binding alongside `Ctrl+Shift+V`, startup
  help highlights `Ctrl+J` / `/plan` / `/build` fallbacks, and a one-time
  startup hint appears under tmux. Documented optional tmux extended-keys config
  under Troubleshooting.

### Fixed

- Prevented a TUI crash when pasting long multi-line text into the composer
  (`OSError: File name too long` from treating paste content as a path).
- Stopped advertising host lifecycle `initialize` as a model-callable tool while
  keeping it available for CLI/TUI startup (LSP setup).
- Bumped `lxml-html-clean` to 0.4.5 (and transitive `lxml` to 6.1.1) to clear
  GHSA-4jhm-jv67-739f from the dependency audit.

## 0.15.0 - 2026-07-07

### Added

- Added Language Server Protocol (LSP) integration with an agent-callable
  read-only `lsp` tool (diagnostics, go-to-definition, references, hover,
  document/workspace symbols, and status) and a trusted `lsp_edit` tool for
  rename, formatting, and code actions. The `edit`, `multi_edit`, and `write`
  tools now append LSP diagnostics to their results, with a `/lsp` status
  command, detected-language status in the TUI wordmark, and a Settings toggle.
- Added bounded skill metadata rendering with a context-aware token budget.
  The prompt catalog now uses skill name and description only, descriptions are
  truncated before skills are omitted, and `list_skills` is queryable with
  `max_results`.
- Documented composable workflow shapes in the Gigacode orchestration guide.

## 0.14.0 - 2026-07-03

### Added

- Adopted Pyright static type checking (basic mode) across the codebase.
  Pyright runs in pre-commit and CI, catching type errors before runtime. The
  `[tool.pyright]` configuration lives in `pyproject.toml`. Run locally with
  `uv run pyright`.
- Added support for Anthropic Claude Fable 5 (`claude-fable-5`) and Claude Sonnet 5
  (`claude-sonnet-5`): 1M-token context, 128K max output, vision input, and
  adaptive thinking with `low`/`medium`/`high`/`xhigh`/`max` effort levels. Both
  are selectable in the Settings UI, `/model` picker, CLI flags, and env vars.
- Added a `/goal` slash command that sets an autonomous completion condition the
  agent works toward. After each turn, a read-only investigation sub-agent
  verifies whether the goal is met; if not, the agent is nudged to continue
  automatically until the goal is met, a turn cap is hit, or the user
  pauses/cancels. Also available as `kolega-code ask --goal <condition>` for
  run-to-completion from the CLI. Goal state persists with the session.

### Fixed

- Handled CRLF line endings in the edit, multi_edit, and write tools so diffs
  and patches apply correctly on Windows-originated files.
- Showed full diffs on the session changes screen instead of a capped preview.
- Kept the turn worker alive during goal verification so the agent does not
  stall while checking autonomous goal conditions.

## 0.13.0 - 2026-07-02

### Added

- Added first-class MCP (Model Context Protocol) support in the CLI and TUI,
  including server settings management.
- Persisted plan artifacts so they survive across CLI invocations.

### Changed

- Made session diff refreshes incremental and asynchronous for smoother TUI
  performance.

### Fixed

- Hid queued follow-up messages from the transcript view.
- Polished status card spacing and made the context-full note generic.
- Avoided tool dispatcher input name collisions and excluded internal tool
  collection methods from dispatch.
- Hardened MCP credential handling, status logging, config output, and server
  settings reliability.
- Preserved gigacode enabled state across session resume.
- Corrected the GPT-5.5 context window to 272K on the Codex backend.

## 0.12.0 - 2026-06-30

### Added

- Added local always-on diagnostics and watchdog support, including `/bug` output
  packaged as a single zip for issue reports.
- Made `search_codebase` regex-by-default and backed it with ripgrep for more
  capable code searches.

### Changed

- Consolidated file-editing tools into a single edit tool interface.
- Reduced per-session memory usage by deferring heavy LLM imports.
- Relicensed the project under the Apache License 2.0.

### Fixed

- Prevented token counting from freezing the UI during LLM activity.
- Improved sub-agent stream handling so long streams accumulate efficiently.
- Replayed reasoning through native provider fields for OpenAI-compatible
  providers.
- Cleaned up duplicate/conflicting footer shortcuts in the TUI.
- Hardened diagnostics crash-log handling and secret scrubbing.

## 0.11.1 - 2026-06-26

### Added

- Made images `@`-mentionable in the TUI and moved file-index walking off the
  event loop.

### Fixed

- Routed DeepSeek through the OpenAI-compatible `/v1` endpoint.
- Bounded LLM streaming timeouts and added retry handling for transport errors.
- Prevented DeepSeek stream freezes in the TUI.

## 0.11.0 - 2026-06-26

### Added

- Added queued follow-up messages in the TUI so users can submit additional
  prompts while an active turn is still running.

### Changed

- Moved task-list status into the sidebar and refined sidebar presentation.
- Updated documentation and positioning copy for gigacode and queued follow-up
  messages.

## 0.10.0 - 2026-06-25

### Added

- Added a session changes inspector to the TUI that shows git diffs for the
  current session, renders added files as diffs, presents an empty state, and
  hides captured edit events.
- Added project prompt overrides, including variable rendering, override status
  shown at startup, validation, and selective dump support.

### Fixed

- Preserved Ollama Cloud reasoning provider metadata.
- Ignored project `.dotenv` files when loading model config.
- Stabilized terminal rendering artifacts in the TUI.
- Restored the generated root CLI help.
- Surfaced prompt override render errors.

### Changed

- Split the LLM specs module into a package for maintainability.

## 0.9.0 - 2026-06-24

### Added

- Added Ollama Cloud as a supported provider, including its model catalog and reasoning-field capture.

### Fixed

- Clarified the disconnected state shown on first run when no provider is configured.

## 0.8.4 - 2026-06-24

### Added

- Added minimal GitHub issue templates for bug reports and feature requests.
- Added project links (homepage, documentation, changelog, security) to package metadata.
- Added pre-commit configuration and a repository coverage badge.
- Added Ruff lint/formatting checks and dependency vulnerability auditing to CI.
- Added coverage reporting and SBOM/provenance attestation to the release workflow.

### Changed

- Hardened local state file permissions for settings, sessions, and project permission files.
- Hardened the release workflow with lockfile-backed installs, version parity checks, and artifact smoke tests.
- Split and reformatted oversized test modules to keep the suite maintainable.
- Reduced the Logs and Terminal TUI scrollback caps to 2000 lines for better performance.
- Updated GitHub Actions to Node 24-compatible versions.

### Fixed

- Fixed the chat composer to auto-grow as text wraps across multiple lines.
- Fixed `ctrl-u` so it clears multiline composer drafts.
- Fixed composer select-all shortcuts.
- Fixed prompt-option focus handoff so focus returns from the composer correctly.
- Fixed chat focus restoration after the app resumes or a turn is cancelled.
- Fixed runtime output clearing when a planning thread is reset.
- Optimized planning sidebar markdown rendering to reduce lag.

## 0.8.3 - 2026-06-23

### Fixed

- Persisted TUI permission mode across sessions.
- Updated the `idna` dependency to a patched version.

### Changed

- Externalized the TUI stylesheet.
- Moved tests into the top-level `tests/` directory.

## 0.8.2 - 2026-06-22

### Added

- Improved gigacode workflow transcript artifacts.
- Documented branch and pull request naming guidance.
- Documented the optional logs sidebar flag.

### Fixed

- Fixed long TUI approval prompt layout.
- Fixed transcript jump-to-bottom locking.

### Changed

- Optimized sidebar terminal/log rendering and streaming render updates.
- Moved TUI session persistence off the event loop.
- Refactored the CLI TUI package layout and controller mixins.

## 0.8.1 - 2026-06-21

### Added

- Added an optional agent iteration cap.

### Changed

- Migrated the API-key OpenAI provider to the Responses API.
- Preserved OpenAI Responses reasoning continuity.
