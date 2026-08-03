# Changelog

All notable changes to Kolega Code are documented here.

This project uses GitHub Releases for detailed generated release notes. This file provides a concise, human-maintained summary of user-visible changes.

## Unreleased

### Added

- Settings → Model & Account gains a **Model Slots** section for pinning the fast
  and thinking models, each with its own provider. Previously these slots were
  reachable only through CLI flags and environment variables, so in the TUI they
  silently inherited the active model — every `web_fetch` answering call ran on
  the main coding model. Slots still inherit by default, and each row now shows
  which model it currently resolves to.

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
