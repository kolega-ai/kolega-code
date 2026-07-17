# Changelog

All notable changes to Kolega Code are documented here.

This project uses GitHub Releases for detailed generated release notes. This file provides a concise, human-maintained summary of user-visible changes.

## Unreleased

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
