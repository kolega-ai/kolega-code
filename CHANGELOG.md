# Changelog

All notable changes to Kolega Code are documented here.

This project uses GitHub Releases for detailed generated release notes. This file provides a concise, human-maintained summary of user-visible changes.

## Unreleased

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
