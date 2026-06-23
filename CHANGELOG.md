# Changelog

All notable changes to Kolega Code are documented here.

This project uses GitHub Releases for detailed generated release notes. This file provides a concise, human-maintained summary of user-visible changes.

## Unreleased

### Added

- Added minimal GitHub issue templates for bug reports and feature requests.
- Added CI coverage reporting and dependency vulnerability auditing.

### Changed

- Hardened local state file permissions for settings, sessions, and project permission files.
- Hardened the release workflow with lockfile-backed installs, version parity checks, artifact smoke tests, provenance attestation, and SBOM generation.
- Added Ruff lint and formatting checks to CI and formatted the codebase.

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

