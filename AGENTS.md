# Repository Guidelines

Multi-agent terminal coding agent: Python 3.11+ package `kolega_code/` (hatchling build, uv-managed) shipping the `kolega-code` CLI (`kolega_code.cli.main:main`) with a Textual TUI.

## Project Structure & Module Organization

- `kolega_code/__init__.py` is the **supported public API** for host apps (KolegaPlatform, kolega-comply) and sandbox providers (kolega-code-e2b); deeper paths are internal. Don't remove or rename top-level re-exports.
- `kolega_code/agent/` — agent core: `baseagent.py` (main loop), `tools.py` (dispatch), `tool_backend/` (one module per tool), specialized agents (`coder.py`, `planningagent.py`, etc.), `orchestration/` (Gigacode fan-out).
- `kolega_code/cli/` — `main.py` (CLI), `app.py` + `tui/` (Textual TUI), sessions, settings, skills. Also: `llm/` (providers), `mcp/`, `sandbox/`, `services/`, `auth/`, `memory/`, `hooks/`.
- `tests/` mirrors the package layout. `benchmarks/edit_tools/fixtures/snapshots/` is excluded from Ruff, pre-commit, and Pyright — don't hand-edit those fixtures.
- `docs/` is a separate Astro/Starlight site (`cd docs && npm ci && npm run build`).

## Build, Test, and Development Commands

- Setup once per clone: `./scripts/setup-dev.sh` (runs `uv sync --extra cli --extra dev` and points `core.hooksPath` at tracked `.githooks/`).
- Fast tests (default, excludes `slow`): `./run_tests.sh`. All tests incl. `slow`: `./run_tests.sh --all`.
- Focused run (args pass through to pytest): `./run_tests.sh tests/agent/llm/test_client.py -ra`, or append `::test_name` for one test.
- Coverage: `./run_tests.sh --coverage` (fails under 45%).
- Lint/format/typecheck: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`.
- CI parity check: `uv run pre-commit run --all-files --show-diff-on-failure` (same command as CI and the tracked git hook).
- Browser-dependent tests need Chromium: `uv run playwright install --with-deps chromium`.
- CI matrix: Python 3.11–3.14, coverage enforced on 3.11 only. Jobs that install Playwright are pinned to `ubuntu-24.04`: `ubuntu-latest` moved to 26.04 ahead of Playwright, which failed the release build and would fail any CI job that misses the browser cache (#763), so keep that pin instead of "modernizing" it.

## Coding Style & Naming Conventions

- Ruff `line-length = 120`, target `py311`; Pyright basic mode (config in `pyproject.toml`). Runtime deps are pinned exactly (`==`).
- Annotate all new function signatures and class attributes. Use `# pyright: ignore[...]` only for third-party stub limitations; narrow `Optional` with `assert`/`if` guards instead.
- `uv.lock` uses relative `exclude-newer = "1 week"`; `uv lock` advances the cutoff and may pull newer deps — expected; don't revert.
- **Comments are rare.** Write one only where the code cannot say it: a non-obvious constraint, a failure mode, a probed vendor behaviour. No comment restating a self-explanatory constant, no docstring that rewords the signature, no multi-paragraph rationale where one line will do. Probe logs and vendor quirks belong in `docs/` or a test name, not a source-file banner.

## Testing Guidelines

- pytest with `asyncio_mode = "strict"`; the wrapper adds `--import-mode=importlib` and deselects `slow` only. Markers: `slow`, `integration`, `live_api`, `e2e`.
- Unit tests use the `isolated_cli_env` fixture (`tests/conftest.py`) so they don't see a developer's env; slow/integration tests read provider keys from repo-root `.env` (copy `.env.example`).
- Mark every module that calls a real provider `pytestmark = [pytest.mark.integration, pytest.mark.live_api]`. Plain `integration` live tests still run under `./run_tests.sh`, so on a machine with `.env` keys the default run reports quota/401/404 failures no PR introduced (#755, #758, #780, #783); keep new live modules out of that noise and treat the existing ones as environmental.
- TUI tests: never wait with `pilot.pause()` or a sleep and then assert layout, geometry (`region`, scroll position) or a click target — poll with `wait_for_layout` from `tests/cli/_app_test_utils.py` instead of copying a local `_wait_for_layout`. `Pilot.click` returns `False` and clicks the screen origin when the widget has no region yet, so assert the click landed (`click_footer_key` in `tests/cli/test_shortcut_bar.py`). A red TUI test gets a root-cause fix, never a CI re-run, a longer sleep, or a looser assertion (#711, #752, #753, #755, #785).

## Provider and Model Catalogs

- Every catalog fact (window, effort ladder, sampling defaults, vision, tool support) comes from probing that exact endpoint. A sibling model's values and a "conservative family budget" are guesses: Ember-1 inherited K3's `max`-only ladder (#714), Workers AI needed two probe rounds for `none` and Gemma 4's reasoning replay (#770), and GPT-6.1 Terra's subscription window was corrected hours after merge (#780 → #783).
- An advertised window can be a compaction threshold rather than a hard reject (#783), so an understated window silently compacts long sessions. If a leg cannot be probed (no credentials, plan gating), say so and record it as unverified rather than asserting the number.
- Append new entries: a model's position is behaviour. The first option for a provider backs the Settings fallback (`model_options[0]` in `settings_panel.py`), so inserting or reordering changes a default nobody asked to change.

## Commit & Pull Request Guidelines

- Conventional Commits with a scope: `feat(cli): ...`, `fix(llm): ...`, `perf(cli): ...`, `test(cli): ...`, `chore: ...`.
- Branches: kebab-case with a type prefix (`fix/tui-large-session-lag`, `chore/release-v0.21.1`).
- PRs stay focused, with tests for behavior changes and docs for user-facing changes. CI (pre-commit, then the Python 3.11–3.14 test matrix) must pass.
- Releases: bump the version in all three places — `pyproject.toml`, `kolega_code/__init__.py`, `kolega_code/agent/__init__.py` (release workflow fails if the `v*` tag doesn't match) — and update `CHANGELOG.md` in the same release PR. Tag only after merge; see `RELEASING.md`.

## Security & Configuration Notes

- Never commit `.env`, credentials, tokens, or private endpoints; use obviously fake credential placeholders in tests and docs.
- CI runs `uv audit --locked` with an explicit `--ignore` list (cryptography on the Intel-mac fork, two transformers CVEs capped by tinker's pin, virtualenv 21.14.x). Each entry needs a dated reason, a revisit condition, and every id form the advisory is reported under (PYSEC/GHSA/CVE), because the primary id varies; only socksio's "archived" status is filtered as a non-vulnerability. A fix inside the `exclude-newer` window needs its own `[tool.uv.exclude-newer-package]` carve-out. When a new advisory reds the audit on `main` it is not the PR's fault: state the cause instead of retrying CI, then land the bump or the documented `--ignore` (#751, #758, #771).
- `docs/src/assets/coverage.svg` is regenerated by an automated CI PR; don't hand-edit.
