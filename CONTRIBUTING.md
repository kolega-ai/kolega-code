# Contributing

## Local Setup

Kolega Code requires Python 3.11 or newer.

Install the CLI, development dependencies, and the tracked Git hooks with the setup script:

```bash
./scripts/setup-dev.sh
```

If you prefer to run the steps manually:

```bash
uv sync --extra cli --extra dev
git config core.hooksPath .githooks
uv run pre-commit install-hooks
uv run pre-commit run --all-files --show-diff-on-failure
```

Installing the `pre-commit` Python package is not enough. This repository uses a tracked
`.githooks/pre-commit` wrapper that runs the exact same all-files command as CI. Run
`./scripts/setup-dev.sh` or `git config core.hooksPath .githooks` once per clone.

If you prefer `pip`, install the package in editable mode:

```bash
pip install -e ".[cli,dev]"
```

Run the CLI locally:

```bash
kolega-code .
```

Some slow and integration tests require provider credentials. Copy the example
environment file only when you need those tests, and never commit real secrets:

```bash
cp .env.example .env
```

## Tests and quality checks

Verify that Git is configured to use the tracked hooks:

```bash
./scripts/check-dev-hooks.sh
```

Local commits run the same all-files pre-commit command as CI when `core.hooksPath` points
to `.githooks`:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

CI runs Ruff through pre-commit. You can also run Ruff directly for one-off linting and
formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

To apply formatting locally:

```bash
uv run ruff format .
```

Run the fast test suite before opening a pull request:

```bash
./run_tests.sh
```

Run slow and integration tests only when you have the required credentials:

```bash
./run_tests.sh --all
```

You can pass additional pytest arguments through the wrapper:

```bash
./run_tests.sh tests/agent/llm/test_client.py -ra
```

## Documentation

The documentation site lives in `docs/`.

```bash
cd docs
npm ci
npm run build
```

Use `npm run dev` from `docs/` for local documentation development.

## Pull Requests

- Keep changes focused and avoid unrelated refactors.
- Add or update tests for behavior changes.
- Update documentation when user-facing behavior changes.
- Do not commit `.env`, local settings, credentials, API keys, tokens, or private
  endpoints.
- Use obviously fake credential placeholders in tests and docs, not strings that
  match real token formats.
- Report security issues privately according to `SECURITY.md`.
