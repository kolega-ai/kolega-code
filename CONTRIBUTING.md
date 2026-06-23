# Contributing

## Local Setup

Kolega Code requires Python 3.11 or newer.

Install the CLI and development dependencies with `uv`:

```bash
uv sync --extra cli --extra dev
```

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

Run Ruff linting and formatting checks before opening a pull request:

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
