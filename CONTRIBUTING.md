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

CI runs Ruff and Pyright through pre-commit. You can also run them directly for
one-off checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

To apply formatting locally:

```bash
uv run ruff format .
```

### Type checking

The project uses [Pyright](https://github.com/microsoft/pyright) in basic mode to
catch type errors before runtime. The configuration lives in `[tool.pyright]` in
`pyproject.toml`.

Run the type checker locally:

```bash
uv run pyright
```

Guidelines for type annotations:

- All new code should have type annotations on function signatures and class
  attributes.
- Use `# pyright: ignore[<rule>]` sparingly and only for third-party stub
  limitations — never to silence errors in first-party code.
- Prefer narrowing `Optional` values with `assert` or `if` guards over ignoring
  the error.

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

## Updating bundled skills

Kolega Code vendors a reviewed, tagged snapshot from
[`kolega-ai/kolega-skills`](https://github.com/kolega-ai/kolega-skills). It never
downloads skills while building, installing, or running.

To update the snapshot, create or choose an immutable upstream tag and run:

```bash
uv run python scripts/sync_bundled_skills.py ../kolega-skills --tag vX.Y.Z
```

The script exports the tag's Git object rather than the sibling working tree, replaces
`kolega_code/_bundled_skills/` atomically, and records the tag, commit, file list, and
SHA-256 hashes in `manifest.json`. Review all generated changes, then verify the
distributions:

```bash
uv build
uv run python scripts/verify_bundled_skill_artifacts.py dist/*
```

Always pin an explicit tag. Do not vendor a branch, a moving `latest` reference, or
uncommitted files from the upstream checkout.

## Documentation

The documentation site lives in `docs/`.

```bash
cd docs
npm ci
npm run build
```

Use `npm run dev` from `docs/` for local documentation development.

## Maintainer automation

The CI workflow updates `docs/src/assets/coverage.svg` by opening an automated pull
request. Configure the repository secret `COVERAGE_BADGE_PR_TOKEN` with a trusted
bot or maintainer token instead of `GITHUB_TOKEN`; GitHub requires manual approval
before workflows run on pull requests opened by `github-actions[bot]`.

Use a fine-grained personal access token or GitHub App token scoped to this
repository with:

- Contents: read and write
- Pull requests: read and write
- Metadata: read-only, included automatically

The token owner should be a trusted repository member or bot account. Do not use
a first-time external contributor account, and never commit the token value.

Optional repository variables can customize the Git commit identity used for the
badge update commit:

- `COVERAGE_BADGE_AUTHOR`
- `COVERAGE_BADGE_COMMITTER`

Use values like `Name <email@example.com>`.

## Pull Requests

- Keep changes focused and avoid unrelated refactors.
- Add or update tests for behavior changes.
- Update documentation when user-facing behavior changes.
- Do not commit `.env`, local settings, credentials, API keys, tokens, or private
  endpoints.
- Use obviously fake credential placeholders in tests and docs, not strings that
  match real token formats.
- Report security issues privately according to `SECURITY.md`.
