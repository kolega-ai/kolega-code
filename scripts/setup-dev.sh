#!/usr/bin/env bash
set -euo pipefail

uv sync --extra cli --extra dev
git config core.hooksPath .githooks
uv run pre-commit install-hooks
uv run pre-commit run --all-files --show-diff-on-failure
