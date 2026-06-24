#!/usr/bin/env bash
set -euo pipefail

expected=".githooks"
actual="$(git config --get core.hooksPath || true)"

if [[ "$actual" != "$expected" ]]; then
  echo "Git hooks are not configured to use $expected."
  echo "Run: git config core.hooksPath $expected"
  exit 1
fi

if [[ ! -x .githooks/pre-commit ]]; then
  echo ".githooks/pre-commit is missing or not executable."
  exit 1
fi

echo "Git hooks are configured correctly."
