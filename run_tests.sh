#!/usr/bin/env bash

set -eo pipefail

echo "Running kolega-code tests..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_SLOW=false
PYTEST_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --all|--slow)
      RUN_SLOW=true
      ;;
    *)
      PYTEST_ARGS+=("$arg")
      ;;
  esac
done

PYTEST_BASE_ARGS=(-ra --durations=50 --import-mode=importlib)

if [ "$RUN_SLOW" = true ]; then
  echo "Running all tests, including slow and integration tests..."
  uv run pytest "${PYTEST_BASE_ARGS[@]}" "${PYTEST_ARGS[@]}"
else
  echo "Running fast tests only. Use './run_tests.sh --all' to include slow and integration tests."
  uv run pytest "${PYTEST_BASE_ARGS[@]}" -m "not slow" "${PYTEST_ARGS[@]}"
fi

echo "All tests completed."
