#!/usr/bin/env bash

set -eo pipefail

echo "Running kolega-code tests..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_SLOW=false
RUN_COVERAGE=false
COVERAGE_FAIL_UNDER=45
PYTEST_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --all|--slow)
      RUN_SLOW=true
      ;;
    --coverage)
      RUN_COVERAGE=true
      ;;
    --coverage-fail-under=*)
      RUN_COVERAGE=true
      COVERAGE_FAIL_UNDER="${arg#*=}"
      ;;
    *)
      PYTEST_ARGS+=("$arg")
      ;;
  esac
done

if [ -n "${PYTEST_WORKERS:-}" ]; then
  NUM_CPUS="$PYTEST_WORKERS"
elif command -v nproc &> /dev/null; then
  NUM_CPUS=$(nproc)
elif command -v sysctl &> /dev/null; then
  NUM_CPUS=$(sysctl -n hw.ncpu 2>/dev/null || echo auto)
else
  NUM_CPUS=auto
fi

echo "Running tests with ${NUM_CPUS} parallel workers"

PYTEST_BASE_ARGS=(-ra --durations=50 --import-mode=importlib -n "$NUM_CPUS")

if [ "$RUN_COVERAGE" = true ]; then
  PYTEST_BASE_ARGS+=(--cov=kolega_code --cov-report=term-missing --cov-report=xml --cov-fail-under="$COVERAGE_FAIL_UNDER")
fi

if [ "$RUN_SLOW" = true ]; then
  echo "Running all tests, including slow and integration tests..."
  uv run pytest "${PYTEST_BASE_ARGS[@]}" "${PYTEST_ARGS[@]}"
else
  echo "Running fast tests only. Use './run_tests.sh --all' to include slow and integration tests."
  uv run pytest "${PYTEST_BASE_ARGS[@]}" -m "not slow" "${PYTEST_ARGS[@]}"
fi

echo "All tests completed."
