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
  NUM_WORKERS="$PYTEST_WORKERS"
else
  if command -v nproc &> /dev/null; then
    CPU_COUNT=$(nproc)
  elif command -v sysctl &> /dev/null; then
    CPU_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo auto)
  else
    CPU_COUNT=auto
  fi

  if [ "$CPU_COUNT" = "auto" ]; then
    NUM_WORKERS=auto
  else
    # One worker beyond the CPU count: a measurable share of the suite's wall
    # time is tests waiting on timers, subprocesses, sockets and browsers, so a
    # spare worker keeps CPU-bound tests running during those waits. Measured
    # on 2 CPUs: 3 workers ran the fast suite in 798s vs 878s (-9.2%), while 4
    # workers only reached 819s, so one extra is the useful amount.
    NUM_WORKERS=$((CPU_COUNT + 1))
  fi
fi

echo "Running tests with ${NUM_WORKERS} parallel workers"

PYTEST_BASE_ARGS=(-ra --durations=50 --import-mode=importlib -n "$NUM_WORKERS")

if [ "$RUN_COVERAGE" = true ]; then
  # coverage.py's sysmon core (Python 3.12+) records the same lines as the C
  # tracer with almost no runtime overhead. An explicit COVERAGE_CORE wins.
  if [ -z "${COVERAGE_CORE:-}" ] && uv run python -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
    export COVERAGE_CORE=sysmon
  fi
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
