"""Opt-in live execution of the repository benchmark's complete provider matrix."""

import os
from pathlib import Path

import pytest

from benchmarks.edit_tools.__main__ import _catalog_smoke_matrix, _provider_smoke_suite
from benchmarks.edit_tools.runner import plan_trials, run_trial


pytestmark = [pytest.mark.integration, pytest.mark.slow]


MATRIX = _catalog_smoke_matrix()
SUITE, TASKS = _provider_smoke_suite()
TRIALS = plan_trials(SUITE, TASKS, MATRIX)


@pytest.mark.parametrize("trial", TRIALS, ids=lambda item: f"{item.model.provider}-{item.protocol}")
@pytest.mark.asyncio
async def test_live_catalog_provider_edit_protocol(trial, tmp_path: Path) -> None:
    if os.getenv("KOLEGA_RUN_LIVE_EDIT_BENCHMARKS") != "1":
        pytest.skip("Set KOLEGA_RUN_LIVE_EDIT_BENCHMARKS=1 to run live edit benchmark trials.")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = await run_trial(
        run_id="live-provider-smoke",
        suite=SUITE,
        trial=trial,
        run_dir=run_dir,
        timeout_seconds=MATRIX.trial_timeout_seconds,
    )
    if record.status == "not_run":
        pytest.skip(record.error or "provider credentials unavailable")
    assert record.status == "passed", record.model_dump(mode="json")
