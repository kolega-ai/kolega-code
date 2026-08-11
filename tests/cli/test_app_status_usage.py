"""The Status tab's Usage card: lifetime baseline + live ledger, rendered
at render time with no pushed state."""

from pathlib import Path

import pytest

from kolega_code.llm.ledger import LlmCallOrigin, llm_call_origin
from kolega_code.llm.usage import normalize_usage

from ._app_test_utils import build_test_config, install_fake_agents


def _build_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import config_summary
    from kolega_code.cli.session_store import SessionStore

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


def _settle(ledger, *, inp=0, out=0, cache_read=None, failed=False):
    metadata = {"input_tokens": inp, "output_tokens": out}
    if cache_read is not None:
        metadata["cache_read_input_tokens"] = cache_read
        metadata["cache_write_input_tokens"] = 0
    with llm_call_origin(LlmCallOrigin(kind="sub_agent", agent_name="Investigator")):
        request_id = ledger.begin("anthropic", "m")
    if failed:
        ledger.record_failure(request_id, "boom")
    else:
        ledger.record_response(request_id, normalize_usage(metadata, "anthropic", "m"))


@pytest.mark.asyncio
async def test_fresh_session_renders_zero_usage_without_markers(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        card = app._usage_summary_lines()
        dashboard = app._format_status_dashboard()

    assert "Session: [bold]0[/bold] tokens" in card
    assert "In 0 · Out 0" in card
    assert "Requests: 0" in card
    assert "(partial)" not in card
    assert "failed" not in card
    assert "Cache reads" not in card
    assert "Cache hit" not in card
    # Usage lives in its own card, not folded into the Status section.
    assert "Session:" not in dashboard


@pytest.mark.asyncio
async def test_live_settlements_render_compact_totals_and_failed_segment(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        _settle(app._usage_ledger, inp=1_000, out=200, cache_read=800)
        _settle(app._usage_ledger, inp=2_000_000, out=100_000)
        _settle(app._usage_ledger, failed=True)
        card = app._usage_summary_lines()

    assert "Session: [bold]2.1M[/bold] tokens" in card
    assert "In 2M · Out 100.2k" in card
    assert "Cache reads 800" in card
    assert "Requests: 3" in card
    assert "1 failed" in card


@pytest.mark.asyncio
async def test_baseline_and_live_usage_combine(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.usage = {
            "requests": 10,
            "failed": 0,
            "input_tokens": 5_000,
            "output_tokens": 1_000,
            "total_tokens": 6_000,
            "cache_read_input_tokens": 0,
            "coverage": {"accounted_runs": 1, "pre_accounting_turns": 0, "full": True},
        }
        _settle(app._usage_ledger, inp=3_000, out=1_000)
        card = app._usage_summary_lines()

    assert "Session: [bold]10k[/bold] tokens" in card
    assert "In 8k · Out 2k" in card
    assert "Requests: 11" in card
    assert "(partial)" not in card


@pytest.mark.asyncio
async def test_cache_hit_percentage_renders(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        _settle(app._usage_ledger, inp=200, out=200, cache_read=800)
        _settle(app._usage_ledger, inp=200, out=200, cache_read=800)
        card = app._usage_summary_lines()

    # Normalized input_tokens is inclusive of cache reads: 1000 per request,
    # 800 served from cache → 1600 of 2000 tokens → 80.00%, two decimals.
    assert "Cache reads 1.6k · Cache hit 80.00%" in card


@pytest.mark.asyncio
async def test_cache_hit_percentage_from_baseline(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.usage = {
            "input_tokens": 10_000,
            "output_tokens": 1_000,
            "total_tokens": 11_000,
            "cache_read_input_tokens": 5_000,
            "coverage": {"accounted_runs": 1, "pre_accounting_turns": 0, "full": True},
        }
        card = app._usage_summary_lines()

    assert "Cache hit 50.00%" in card


@pytest.mark.asyncio
async def test_partial_marker_only_when_turns_predate_accounting(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.usage = {
            "total_tokens": 100,
            "coverage": {"accounted_runs": 0, "pre_accounting_turns": 3, "full": False},
        }
        with_marker = app._usage_summary_lines()
        app.session.usage["coverage"]["pre_accounting_turns"] = 0
        without_marker = app._usage_summary_lines()

    assert "(partial)" in with_marker
    assert "(partial)" not in without_marker


@pytest.mark.asyncio
async def test_usage_card_widget_renders_and_refreshes(tmp_path, monkeypatch):
    from textual.containers import Vertical
    from textual.widgets import Static

    app = _build_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        section = app.query_one("#status_usage_section", Vertical)
        assert section.border_title == "Usage"

        _settle(app._usage_ledger, inp=1_200, out=100)
        app._refresh_status_dashboard()
        rendered = str(app.query_one("#status_usage", Static).render())

    assert "1.3k" in rendered  # 1200 + 100 total, compact
    assert "Requests: 1" in rendered
    assert "Session:" in rendered


def test_format_token_count_boundaries(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    assert app._format_token_count(999) == "999"
    assert app._format_token_count(1_200) == "1.2k"
    assert app._format_token_count(2_000_000) == "2M"
