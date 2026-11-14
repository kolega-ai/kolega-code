from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, expect

from kolega_code.cli.session_usage import derive_session_usage
from kolega_code.events import KnownEventType
from tests.web.test_player_task_list import (
    WORKER,
    _Recording,
    _delivered,
    _exported,
    _focus,
    _recording,
    _seek,
    _served,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)
MALICIOUS = (
    "</script><script>globalThis.usageExecuted=true</script>"
    '<img src="https://usage.invalid/image" onerror="globalThis.usageExecuted=true">'
    '<iframe src="https://usage.invalid/frame"></iframe>'
    '<style>@import url("https://usage.invalid/style");</style>'
    "[origin](javascript:globalThis.usageExecuted=true)"
)


def _amounts(reported: str, estimated: str, total: str, *, reports: int = 1, estimates: int = 0) -> dict[str, Any]:
    return {
        "reported": reported,
        "estimated": estimated,
        "total": total,
        "requests": reports + estimates,
        "reported_requests": reports,
        "estimated_requests": estimates,
    }


def _usage() -> dict[str, Any]:
    return {
        "requests": 1,
        "responses": 1,
        "reported": 1,
        "unreported": 0,
        "failed": 0,
        "open": 0,
        "abandoned": 0,
        "partial": 0,
        "missing": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "reasoning_output_tokens": 0,
        "known_requests": {field: int(field in TOKEN_FIELDS[:3]) for field in TOKEN_FIELDS},
        "complete": True,
        "coverage": {"full": True, "accounted_runs": 1, "unclosed_runs": 0, "lost_requests": 0},
        "costs": {
            "reported": 1,
            "estimated": 0,
            "missing": 0,
            "partial": 0,
            "missing_requests": 0,
            "partial_requests": 0,
            "currencies": {"USD": _amounts("0.1000", "0", "0.1000")},
            "non_monetary": {},
            "complete": True,
        },
        "persist_failures": 0,
        "persistence": {"healthy": True, "persist_failures": 0, "unclosed_runs": 0, "lost_requests": 0},
    }


def _partial() -> dict[str, Any]:
    usage = _usage()
    usage.update(
        requests=3,
        responses=2,
        reported=0,
        unreported=2,
        failed=1,
        partial=2,
        missing=1,
        input_tokens=19,
        output_tokens=0,
        total_tokens=0,
        known_requests={field: 2 if field == "input_tokens" else 0 for field in TOKEN_FIELDS},
        complete=False,
    )
    usage["coverage"].update(full=False, pre_accounting_turns=1, uninstrumented_turns=1, legacy_requests=1)
    usage["costs"].update(
        estimated=1,
        missing=1,
        missing_requests=1,
        partial=1,
        partial_requests=1,
        complete=False,
        currencies={
            "USD": _amounts("1.0000", "0.0000000000000000001234500", "1.0000000000000000001234500", estimates=1)
        },
    )
    return usage


def _zero(*, empty: bool = False) -> dict[str, Any]:
    usage = _usage()
    for field in TOKEN_FIELDS:
        usage[field] = 0
        usage["known_requests"][field] = 0 if empty else 1
    usage["costs"]["currencies"] = {} if empty else {"USD": _amounts("0.0000", "0", "0.0000")}
    if empty:
        usage.update(requests=0, responses=0, reported=0)
        usage["costs"]["reported"] = 0
    return usage


async def _snapshot(
    recording: _Recording,
    usage: dict[str, Any],
    *,
    source_seq: int | None = None,
    elapsed_ms: int | None = None,
    info: dict[str, Any] | None = None,
) -> None:
    await recording.append(
        "usage_updated",
        scope="session",
        semantics="snapshot",
        source_seq=source_seq if source_seq is not None else len(recording.recorded) + 1,
        usage=deepcopy(usage),
        elapsed_ms=elapsed_ms,
        info=info,
    )


async def _value(page: Page, group: str, field: str, expected: str) -> None:
    await expect(page.locator(f'#usage [data-group="{group}"] > [data-field="{field}"] .kc-usage-value')).to_have_text(
        expected
    )


async def _cost(page: Page, group: str, denomination: str, field: str, expected: str) -> None:
    rows = page.locator(f'#usage [data-group="{group}"] > section')
    row = rows.filter(has=page.get_by_role("heading", name=denomination, exact=True))
    await expect(row.locator(f'[data-field="{field}"] .kc-usage-value')).to_have_text(expected)


async def _unknown(page: Page) -> None:
    panel = page.locator("#usage")
    await expect(panel).to_have_attribute("data-state", "unknown")
    await expect(panel.locator(".kc-empty")).to_have_text("Usage unavailable. No usage recorded.")
    assert await panel.locator(".kc-usage-row").count() == 0
    await expect(panel.locator(".kc-usage-summary")).to_contain_text("Session-wide")


async def _check_partial(page: Page) -> None:
    await _value(page, "requests", "requests", "3")
    await _value(page, "requests", "partial", "2")
    await _value(page, "requests", "missing", "1")
    await _value(page, "tokens", "input_tokens", "19 · 2/3 requests · partial subtotal")
    for field in TOKEN_FIELDS[1:]:
        await _value(page, "tokens", field, "Unavailable · 0/3 requests")
    await expect(page.locator('#usage [data-group="tokens"] .kc-usage-note')).to_have_text(
        "Incomplete primary token evidence · known subtotals only"
    )
    await expect(page.locator('#usage [data-group="coverage"] .kc-usage-note')).to_have_text(
        "Incomplete recording coverage"
    )
    await _value(page, "coverage", "pre_accounting_turns", "1")
    await _value(page, "coverage", "legacy_requests", "1")
    await _value(page, "costs", "missing", "1")
    await _value(page, "costs", "partial", "1")
    await _cost(page, "currencies", "USD", "total", "1.0000000000000000001234500")
    await _cost(page, "currencies", "USD", "reported", "1.0000 · 1 requests")
    await _cost(page, "currencies", "USD", "estimated", "0.0000000000000000001234500 · 1 requests")
    await expect(page.locator('#usage [data-group="persistence"] .kc-usage-note')).to_have_text("Healthy")


async def test_live_snapshots_replace_reload_and_preserve_historical_view(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    async with _served(recording) as page:
        await _unknown(page)
        await _snapshot(recording, _partial(), elapsed_ms=1000)
        await _delivered(page, recording)
        await _check_partial(page)
        await _snapshot(recording, _zero(), elapsed_ms=2000)
        await _delivered(page, recording)
        await _value(page, "requests", "requests", "1")
        for field in TOKEN_FIELDS:
            await _value(page, "tokens", field, "0 · 1/1 requests")
        await _cost(page, "currencies", "USD", "reported", "0.0000 · 1 requests")
        await _cost(page, "currencies", "USD", "estimated", "Unavailable · 0 requests")
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _value(page, "tokens", "total_tokens", "0 · 1/1 requests")
        await _seek(page, 500)
        await _check_partial(page)
        await _snapshot(recording, _usage(), elapsed_ms=3000)
        await _delivered(page, recording)
        await _check_partial(page)
        await expect(page.locator("#live")).to_have_attribute("data-state", "behind")
        await page.locator("#live").click()
        await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
        await _cost(page, "currencies", "USD", "total", "0.1000")
        await _seek(page, 0)
        await _unknown(page)


async def test_empty_new_snapshot_is_distinct_from_old_recording_unknown(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    async with _served(recording) as page:
        await _unknown(page)
        await _snapshot(recording, _zero(empty=True))
        await _delivered(page, recording)
        await expect(page.locator("#usage")).to_have_attribute("data-state", "recorded")
        await _value(page, "requests", "requests", "0")
        await _value(page, "tokens", "total_tokens", "0 · 0/0 requests")
        await _value(page, "costs", "missing", "0")


@pytest.mark.parametrize("missing", ["tokens", "costs"])
async def test_token_and_cost_availability_are_independent(tmp_path: Path, missing: str) -> None:
    recording = await _recording(tmp_path)
    usage = _usage()
    if missing == "tokens":
        usage.update(complete=False, reported=0, unreported=1, missing=1)
        for field in TOKEN_FIELDS:
            usage[field] = 0
            usage["known_requests"][field] = 0
    else:
        usage["costs"].update(complete=False, reported=0, missing=1, missing_requests=1, currencies={})
    await _snapshot(recording, usage)
    async with _served(recording) as page:
        if missing == "tokens":
            await _value(page, "tokens", "total_tokens", "Unavailable · 0/1 requests")
            await _cost(page, "currencies", "USD", "total", "0.1000")
            await expect(page.locator('#usage [data-group="costs"] .kc-usage-note')).to_have_attribute(
                "data-state", "complete"
            )
        else:
            await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
            await expect(page.locator('#usage [data-group="currencies"] .kc-empty')).to_have_text(
                "Unavailable · no cost recorded."
            )
            await _value(page, "costs", "missing", "1")
        await expect(page.locator('#usage [data-group="persistence"] .kc-usage-note')).to_have_text("Healthy")
        await expect(page.locator('#usage [data-group="coverage"] .kc-usage-note')).to_have_text(
            "Full recording coverage"
        )


async def test_worker_focus_does_not_change_session_usage_or_context(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(KnownEventType.LLM_CONTEXT_UPDATE, input_tokens=40, max_tokens=100, usage_percentage=40)
    await recording.append(KnownEventType.ASSISTANT_DELTA, info=WORKER, text="worker answer", complete=True)
    await _snapshot(recording, _usage())
    async with _served(recording) as page:
        context = await page.locator("#gaugeLabel").text_content()
        await _focus(page)
        await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
        await _snapshot(recording, _zero(), info=WORKER, source_seq=999)
        await _delivered(page, recording)
        await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
        await _snapshot(recording, _partial())
        await _delivered(page, recording)
        await _check_partial(page)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
        await page.locator(".kc-agent-back").click()
        await _check_partial(page)
        assert await page.locator("#gaugeLabel").text_content() == context
        assert await page.locator("#usage button, #usage input, #usage a").count() == 0


async def _replay(directory: Path) -> _Recording:
    recording = await _recording(directory)
    await _snapshot(recording, _partial(), elapsed_ms=1000)
    await recording.append(
        KnownEventType.ASSISTANT_DELTA, info=WORKER, elapsed_ms=2000, text="worker answer", complete=True
    )
    await _snapshot(recording, _usage(), elapsed_ms=3000)
    await recording.append(KnownEventType.TURN_ENDED, elapsed_ms=4000, turn_id="root-turn", status="completed")
    return recording


async def _check_replay(page: Page) -> None:
    await _unknown(page)
    await _seek(page, 250)
    await _check_partial(page)
    await _seek(page, 750)
    await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
    await _focus(page)
    await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
    await _seek(page, 500)
    await _check_partial(page)
    await _seek(page, 0)
    await _unknown(page)
    await _seek(page, 1000)
    await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
    await page.reload()
    await expect(page.locator("#transcript .kc-entry").first).to_be_visible()
    await _unknown(page)
    await _seek(page, 250)
    await _check_partial(page)


async def test_served_replay_rebuilds_usage_on_backward_seek_and_reload(tmp_path: Path) -> None:
    recording = await _replay(tmp_path)
    async with _served(recording, live=False) as page:
        await _check_replay(page)


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "offline-single-html"])
async def test_export_usage_replay_is_offline_and_session_scoped(tmp_path: Path, single_file: bool) -> None:
    recording = await _replay(tmp_path)
    async with _exported(recording, tmp_path, single_file=single_file) as page:
        await expect(page.locator("#live")).to_be_hidden()
        sockets: list[str] = []
        requests: list[str] = []
        page.on("websocket", lambda socket: sockets.append(socket.url))
        page.on("request", lambda request: requests.append(request.url))
        await _check_replay(page)
        assert sockets == []
        if single_file:
            assert all(url.startswith("file:") for url in requests)
        await expect(page.locator("#live")).to_be_hidden()


async def test_independent_denominations_provenance_precision_health_and_fallback_counts(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    usage = _partial()
    usage["costs"]["currencies"]["EUR"] = _amounts("9007199254740993.230000", "0", "9007199254740993.230000")
    usage["costs"]["non_monetary"]["provider_credits"] = _amounts("0", "3.000000", "3.000000", reports=0, estimates=1)
    usage["costs"]["currencies"]["USD"]["token_cost_subtotal"] = "1.0000000000000000000234500"
    usage["costs"].update(reported=2, estimated=2)
    usage.update(requests=5, responses=4)
    async with _served(recording) as page:
        await _snapshot(recording, usage)
        await _delivered(page, recording)
        await _cost(page, "currencies", "EUR", "total", "9007199254740993.230000")
        await _cost(page, "currencies", "USD", "total", "1.0000000000000000001234500")
        await _cost(page, "currencies", "USD", "token_cost_subtotal", "1.0000000000000000000234500")
        await _cost(page, "non_monetary", "provider_credits", "estimated", "3.000000 · 1 requests")
        await _cost(page, "non_monetary", "provider_credits", "reported", "Unavailable · 0 requests")
        await _value(page, "costs", "reported", "2")
        await _value(page, "costs", "estimated", "2")
        await expect(page.locator('#usage [data-group="persistence"] .kc-usage-note')).to_have_text("Healthy")
        usage["persistence"] = {"healthy": False, "persist_failures": 2, "unclosed_runs": 1, "lost_requests": 3}
        await _snapshot(recording, usage)
        await _delivered(page, recording)
        await expect(page.locator('#usage [data-group="persistence"] .kc-usage-note')).to_have_attribute(
            "data-state", "unhealthy"
        )
        await _value(page, "persistence", "persist_failures", "2")
        await _value(page, "persistence", "unclosed_runs", "1")
        await _value(page, "persistence", "lost_requests", "3")
        usage.pop("persistence")
        usage["persist_failures"] = 4
        usage["coverage"].update(unclosed_runs=2, lost_requests=5)
        await _snapshot(recording, usage)
        await _delivered(page, recording)
        await _value(page, "persistence", "persist_failures", "4")
        await _value(page, "persistence", "unclosed_runs", "2")
        await _value(page, "persistence", "lost_requests", "5")
        await expect(page.locator('#usage [data-group="persistence"] .kc-usage-note')).to_have_text(
            "Persistence health unavailable"
        )


async def test_semantic_journal_evidence_reaches_browser_without_inventing_missing_totals(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    journal = recording.store.journal(recording.store.list()[0].session_id)
    journal.append("llm.run_started", actor="system", payload={"run_id": "run", "accounting_version": 2})
    for request_id, observation, cost in [
        (
            "reported",
            {"input_tokens": 12, "output_tokens": 0},
            {
                "source": "provider_reported",
                "amount": "0.0000",
                "currency": "USD",
                "complete": True,
            },
        ),
        (
            "partial",
            {"input_tokens": 7, "complete": False},
            {
                "source": "estimated",
                "known_subtotal": "0.0000000000000000001234500",
                "currency": "USD",
                "complete": False,
            },
        ),
        ("missing", {}, None),
    ]:
        journal.append(
            "llm.request_settled",
            actor="system",
            payload={
                "run_id": "run",
                "request_id": request_id,
                "revision": 1,
                "state": "responded",
                "provider": "test",
                "model": "recorded",
                "observation": observation,
                "cost": cost,
            },
        )
    journal.append(
        "llm.run_finished", actor="system", payload={"run_id": "run", "counts": {"requests": 3}, "persist_failures": 0}
    )
    semantic_events = journal.read_events(repair_tail=False)
    usage = derive_session_usage(semantic_events)
    await _snapshot(recording, usage, source_seq=semantic_events[-1].seq)
    async with _served(recording) as page:
        await _value(page, "requests", "requests", "3")
        await _value(page, "requests", "reported", "1")
        await _value(page, "requests", "partial", "1")
        await _value(page, "requests", "missing", "1")
        await _value(page, "tokens", "input_tokens", "19 · 2/3 requests · partial subtotal")
        await _value(page, "tokens", "output_tokens", "0 · 1/3 requests · partial subtotal")
        await _value(page, "tokens", "total_tokens", "12 · 1/3 requests · partial subtotal")
        await _value(page, "tokens", "cache_read_input_tokens", "Unavailable · 0/3 requests")
        await _cost(page, "currencies", "USD", "total", "1.234500E-19")
        await _cost(page, "currencies", "USD", "reported", "0.0000 · 1 requests")
        await _cost(page, "currencies", "USD", "estimated", "1.234500E-19 · 1 requests")
        await _value(page, "costs", "missing", "1")
        await _value(page, "costs", "partial", "1")
        await expect(page.locator('#usage [data-group="coverage"] .kc-usage-note')).to_have_text(
            "Full recording coverage"
        )
        await expect(page.locator('#usage [data-group="tokens"] .kc-usage-note')).to_have_attribute(
            "data-state", "incomplete"
        )
        await expect(page.locator('#usage [data-group="persistence"] .kc-usage-note')).to_have_text("Healthy")


@pytest.mark.parametrize("mode", ["live", "directory", "single-html"])
async def test_usage_text_is_literal_read_only_and_never_loads_resources(tmp_path: Path, mode: str) -> None:
    recording = await _recording(tmp_path)
    usage = _usage()
    usage["costs"]["currencies"] = {MALICIOUS: _amounts("1.2300", "0", "1.2300")}
    usage["costs"]["non_monetary"] = {MALICIOUS: _amounts("0", "3.0000", "3.0000", reports=0, estimates=1)}
    usage["costs"]["origin"] = MALICIOUS
    usage["breakdown"] = [{"provider": MALICIOUS, "model": MALICIOUS, "origin": MALICIOUS}]
    if mode != "live":
        await _snapshot(recording, usage, elapsed_ms=1000)
        await recording.append(KnownEventType.TURN_ENDED, turn_id="root-turn", status="completed", elapsed_ms=2000)
    opened = _served(recording) if mode == "live" else _exported(recording, tmp_path, single_file=mode == "single-html")
    async with opened as page:
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url))
        await page.route("https://usage.invalid/**", lambda route: route.abort())
        if mode == "live":
            await _snapshot(recording, usage)
            await _delivered(page, recording)
        else:
            await _seek(page, 1000)
        await expect(page.locator("#usage h4")).to_have_text([MALICIOUS, MALICIOUS])
        assert await page.locator("#usage h4").all_text_contents() == [MALICIOUS, MALICIOUS]
        await _cost(page, "currencies", MALICIOUS, "total", "1.2300")
        await _cost(page, "non_monetary", MALICIOUS, "estimated", "3.0000 · 1 requests")
        assert (
            await page.locator(
                "#usage script, #usage img, #usage iframe, #usage svg, #usage style, #usage link, "
                "#usage [src], #usage [href], #usage button, #usage input, #usage a"
            ).count()
            == 0
        )
        assert await page.evaluate("globalThis.usageExecuted === undefined")
        assert requests == []


async def test_unrelated_streaming_does_not_rebuild_usage(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await _snapshot(recording, _usage())
    async with _served(recording) as page:
        await _value(page, "tokens", "total_tokens", "15 · 1/1 requests")
        await page.evaluate(
            """() => {
                globalThis.usageMutations = 0;
                new MutationObserver(records => { globalThis.usageMutations += records.length; })
                    .observe(document.getElementById('usage'),
                        {childList: true, subtree: true, characterData: true, attributes: true});
            }"""
        )
        await recording.append(KnownEventType.ASSISTANT_DELTA, text="unrelated stream", complete=False)
        await _delivered(page, recording)
        assert await page.evaluate("globalThis.usageMutations === 0")
