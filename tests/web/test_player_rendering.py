"""The shipped player must render what the fold actually holds.

The fold is covered by parity tests, but the player draws from it incrementally:
it appends DOM for newly reached entries and refreshes the ones that changed.
That optimisation is where a replay can silently disagree with its own event log,
so it is tested against the real assets in a real browser rather than a stub.

The case that matters is interleaving. Streaming segments are keyed by uuid, so a
reasoning segment stays open across the assistant prose and tool calls that
follow it — every turn does this — and a renderer that only refreshes its newest
entry freezes reasoning mid-sentence, spinner still running.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.web.bundle import export_bundle

playwright_api = pytest.importorskip("playwright.async_api", reason="playwright is required for player tests")


def _event(event_type: str, seq: int, *, elapsed_ms: int, uuid: str | None = None, **content) -> AgentEvent:
    event = AgentEvent(
        session_id="s1",
        sender="agent",
        event_type=event_type,
        content=dict(content),
        seq=seq,
        elapsed_ms=elapsed_ms,
    )
    if uuid is not None:
        event.uuid = uuid
    return event


#: One turn in which reasoning, prose, and a tool call are all open at once, each
#: finishing only after later entries exist. Spread over whole seconds so stepping
#: the scrub lands between them instead of applying the log in one jump.
def _interleaved_turn() -> list[AgentEvent]:
    return [
        _event(KnownEventType.TURN_STARTED, 1, elapsed_ms=0, turn_id="t1", user_text="explain the fix"),
        _event(
            KnownEventType.THINKING_DELTA, 2, elapsed_ms=1000, uuid="th", text="Reasoning first half. ", complete=False
        ),
        _event(
            KnownEventType.ASSISTANT_DELTA, 3, elapsed_ms=2000, uuid="as", text="Answer first half. ", complete=False
        ),
        _event(
            KnownEventType.CHAT_MESSAGE,
            4,
            elapsed_ms=2500,
            message_type="tool_call",
            text="reading",
            tool_description="read_file",
            tool_call_id="c1",
        ),
        _event(
            KnownEventType.THINKING_DELTA, 5, elapsed_ms=3000, uuid="th", text="Reasoning second half.", complete=True
        ),
        _event(
            KnownEventType.ASSISTANT_DELTA, 6, elapsed_ms=3500, uuid="as", text="Answer second half.", complete=True
        ),
        _event(
            KnownEventType.CHAT_MESSAGE,
            7,
            elapsed_ms=4000,
            message_type="tool_result",
            text="12 lines",
            tool_call_id="c1",
        ),
        _event(KnownEventType.TURN_ENDED, 8, elapsed_ms=5000, turn_id="t1", status="completed"),
    ]


@pytest.fixture
def bundle_url(tmp_path: Path) -> Iterator[str]:
    """Export the interleaved turn and serve it over HTTP for the browser."""
    asyncio.run(
        export_bundle(
            _interleaved_turn(),
            tmp_path / "bundle",
            session_id="s1",
            title="interleaved",
        )
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "bundle"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/player.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


#: Read back exactly what a viewer would see, per transcript entry.
_ENTRIES_JS = """
() => Array.from(document.querySelectorAll('.kc-entry')).map((entry) => ({
  kind: entry.dataset.kind,
  text: entry.querySelector('.kc-body')?.textContent ?? '',
  spinning: !!entry.querySelector('.kc-spinner'),
  status: entry.querySelector('.kc-tool-status')?.textContent ?? null,
}))
"""


async def _step_to_end(page: Any) -> None:
    """Advance the scrub in small steps so playback is applied incrementally.

    Seeking straight to the end would append every entry in its final form and
    never exercise the refresh path this test exists to cover.
    """
    for value in range(0, 1001, 50):
        await page.eval_on_selector(
            "#scrub",
            "(el, v) => { el.value = String(v); el.dispatchEvent(new Event('input')); }",
            value,
        )


@pytest.mark.asyncio
async def test_interleaved_reasoning_is_complete_after_playback(bundle_url: str) -> None:
    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(bundle_url)
            await page.wait_for_selector(".kc-entry")

            await _step_to_end(page)
            entries = await page.evaluate(_ENTRIES_JS)
        finally:
            await browser.close()

    assert not errors, f"the player raised: {errors}"
    by_kind = {entry["kind"]: entry for entry in entries}

    # The defect this guards against left these frozen at their first delta.
    assert by_kind["thinking"]["text"] == "Reasoning first half. Reasoning second half."
    assert by_kind["assistant"]["text"] == "Answer first half. Answer second half."
    assert by_kind["tool"]["status"] == "done"
    assert not any(entry["spinning"] for entry in entries), "a settled entry is still showing a spinner"


@pytest.mark.asyncio
async def test_seeking_backwards_then_forwards_rebuilds_the_transcript(bundle_url: str) -> None:
    """A backward seek rebuilds from empty; the result must match a forward pass."""
    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            await page.goto(bundle_url)
            await page.wait_for_selector(".kc-entry")

            await _step_to_end(page)
            forward = await page.evaluate(_ENTRIES_JS)

            for value in (300, 0, 1000):
                await page.eval_on_selector(
                    "#scrub",
                    "(el, v) => { el.value = String(v); el.dispatchEvent(new Event('input')); }",
                    value,
                )
            rewound = await page.evaluate(_ENTRIES_JS)
        finally:
            await browser.close()

    assert rewound == forward, "the transcript differs after seeking back and forward again"
