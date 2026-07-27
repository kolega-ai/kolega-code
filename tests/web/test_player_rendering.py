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


def _delegated(event: AgentEvent, **info: Any) -> AgentEvent:
    event.sub_agent_info = {"dispatch_id": "d1", "agent_name": "investigator", **info}
    return event


#: A dispatch: the main agent calls a tool, the sub-agent reasons and answers
#: inside it, then the main agent resumes. The sub-agent's work is folded out of
#: the main conversation, so a player that renders only the conversation shows
#: none of it.
def _delegating_turn() -> list[AgentEvent]:
    return [
        _event(KnownEventType.TURN_STARTED, 1, elapsed_ms=0, turn_id="t1", user_text="count the lines"),
        _event(
            KnownEventType.CHAT_MESSAGE,
            2,
            elapsed_ms=500,
            message_type="tool_call",
            text="delegating",
            tool_description="dispatch_investigation_agent",
            tool_call_id="c1",
        ),
        _delegated(
            _event(KnownEventType.TURN_STARTED, 3, elapsed_ms=600, turn_id="sub", user_text="count the lines"),
            task="count the lines",
        ),
        _delegated(
            _event(
                KnownEventType.THINKING_DELTA,
                4,
                elapsed_ms=700,
                uuid="subth",
                text="Delegated reasoning that must be visible.",
                complete=True,
            )
        ),
        _delegated(_event(KnownEventType.ASSISTANT_DELTA, 5, elapsed_ms=800, uuid="subas", text="241", complete=True)),
        _delegated(_event(KnownEventType.CHAT_MESSAGE, 6, elapsed_ms=900, status="STOPPED", message="done")),
        _event(
            KnownEventType.CHAT_MESSAGE,
            7,
            elapsed_ms=1000,
            message_type="tool_result",
            text="241",
            tool_call_id="c1",
        ),
        _event(KnownEventType.ASSISTANT_DELTA, 8, elapsed_ms=1100, uuid="as", text="It has 241 lines.", complete=True),
        _event(KnownEventType.TURN_ENDED, 9, elapsed_ms=1200, turn_id="t1", status="completed"),
    ]


def _serve(directory: Path) -> Iterator[str]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/player.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _bundle(events: list[AgentEvent], destination: Path, title: str) -> Iterator[str]:
    asyncio.run(export_bundle(events, destination, session_id="s1", title=title))
    yield from _serve(destination)


@pytest.fixture
def bundle_url(tmp_path: Path) -> Iterator[str]:
    """Export the interleaved turn and serve it over HTTP for the browser."""
    yield from _bundle(_interleaved_turn(), tmp_path / "bundle", "interleaved")


@pytest.fixture
def delegating_url(tmp_path: Path) -> Iterator[str]:
    yield from _bundle(_delegating_turn(), tmp_path / "bundle", "delegating")


#: Read back exactly what a viewer would see, per transcript entry.
_ENTRIES_JS = """
() => Array.from(document.querySelectorAll('.kc-entry')).map((entry) => ({
  kind: entry.dataset.kind,
  agent: entry.dataset.subAgent ?? null,
  lead: entry.querySelector('.kc-sub-agent-lead')?.textContent ?? null,
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


@pytest.mark.asyncio
async def test_a_dispatch_is_one_openable_entry_not_spliced_into_the_thread(delegating_url: str) -> None:
    """Delegated work is reachable from the main thread without invading it.

    Splicing a delegate's steps into the transcript reads fine for one agent
    and falls apart for several: parallel agents interleave line by line, so
    answers arrive in an order unrelated to the questions above them. The
    terminal collapses a dispatch to one entry you open, and so does this.
    """
    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            await page.goto(delegating_url)
            await page.wait_for_selector(".kc-entry")
            await _step_to_end(page)
            main_thread = await page.evaluate(_ENTRIES_JS)
            card = await page.evaluate(
                """() => {
                    const el = document.querySelector('.kc-agent-card');
                    return el && {text: el.innerText, status: el.dataset.status, key: el.dataset.agentKey};
                }"""
            )
            await page.click(".kc-agent-card")
            opened = await page.evaluate(_ENTRIES_JS)
            header = await page.inner_text(".kc-agent-header")
            await page.click(".kc-agent-back")
            restored = await page.evaluate(_ENTRIES_JS)
        finally:
            await browser.close()

    assert card is not None, "the dispatch left no entry in the main thread"
    assert "investigator" in card["text"] and card["status"] == "completed"
    assert "3 steps" in card["text"], "the card should say how much the agent did"
    assert not any(entry["agent"] == "investigator" for entry in main_thread), (
        "the delegate's own steps must not be spliced into the main thread"
    )

    # The card sits where the dispatch happened.
    kinds = [entry["kind"] for entry in main_thread]
    assert kinds.index("agent") > kinds.index("tool")
    assert kinds.index("agent") < len(kinds) - 1

    # Opening it shows that agent's thread, and only that.
    assert [entry["kind"] for entry in opened] == ["user", "thinking", "assistant"]
    assert opened[1]["text"] == "Delegated reasoning that must be visible."
    assert "investigator" in header and "Back" in header

    assert restored == main_thread, "going back did not restore the main thread"

    # The main transcript keeps only what the session itself said.
    assert [entry["text"] for entry in main_thread if entry["kind"] == "user"] == ["count the lines"]


def _workflow_agent(event: AgentEvent, *, agent_id: str, label: str) -> AgentEvent:
    event.sub_agent_info = {
        "agent_id": agent_id,
        "agent_name": "general-agent",
        "dispatch_id": None,
        "label": label,
        "phase": "Classify",
        "task": f"classify {label}",
        "workflow_run_id": "run-1",
    }
    return event


def _workflow_turn() -> list[AgentEvent]:
    """A gigacode fan-out: two agents that share a name and run in parallel."""
    return [
        _event(KnownEventType.TURN_STARTED, 1, elapsed_ms=0, turn_id="t1", user_text="triage these"),
        _event(
            KnownEventType.CHAT_MESSAGE,
            2,
            elapsed_ms=100,
            message_type="workflow_start",
            workflow_run_id="run-1",
            name="triage",
            description="classify in parallel",
            text="",
        ),
        _event(
            KnownEventType.CHAT_MESSAGE,
            3,
            elapsed_ms=200,
            message_type="workflow_phase",
            workflow_run_id="run-1",
            text="Classify",
        ),
        _workflow_agent(
            _event(KnownEventType.ASSISTANT_DELTA, 4, elapsed_ms=300, uuid="a", text="alpha done", complete=True),
            agent_id="wf-alpha",
            label="alpha",
        ),
        _workflow_agent(
            _event(KnownEventType.ASSISTANT_DELTA, 5, elapsed_ms=400, uuid="b", text="beta done", complete=True),
            agent_id="wf-beta",
            label="beta",
        ),
        _event(
            KnownEventType.CHAT_MESSAGE,
            6,
            elapsed_ms=500,
            message_type="workflow_end",
            workflow_run_id="run-1",
            status="completed",
            text="",
        ),
        _event(KnownEventType.TURN_ENDED, 7, elapsed_ms=600, turn_id="t1", status="completed"),
    ]


@pytest.fixture
def workflow_url(tmp_path: Path) -> Iterator[str]:
    yield from _bundle(_workflow_turn(), tmp_path / "bundle", "workflow")


@pytest.mark.asyncio
async def test_tool_output_is_collapsed_until_asked_for(bundle_url: str) -> None:
    """A transcript of expanded tool results is unreadable.

    Every result rendered in full, so following the conversation meant scrolling
    past screens of output to find the sentence between two calls.
    """
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

            count = "() => document.querySelectorAll('.kc-tool-output').length"
            collapsed = await page.evaluate(count)
            # The head still says what ran and how it went.
            head = await page.inner_text(".kc-tool-head")
            await page.click("[data-tool-key] >> nth=0")
            expanded = await page.evaluate(count)
            opened_text = await page.inner_text(".kc-tool-output")
            await page.click("[data-tool-key] >> nth=0")
            recollapsed = await page.evaluate(count)
        finally:
            await browser.close()

    assert collapsed == 0, "tool output was rendered before anyone asked for it"
    assert "read_file" in head and "done" in head.lower()
    assert expanded == 1 and "12 lines" in opened_text
    assert recollapsed == 0, "clicking again did not put it away"


@pytest.mark.asyncio
async def test_clicking_an_agent_narrows_the_transcript_to_its_own_work(workflow_url: str) -> None:
    """A fan-out is unreadable as one merged stream.

    Parallel agents interleave line by line, and every agent of a workflow
    shares the name "general-agent", so without per-agent identity and a way to
    isolate one there is no way to follow what any single agent did.
    """
    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            await page.goto(workflow_url)
            await page.wait_for_selector(".kc-entry")
            await _step_to_end(page)

            labels = await page.evaluate(
                "() => [...document.querySelectorAll('#subAgents [data-agent-key]')]"
                ".map(b => b.querySelector('.kc-sub-agent-name').innerText)"
            )
            everything = await page.evaluate(_ENTRIES_JS)

            await page.click('[data-agent-key="wf-beta"]')
            focused = await page.evaluate(_ENTRIES_JS)
            back_label = await page.inner_text(".kc-sub-agent-all")

            await page.click(".kc-sub-agent-all")
            restored = await page.evaluate(_ENTRIES_JS)

            workflow_rows = [entry for entry in everything if entry["kind"] == "workflow"]
        finally:
            await browser.close()

    assert len(labels) == 2, f"parallel agents were not listed separately: {labels}"
    assert any("alpha" in label for label in labels) and any("beta" in label for label in labels)

    assert [entry["text"] for entry in focused] == ["beta done"], (
        "clicking an agent did not narrow the transcript to that agent"
    )
    assert "Back" in back_label
    assert restored == everything, "going back did not restore the whole session"

    # The run's own lifecycle is visible, not silently dropped.
    assert [entry["text"] for entry in workflow_rows] == [
        "triage — classify in parallel",
        "Classify",
        "completed",
    ]
