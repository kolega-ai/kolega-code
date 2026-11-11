from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, async_playwright, expect

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.web.bundle import export_bundle
from kolega_code.web.hosting import ShareServer

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

SESSION_ID = "c" * 32
TASK_EVENT = "task_list_updated"
WORKER = {
    "dispatch_id": "task-worker",
    "agent_id": "worker-agent",
    "agent_name": "general-agent",
    "label": "Worker",
    "parent_agent_id": "root",
    "depth": 1,
    "background": True,
}
FIRST = "## Session tasks\nKeep this explanatory note.\n\n- [ ] Implement the change\n- [X] Read the requirements"
FIRST_ENTRIES = [("Implement the change", "pending"), ("Read the requirements", "completed")]
SECOND = "- [x] Implement the change\n- [ ] Run the browser tests"
SECOND_ENTRIES = [("Implement the change", "completed"), ("Run the browser tests", "pending")]
LONG_TEXT = "Verify 測試 e\u0301 " + "complete task content " * 60 + "VISIBLE END"
LONG_MARKDOWN = f"## Full plan\nNotes remain visible.\n- [ ] {LONG_TEXT}\n- [x] Finished"
MALICIOUS_PENDING = (
    '<img src="https://task-list.invalid/image.png" onerror="globalThis.taskListExecuted=true">'
    " [execute](javascript:globalThis.taskListExecuted=true)"
)
MALICIOUS_COMPLETED = (
    '![remote](https://task-list.invalid/markdown.png) <svg onload="globalThis.taskListExecuted=true"></svg>'
)
MALICIOUS = (
    "# Literal task content\n"
    "<script>globalThis.taskListExecuted=true</script>\n"
    '<iframe src="https://task-list.invalid/frame"></iframe>\n'
    '<style>@import url("https://task-list.invalid/style.css");</style>\n'
    f"- [ ] {MALICIOUS_PENDING}\n"
    f"- [x] {MALICIOUS_COMPLETED}\n"
    "</script><script>globalThis.taskListExecuted=true</script>\n"
)


class _Recording:
    def __init__(self, directory: Path) -> None:
        self.store: SessionStore = SessionStore(directory / "state")
        self.store.create(directory, "cli", {}, session_id=SESSION_ID, title="task list replay")
        self.events: FileSessionEventStore = FileSessionEventStore(self.store.journal(SESSION_ID))
        self.recorded: list[AgentEvent] = []

    async def append(
        self,
        event_type: str,
        *,
        info: dict[str, Any] | None = None,
        elapsed_ms: int | None = None,
        **content: Any,
    ) -> None:
        event = AgentEvent(
            session_id=SESSION_ID,
            sender="agent",
            event_type=event_type,
            content=content,
            sub_agent_info=info,
            elapsed_ms=elapsed_ms if elapsed_ms is not None else len(self.recorded) * 100,
        )
        await self.events.append(event)
        self.recorded.append(event)


async def _recording(directory: Path) -> _Recording:
    recording = _Recording(directory)
    await recording.append(KnownEventType.TURN_STARTED, turn_id="root-turn", user_text="root prompt")
    return recording


@asynccontextmanager
async def _page() -> AsyncGenerator[Page, None]:
    async with async_playwright() as driver:
        browser = await driver.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1200, "height": 1200})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            yield page
            assert not errors, f"player pageerrors: {errors}"
        finally:
            await browser.close()


async def _open(page: Page, url: str, *, live: bool = False) -> None:
    await page.goto(url)
    await expect(page.locator("#transcript .kc-entry").first).to_be_visible()
    if live:
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")


@asynccontextmanager
async def _served(recording: _Recording, *, live: bool = True) -> AsyncGenerator[Page, None]:
    server = ShareServer(recording.store, session_id=SESSION_ID)
    await server.start()
    try:
        async with _page() as page:
            await _open(page, server.session_url(SESSION_ID), live=live)
            yield page
    finally:
        await server.stop()


@contextmanager
def _static_url(directory: Path) -> Generator[str, None, None]:
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


@asynccontextmanager
async def _exported(recording: _Recording, directory: Path, *, single_file: bool) -> AsyncGenerator[Page, None]:
    result = await export_bundle(
        recording.recorded,
        directory / ("replay.html" if single_file else "bundle"),
        session_id=SESSION_ID,
        single_file=single_file,
    )
    async with _page() as page:
        if single_file:
            await _open(page, result.path.as_uri())
            yield page
        else:
            with _static_url(result.path) as url:
                await _open(page, url)
                yield page


async def _delivered(page: Page, recording: _Recording) -> None:
    await page.wait_for_function(
        "(count) => document.getElementById('meta').textContent.startsWith(`${count} events`)",
        arg=len(recording.recorded),
    )


async def _seek(page: Page, value: int) -> None:
    await page.eval_on_selector(
        "#scrub", "(el, value) => { el.value = String(value); el.dispatchEvent(new Event('input')); }", value
    )


async def _focus(page: Page) -> None:
    await page.locator('#subAgents [data-agent-key="task-worker"]').click()
    await expect(page.locator("#agentHeader")).to_be_visible()


async def _empty(page: Page, *, cleared: bool = False) -> None:
    panel = page.locator("#taskList")
    await expect(panel).to_have_attribute("data-state", "cleared" if cleared else "unknown")
    await expect(panel.locator(".kc-empty")).to_have_text(
        "Task list cleared." if cleared else "No task list recorded yet."
    )
    await expect(panel.locator(".kc-empty")).to_be_visible()
    await expect(panel.locator(".kc-task-summary")).to_have_text("Session-wide")
    assert await panel.locator(".kc-task-entry, .kc-task-source").count() == 0


async def _tasks(page: Page, markdown: str, entries: list[tuple[str, str]]) -> None:
    panel = page.locator("#taskList")
    await expect(panel).to_have_attribute("data-state", "populated")
    await expect(panel.locator(".kc-task-text")).to_have_text([text for text, _ in entries])
    assert await panel.locator(".kc-task-text").all_text_contents() == [text for text, _ in entries]
    assert await panel.locator(".kc-task-entry").evaluate_all("(rows) => rows.map(row => row.dataset.status)") == [
        status for _, status in entries
    ]
    assert await panel.locator(".kc-task-mark").all_text_contents() == [
        "✓" if status == "completed" else "○" for _, status in entries
    ]
    assert await panel.locator(".kc-task-mark").evaluate_all(
        "(marks) => marks.map(mark => mark.getAttribute('aria-label'))"
    ) == ["Completed" if status == "completed" else "Pending" for _, status in entries]
    completed = sum(status == "completed" for _, status in entries)
    await expect(panel.locator(".kc-task-summary")).to_have_text(f"Session-wide · {completed}/{len(entries)} completed")
    await expect(panel.locator(".kc-task-source")).to_be_visible()
    assert await panel.locator(".kc-task-source").text_content() == markdown
    assert await panel.locator("input, button, a, [role=checkbox]").count() == 0


async def test_served_live_updates_replace_complete_content_clear_and_survive_reload(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    async with _served(recording) as page:
        await _empty(page)
        await recording.append(TASK_EVENT, markdown=FIRST)
        await _delivered(page, recording)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await recording.append(TASK_EVENT, markdown=LONG_MARKDOWN)
        await _delivered(page, recording)
        await _tasks(page, LONG_MARKDOWN, [(LONG_TEXT, "pending"), ("Finished", "completed")])
        assert await page.locator(".kc-task-text").first.evaluate(
            "(node) => getComputedStyle(node).whiteSpace === 'pre-wrap' && "
            "getComputedStyle(node).overflowWrap === 'anywhere'"
        )
        await recording.append(TASK_EVENT, markdown=SECOND)
        await _delivered(page, recording)
        await _tasks(page, SECOND, SECOND_ENTRIES)
        await recording.append(TASK_EVENT, markdown="")
        await _delivered(page, recording)
        await _empty(page, cleared=True)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _empty(page, cleared=True)
        await recording.append(TASK_EVENT, markdown=FIRST)
        await _delivered(page, recording)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _tasks(page, FIRST, FIRST_ENTRIES)


async def test_task_list_is_session_scoped_while_worker_is_focused(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(KnownEventType.ASSISTANT_DELTA, info=WORKER, text="worker answer", complete=True)
    async with _served(recording) as page:
        await _focus(page)
        await _empty(page)
        worker_updates: list[tuple[str, dict[str, Any]]] = [
            (TASK_EVENT, {"markdown": "- [ ] worker must not replace root"}),
            (KnownEventType.CHAT_MESSAGE, {"message_type": "task_list_update", "text": "- [x] legacy worker"}),
        ]
        for event_type, content in worker_updates:
            await recording.append(event_type, info=WORKER, **content)
            await _delivered(page, recording)
            await _empty(page)
        await recording.append(TASK_EVENT, markdown=FIRST)
        await _delivered(page, recording)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await recording.append(TASK_EVENT, info=WORKER, markdown="")
        await _delivered(page, recording)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await recording.append(
            KnownEventType.CHAT_MESSAGE, info=WORKER, message_type="task_list_update", text="- [ ] still worker"
        )
        await _delivered(page, recording)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
        await recording.append(KnownEventType.CHAT_MESSAGE, message_type="task_list_update", text=SECOND)
        await _delivered(page, recording)
        await _tasks(page, SECOND, SECOND_ENTRIES)
        await page.locator(".kc-agent-back").click()
        await _tasks(page, SECOND, SECOND_ENTRIES)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await _focus(page)
        await recording.append(TASK_EVENT, markdown="")
        await _delivered(page, recording)
        await _empty(page, cleared=True)
        await page.locator(".kc-agent-back").click()
        await _empty(page, cleared=True)


async def test_unrelated_streaming_and_identical_updates_do_not_rebuild_the_task_list(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(TASK_EVENT, markdown=LONG_MARKDOWN)
    async with _served(recording) as page:
        await _tasks(page, LONG_MARKDOWN, [(LONG_TEXT, "pending"), ("Finished", "completed")])
        await page.evaluate(
            """() => {
                globalThis.taskRow = document.querySelector('.kc-task-entry');
                globalThis.taskSource = document.querySelector('.kc-task-source');
                globalThis.taskMutations = 0;
                new MutationObserver(records => { globalThis.taskMutations += records.length; })
                    .observe(document.getElementById('taskList'),
                        {childList: true, subtree: true, characterData: true, attributes: true});
                document.getElementById('taskList').scrollTop = 75;
                globalThis.taskScroll = document.getElementById('taskList').scrollTop;
            }"""
        )
        assert await page.evaluate("globalThis.taskScroll > 0")
        for text in ("Streaming", " an answer", " in chunks"):
            await recording.append(KnownEventType.ASSISTANT_DELTA, text=text, complete=False)
            await _delivered(page, recording)
        await recording.append(TASK_EVENT, markdown=LONG_MARKDOWN)
        await _delivered(page, recording)
        await recording.append(KnownEventType.STATUS_UPDATE, message="unrelated status")
        await _delivered(page, recording)
        assert await page.evaluate(
            """() => globalThis.taskMutations === 0 &&
                document.querySelector('.kc-task-entry') === globalThis.taskRow &&
                document.querySelector('.kc-task-source') === globalThis.taskSource &&
                document.getElementById('taskList').scrollTop === globalThis.taskScroll"""
        )
        changed = LONG_MARKDOWN.replace("- [ ]", "- [x]")
        await recording.append(TASK_EVENT, markdown=changed)
        await _delivered(page, recording)
        await _tasks(page, changed, [(LONG_TEXT, "completed"), ("Finished", "completed")])
        assert await page.evaluate("globalThis.taskMutations > 0")
        changed_notes = changed.replace("Notes remain visible.", "New notes with the same checklist.")
        await recording.append(TASK_EVENT, markdown=changed_notes)
        await _delivered(page, recording)
        await _tasks(page, changed_notes, [(LONG_TEXT, "completed"), ("Finished", "completed")])


async def _replay_recording(directory: Path) -> _Recording:
    recording = await _recording(directory)
    await recording.append(KnownEventType.CHAT_MESSAGE, elapsed_ms=1000, message_type="task_list_update", text=FIRST)
    await recording.append(
        KnownEventType.ASSISTANT_DELTA, elapsed_ms=2000, info=WORKER, text="worker answer", complete=True
    )
    await recording.append(TASK_EVENT, elapsed_ms=3000, info=WORKER, markdown="- [ ] not the root task list")
    await recording.append(KnownEventType.CHAT_MESSAGE, elapsed_ms=4000, message_type="task_list_update", text="")
    await recording.append(TASK_EVENT, elapsed_ms=5000, markdown=SECOND)
    await recording.append(
        KnownEventType.CHAT_MESSAGE,
        elapsed_ms=6000,
        info=WORKER,
        message_type="task_list_update",
        text="- [x] ignored legacy worker",
    )
    await recording.append(TASK_EVENT, elapsed_ms=7000, markdown="")
    await recording.append(KnownEventType.TURN_ENDED, elapsed_ms=8000, turn_id="root-turn", status="completed")
    return recording


async def _check_replay(page: Page) -> None:
    await _empty(page)
    await _seek(page, 125)
    await _tasks(page, FIRST, FIRST_ENTRIES)
    await _seek(page, 375)
    await _tasks(page, FIRST, FIRST_ENTRIES)
    await _focus(page)
    assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
    await _tasks(page, FIRST, FIRST_ENTRIES)
    await _seek(page, 500)
    await _empty(page, cleared=True)
    await _seek(page, 625)
    await _tasks(page, SECOND, SECOND_ENTRIES)
    await _seek(page, 750)
    await _tasks(page, SECOND, SECOND_ENTRIES)
    assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
    await _seek(page, 1000)
    await _empty(page, cleared=True)
    await _seek(page, 125)
    await _tasks(page, FIRST, FIRST_ENTRIES)
    assert "not present at this position" in await page.locator("#agentHeader").inner_text()
    await _seek(page, 0)
    await _empty(page)
    await _seek(page, 625)
    await _tasks(page, SECOND, SECOND_ENTRIES)
    await page.locator(".kc-agent-back").click()
    assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
    await page.reload()
    await expect(page.locator("#transcript .kc-entry").first).to_be_visible()
    await _empty(page)
    await _seek(page, 625)
    await _tasks(page, SECOND, SECOND_ENTRIES)
    await _seek(page, 1000)
    await _empty(page, cleared=True)


async def test_served_legacy_replay_rebuilds_visible_prefixes_on_seek_and_reload(tmp_path: Path) -> None:
    recording = await _replay_recording(tmp_path)
    async with _served(recording, live=False) as page:
        await _check_replay(page)


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "single-html"])
async def test_exports_replay_task_lists_without_a_live_connection(tmp_path: Path, single_file: bool) -> None:
    recording = await _replay_recording(tmp_path)
    async with _exported(recording, tmp_path, single_file=single_file) as page:
        await expect(page.locator("#live")).to_be_hidden()
        await _check_replay(page)
        await expect(page.locator("#live")).to_be_hidden()


async def test_live_scrubbing_retains_historical_tasks_until_jump_to_live(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(TASK_EVENT, elapsed_ms=1000, markdown=FIRST)
    async with _served(recording) as page:
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await _seek(page, 0)
        await _empty(page)
        await recording.append(TASK_EVENT, elapsed_ms=2000, markdown=SECOND)
        await _delivered(page, recording)
        await _empty(page)
        await expect(page.locator("#live")).to_have_attribute("data-state", "behind")
        await page.locator("#live").click()
        await _tasks(page, SECOND, SECOND_ENTRIES)
        await _seek(page, 500)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await recording.append(TASK_EVENT, elapsed_ms=3000, markdown="")
        await _delivered(page, recording)
        await _tasks(page, FIRST, FIRST_ENTRIES)
        await page.locator("#live").click()
        await _empty(page, cleared=True)


@pytest.mark.parametrize("mode", ["live", "directory", "single-html"])
async def test_task_content_is_literal_text_and_never_executes_or_loads_resources(tmp_path: Path, mode: str) -> None:
    recording = await _recording(tmp_path)
    if mode != "live":
        await recording.append(TASK_EVENT, markdown=MALICIOUS)
        await recording.append(KnownEventType.TURN_ENDED, turn_id="root-turn", status="completed")
    opened = _served(recording) if mode == "live" else _exported(recording, tmp_path, single_file=mode == "single-html")
    async with opened as page:
        remote_requests: list[str] = []
        page.on("request", lambda request: remote_requests.append(request.url))
        await page.route("https://task-list.invalid/**", lambda route: route.abort())
        if mode == "live":
            await recording.append(TASK_EVENT, markdown=MALICIOUS)
            await _delivered(page, recording)
        else:
            await _seek(page, 1000)
        await _tasks(page, MALICIOUS, [(MALICIOUS_PENDING, "pending"), (MALICIOUS_COMPLETED, "completed")])
        assert await page.locator("#taskList script, #taskList img, #taskList iframe, #taskList svg").count() == 0
        assert await page.locator("#taskList style, #taskList link, #taskList [src], #taskList [href]").count() == 0
        assert await page.evaluate("globalThis.taskListExecuted === undefined")
        assert remote_requests == []
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
