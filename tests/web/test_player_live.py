"""The player following a session that is still being written.

A share link is only worth sending if the person who opens it sees what is
happening now. That means the whole chain has to hold at once: the manifest says
the session is open, the link's token survives into the player's own subresources
and its WebSocket handshake, and events appended after the page loaded reach the
transcript with no reload and no gap.

Driven against a real browser and the real in-process share server, because every
one of those links has failed silently at some point and none of them are visible
from a unit test of either end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.web.hosting import ShareServer

playwright_api = pytest.importorskip("playwright.async_api", reason="playwright is required for player tests")

SESSION_ID = "a" * 32

#: Read the parts of the page a viewer would judge "is this live?" by.
_SNAPSHOT_JS = """
() => ({
  entries: document.querySelectorAll('.kc-entry').length,
  text: Array.from(document.querySelectorAll('.kc-body')).map((b) => b.textContent).join('|'),
  live: document.getElementById('live').textContent,
  liveState: document.getElementById('live').dataset.state,
  liveHidden: document.getElementById('live').hidden,
})
"""


async def _append(
    events: FileSessionEventStore, event_type: str, *, elapsed_ms: int, uuid: str | None = None, **content
) -> None:
    event = AgentEvent(
        session_id=SESSION_ID,
        sender="agent",
        event_type=event_type,
        content=dict(content),
        elapsed_ms=elapsed_ms,
    )
    if uuid is not None:
        event.uuid = uuid
    await events.append(event)


async def _wait_for(page: Any, predicate, *, timeout: float = 10.0) -> Any:
    """Poll the page until the live stream has delivered. Never assert on a sleep."""
    deadline = asyncio.get_running_loop().time() + timeout
    snapshot = await page.evaluate(_SNAPSHOT_JS)
    while asyncio.get_running_loop().time() < deadline:
        if predicate(snapshot):
            return snapshot
        await asyncio.sleep(0.1)
        snapshot = await page.evaluate(_SNAPSHOT_JS)
    return snapshot


@pytest.mark.asyncio
async def test_a_share_link_follows_the_session_as_it_runs(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "cli", {}, session_id=SESSION_ID, title="live")
    events = FileSessionEventStore(store.journal(SESSION_ID))
    # An open turn: the manifest reports "open", so the player starts at the edge.
    await _append(events, KnownEventType.TURN_STARTED, elapsed_ms=0, turn_id="t1", user_text="watch this")
    await _append(events, KnownEventType.THINKING_DELTA, elapsed_ms=100, uuid="th", text="before ", complete=False)

    server = ShareServer(store)
    await server.start()
    url = server.session_url(SESSION_ID)

    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            await server.stop()
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(url)
            await page.wait_for_selector(".kc-entry")

            # Entries render from the initial HTTP fetch while the live socket is
            # still CONNECTING, so the first rendered state is "detached"; the
            # badge only flips to "following" when the socket opens. Asserting
            # right after the entries appeared raced on loaded CI runners.
            opened = await _wait_for(page, lambda snap: snap["liveState"] == "following")
            assert opened["liveState"] == "following", "the player never attached its live stream"
            assert not opened["liveHidden"], "a live session must advertise itself"

            # Appended only now, with the page already open.
            await _append(
                events, KnownEventType.THINKING_DELTA, elapsed_ms=200, uuid="th", text="and after.", complete=True
            )
            await _append(
                events, KnownEventType.ASSISTANT_DELTA, elapsed_ms=300, uuid="as", text="Answered live.", complete=True
            )

            arrived = await _wait_for(page, lambda snap: "Answered live." in snap["text"])
            assert "before and after." in arrived["text"], "reasoning must accumulate across the join"
            assert "Answered live." in arrived["text"]

            # Scrubbing back releases the follow, and the view stays put.
            await page.eval_on_selector("#scrub", "(el) => { el.value = '0'; el.dispatchEvent(new Event('input')); }")
            behind = await page.evaluate(_SNAPSHOT_JS)
            assert behind["liveState"] == "behind"
            await _append(
                events,
                KnownEventType.ASSISTANT_DELTA,
                elapsed_ms=400,
                uuid="as2",
                text="Missed while reading.",
                complete=True,
            )
            await asyncio.sleep(0.6)
            held = await page.evaluate(_SNAPSHOT_JS)
            assert "Missed while reading." not in held["text"], "a viewer reading history must not be yanked forward"

            # ...and the button brings them back to the edge.
            await page.click("#live")
            caught_up = await _wait_for(page, lambda snap: "Missed while reading." in snap["text"])
            assert caught_up["liveState"] == "following"
        finally:
            await browser.close()
            await server.stop()

    assert not errors, f"the player raised: {errors}"


@pytest.mark.asyncio
async def test_a_finished_session_opens_at_the_start_and_does_not_claim_to_be_live(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "cli", {}, session_id=SESSION_ID, title="done")
    events = FileSessionEventStore(store.journal(SESSION_ID))
    await _append(events, KnownEventType.TURN_STARTED, elapsed_ms=0, turn_id="t1", user_text="all done")
    await _append(events, KnownEventType.ASSISTANT_DELTA, elapsed_ms=100, uuid="as", text="Finished.", complete=True)
    await _append(events, KnownEventType.TURN_ENDED, elapsed_ms=200, turn_id="t1", status="completed")

    server = ShareServer(store)
    await server.start()

    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            await server.stop()
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            await page.goto(server.session_url(SESSION_ID))
            await page.wait_for_selector(".kc-entry")
            snapshot = await page.evaluate(_SNAPSHOT_JS)
        finally:
            await browser.close()
            await server.stop()

    # The socket stays attachable forever, so being attached is not evidence of
    # life. Nothing arrived, so the player must not offer to "jump to live", and
    # must open where a replay opens rather than at the end.
    assert snapshot["liveHidden"], "a session with nothing happening must not claim to be live"
    assert "Finished." not in snapshot["text"], "a closed session opens at the beginning"


@pytest.mark.asyncio
async def test_a_served_session_renders_an_image_from_the_artifact_endpoint(tmp_path: Path) -> None:
    """A shared link should show a screenshot, not a badge describing one.

    Image bytes are not inline in a served session the way they are in a
    single-file export, but the artifact endpoint has always been there — gated
    on the same token and the same shareable-purpose allowlist. The player was
    simply not asking it for them.
    """
    import base64

    from kolega_code.cli.session_event_store import FileArtifactStore
    from kolega_code.events import ArtifactPurpose

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "cli", {}, session_id=SESSION_ID, title="pictures")
    journal = store.journal(SESSION_ID)
    ref = await FileArtifactStore(journal).put(
        png, media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64"
    )
    events = FileSessionEventStore(journal)
    await _append(events, KnownEventType.TURN_STARTED, elapsed_ms=0, turn_id="t1", user_text="look")
    shot = AgentEvent(
        session_id=SESSION_ID,
        sender="agent",
        event_type=KnownEventType.CHAT_MESSAGE,
        content={"message_type": "tool_result", "text": "# shot.png", "tool_description": "read_image"},
        elapsed_ms=100,
    )
    shot.artifacts = [ref]
    await events.append(shot)
    await _append(events, KnownEventType.TURN_ENDED, elapsed_ms=200, turn_id="t1", status="completed")

    server = ShareServer(store, session_id=SESSION_ID)
    await server.start()
    url = server.session_url(SESSION_ID)

    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            await server.stop()
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            page = await browser.new_page()
            failed: list[str] = []
            page.on("requestfailed", lambda request: failed.append(request.url))
            await page.goto(url)
            await page.wait_for_selector(".kc-entry")
            await page.eval_on_selector(
                "#scrub", "(el) => { el.value = el.max; el.dispatchEvent(new Event('input')); }"
            )
            shown = await _wait_for_image(page)
        finally:
            await browser.close()
            await server.stop()

    assert failed == [], f"the player requested something that 404'd: {failed}"
    assert shown is not None, "a served session still shows a badge where the image should be"
    assert shown["naturalWidth"] == 1, "the artifact endpoint did not return decodable image bytes"
    assert f"/api/sessions/{SESSION_ID}/artifacts/{ref.sha256}" in shown["src"]


async def _wait_for_image(page: Any, *, timeout: float = 10.0):
    read = """
    () => {
      const image = document.querySelector('#transcript img.kc-artifact-image');
      if (!image) return null;
      return {src: image.getAttribute('src'), naturalWidth: image.naturalWidth, complete: image.complete};
    }
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        found = await page.evaluate(read)
        if found and found["complete"]:
            return found
        await asyncio.sleep(0.1)
    return await page.evaluate(read)


@pytest.mark.asyncio
async def test_losing_the_network_stops_the_badge_claiming_to_be_live(tmp_path: Path) -> None:
    """A frozen transcript under a "LIVE" badge is the one lie it must not tell.

    Dropping a connection does not close the WebSocket — the browser holds it
    open until TCP gives up, which can be minutes — so the badge kept reading
    LIVE over a view that had stopped receiving anything.
    """
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "cli", {}, session_id=SESSION_ID, title="live")
    events = FileSessionEventStore(store.journal(SESSION_ID))
    await _append(events, KnownEventType.TURN_STARTED, elapsed_ms=0, turn_id="t1", user_text="watch")
    await _append(events, KnownEventType.ASSISTANT_DELTA, elapsed_ms=100, uuid="as", text="working", complete=False)

    server = ShareServer(store, session_id=SESSION_ID)
    await server.start()
    url = server.session_url(SESSION_ID)

    async with playwright_api.async_playwright() as driver:
        try:
            browser = await driver.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            await server.stop()
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url)
            await page.wait_for_selector(".kc-entry")
            attached = await _wait_for(page, lambda snap: snap["liveState"] == "following")
            assert attached["liveState"] == "following", "the player never attached its live stream"

            await context.set_offline(True)
            dropped = await _wait_for(page, lambda snapshot: snapshot["liveState"] != "following")
            assert dropped["liveState"] == "detached", "an offline viewer was still told the session was live"
            assert "LIVE" not in dropped["live"]

            await context.set_offline(False)
            back = await _wait_for(page, lambda snapshot: snapshot["liveState"] == "following", timeout=15.0)
            assert back["liveState"] == "following", "the viewer never reattached after the network returned"
        finally:
            await browser.close()
            await server.stop()
