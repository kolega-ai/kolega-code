from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, expect

from kolega_code.web.bundle import export_bundle
from tests.web.test_player_task_list import (
    SESSION_ID,
    WORKER,
    _Recording,
    _delivered,
    _exported,
    _focus,
    _open,
    _page,
    _recording,
    _seek,
    _served,
    _static_url,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

QUEUE_EVENT = "queue_updated"
CAVEAT = "Historical preview only — not recoverable pending input or semantic-delivery confirmation."
HOSTILE = (
    '<img src="https://queue.invalid/image" onerror="globalThis.queueExecuted=true">\n'
    '<iframe src="https://queue.invalid/frame"></iframe>\n'
    '<style>@import url("https://queue.invalid/style");</style>\n'
    '<svg onload="globalThis.queueExecuted=true"></svg>\n'
    "![remote](https://queue.invalid/markdown)\n"
    "[execute](javascript:globalThis.queueExecuted=true)\n"
    "</script><script>globalThis.queueExecuted=true</script>"
)
FIRST = {
    "queue_id": "entry-a",
    "text": "First preview\n  Keep literal 測試 e\u0301 **Markdown**",
    "image_count": 0,
    "origin": None,
}
PEER = {
    "queue_id": "entry-b",
    "text": "Second preview",
    "image_count": 1,
    "origin": {"kind": "peer", "session_id": "peer-session-b", "title": "Actual peer title"},
}
IMAGES = {"queue_id": "entry-images", "text": "", "image_count": 3, "origin": None}


async def _queue(
    recording: _Recording,
    items: list[dict[str, Any]],
    revision: int,
    *,
    reason: str = "snapshot",
    source: str = "tui",
    generation: str = "publisher-generation-a",
    elapsed_ms: int | None = None,
    info: dict[str, Any] | None = None,
) -> None:
    await recording.append(
        QUEUE_EVENT,
        info=info,
        elapsed_ms=elapsed_ms,
        queue_id=generation,
        revision=revision,
        source=source,
        reason=reason,
        items=items,
    )


async def _panel(
    page: Page,
    items: list[dict[str, Any]] | None,
    *,
    reason: str = "snapshot",
    source: str = "TUI",
) -> None:
    panel = page.locator("#queue")
    await expect(panel).to_have_attribute(
        "data-state", "unknown" if items is None else "populated" if items else "empty"
    )
    await expect(panel.locator(".kc-queue-caveat")).to_have_text(CAVEAT)
    await expect(panel.locator(".kc-queue-caveat")).to_be_visible()
    await expect(panel.locator(".kc-queue-summary")).to_have_text(
        "Session-wide"
        if items is None
        else f"Session-wide · {len(items)} recorded pending {'input' if len(items) == 1 else 'inputs'}"
    )
    if items is None:
        await expect(panel.locator(".kc-empty")).to_have_text("Queue unknown. No queue recorded yet.")
        await expect(panel.locator(".kc-queue-observation")).to_have_count(0)
    else:
        observations = {
            "snapshot": "Snapshot recorded.",
            "enqueued": "Input enqueued.",
            "dequeued": "Dequeued — removed for processing, not delivery confirmation.",
            "cleared": "Queue cleared.",
            "removed": "Input removed.",
        }
        await expect(panel.locator(".kc-queue-observation")).to_have_text(f"{source} · {observations[reason]}")
        if not items:
            await expect(panel.locator(".kc-empty")).to_have_text("Recorded queue is empty.")
        else:
            await expect(panel.locator(".kc-empty")).to_have_count(0)
            await expect(panel.locator("ol.kc-queue-entries > li")).to_have_count(len(items))
            assert await panel.locator(".kc-queue-text").all_text_contents() == [item["text"] for item in items]
            assert await panel.locator(".kc-queue-origin").all_text_contents() == [
                f"Peer: {item['origin']['title']} ({item['origin']['session_id']})"
                if item["origin"] is not None
                else "Origin not recorded"
                for item in items
            ]
            assert await panel.locator(".kc-queue-images").all_text_contents() == [
                f"{item['image_count']} {'image' if item['image_count'] == 1 else 'images'}" for item in items
            ]
    await expect(panel.locator(".kc-queue-entry")).to_have_count(len(items or []))
    assert (
        await panel.locator(
            "input, button, select, textarea, a, [role=button], [contenteditable], [onclick], [tabindex]"
        ).count()
        == 0
    )
    assert await panel.locator("script, img, iframe, svg, style, link, [src], [href], [download]").count() == 0


async def test_live_fifo_full_snapshots_empty_clear_and_generation_reload(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    async with _served(recording) as page:
        await _panel(page, None)
        await _queue(recording, [], 0)
        await _delivered(page, recording)
        await _panel(page, [])
        for items, revision, reason in [
            ([FIRST], 1, "enqueued"),
            ([FIRST, PEER, IMAGES], 2, "enqueued"),
            ([PEER, IMAGES], 3, "dequeued"),
            ([IMAGES], 4, "removed"),
            ([], 5, "cleared"),
        ]:
            await _queue(recording, items, revision, reason=reason)
            await _delivered(page, recording)
            await _panel(page, items, reason=reason)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await expect(page.locator("#turns .kc-turn")).to_have_count(1)
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _panel(page, [], reason="cleared")
        await _queue(recording, [FIRST, PEER], 0, source="ask", generation="publisher-generation-b")
        await _delivered(page, recording)
        await _panel(page, [FIRST, PEER], source="ask")
        assert await page.locator(".kc-queue-text").first.evaluate(
            "(node) => getComputedStyle(node).whiteSpace === 'pre-wrap' && "
            "getComputedStyle(node).overflowWrap === 'anywhere'"
        )
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _panel(page, [FIRST, PEER], source="ask")


async def test_queue_remains_session_wide_during_child_focus_and_live_scrubbing(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append("assistant_delta", elapsed_ms=1000, info=WORKER, text="worker answer", complete=True)
    async with _served(recording) as page:
        await _focus(page)
        await _panel(page, None)
        await _queue(recording, [FIRST], 0, elapsed_ms=2000, info=WORKER)
        await _delivered(page, recording)
        await _panel(page, None)
        await _queue(recording, [FIRST, PEER], 0, elapsed_ms=3000)
        await _delivered(page, recording)
        await _panel(page, [FIRST, PEER])
        await _queue(recording, [], 1, reason="cleared", elapsed_ms=4000, info=WORKER)
        await _delivered(page, recording)
        await _panel(page, [FIRST, PEER])
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
        await _seek(page, 750)
        await _panel(page, [FIRST, PEER])
        await _queue(recording, [PEER], 1, reason="dequeued", elapsed_ms=5000)
        await _delivered(page, recording)
        await _panel(page, [FIRST, PEER])
        await expect(page.locator("#live")).to_have_attribute("data-state", "behind")
        await page.locator("#live").click()
        await _panel(page, [PEER], reason="dequeued")
        await page.locator(".kc-agent-back").click()
        await _panel(page, [PEER], reason="dequeued")
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await _focus(page)
        await _queue(recording, [], 2, reason="cleared", elapsed_ms=6000)
        await _delivered(page, recording)
        await _panel(page, [], reason="cleared")


async def _replay_recording(directory: Path) -> _Recording:
    recording = await _recording(directory)
    await _queue(recording, [], 0, elapsed_ms=1000)
    await _queue(recording, [FIRST, PEER, IMAGES], 1, reason="enqueued", elapsed_ms=2000)
    await recording.append("assistant_delta", elapsed_ms=3000, info=WORKER, text="worker answer", complete=True)
    await _queue(recording, [PEER, IMAGES], 2, reason="dequeued", elapsed_ms=4000)
    await _queue(recording, [], 3, reason="cleared", elapsed_ms=5000)
    await recording.append("turn_ended", elapsed_ms=6000, turn_id="root-turn", status="completed")
    return recording


@pytest.mark.parametrize("mode", ["served", "directory", "inline"])
async def test_replay_seeks_rebuilds_and_reloads_queue_observations(tmp_path: Path, mode: str) -> None:
    recording = await _replay_recording(tmp_path)
    opened = (
        _served(recording, live=False)
        if mode == "served"
        else _exported(recording, tmp_path, single_file=mode == "inline")
    )
    async with opened as page:
        await _panel(page, None)
        await _seek(page, 180)
        await _panel(page, [])
        await _seek(page, 520)
        await _panel(page, [FIRST, PEER, IMAGES], reason="enqueued")
        await _focus(page)
        await _panel(page, [FIRST, PEER, IMAGES], reason="enqueued")
        await _seek(page, 680)
        await _panel(page, [PEER, IMAGES], reason="dequeued")
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
        await _seek(page, 1000)
        await _panel(page, [], reason="cleared")
        await _seek(page, 350)
        await _panel(page, [FIRST, PEER, IMAGES], reason="enqueued")
        assert "not present at this position" in await page.locator("#agentHeader").inner_text()
        await _seek(page, 0)
        await _panel(page, None)
        await page.locator(".kc-agent-back").click()
        await _seek(page, 350)
        await _panel(page, [FIRST, PEER, IMAGES], reason="enqueued")
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        for _ in range(3):
            await page.locator("#speed").click()
        await page.locator("#play").click()
        await _panel(page, [], reason="cleared")
        await page.reload()
        await expect(page.locator("#transcript .kc-entry").first).to_be_visible()
        await _panel(page, None)
        await _seek(page, 680)
        await _panel(page, [PEER, IMAGES], reason="dequeued")
        if mode != "served":
            await expect(page.locator("#live")).to_be_hidden()


@pytest.mark.parametrize("mode", ["live", "directory", "inline"])
async def test_hostile_previews_and_peer_attribution_are_literal_without_fetches(tmp_path: Path, mode: str) -> None:
    recording = await _recording(tmp_path)
    item = {
        "queue_id": "hostile-entry",
        "text": HOSTILE,
        "image_count": 2,
        "origin": {"kind": "peer", "session_id": HOSTILE, "title": HOSTILE},
    }
    if mode != "live":
        await _queue(recording, [item, IMAGES], 0)
        await recording.append("turn_ended", turn_id="root-turn", status="completed")
    opened = _served(recording) if mode == "live" else _exported(recording, tmp_path, single_file=mode == "inline")
    async with opened as page:
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url))
        await page.route("https://queue.invalid/**", lambda route: route.abort())
        if mode == "live":
            await _queue(recording, [item, IMAGES], 0)
            await _delivered(page, recording)
        else:
            await _seek(page, 1000)
        await _panel(page, [item, IMAGES])
        assert await page.evaluate("globalThis.queueExecuted === undefined")
        assert requests == []
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await expect(page.locator("#turns .kc-turn")).to_have_count(1)
        await expect(page.locator("#subAgents [data-agent-key]")).to_have_count(0)


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "inline"])
async def test_export_redacts_queue_previews_and_peer_metadata_without_mutating_recording(
    tmp_path: Path, single_file: bool
) -> None:
    recording = await _recording(tmp_path)
    secret = "obviously-fake-queue-secret-for-export-test"
    home = str(Path.home())
    item = {
        "queue_id": "redacted-entry",
        "text": f"Preview {secret} in {home}/queue-preview.txt",
        "image_count": 1,
        "origin": {"kind": "peer", "session_id": f"peer-{secret}", "title": f"Peer {secret} at {home}"},
    }
    await _queue(recording, [item], 0)
    await recording.append("turn_ended", turn_id="root-turn", status="completed")
    result = await export_bundle(
        recording.recorded,
        tmp_path / ("replay.html" if single_file else "bundle"),
        session_id=SESSION_ID,
        single_file=single_file,
        extra_secrets=[secret],
    )
    assert result.report.strings_redacted >= 3
    assert recording.recorded[1].content["items"][0] == item
    with nullcontext(result.path.as_uri()) if single_file else _static_url(result.path) as url:
        async with _page() as page:
            await _open(page, url)
            await _seek(page, 1000)
            panel = page.locator("#queue")
            await expect(panel).to_have_attribute("data-state", "populated")
            content = await panel.text_content()
            assert content is not None
            assert secret not in content
            assert home not in content
            assert "~/queue-preview.txt" in content
            assert "‹secret›" in content
            assert await panel.locator(".kc-queue-images").text_content() == "1 image"
            assert await panel.locator(".kc-queue-origin").text_content() == (
                "Peer: Peer ‹secret› at ~ (peer-‹secret›)"
            )
            assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
