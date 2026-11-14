from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, expect

from kolega_code.cli.session_event_store import FileArtifactStore
from kolega_code.events import AgentEvent, ArtifactRef
from kolega_code.session.recording import _preview
from kolega_code.web.bundle import export_bundle
from tests.web.test_background_delivery_fold import DELIVERY, DELIVERY_EVENT, METADATA, WORKER
from tests.web.test_player_task_list import (
    SESSION_ID,
    _Recording,
    _delivered,
    _exported,
    _open,
    _page,
    _recording,
    _seek,
    _served,
    _static_url,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

HOSTILE = (
    '<img src="https://background-delivery.invalid/image" onerror="globalThis.deliveryExecuted=true">\n'
    '<iframe src="https://background-delivery.invalid/frame"></iframe>\n'
    '<style>@import url("https://background-delivery.invalid/style");</style>\n'
    '<svg onload="globalThis.deliveryExecuted=true"></svg>\n'
    "[execute](javascript:globalThis.deliveryExecuted=true)\n"
    "</script><script>globalThis.deliveryExecuted=true</script>"
)
LONG_REPORT = "Complete report 測試 e\u0301\n" + "Full report text, not a preview.\n" * 1200 + "VISIBLE REPORT END"


async def _worker_stopped(recording: _Recording) -> None:
    await recording.append("assistant_delta", info=WORKER, text="Worker-only answer", complete=True)
    await recording.append("chat_message", info=WORKER, status="STOPPED", message="Worker finished")
    await recording.append("turn_ended", turn_id="root-turn", status="completed")
    await recording.append("turn_started", turn_id="later-turn", user_text="A later request")


async def _delivery(recording: _Recording, **changes: Any) -> None:
    await recording.append(DELIVERY_EVENT, **{**DELIVERY, **changes})


async def _assert_delivery(page: Page, text: str, *, archived: bool = False, completion: str = "completion-a") -> None:
    report = page.locator(f'.kc-background-delivery[data-completion-id="{completion}"]')
    await expect(report).to_have_count(1)
    await expect(report.locator(".kc-background-report")).to_have_text(text)
    assert await report.locator(".kc-background-report").text_content() == text
    await expect(report.locator(".kc-background-disposition")).to_have_text(
        "Archived — NOT added to live context." if archived else "Delivered — added to live context."
    )
    assert await report.get_attribute("data-disposition") == ("archived" if archived else "context")
    assert await report.locator(".kc-background-report").evaluate(
        "(node) => getComputedStyle(node).whiteSpace === 'pre-wrap' && "
        "getComputedStyle(node).overflowWrap === 'anywhere'"
    )


async def _assert_attribution(page: Page) -> None:
    report = page.locator('.kc-background-delivery[data-completion-id="completion-a"]')
    await expect(report.locator(".kc-background-heading")).to_have_text("Background report · general · completed")
    await expect(report.locator(".kc-background-task")).to_have_text("Task: Inspect the build")
    await report.locator("summary").click()
    facts = await report.locator(".kc-background-details").inner_text()
    for fact in (
        "Job: job-a",
        "Completion: completion-a",
        "Epoch: epoch-a",
        "Launch turn: launch-turn",
        "Parent tool call: launch-call",
        'Effective routing: {"provider":"offline","model":"worker-model"}',
        "Workspace: workspace-a",
        "Thread: thread-a",
        "Project: /example/project",
    ):
        assert fact in facts
    assert (
        await report.locator(".kc-background-metadata").evaluate("(node) => JSON.parse(node.textContent)") == METADATA
    )


async def test_live_completion_is_not_delivery_then_later_turn_report_survives_reload_and_scrub(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await _worker_stopped(recording)
    async with _served(recording) as page:
        await expect(page.locator(".kc-background-delivery")).to_have_count(0)
        await expect(page.locator(".kc-agent-card")).to_have_attribute("data-status", "completed")
        await _delivery(recording)
        await _delivered(page, recording)
        await _assert_delivery(page, DELIVERY["text"])
        await _assert_attribution(page)
        await _delivery(recording, text="Duplicated report must never appear")
        await _delivery(
            recording,
            job_id="job-b",
            completion_id="completion-b",
            status="cancelled",
            disposition="archived",
            text="Archived report",
        )
        await _delivered(page, recording)
        await expect(page.locator(".kc-background-delivery")).to_have_count(2)
        await _assert_delivery(page, "Archived report", archived=True, completion="completion-b")
        assert "Duplicated report" not in await page.locator("#transcript").inner_text()
        await page.locator('#subAgents [data-agent-key="background-worker"]').click()
        await expect(page.locator("#agentHeader")).to_be_visible()
        await expect(page.locator(".kc-background-delivery")).to_have_count(0)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["Worker-only answer"]
        facts = await page.locator("#agentHeader").inner_text()
        assert 'Effective routing: {"provider":"offline","model":"worker-model"}' in facts
        await page.locator(".kc-agent-back").click()
        await _assert_delivery(page, DELIVERY["text"])
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _assert_delivery(page, DELIVERY["text"])
        await _assert_delivery(page, "Archived report", archived=True, completion="completion-b")
        await _seek(page, 500)
        await expect(page.locator(".kc-background-delivery")).to_have_count(0)
        await expect(page.locator(".kc-agent-card")).to_have_attribute("data-status", "completed")
        await page.locator("#live").click()
        await expect(page.locator(".kc-background-delivery")).to_have_count(2)
        await _assert_delivery(page, DELIVERY["text"])


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "inline"])
async def test_exports_scrub_replay_complete_reports_archival_and_unknown_legacy_metadata(
    tmp_path: Path, single_file: bool
) -> None:
    recording = await _recording(tmp_path)
    await _worker_stopped(recording)
    await _delivery(recording, text=LONG_REPORT)
    await _delivery(recording, text="Duplicate must stay hidden")
    await _delivery(
        recording,
        job_id="archive-job",
        completion_id="archive-completion",
        disposition="archived",
        status="failed",
        text="Archived failure",
        metadata={},
        epoch_id=None,
    )
    await recording.append("turn_ended", turn_id="later-turn", status="completed")
    async with _exported(recording, tmp_path, single_file=single_file) as page:
        await expect(page.locator("#live")).to_be_hidden()
        await _seek(page, 500)
        await expect(page.locator(".kc-background-delivery")).to_have_count(0)
        await _seek(page, 625)
        await _assert_delivery(page, LONG_REPORT)
        await _assert_attribution(page)
        await _seek(page, 1000)
        await expect(page.locator(".kc-background-delivery")).to_have_count(2)
        await _assert_delivery(page, "Archived failure", archived=True, completion="archive-completion")
        legacy = page.locator('.kc-background-delivery[data-completion-id="archive-completion"]')
        await legacy.locator("summary").click()
        facts = await legacy.locator(".kc-background-details").inner_text()
        for label in (
            "Epoch",
            "Launch turn",
            "Parent tool call",
            "Effective routing",
            "Workspace",
            "Thread",
            "Project",
        ):
            assert f"{label}: unknown" in facts
        assert await legacy.locator(".kc-background-metadata").text_content() == "{}"
        await page.locator('#subAgents [data-agent-key="background-worker"]').click()
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["Worker-only answer"]
        await page.locator(".kc-agent-back").click()
        await _seek(page, 500)
        await expect(page.locator(".kc-background-delivery")).to_have_count(0)
        await page.locator("#speed").click()
        await page.locator("#speed").click()
        await page.locator("#speed").click()
        await page.locator("#play").click()
        await expect(page.locator(".kc-background-delivery")).to_have_count(2)
        await _assert_delivery(page, LONG_REPORT)
        await page.reload()
        await expect(page.locator("#transcript .kc-entry").first).to_be_visible()
        await _seek(page, 1000)
        await expect(page.locator(".kc-background-delivery")).to_have_count(2)
        await _assert_delivery(page, LONG_REPORT)


@pytest.mark.parametrize("mode", ["live", "directory", "inline"])
async def test_reports_and_attribution_are_literal_and_do_not_execute_or_request_remote_resources(
    tmp_path: Path, mode: str
) -> None:
    recording = await _recording(tmp_path)
    hostile_delivery = {
        "agent_type": HOSTILE,
        "task": HOSTILE,
        "text": HOSTILE,
        "metadata": {"effective_routing": {"model": HOSTILE}, "project_path": HOSTILE},
    }
    if mode != "live":
        await _delivery(recording, **hostile_delivery)
    opened = _served(recording) if mode == "live" else _exported(recording, tmp_path, single_file=mode == "inline")
    async with opened as page:
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url))
        await page.route("https://background-delivery.invalid/**", lambda route: route.abort())
        if mode == "live":
            await _delivery(recording, **hostile_delivery)
            await _delivered(page, recording)
        else:
            await _seek(page, 1000)
        await _assert_delivery(page, HOSTILE)
        report = page.locator(".kc-background-delivery")
        await report.locator("summary").click()
        assert HOSTILE in await report.locator(".kc-background-details").inner_text()
        assert await report.locator("script, img, iframe, svg, style, link, [src], [href]").count() == 0
        assert await page.evaluate("globalThis.deliveryExecuted === undefined")
        assert requests == []
        assert await page.locator("#subAgents [data-agent-key]").count() == 0
        await _delivery(recording, **hostile_delivery)
        if mode == "live":
            await _delivered(page, recording)
        await expect(report).to_have_count(1)


@pytest.mark.parametrize("mode", ["live", "directory", "inline"])
async def test_offloaded_report_is_loaded_completely_as_text(tmp_path: Path, mode: str) -> None:
    recording = await _recording(tmp_path)
    artifacts = FileArtifactStore(recording.store.journal(SESSION_ID))
    text = LONG_REPORT + "\n" + HOSTILE
    ref = await artifacts.put(
        text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        purpose="tool_result",
        encoding="utf-8",
        chars=len(text),
    )
    preview = _preview(text, 256, ref)
    event = AgentEvent(
        session_id=SESSION_ID,
        sender="agent",
        event_type=DELIVERY_EVENT,
        elapsed_ms=100,
        content={**DELIVERY, "text": preview},
        artifacts=[ref],
    )
    await recording.events.append(event)
    recording.recorded.append(event)
    if mode == "live":
        async with _served(recording) as page:
            await _assert_delivery(page, text)
            await expect(page.locator(".kc-background-artifact-notice")).to_have_count(0)
            assert await page.locator(".kc-background-report img, .kc-background-report script").count() == 0
            assert await page.evaluate("globalThis.deliveryExecuted === undefined")
            await page.reload()
            await expect(page.locator("#live")).to_have_attribute("data-state", "following")
            await _assert_delivery(page, text)
    else:
        result = await export_bundle(
            recording.recorded,
            tmp_path / "bundle",
            session_id=SESSION_ID,
            artifact_store=artifacts,
            single_file=mode == "inline",
        )
        with _static_url(result.path) if mode == "directory" else nullcontext(result.path.as_uri()) as url:
            async with _page() as page:
                await _open(page, url)
                await _seek(page, 1000)
                await _assert_delivery(page, text)
                await _seek(page, 0)
                await expect(page.locator(".kc-background-delivery")).to_have_count(0)
                await _seek(page, 1000)
                await _assert_delivery(page, text)
                assert await page.locator(".kc-background-report img, .kc-background-report script").count() == 0
                assert await page.evaluate("globalThis.deliveryExecuted === undefined")


@pytest.mark.parametrize("mode", ["live", "directory", "inline"])
async def test_missing_report_artifact_keeps_an_explicitly_labelled_recorded_preview(tmp_path: Path, mode: str) -> None:
    recording = await _recording(tmp_path)
    ref = ArtifactRef(
        sha256="f" * 64,
        bytes=len(LONG_REPORT.encode()),
        media_type="text/plain",
        purpose="tool_result",
        encoding="utf-8",
        chars=len(LONG_REPORT),
    )
    preview = _preview(LONG_REPORT, 256, ref)
    event = AgentEvent(
        session_id=SESSION_ID,
        sender="agent",
        event_type=DELIVERY_EVENT,
        elapsed_ms=100,
        content={**DELIVERY, "text": preview},
        artifacts=[ref],
    )
    await recording.events.append(event)
    recording.recorded.append(event)
    opened = _served(recording) if mode == "live" else _exported(recording, tmp_path, single_file=mode == "inline")
    async with opened as page:
        if mode != "live":
            await _seek(page, 1000)
        await _assert_delivery(page, preview)
        await expect(page.locator(".kc-background-artifact-notice")).to_have_text(
            "Complete report unavailable; recorded preview shown."
        )
        await expect(page.locator("#subAgents [data-agent-key]")).to_have_count(0)
