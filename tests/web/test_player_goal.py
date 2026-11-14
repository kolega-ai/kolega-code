from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, expect

from kolega_code.web.bundle import export_bundle
from tests.web.test_goal_fold import GOAL
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

CAVEAT = "Read-only historical observation — not execution authority or recoverable goal state."
HOSTILE = (
    '<img src="https://goal.invalid/image" onerror="globalThis.goalExecuted=true">\n'
    '<iframe src="https://goal.invalid/frame"></iframe>\n'
    '<style>@import url("https://goal.invalid/style");</style>\n'
    '<svg onload="globalThis.goalExecuted=true"></svg>\n'
    "![remote](https://goal.invalid/markdown)\n"
    "[execute](javascript:globalThis.goalExecuted=true)\n"
    "</script><script>globalThis.goalExecuted=true</script>"
)


async def _goal(
    recording: _Recording,
    goal: dict[str, Any] | None,
    revision: int,
    *,
    reason: str = "snapshot",
    source: str = "tui",
    generation: str = "goal-generation-a",
    evaluation_met: bool | None = None,
    elapsed_ms: int | None = None,
    info: dict[str, Any] | None = None,
) -> None:
    await recording.append(
        "goal_updated",
        info=info,
        elapsed_ms=elapsed_ms,
        generation_id=generation,
        revision=revision,
        source=source,
        reason=reason,
        goal=deepcopy(goal),
        evaluation_met=evaluation_met,
    )


async def _panel(
    page: Page,
    status: str,
    goal: dict[str, Any] | None = None,
    *,
    reason: str = "snapshot",
    source: str = "TUI",
    evaluation_met: bool | None = None,
) -> None:
    panel = page.locator("#goal")
    labels = {
        "unknown": "Goal unknown",
        "no-goal": "No goal",
        "active": "Active",
        "paused": "Paused",
        "met": "Met",
        "ended-unmet": "Ended — unmet",
    }
    await expect(panel).to_have_attribute("data-state", status)
    await expect(panel.locator(".kc-goal-summary")).to_have_text(f"Session-wide · {labels[status]}")
    await expect(panel.locator(".kc-goal-caveat")).to_have_text(CAVEAT)
    await expect(panel.locator(".kc-goal-caveat")).to_be_visible()
    await expect(panel.locator(".kc-goal-observation")).to_have_text(
        "Source / reason unavailable" if status == "unknown" else f"{source} · {reason}"
    )
    assert await panel.locator(".kc-goal-condition").text_content() == (
        "Condition unavailable." if goal is None else goal["condition"]
    )
    values = {
        "turns_evaluated": "Unavailable" if goal is None else str(goal["turns_evaluated"]),
        "tokens_spent": "Unavailable" if goal is None else str(goal["tokens_spent"]),
        "max_turns": "Unavailable" if goal is None else str(goal["max_turns"]),
        "run_to_completion": "Unavailable" if goal is None else "Yes" if goal["run_to_completion"] else "No",
        "evaluation_met": "Yes" if evaluation_met is True else "No" if evaluation_met is False else "Unavailable",
        **{
            field: "Unavailable" if goal is None else goal[field] or "Unavailable"
            for field in ("last_reason", "status_note", "started_at", "last_evaluated_at")
        },
    }
    for field, value in values.items():
        await expect(panel.locator(f'[data-field="{field}"]')).to_have_text(value)
        assert await panel.locator(f'[data-field="{field}"]').text_content() == value
    assert await panel.locator(".kc-goal-value").count() == len(values)
    assert (
        await panel.locator(
            "input, button, select, textarea, a, [role=button], [contenteditable], [onclick], [tabindex], "
            "script, img, iframe, svg, style, link, [src], [href], [download]"
        ).count()
        == 0
    )


async def test_live_lifecycle_verdicts_completed_unmet_generation_changes_and_reload(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    evaluated = {
        **GOAL,
        "turns_evaluated": 1,
        "tokens_spent": 123,
        "last_reason": "Still one check remaining.",
        "last_evaluated_at": "2026-11-12T10:01:00+00:00",
    }
    paused = {**evaluated, "paused": True, "status_note": "Stopped by user"}
    rtc = {**evaluated, "run_to_completion": True, "turns_evaluated": 2}
    met = {**rtc, "met": True, "last_reason": "All checks passed."}
    async with _served(recording) as page:
        await _panel(page, "unknown")
        await _goal(recording, None, 0)
        await _delivered(page, recording)
        await _panel(page, "no-goal")
        for goal, revision, reason, verdict, status in [
            (GOAL, 1, "started", None, "active"),
            (evaluated, 2, "evaluated", False, "active"),
            (paused, 3, "paused", False, "paused"),
            (evaluated, 4, "resumed", False, "active"),
            (rtc, 5, "evaluated", True, "active"),
            (met, 6, "completed", True, "met"),
            (None, 7, "cleared", None, "no-goal"),
            (GOAL, 8, "replaced", None, "active"),
            (evaluated, 9, "completed", False, "ended-unmet"),
        ]:
            await _goal(recording, goal, revision, reason=reason, source="ask", evaluation_met=verdict)
            await _delivered(page, recording)
            await _panel(page, status, goal, reason=reason, source="ask", evaluation_met=verdict)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await expect(page.locator("#turns .kc-turn")).to_have_count(1)
        await expect(page.locator("#subAgents [data-agent-key]")).to_have_count(0)
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _panel(page, "ended-unmet", evaluated, reason="completed", source="ask", evaluation_met=False)
        await _goal(recording, GOAL, 0, generation="goal-generation-b")
        await _delivered(page, recording)
        await _panel(page, "active", GOAL)
        assert await page.locator(".kc-goal-condition").evaluate(
            "(node) => getComputedStyle(node).whiteSpace === 'pre-wrap' && "
            "getComputedStyle(node).overflowWrap === 'anywhere'"
        )
        await _goal(recording, None, 10, reason="cleared")
        await _delivered(page, recording)
        await _panel(page, "active", GOAL)
        await page.reload()
        await expect(page.locator("#live")).to_have_attribute("data-state", "following")
        await _panel(page, "active", GOAL)


async def test_session_wide_goal_ignores_child_updates_and_live_scrubbing_keeps_historical_prefix(
    tmp_path: Path,
) -> None:
    recording = await _recording(tmp_path)
    paused = {**GOAL, "paused": True, "status_note": "Paused"}
    await recording.append("assistant_delta", elapsed_ms=1000, info=WORKER, text="worker answer", complete=True)
    async with _served(recording) as page:
        await _focus(page)
        await _panel(page, "unknown")
        await _goal(recording, GOAL, 0, elapsed_ms=2000, info=WORKER)
        await _delivered(page, recording)
        await _panel(page, "unknown")
        await _goal(recording, GOAL, 0, elapsed_ms=3000)
        await _delivered(page, recording)
        await _panel(page, "active", GOAL)
        await _goal(recording, None, 1, reason="cleared", elapsed_ms=4000, info=WORKER)
        await _delivered(page, recording)
        await _panel(page, "active", GOAL)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
        await _seek(page, 750)
        await _panel(page, "active", GOAL)
        await _goal(recording, paused, 1, reason="paused", elapsed_ms=5000)
        await _delivered(page, recording)
        await _panel(page, "active", GOAL)
        await expect(page.locator("#live")).to_have_attribute("data-state", "behind")
        await page.locator("#live").click()
        await _panel(page, "paused", paused, reason="paused")
        await page.locator(".kc-agent-back").click()
        await _panel(page, "paused", paused, reason="paused")
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await _focus(page)
        await _goal(recording, None, 2, reason="cleared", elapsed_ms=6000)
        await _delivered(page, recording)
        await _panel(page, "no-goal", reason="cleared")
        await expect(page.locator("#turns .kc-turn")).to_have_count(1)
        await expect(page.locator("#subAgents [data-agent-key]")).to_have_count(1)


async def test_ignored_and_unrelated_events_do_not_rebuild_goal_dom(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await _goal(recording, GOAL, 0)
    async with _served(recording) as page:
        await _panel(page, "active", GOAL)
        await page.evaluate(
            """() => {
                globalThis.goalCondition = document.querySelector('.kc-goal-condition');
                globalThis.goalMutations = 0;
                new MutationObserver(records => { globalThis.goalMutations += records.length; })
                    .observe(document.getElementById('goal'),
                        {childList: true, subtree: true, characterData: true, attributes: true});
            }"""
        )
        await recording.append("assistant_delta", text="Unrelated streamed answer", complete=False)
        await _goal(recording, GOAL, 0)
        await _goal(recording, None, 1, reason="cleared", info={})
        await recording.append("status_update", message="Unrelated notice")
        await _delivered(page, recording)
        await _panel(page, "active", GOAL)
        assert await page.evaluate(
            "globalThis.goalMutations === 0 && "
            "document.querySelector('.kc-goal-condition') === globalThis.goalCondition"
        )
        await _goal(recording, None, 1, reason="cleared")
        await _delivered(page, recording)
        await _panel(page, "no-goal", reason="cleared")
        assert await page.evaluate("globalThis.goalMutations > 0")


@pytest.mark.parametrize("mode", ["served", "directory", "inline"])
async def test_replay_goal_prefixes_seek_playback_focus_and_reload_in_all_modes(tmp_path: Path, mode: str) -> None:
    recording = await _recording(tmp_path)
    rtc = {**GOAL, "run_to_completion": True, "turns_evaluated": 1, "last_reason": "Verifier says yes"}
    paused = {**rtc, "paused": True, "status_note": "Stopped"}
    replacement = {**GOAL, "condition": "Replacement goal"}
    await _goal(recording, None, 0, elapsed_ms=1000)
    await _goal(recording, GOAL, 1, reason="started", elapsed_ms=2000)
    await recording.append("assistant_delta", elapsed_ms=3000, info=WORKER, text="worker answer", complete=True)
    await _goal(recording, rtc, 2, reason="evaluated", evaluation_met=True, elapsed_ms=4000)
    await _goal(recording, paused, 3, reason="paused", evaluation_met=True, elapsed_ms=5000)
    await _goal(recording, rtc, 4, reason="resumed", evaluation_met=True, elapsed_ms=6000)
    await _goal(recording, rtc, 5, reason="completed", evaluation_met=True, elapsed_ms=7000)
    await _goal(recording, replacement, 6, reason="replaced", elapsed_ms=8000)
    await _goal(recording, None, 7, reason="cleared", elapsed_ms=9000)
    await recording.append("turn_ended", elapsed_ms=10000, turn_id="root-turn", status="completed")
    opened = (
        _served(recording, live=False)
        if mode == "served"
        else _exported(recording, tmp_path, single_file=mode == "inline")
    )
    async with opened as page:
        await _panel(page, "unknown")
        await _seek(page, 100)
        await _panel(page, "no-goal")
        await _seek(page, 200)
        await _panel(page, "active", GOAL, reason="started")
        await _seek(page, 400)
        await _panel(page, "active", rtc, reason="evaluated", evaluation_met=True)
        await _focus(page)
        await _panel(page, "active", rtc, reason="evaluated", evaluation_met=True)
        await _seek(page, 500)
        await _panel(page, "paused", paused, reason="paused", evaluation_met=True)
        await _seek(page, 600)
        await _panel(page, "active", rtc, reason="resumed", evaluation_met=True)
        await _seek(page, 700)
        await _panel(page, "ended-unmet", rtc, reason="completed", evaluation_met=True)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["worker answer"]
        await _seek(page, 800)
        await _panel(page, "active", replacement, reason="replaced")
        await _seek(page, 1000)
        await _panel(page, "no-goal", reason="cleared")
        await _seek(page, 200)
        await _panel(page, "active", GOAL, reason="started")
        assert "not present at this position" in await page.locator("#agentHeader").inner_text()
        await _seek(page, 0)
        await _panel(page, "unknown")
        await page.locator(".kc-agent-back").click()
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        for _ in range(3):
            await page.locator("#speed").click()
        await page.locator("#play").click()
        await _panel(page, "no-goal", reason="cleared")
        await expect(page.locator("#turns .kc-turn")).to_have_count(1)
        await page.reload()
        await expect(page.locator("#transcript .kc-entry").first).to_be_visible()
        await _panel(page, "unknown")
        await _seek(page, 500)
        await _panel(page, "paused", paused, reason="paused", evaluation_met=True)
        if mode != "served":
            await expect(page.locator("#live")).to_be_hidden()


@pytest.mark.parametrize("mode", ["live", "directory", "inline"])
async def test_all_free_text_is_literal_without_execution_requests_or_transcript_inflation(
    tmp_path: Path, mode: str
) -> None:
    recording = await _recording(tmp_path)
    hostile = {
        **GOAL,
        **{field: HOSTILE for field in ("condition", "started_at", "last_reason", "last_evaluated_at", "status_note")},
    }
    if mode != "live":
        await _goal(recording, hostile, 0)
        await recording.append("turn_ended", turn_id="root-turn", status="completed")
    opened = _served(recording) if mode == "live" else _exported(recording, tmp_path, single_file=mode == "inline")
    async with opened as page:
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url))
        await page.route("https://goal.invalid/**", lambda route: route.abort())
        if mode == "live":
            await _goal(recording, hostile, 0)
            await _delivered(page, recording)
        else:
            await _seek(page, 1000)
        await _panel(page, "active", hostile)
        await page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        assert await page.evaluate("globalThis.goalExecuted === undefined")
        assert requests == []
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        await expect(page.locator("#turns .kc-turn")).to_have_count(1)
        await expect(page.locator("#subAgents [data-agent-key]")).to_have_count(0)


@pytest.mark.parametrize("single_file", [False, True], ids=["directory", "inline"])
async def test_redacted_export_preserves_observation_and_does_not_mutate_source(
    tmp_path: Path, single_file: bool
) -> None:
    recording = await _recording(tmp_path)
    secret = "obviously-fake-goal-secret-for-export-test"
    home = str(Path.home())
    goal = {
        **GOAL,
        **{
            field: f"{field}: {secret} in {home}/goal.txt"
            for field in ("condition", "started_at", "last_reason", "last_evaluated_at", "status_note")
        },
    }
    await _goal(recording, goal, 0, reason="evaluated", evaluation_met=False)
    await recording.append("turn_ended", turn_id="root-turn", status="completed")
    original = deepcopy(recording.recorded[1].content)
    result = await export_bundle(
        recording.recorded,
        tmp_path / ("replay.html" if single_file else "bundle"),
        session_id=SESSION_ID,
        single_file=single_file,
        extra_secrets=[secret],
    )
    assert result.report.strings_redacted >= 5
    assert recording.recorded[1].content == original
    assert recording.recorded[1].content["goal"] == goal
    with nullcontext(result.path.as_uri()) if single_file else _static_url(result.path) as url:
        async with _page() as page:
            await _open(page, url)
            await _seek(page, 1000)
            redacted = {
                **goal,
                **{
                    field: f"{field}: ‹secret› in ~/goal.txt"
                    for field in ("condition", "started_at", "last_reason", "last_evaluated_at", "status_note")
                },
            }
            await _panel(page, "active", redacted, reason="evaluated", evaluation_met=False)
            text = await page.locator("#goal").text_content()
            assert text is not None and secret not in text and home not in text
            assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
    assert recording.recorded[1].content == original
