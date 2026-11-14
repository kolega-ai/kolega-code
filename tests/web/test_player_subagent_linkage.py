from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.web.bundle import export_bundle
from kolega_code.web.hosting import ShareServer

playwright_api = pytest.importorskip("playwright.async_api", reason="playwright is required for player tests")

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

SESSION_ID = "b" * 32
OWNERS = {
    "alpha": {
        "dispatch_id": "dispatch-alpha",
        "agent_id": "agent-alpha",
        "agent_name": "general-agent",
        "label": "Alpha",
        "parent_agent_id": "root",
        "parent_tool_call_id": "call-alpha",
        "depth": 1,
        "background": False,
    },
    "parallel": {
        "dispatch_id": "dispatch-parallel",
        "agent_id": "agent-parallel",
        "agent_name": "general-agent",
        "label": "Parallel",
        "parent_agent_id": "root",
        "parent_tool_call_id": "call-parallel",
        "depth": 1,
        "background": False,
    },
    "background": {
        "dispatch_id": "dispatch-background",
        "agent_id": "agent-background",
        "agent_name": "general-agent",
        "label": "Background",
        "parent_agent_id": "root",
        "parent_tool_call_id": "call-background",
        "depth": 1,
        "background": True,
    },
    "nested": {
        "dispatch_id": "dispatch-nested",
        "agent_id": "agent-nested",
        "agent_name": "general-agent",
        "label": "Nested",
        "parent_agent_id": "agent-alpha",
        "parent_tool_call_id": "call-nested",
        "depth": 2,
        "background": False,
    },
}


class _Recording:
    def __init__(self, directory: Path) -> None:
        self.store: SessionStore = SessionStore(directory / "state")
        self.store.create(directory, "cli", {}, session_id=SESSION_ID, title="worker linkage")
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
        if elapsed_ms is None:
            elapsed_ms = self.recorded[-1].elapsed_ms + 100 if self.recorded else 0
        event = AgentEvent(
            session_id=SESSION_ID,
            sender="agent",
            event_type=event_type,
            content=content,
            sub_agent_info=info,
            elapsed_ms=elapsed_ms,
        )
        await self.events.append(event)
        self.recorded.append(event)


async def _recording(directory: Path) -> _Recording:
    recording = _Recording(directory)
    await recording.append(KnownEventType.TURN_STARTED, elapsed_ms=0, turn_id="root-turn", user_text="root prompt")
    return recording


@asynccontextmanager
async def _served(recording: _Recording, *, live: bool = True) -> AsyncGenerator[Any, None]:
    server = ShareServer(recording.store, session_id=SESSION_ID)
    await server.start()
    try:
        async with playwright_api.async_playwright() as driver:
            browser = await driver.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport={"width": 1200, "height": 1200})
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(server.session_url(SESSION_ID))
                await page.wait_for_selector("#transcript .kc-entry")
                if live:
                    await page.wait_for_function("() => document.getElementById('live').dataset.state === 'following'")
                yield page
                assert not errors, f"player pageerrors: {errors}"
            finally:
                await browser.close()
    finally:
        await server.stop()


async def _focus(page: Any, key: str) -> None:
    escaped = await page.evaluate("(key) => CSS.escape(key)", key)
    await page.locator(f'#subAgents [data-agent-key="{escaped}"]').click()
    await page.wait_for_selector("#agentHeader")


async def _seek(page: Any, value: int) -> None:
    await page.eval_on_selector(
        "#scrub", "(el, value) => { el.value = String(value); el.dispatchEvent(new Event('input')); }", value
    )


async def _delivered(page: Any, recording: _Recording) -> None:
    await page.wait_for_function(
        "(count) => document.getElementById('meta').textContent.startsWith(`${count} events`)",
        arg=len(recording.recorded),
    )


async def _terminal(page: Any, text: str, owner: str, *, truncated: bool = False) -> None:
    await page.wait_for_function(
        """({text, owner, truncated}) =>
            document.getElementById('terminal').textContent === text &&
            document.getElementById('terminalOwner').dataset.owner === owner &&
            document.getElementById('terminalTruncated').hidden === !truncated""",
        arg={"text": text, "owner": owner, "truncated": truncated},
    )
    assert await page.locator("#terminalOwner").is_visible()
    assert await page.locator("#terminalTruncated").is_visible() is truncated


async def _facts(page: Any, selector: str = "#agentHeader") -> list[str]:
    return await page.locator(f"{selector} .kc-agent-fact").all_text_contents()


async def _seed(recording: _Recording) -> None:
    await recording.append(KnownEventType.TERMINAL_COMMAND, command="printf root")
    await recording.append(KnownEventType.TERMINAL_OUTPUT, output="ROOT ONLY\n")
    for scope in ("nested", "alpha", "parallel", "background"):
        await recording.append(KnownEventType.TERMINAL_COMMAND, info=OWNERS[scope], command=f"printf {scope}")
        await recording.append(KnownEventType.TERMINAL_OUTPUT, info=OWNERS[scope], output=f"{scope.upper()} ONLY\n")
        await recording.append(
            KnownEventType.ASSISTANT_DELTA, info=OWNERS[scope], text=f"{scope} answer", complete=True
        )


@pytest.mark.parametrize("scope", list(OWNERS))
async def test_root_parallel_background_and_nested_terminals_are_isolated(tmp_path: Path, scope: str) -> None:
    recording = await _recording(tmp_path)
    await _seed(recording)
    async with _served(recording) as page:
        await _terminal(page, "$ printf root\nROOT ONLY\n", "root")
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["root prompt"]
        info = OWNERS[scope]
        await _focus(page, info["dispatch_id"])
        await _terminal(page, f"$ printf {scope}\n{scope.upper()} ONLY\n", info["agent_id"])
        assert await page.locator("#terminalOwner").text_content() == f"Owner: {info['label']} ({info['agent_id']})"
        assert await page.locator("#transcript .kc-body").all_text_contents() == [f"{scope} answer"]
        facts = await _facts(page)
        assert f"Agent: {info['agent_id']}" in facts
        assert f"Parent tool call: {info['parent_tool_call_id']}" in facts
        assert f"Depth: {info['depth']}" in facts
        assert f"Background: {'yes' if info['background'] else 'no'}" in facts
        assert f"Parent: {info['parent_agent_id']}" + (" · Alpha" if scope == "nested" else "") in facts
        await page.locator(".kc-agent-back").click()
        await _terminal(page, "$ printf root\nROOT ONLY\n", "root")


async def test_empty_worker_never_falls_back_to_root_and_unknown_routes_stay_unknown(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(KnownEventType.TERMINAL_OUTPUT, output="PRIVATE ROOT OUTPUT")
    legacy = {"dispatch_id": "legacy", "agent_name": "legacy-model-looking-name", "total_tokens": 0}
    await recording.append(KnownEventType.TERMINAL_LAUNCHED, info=legacy)
    async with _served(recording) as page:
        await _focus(page, "legacy")
        await _terminal(page, "No terminal output yet.", "legacy")
        facts = await _facts(page)
        assert "Parent: unknown" in facts
        assert "Parent tool call: unknown" in facts
        assert "Requested routing: unknown" in facts
        assert "Effective routing: unknown" in facts
        assert "Total tokens: 0" in facts
        assert await page.locator(".kc-agent-parent").count() == 0
        await page.locator(".kc-agent-back").click()
        await _terminal(page, "PRIVATE ROOT OUTPUT", "root")


async def test_live_appends_remain_scoped_while_worker_is_focused_and_survive_reload(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await _seed(recording)
    async with _served(recording) as page:
        await _focus(page, "dispatch-alpha")
        await page.locator('#subAgents [data-agent-key="dispatch-alpha"]').focus()
        await page.evaluate("globalThis.focusedRail = document.activeElement")
        for scope in (None, "parallel", "background", "nested", "alpha"):
            await recording.append(
                KnownEventType.TERMINAL_OUTPUT,
                info=None if scope is None else OWNERS[scope],
                output=f"LATE {scope or 'root'}\n",
            )
            await _delivered(page, recording)
            expected = "$ printf alpha\nALPHA ONLY\n" + ("LATE alpha\n" if scope == "alpha" else "")
            await _terminal(page, expected, "agent-alpha")
            assert await page.evaluate("document.activeElement === globalThis.focusedRail")
        await recording.append(KnownEventType.ASSISTANT_DELTA, info=OWNERS["alpha"], text="alpha live", complete=True)
        await _delivered(page, recording)
        assert await page.locator("#transcript .kc-body").all_text_contents() == ["alpha answer", "alpha live"]
        await page.locator(".kc-agent-back").click()
        await _terminal(page, "$ printf root\nROOT ONLY\nLATE root\n", "root")
        for scope in ("parallel", "background", "nested"):
            await _focus(page, OWNERS[scope]["dispatch_id"])
            await _terminal(
                page,
                f"$ printf {scope}\n{scope.upper()} ONLY\nLATE {scope}\n",
                OWNERS[scope]["agent_id"],
            )
        await page.reload()
        await page.wait_for_selector("#transcript .kc-entry")
        await _terminal(page, "$ printf root\nROOT ONLY\nLATE root\n", "root")
        await _focus(page, "dispatch-alpha")
        await _terminal(page, "$ printf alpha\nALPHA ONLY\nLATE alpha\n", "agent-alpha")


async def test_late_metadata_and_zero_totals_refresh_cards_headers_and_preserve_keyboard_focus(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(KnownEventType.ASSISTANT_DELTA, info=OWNERS["alpha"], text="one step", complete=True)
    requested = {"provider": "requested-provider", "model": "requested-model", "thinking_effort": "high"}
    effective = {"provider": "effective-provider", "model": "effective-model", "thinking_effort": "low"}
    updated = {
        **OWNERS["alpha"],
        "label": "Updated Alpha",
        "task": "late task",
        "phase": "Verify",
        "requested_routing": requested,
        "effective_routing": effective,
    }
    async with _served(recording) as page:
        card = page.locator('.kc-agent-card[data-agent-key="dispatch-alpha"]')
        await card.focus()
        await recording.append(KnownEventType.LLM_STATUS_UPDATE, info=updated, total_tokens=0)
        await _delivered(page, recording)
        assert "Total tokens: 0" in await _facts(page, '.kc-agent-card[data-agent-key="dispatch-alpha"]')
        assert await card.locator(".kc-agent-card-name").text_content() == "Updated Alpha"
        assert await card.locator(".kc-agent-card-task").text_content() == "late task"
        assert await card.locator(".kc-sub-agent-phase").text_content() == "Verify"
        assert await card.locator(".kc-agent-card-steps").text_content() == "1 step · open"
        assert await card.evaluate("(el) => document.activeElement === el")
        await card.press("Space")
        assert await page.locator("#play").get_attribute("aria-label") == "Play"
        assert await page.locator(".kc-agent-back").evaluate("(el) => document.activeElement === el")
        facts = await _facts(page)
        assert "Total tokens: 0" in facts
        assert "Requested routing: " + json.dumps(requested, separators=(",", ":")) in facts
        assert "Effective routing: " + json.dumps(effective, separators=(",", ":")) in facts
        await recording.append(
            KnownEventType.LLM_STATUS_UPDATE, info={**updated, "label": "Final Alpha"}, total_tokens=1234
        )
        await _delivered(page, recording)
        assert "Total tokens: 1,234" in await _facts(page)
        assert await page.locator(".kc-agent-header-title").text_content() == "Final Alpha"
        assert await page.locator(".kc-agent-back").evaluate("(el) => document.activeElement === el")
        await page.locator(".kc-agent-back").press("Space")
        await _terminal(page, "No terminal output yet.", "root")
        assert await page.locator("#play").get_attribute("aria-label") == "Play"
        rail = page.locator('#subAgents [data-agent-key="dispatch-alpha"]')
        await rail.focus()
        await rail.press("Enter")
        assert await page.locator(".kc-agent-header-title").text_content() == "Final Alpha"


async def test_child_before_parent_gains_exact_parent_link_without_inferred_depth(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    child = {**OWNERS["nested"], "depth": 7}
    await recording.append(KnownEventType.TERMINAL_OUTPUT, info=child, output="CHILD\n")
    async with _served(recording) as page:
        await _focus(page, "dispatch-nested")
        assert "Parent: agent-alpha (unavailable)" in await _facts(page)
        assert await page.locator(".kc-agent-parent").count() == 0
        await recording.append(KnownEventType.TERMINAL_OUTPUT, info=OWNERS["alpha"], output="PARENT\n")
        await _delivered(page, recording)
        assert "Parent: agent-alpha · Alpha" in await _facts(page)
        assert "Depth: 7" in await _facts(page)
        await page.locator(".kc-agent-parent").click()
        await _terminal(page, "PARENT\n", "agent-alpha")
        assert "Parent: root" in await _facts(page)
        await page.locator(".kc-agent-parent").click()
        await _terminal(page, "No terminal output yet.", "root")
        child_card = '.kc-agent-card[data-agent-key="dispatch-nested"]'
        assert "Parent: agent-alpha · Alpha" in await _facts(page, child_card)
        assert "Depth: 7" in await _facts(page, child_card)


async def test_late_first_step_reorders_cards_without_moving_keyboard_focus_to_another_worker(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await recording.append(KnownEventType.TERMINAL_OUTPUT, info=OWNERS["alpha"], output="ALPHA\n")
    await recording.append(KnownEventType.TERMINAL_OUTPUT, info=OWNERS["parallel"], output="PARALLEL\n")
    async with _served(recording) as page:
        assert await page.locator(".kc-agent-card-name").all_text_contents() == ["Alpha", "Parallel"]
        alpha = page.locator('.kc-agent-card[data-agent-key="dispatch-alpha"]')
        await alpha.focus()
        await recording.append(
            KnownEventType.ASSISTANT_DELTA, info=OWNERS["parallel"], text="first step", complete=True
        )
        await _delivered(page, recording)
        assert await page.locator(".kc-agent-card-name").all_text_contents() == ["Parallel", "Alpha"]
        assert await alpha.evaluate("(el) => document.activeElement === el")
        await alpha.press("Space")
        await _terminal(page, "ALPHA\n", "agent-alpha")
        assert await page.locator("#play").get_attribute("aria-label") == "Play"


@pytest.mark.parametrize("parent_kind", ["missing", "self-cycle", "cycle"])
async def test_missing_or_cyclic_parents_and_hostile_metadata_are_safe(tmp_path: Path, parent_kind: str) -> None:
    recording = await _recording(tmp_path)
    hostile = """worker'"]<img src=x onerror="globalThis.injected=true">"""
    markup = "<script>globalThis.injected=true</script>"
    parent_id = {"missing": "unrecorded-parent", "self-cycle": "hostile-id", "cycle": "other-id"}[parent_kind]
    info = {
        "dispatch_id": hostile,
        "agent_id": "hostile-id",
        "agent_name": "worker",
        "label": markup,
        "task": markup,
        "parent_agent_id": parent_id,
        "parent_tool_call_id": markup,
        "depth": 9,
        "background": True,
        "requested_routing": {"model": markup},
        "effective_routing": {"model": hostile},
        "total_tokens": 0,
    }
    await recording.append(KnownEventType.TERMINAL_OUTPUT, info=info, output=markup)
    if parent_kind == "cycle":
        await recording.append(
            KnownEventType.TERMINAL_LAUNCHED,
            info={"agent_id": "other-id", "agent_name": "Other", "parent_agent_id": "hostile-id"},
        )
    async with _served(recording) as page:
        await _focus(page, hostile)
        await _terminal(page, markup, "hostile-id")
        facts = await _facts(page)
        assert "Parent tool call: " + markup in facts
        assert "Requested routing: " + json.dumps({"model": markup}, separators=(",", ":")) in facts
        assert await page.locator(".kc-agent-header-title").text_content() == markup
        assert await page.locator("#agentHeader script, #agentHeader img, #terminal script, #terminal img").count() == 0
        assert not await page.evaluate("Boolean(globalThis.injected)")
        if parent_kind == "missing":
            assert "Parent: unrecorded-parent (unavailable)" in facts
            assert await page.locator(".kc-agent-parent").count() == 0
        else:
            assert any(fact.startswith(f"Parent: {parent_id}") and fact.endswith("(cycle)") for fact in facts)
            await page.locator(".kc-agent-parent").click()
            await page.wait_for_selector("#agentHeader")
        await page.locator(".kc-agent-back").click()
        await _terminal(page, "No terminal output yet.", "root")


async def _replay_recording(directory: Path) -> _Recording:
    recording = await _recording(directory)
    await recording.append(KnownEventType.TERMINAL_OUTPUT, elapsed_ms=1000, output="ROOT FIRST\n")
    await recording.append(
        KnownEventType.TERMINAL_OUTPUT, elapsed_ms=2000, info=OWNERS["alpha"], output="ALPHA FIRST\n"
    )
    await recording.append(
        KnownEventType.TERMINAL_OUTPUT, elapsed_ms=3000, info=OWNERS["parallel"], output="PARALLEL\n"
    )
    await recording.append(KnownEventType.TERMINAL_OUTPUT, elapsed_ms=4000, info=OWNERS["alpha"], output="ALPHA LAST\n")
    await recording.append(KnownEventType.TERMINAL_OUTPUT, elapsed_ms=5000, output="ROOT LAST\n")
    await recording.append(KnownEventType.LLM_STATUS_UPDATE, elapsed_ms=6000, info=OWNERS["alpha"], total_tokens=0)
    await recording.append(KnownEventType.TURN_ENDED, elapsed_ms=8000, turn_id="root-turn", status="completed")
    return recording


async def test_seeking_rebuilds_prefixes_and_absent_worker_never_exposes_root(tmp_path: Path) -> None:
    recording = await _replay_recording(tmp_path)
    async with _served(recording, live=False) as page:
        await _terminal(page, "No terminal output yet.", "root")
        await _seek(page, 1000)
        await _focus(page, "dispatch-alpha")
        await _terminal(page, "ALPHA FIRST\nALPHA LAST\n", "agent-alpha")
        assert "Total tokens: 0" in await _facts(page)
        await _seek(page, 375)
        await _terminal(page, "ALPHA FIRST\n", "agent-alpha")
        assert "Total tokens: unknown" in await _facts(page)
        prefix = await page.locator("#agentHeader").text_content()
        await _seek(page, 125)
        await _terminal(page, "No terminal output yet.", "dispatch-alpha")
        assert "not present at this position" in await page.locator("#agentHeader").inner_text()
        await _seek(page, 0)
        await _terminal(page, "No terminal output yet.", "dispatch-alpha")
        await _seek(page, 1000)
        await _terminal(page, "ALPHA FIRST\nALPHA LAST\n", "agent-alpha")
        await page.locator(".kc-agent-back").click()
        await _terminal(page, "ROOT FIRST\nROOT LAST\n", "root")
        await page.reload()
        await page.wait_for_selector("#transcript .kc-entry")
        await _seek(page, 375)
        await _focus(page, "dispatch-alpha")
        await _terminal(page, "ALPHA FIRST\n", "agent-alpha")
        assert await page.locator("#agentHeader").text_content() == prefix


async def test_single_file_export_keeps_owner_metadata_terminal_isolation_and_replay(tmp_path: Path) -> None:
    recording = await _recording(tmp_path)
    await _seed(recording)
    await recording.append(KnownEventType.LLM_STATUS_UPDATE, info=OWNERS["background"], total_tokens=0)
    await recording.append(KnownEventType.TURN_ENDED, turn_id="root-turn", status="completed")
    result = await export_bundle(recording.recorded, tmp_path / "replay.html", session_id=SESSION_ID, single_file=True)
    async with _served(recording, live=False) as page:
        await page.goto(result.path.as_uri())
        await page.wait_for_selector("#transcript .kc-entry")
        await _seek(page, 1000)
        assert not await page.locator("#live").is_visible()
        await _terminal(page, "$ printf root\nROOT ONLY\n", "root")
        for scope, info in OWNERS.items():
            await _focus(page, info["dispatch_id"])
            await _terminal(page, f"$ printf {scope}\n{scope.upper()} ONLY\n", info["agent_id"])
            if scope == "background":
                assert "Total tokens: 0" in await _facts(page)
                assert "Background: yes" in await _facts(page)
            if scope == "nested":
                await page.locator(".kc-agent-parent").click()
                await _terminal(page, "$ printf alpha\nALPHA ONLY\n", "agent-alpha")
        await page.reload()
        await page.wait_for_selector("#transcript .kc-entry")
        await _seek(page, 1000)
        await _terminal(page, "$ printf root\nROOT ONLY\n", "root")


@pytest.mark.parametrize("truncated_scope", ["root", "alpha"])
async def test_terminal_truncation_badge_belongs_only_to_its_owner(tmp_path: Path, truncated_scope: str) -> None:
    recording = await _recording(tmp_path)
    await recording.append(KnownEventType.TERMINAL_OUTPUT, output="ROOT\n")
    await recording.append(KnownEventType.TERMINAL_OUTPUT, info=OWNERS["alpha"], output="ALPHA\n")
    async with _served(recording) as page:
        payload = "DROPPED PREFIX" + "x" * 2_000_000 + "RETAINED END"
        await recording.append(
            KnownEventType.TERMINAL_OUTPUT,
            info=None if truncated_scope == "root" else OWNERS["alpha"],
            output=payload,
        )
        await _delivered(page, recording)
        if truncated_scope == "root":
            await _terminal(page, payload[-2_000_000:], "root", truncated=True)
            await _focus(page, "dispatch-alpha")
            await _terminal(page, "ALPHA\n", "agent-alpha")
        else:
            await _terminal(page, "ROOT\n", "root")
            await _focus(page, "dispatch-alpha")
            await _terminal(page, payload[-2_000_000:], "agent-alpha", truncated=True)
        await page.locator(".kc-agent-back").click()
        await _terminal(
            page,
            payload[-2_000_000:] if truncated_scope == "root" else "ROOT\n",
            "root",
            truncated=truncated_scope == "root",
        )
        await _seek(page, 0)
        await _terminal(page, "No terminal output yet.", "root")
        await _seek(page, 1000)
        await _terminal(
            page,
            payload[-2_000_000:] if truncated_scope == "root" else "ROOT\n",
            "root",
            truncated=truncated_scope == "root",
        )
        await _focus(page, "dispatch-alpha")
        await _terminal(
            page,
            payload[-2_000_000:] if truncated_scope == "alpha" else "ALPHA\n",
            "agent-alpha",
            truncated=truncated_scope == "alpha",
        )
