"""MCP edits have their own protected draft and explicit verification contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import Button, Input, OptionList, Select, Static, Switch

from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.tui import settings_panel
from kolega_code.cli.tui.memory_screen import MemoryScreen
from kolega_code.cli.tui.settings_screen import (
    SETTINGS_CATEGORIES,
    ConfirmDiscardSettingsScreen,
    ConfirmMCPDraftScreen,
    SettingsScreen,
)
from kolega_code.mcp.config import (
    MCPConfigFile,
    MCPServerConfig,
    global_mcp_config_path,
    project_mcp_config_path,
    save_config_file,
)
from kolega_code.mcp.service import MCPService, MCPVerificationResult
from kolega_code.session.inbox import PeerMessage

from ._app_test_utils import FakeCoderAgent, install_fake_agents, wait_for_turn_idle
from .test_settings_dropdown_interactions import _wait_for_layout
from .test_tui_settings_screens import _configured_app


def _configured_mcp_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[KolegaCodeApp, Path]:
    app, store = _configured_app(tmp_path, monkeypatch)
    path = global_mcp_config_path(store.root)
    save_config_file(
        path,
        MCPConfigFile(
            servers=[
                MCPServerConfig(id="first", name="First", url="https://first.invalid/mcp"),
                MCPServerConfig(id="second", name="Second", url="https://second.invalid/mcp"),
            ]
        ),
    )
    return app, path


async def _open_mcp(app: KolegaCodeApp, pilot: Pilot, *, server: str = "first") -> SettingsScreen:
    app.action_open_settings(category="mcp")
    await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    screen.query_one("#mcp_server_select", Select).value = server
    await _wait_for_layout(
        pilot, lambda: screen.query_one("#mcp_url_input", Input).value == f"https://{server}.invalid/mcp"
    )
    return screen


async def _press(pilot: Pilot, screen: Screen, selector: str) -> None:
    await _wait_for_layout(pilot, lambda: bool(screen.query(selector)))
    button = screen.query_one(selector, Button)
    button.focus()
    await _wait_for_layout(pilot, lambda: button.has_focus)
    await pilot.press("enter")


def _saved_url(path: Path, server_id: str = "first") -> str:
    return next(server["url"] for server in json.loads(path.read_text())["servers"] if server["id"] == server_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("navigation", ["select", "reload", "close"])
@pytest.mark.parametrize("decision", ["keep_editing", "discard", "save"])
async def test_mcp_navigation_protects_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    navigation: str,
    decision: str,
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_mcp(app, pilot)
        field = screen.query_one("#mcp_url_input", Input)
        field.value = "https://edited.invalid/mcp"
        await pilot.pause()
        if navigation == "select":
            screen.query_one("#mcp_server_select", Select).value = "second"
        elif navigation == "reload":
            await _press(pilot, screen, "#mcp_refresh")
        else:
            await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: app.screen is not screen and app.screen.is_mounted)
        assert app.screen.query("#mcp_draft_save"), "Expected the MCP Save / Discard / Keep Editing prompt"
        assert field.value == "https://edited.invalid/mcp"
        assert screen.query_one("#mcp_server_select", Select).value == "first"
        await _press(pilot, app.screen, f"#mcp_draft_{decision}")

        if decision == "keep_editing":
            await _wait_for_layout(pilot, lambda: app.screen is screen)
            assert field.value == "https://edited.invalid/mcp"
            assert screen.mcp_dirty
        elif navigation == "close":
            await _wait_for_layout(pilot, lambda: app._settings_screen is None)
        else:
            expected = (
                "https://second.invalid/mcp"
                if navigation == "select"
                else ("https://edited.invalid/mcp" if decision == "save" else "https://first.invalid/mcp")
            )
            await _wait_for_layout(
                pilot, lambda: app.screen is screen and field.value == expected and not screen.mcp_busy
            )
            assert not screen.mcp_dirty
        assert _saved_url(path) == ("https://edited.invalid/mcp" if decision == "save" else "https://first.invalid/mcp")
        assert _saved_url(path, "second") == "https://second.invalid/mcp"


@pytest.mark.asyncio
async def test_mcp_dirty_is_separate_from_global_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        await pilot.pause()
        assert screen.mcp_dirty
        assert not screen.dirty
        assert "Unsaved server changes" in str(screen.query_one("#mcp_draft_status", Static).render())
        assert not screen.query_one("#save_settings", Button).display
        assert screen.query_one("#mcp_save_verify_server", Button).display
        assert not screen.query_one("#mcp_verify_server", Button).display

        # Neither global Apply nor category browsing may clear or save the MCP draft.
        tips = screen.query_one("#discovery_tips_switch", Switch)
        tips.value = not tips.value
        screen._show_category("appearance")
        await pilot.pause()
        await app._save_settings_from_ui()
        assert not screen.dirty
        assert screen.mcp_dirty
        assert _saved_url(path) == "https://first.invalid/mcp"
        screen._show_category("mcp")
        assert screen.query_one("#mcp_url_input", Input).value == "https://edited.invalid/mcp"


@pytest.mark.asyncio
async def test_verify_saved_refuses_to_test_an_edited_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    verify = AsyncMock(return_value=MCPVerificationResult("first", True, "Verified", 2))
    monkeypatch.setattr(MCPService, "verify_server", verify)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        await app._verify_mcp_server_from_ui()
        verify.assert_not_called()
        assert screen.query_one("#mcp_url_input", Input).value == "https://edited.invalid/mcp"
        assert _saved_url(path) == "https://first.invalid/mcp"


@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [False, True])
async def test_save_and_verify_tests_the_visible_configuration_and_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, ok: bool
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    seen: list[str | None] = []

    async def verify(service: MCPService, server_id: str, **kwargs: Any) -> MCPVerificationResult:
        seen.append(service.config.servers[server_id].url)
        assert _saved_url(path) == "https://edited.invalid/mcp"  # Saved before connecting.
        return MCPVerificationResult(server_id, ok, "Connection refused." if not ok else "Verified", 2)

    monkeypatch.setattr(MCPService, "verify_server", verify)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        await pilot.pause()
        await _press(pilot, screen, "#mcp_save_verify_server")
        await _wait_for_layout(pilot, lambda: bool(seen) and not screen.mcp_busy)
        assert seen == ["https://edited.invalid/mcp"]
        assert screen.query_one("#mcp_url_input", Input).value == "https://edited.invalid/mcp"
        assert not screen.mcp_dirty
        feedback = str(screen.query_one("#mcp_draft_status", Static).render()).lower()
        assert "saved" in feedback
        assert ("verified" if ok else "verification failed") in feedback


@pytest.mark.asyncio
async def test_failed_save_keeps_navigation_and_draft_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = ""  # Required URL is missing.
        await pilot.pause()
        screen.query_one("#mcp_server_select", Select).value = "second"
        await _wait_for_layout(pilot, lambda: app.screen is not screen and app.screen.is_mounted)
        await _press(pilot, app.screen, "#mcp_draft_save")
        await _wait_for_layout(pilot, lambda: app.screen is screen and not screen.mcp_busy)
        assert screen.query_one("#mcp_server_select", Select).value == "first"
        assert screen.query_one("#mcp_url_input", Input).value == ""
        assert screen.mcp_dirty
        assert _saved_url(path) == "https://first.invalid/mcp"


@pytest.mark.asyncio
async def test_new_mcp_save_and_failed_verify_preserves_raw_form_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    verify = AsyncMock(side_effect=RuntimeError("fake-sensitive-transport-detail"))
    monkeypatch.setattr(MCPService, "verify_server", verify)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_server_select", Select).value = settings_panel.MCP_NEW_SERVER_VALUE
        await _wait_for_layout(pilot, lambda: screen.query_one("#mcp_url_input", Input).value == "")
        screen.query_one("#mcp_name_input", Input).value = "  New server  "
        screen.query_one("#mcp_url_input", Input).value = " https://new.invalid/mcp "
        screen.query_one("#mcp_headers_input", Input).value = '{ "X-Test" : "fake-value" }'
        await pilot.pause()
        await _press(pilot, screen, "#mcp_save_verify_server")
        await _wait_for_layout(pilot, lambda: verify.await_count == 1 and not screen.mcp_busy)
        assert verify.await_args is not None and verify.await_args.args == ("new-server",)
        assert _saved_url(path, "new-server") == "https://new.invalid/mcp"
        assert screen.query_one("#mcp_server_select", Select).value == "new-server"
        assert screen.query_one("#mcp_name_input", Input).value == "  New server  "
        assert screen.query_one("#mcp_url_input", Input).value == " https://new.invalid/mcp "
        assert screen.query_one("#mcp_headers_input", Input).value == '{ "X-Test" : "fake-value" }'
        assert not screen.mcp_dirty
        feedback = str(screen.query_one("#mcp_draft_status", Static).render()).lower()
        assert "saved" in feedback and "verification failed" in feedback
        assert "fake-sensitive" not in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["validation", "write", "active-turn"])
async def test_failed_save_and_verify_never_connects_or_clears_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, failure: str
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    original = path.read_bytes()
    verify = AsyncMock()
    monkeypatch.setattr(MCPService, "verify_server", verify)

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic write failure")

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        value = "" if failure == "validation" else "https://edited.invalid/mcp"
        screen.query_one("#mcp_url_input", Input).value = value
        if failure == "write":
            monkeypatch.setattr(settings_panel, "upsert_server_config", fail_write)
        app._turn_active = failure == "active-turn"
        try:
            await app._handle_mcp_settings_button("mcp_save_verify_server")
        finally:
            app._turn_active = False
        verify.assert_not_called()
        assert path.read_bytes() == original
        assert screen.query_one("#mcp_url_input", Input).value == value
        assert screen.mcp_dirty
        status = str(screen.query_one("#mcp_status", Static).render()).lower()
        assert {"validation": "requires url", "write": "could not save", "active-turn": "stop the active turn"}[
            failure
        ] in status


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["save", "discard"])
async def test_closing_resolves_mcp_and_ordinary_drafts_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, decision: str
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    original_tips = app.settings.discovery_tips
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        screen.query_one("#discovery_tips_switch", Switch).value = not original_tips
        screen._show_category("appearance")
        await pilot.pause()
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, ConfirmMCPDraftScreen) and app.screen.is_mounted)
        await _press(pilot, app.screen, f"#mcp_draft_{decision}")
        await _wait_for_layout(
            pilot, lambda: isinstance(app.screen, ConfirmDiscardSettingsScreen) and app.screen.is_mounted
        )
        await pilot.press("escape")  # Keep the ordinary draft.
        await _wait_for_layout(pilot, lambda: app.screen is screen and not screen.mcp_busy)
        assert screen.dirty and not screen.mcp_dirty
        assert screen.query_one("#discovery_tips_switch", Switch).value is not original_tips
        assert app.settings_store.load().discovery_tips is original_tips
        assert _saved_url(path) == ("https://edited.invalid/mcp" if decision == "save" else "https://first.invalid/mcp")


@pytest.mark.asyncio
async def test_inflight_verification_blocks_navigation_and_does_not_reload_later_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def verify(service: MCPService, server_id: str, **kwargs: Any) -> MCPVerificationResult:
        started.set()
        await release.wait()
        return MCPVerificationResult(server_id, True, "Verified", 3)

    monkeypatch.setattr(MCPService, "verify_server", verify)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        await pilot.pause()
        await _press(pilot, screen, "#mcp_save_verify_server")
        await _wait_for_layout(pilot, started.is_set)
        try:
            assert screen.mcp_busy
            assert screen.query_one("#mcp_url_input", Input).disabled
            assert screen.query_one("#mcp_refresh", Button).disabled
            assert screen.query_one("#mcp_server_select", Select).disabled
            await pilot.press("escape")
            assert app.screen is screen
            screen.request_mcp_selection("second")
            assert screen.query_one("#mcp_server_select", Select).value == "first"
            # Simulate a queued/programmatic edit, even though the real controls are locked.
            screen.query_one("#mcp_url_input", Input).value = "https://later.invalid/mcp"
        finally:
            release.set()
        await _wait_for_layout(pilot, lambda: not screen.mcp_busy)
        assert screen.query_one("#mcp_url_input", Input).value == "https://later.invalid/mcp"
        assert screen.mcp_dirty
        assert _saved_url(path) == "https://edited.invalid/mcp"
        assert "Unsaved server changes" in str(screen.query_one("#mcp_draft_status", Static).render())
        assert not screen.query_one("#mcp_url_input", Input).disabled


@pytest.mark.asyncio
async def test_stale_mcp_select_events_and_escape_cannot_replace_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _configured_mcp_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        picker = screen.query_one("#mcp_server_select", Select)
        transport = screen.query_one("#mcp_transport_select", Select)
        # Textual's nested Changed class uses an unbound outer TypeVar in its signature.
        app.on_select_changed(Select.Changed(picker, "second"))  # pyright: ignore[reportArgumentType]
        app.on_select_changed(Select.Changed(transport, "stdio"))  # pyright: ignore[reportArgumentType]
        await pilot.pause()
        assert app.screen is screen
        assert screen.query_one("#mcp_url_input", Input).display
        assert screen.query_one("#mcp_url_input", Input).value == "https://edited.invalid/mcp"
        picker.value = "second"
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, ConfirmMCPDraftScreen) and app.screen.is_mounted)
        # More queued selection changes while the modal is active must not open another modal.
        screen.request_mcp_selection(settings_panel.MCP_NEW_SERVER_VALUE)
        assert sum(isinstance(item, ConfirmMCPDraftScreen) for item in app.screen_stack) == 1
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: app.screen is screen)
        assert picker.value == "first"
        assert screen.query_one("#mcp_url_input", Input).value == "https://edited.invalid/mcp"
        assert screen.mcp_dirty


@pytest.mark.asyncio
@pytest.mark.parametrize("save_first", [False, True], ids=["verify-saved", "save-and-verify"])
async def test_verification_defers_peer_turns_and_preserves_the_queue_through_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, save_first: bool
) -> None:
    app, _ = _configured_mcp_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    started, release = asyncio.Event(), asyncio.Event()

    async def verify(service: MCPService, server_id: str, **kwargs: Any) -> MCPVerificationResult:
        started.set()
        await release.wait()
        return MCPVerificationResult(server_id, True, "Verified", 1)

    monkeypatch.setattr(MCPService, "verify_server", verify)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        if save_first:
            screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        await pilot.pause()
        await _press(pilot, screen, "#mcp_save_verify_server" if save_first else "#mcp_verify_server")
        await _wait_for_layout(pilot, started.is_set)
        old_agent = app.agent
        assert isinstance(old_agent, FakeCoderAgent)
        try:
            for text in ("first peer message", "second peer message"):
                app._enqueue_peer_message(
                    PeerMessage.create(sender_session_id="fake-peer", sender_title="Fake peer", text=text)
                )
            assert not app._maybe_start_queued_message()
            await pilot.pause()
            assert len(app._queued_messages) == 2
            assert not app._turn_active and app.agent_worker is None
            assert old_agent.cleanup_calls == 0
            mode = app.interaction_mode
            await pilot.press("shift+tab")  # A priority app binding, even over Settings.
            assert app.interaction_mode == mode
            assert len(app._queued_messages) == 2
            assert old_agent.cleanup_calls == 0
        finally:
            release.set()
            await _wait_for_layout(pilot, lambda: not screen.mcp_busy)
            await wait_for_turn_idle(app, pilot)
        await _wait_for_layout(pilot, lambda: isinstance(app.agent, FakeCoderAgent) and len(app.agent.messages) == 2)
        await wait_for_turn_idle(app, pilot)
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent is not old_agent and old_agent.cleanup_calls == 1
        assert old_agent.messages == []
        assert "first peer message" in app.agent.messages[0]
        assert "second peer message" in app.agent.messages[1]
        assert not app._queued_messages


@pytest.mark.asyncio
async def test_mcp_operation_defers_scheduled_loop_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _configured_mcp_app(tmp_path, monkeypatch)
    release = asyncio.Event()
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        assert app._loop_ready_to_fire()
        screen._start_mcp_action(release.wait)
        try:
            assert not app._loop_ready_to_fire()
        finally:
            release.set()
            await _wait_for_layout(pilot, lambda: not screen.mcp_busy)
        assert app._loop_ready_to_fire()


@pytest.mark.asyncio
@pytest.mark.parametrize("navigation", ["select", "reload", "close"])
async def test_new_server_can_be_saved_before_navigation_without_overwriting_existing_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, navigation: str
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open_mcp(app, pilot)
        picker = screen.query_one("#mcp_server_select", Select)
        picker.value = settings_panel.MCP_NEW_SERVER_VALUE
        await _wait_for_layout(pilot, lambda: screen.mcp_server_id == settings_panel.MCP_NEW_SERVER_VALUE)
        screen.query_one("#mcp_name_input", Input).value = "First"  # Collides with a saved server's generated ID.
        screen.query_one("#mcp_url_input", Input).value = "https://new.invalid/mcp"
        await pilot.pause()
        if navigation == "select":
            picker.value = "second"
        elif navigation == "reload":
            await _press(pilot, screen, "#mcp_refresh")
        else:
            screen.action_close()
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, ConfirmMCPDraftScreen) and app.screen.is_mounted)
        await _press(pilot, app.screen, "#mcp_draft_save")
        await _wait_for_layout(pilot, lambda: not screen.mcp_busy)
        if navigation == "close":
            await _wait_for_layout(pilot, lambda: app._settings_screen is None)
        else:
            assert picker.value == ("second" if navigation == "select" else "first-2")
            assert not screen.mcp_dirty
        assert _saved_url(path, "first-2") == "https://new.invalid/mcp"
        assert _saved_url(path, "first") == "https://first.invalid/mcp"


@pytest.mark.asyncio
async def test_project_server_stays_read_only_and_can_verify_saved_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    project_file = project_mcp_config_path(app.project_path)
    save_config_file(
        project_file, MCPConfigFile(servers=[MCPServerConfig(id="project", url="https://project.invalid/mcp")])
    )
    app.settings.trust_mcp_project(app.project_path)
    app.settings_store.save(app.settings)
    verify = AsyncMock(return_value=MCPVerificationResult("project", True, "Verified", 1))
    monkeypatch.setattr(MCPService, "verify_server", verify)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot, server="project")
        assert screen.query_one("#mcp_url_input", Input).disabled
        assert screen.query_one("#mcp_save_server", Button).disabled
        assert screen.query_one("#mcp_delete_server", Button).disabled
        before = path.read_bytes(), project_file.read_bytes()
        assert not await app._save_mcp_server_from_ui()
        await _press(pilot, screen, "#mcp_verify_server")
        await _wait_for_layout(pilot, lambda: verify.await_count == 1 and not screen.mcp_busy)
        assert verify.await_args is not None and verify.await_args.args == ("project",)
        assert (path.read_bytes(), project_file.read_bytes()) == before
        assert not screen.mcp_dirty
        screen.query_one("#mcp_server_select", Select).value = "second"
        await _wait_for_layout(pilot, lambda: screen.mcp_server_id == "second")
        assert not screen.query_one("#mcp_url_input", Input).disabled


@pytest.mark.asyncio
async def test_save_and_close_does_not_dismiss_a_screen_opened_during_the_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, path = _configured_mcp_app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_mcp(app, pilot)
        screen.query_one("#mcp_url_input", Input).value = "https://edited.invalid/mcp"
        await pilot.pause()
        original_ensure = app._ensure_agent_from_settings

        async def gated_ensure(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await release.wait()
            await original_ensure(*args, **kwargs)

        monkeypatch.setattr(app, "_ensure_agent_from_settings", gated_ensure)
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, ConfirmMCPDraftScreen) and app.screen.is_mounted)
        await _press(pilot, app.screen, "#mcp_draft_save")
        await _wait_for_layout(pilot, entered.is_set)
        try:
            categories = screen.query_one("#settings_categories", OptionList)
            categories.highlighted = next(i for i, (_, value) in enumerate(SETTINGS_CATEGORIES) if value == "memory")
            categories.focus()
            await pilot.press("enter")
            await _wait_for_layout(pilot, lambda: screen.category == "memory")
            await _press(pilot, screen, "#memory_settings_browse")
            await _wait_for_layout(pilot, lambda: isinstance(app.screen, MemoryScreen) and app.screen.is_mounted)
            memory_screen = app.screen
            assert screen.mcp_busy
        finally:
            release.set()
            await _wait_for_layout(pilot, lambda: not screen.mcp_busy)
        assert app.screen is memory_screen
        assert app._settings_screen is screen
        assert screen in app.screen_stack
        assert not screen.mcp_dirty
        assert _saved_url(path) == "https://edited.invalid/mcp"
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: app.screen is screen)
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: app._settings_screen is None)
