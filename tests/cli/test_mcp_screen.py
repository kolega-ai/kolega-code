"""The ``/mcp`` panel, its text forms, and the Settings page's scope-aware MCP edits."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from rich.text import Text
from textual.widgets import Button, Input, OptionList, Select, SelectionList, Static

from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.settings import SettingsStore
from kolega_code.cli.tui import mcp_ops
from kolega_code.cli.tui import mcp_import_screen as import_screen_module
from kolega_code.cli.tui.mcp_import_screen import MCPImportScreen
from kolega_code.cli.tui.mcp_screen import MCPScreen
from kolega_code.cli.tui.settings_screen import ConfirmSettingsActionScreen, SettingsScreen
from kolega_code.mcp.config import (
    MCPConfigFile,
    MCPServerConfig,
    global_mcp_config_path,
    save_config_file,
    server_fingerprint,
)
from kolega_code.mcp.failure_messages import MCP_FAILURE_MESSAGE_GENERIC
from kolega_code.mcp.importers import ImportCandidate, ImportSource
from kolega_code.mcp.service import MCPService, MCPVerificationResult
from kolega_code.mcp.state import MCPServerStatus, MCPToolStatus

from ._app_test_utils import install_fake_agents, wait_for_layout as _wait_for_layout
from .test_tui_settings_screens import _configured_app

SHARED_FILE = '{"mcpServers": {"repo": {"command": "npx", "args": ["repo-tools"]}}}\n'


def _mcp_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[KolegaCodeApp, SettingsStore]:
    """A project with two user servers and an untrusted shared ``.mcp.json``."""
    app, settings_store = _configured_app(tmp_path, monkeypatch)
    save_config_file(
        global_mcp_config_path(settings_store.root),
        MCPConfigFile(
            servers=[
                MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.invalid/mcp"),
                MCPServerConfig(id="off", transport="streamable_http", url="https://off.invalid/mcp", enabled=False),
            ]
        ),
    )
    (app.project_path / ".mcp.json").write_text(SHARED_FILE, encoding="utf-8")
    return app, settings_store


def _text(widget: Static) -> str:
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


def _rows(screen: MCPScreen) -> list[str]:
    listing = screen.query_one("#mcp_panel_servers", OptionList)
    return [_text_of(option.prompt) for option in listing.options]


def _text_of(visual: object) -> str:
    return visual.plain if isinstance(visual, Text) else str(visual)


def _row(screen: MCPScreen, name: str) -> str:
    return next(row for row in _rows(screen) if row.split()[1] == name)


async def _open_panel(app: KolegaCodeApp, pilot: Any) -> MCPScreen:
    def listed() -> bool:
        screen = app.screen
        return isinstance(screen, MCPScreen) and bool(screen.query("#mcp_panel_servers")) and len(_rows(screen)) == 3

    app.action_open_mcp()
    await _wait_for_layout(pilot, listed)
    screen = app.screen
    assert isinstance(screen, MCPScreen)
    return screen


async def _select(screen: MCPScreen, pilot: Any, name: str) -> None:
    listing = screen.query_one("#mcp_panel_servers", OptionList)
    index = next(position for position, row in enumerate(_rows(screen)) if row.split()[1] == name)
    listing.highlighted = index
    await _wait_for_layout(pilot, lambda: listing.highlighted == index)


async def _confirm(pilot: Any) -> None:
    await _wait_for_layout(pilot, lambda: isinstance(pilot.app.screen, ConfirmSettingsActionScreen))
    button = pilot.app.screen.query_one("#settings_action_confirm", Button)
    button.focus()
    await _wait_for_layout(pilot, lambda: button.has_focus)
    await pilot.press("enter")


def _fake_verify(monkeypatch: pytest.MonkeyPatch, *, ok: bool = True) -> list[tuple[str, dict[str, Any]]]:
    """Replace verification with a canned result that also writes the status it implies."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_verify(self: MCPService, server_id: str, **kwargs: Any) -> MCPVerificationResult:
        calls.append((server_id, kwargs))
        server = self.config.servers[server_id]
        fingerprint = server_fingerprint(server)
        status = (
            MCPServerStatus.verified(
                fingerprint=fingerprint,
                transport=server.transport,
                source=server.source,
                tools=[MCPToolStatus(id="ping", description="Ping it")],
            )
            if ok
            else MCPServerStatus.failed(
                fingerprint=fingerprint,
                transport=server.transport,
                source=server.source,
                message=MCP_FAILURE_MESSAGE_GENERIC,
            )
        )
        self.status_store.update(server_id, status)
        return MCPVerificationResult(
            server_id, ok, "Verified 1 tool(s)." if ok else MCP_FAILURE_MESSAGE_GENERIC, 1 if ok else 0
        )

    monkeypatch.setattr(MCPService, "verify_server", fake_verify)
    return calls


@pytest.mark.asyncio
async def test_mcp_panel_lists_every_scope_and_waits_for_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)

        assert [row.split()[1] for row in _rows(screen)] == ["docs", "off", "repo"]
        assert "user" in _row(screen, "docs") and "Not connected" in _row(screen, "docs")
        assert "Disabled" in _row(screen, "off")
        assert "project" in _row(screen, "repo") and "Pending approval" in _row(screen, "repo")
        assert "project not trusted" in _text(screen.query_one("#mcp_panel_title", Static))

        await _select(screen, pilot, "repo")
        await _wait_for_layout(
            pilot, lambda: "Pending approval" in _text(screen.query_one("#mcp_panel_detail", Static))
        )
        detail = _text(screen.query_one("#mcp_panel_detail", Static))
        assert str(app.project_path / ".mcp.json") in detail
        assert "npx repo-tools" in detail


@pytest.mark.asyncio
async def test_mcp_panel_reconnect_verifies_the_selected_server_and_lists_its_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)
    calls = _fake_verify(monkeypatch)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)

        await pilot.press("v")
        await _wait_for_layout(pilot, lambda: len(calls) == 1 and not app._mcp_operation_in_flight())

        assert calls[0][0] == "docs"
        assert calls[0][1] == {"interactive_oauth": True, "open_browser": True, "output": app.console}
        await _wait_for_layout(pilot, lambda: "Connected" in _row(screen, "docs"))
        assert "1 tool" in _row(screen, "docs")

        tools = screen.query_one("#mcp_panel_tools", OptionList)
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: tools.display and len(tools.options) == 1)
        assert _text_of(tools.options[0].prompt) == "mcp__docs__ping  Ping it"
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: not tools.display)

        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: app.screen is not screen)
        assert app._mcp_screen is None


@pytest.mark.asyncio
async def test_mcp_panel_trust_then_disable_never_edits_the_shared_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, settings_store = _mcp_app(tmp_path, monkeypatch)
    shared = app.project_path / ".mcp.json"
    original = shared.read_bytes()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)
        await _select(screen, pilot, "repo")

        await pilot.press("t")
        await _confirm(pilot)
        await _wait_for_layout(
            pilot,
            lambda: (
                app.settings.is_mcp_project_trusted(app.project_path)
                and not app._mcp_operation_in_flight()
                and app.screen is screen
            ),
        )
        assert "project trusted" in _text(screen.query_one("#mcp_panel_title", Static))
        await _wait_for_layout(pilot, lambda: "Not connected" in _row(screen, "repo"))

        await pilot.press("space")
        await _wait_for_layout(
            pilot, lambda: "disabled for this project" in _row(screen, "repo") and not app._mcp_operation_in_flight()
        )

        assert shared.read_bytes() == original
        saved = SettingsStore(settings_store.root).load()
        assert saved.mcp_project_server_overrides == {str(app.project_path.resolve()): {"repo": False}}
        assert saved.is_mcp_project_trusted(app.project_path)

        await pilot.press("space")
        await _wait_for_layout(pilot, lambda: "Not connected" in _row(screen, "repo"))
        assert SettingsStore(settings_store.root).load().mcp_project_server_overrides == {
            str(app.project_path.resolve()): {"repo": True}
        }


@pytest.mark.asyncio
async def test_mcp_panel_refuses_changes_while_a_turn_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)
    calls = _fake_verify(monkeypatch)
    notices: list[str] = []

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)
        monkeypatch.setattr(app, "_notify_user", lambda message, **kwargs: notices.append(message))
        app._turn_active = True
        try:
            await pilot.press("v")
            await pilot.press("space")
            await pilot.press("t")
            await pilot.pause()
        finally:
            app._turn_active = False

        assert calls == []
        assert notices and all(message == mcp_ops.BLOCKED_MESSAGE for message in notices)
        assert not app._mcp_operation_in_flight()
        assert app.screen is screen


@pytest.mark.asyncio
async def test_mcp_panel_operation_holds_peer_turns_and_mode_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.session.inbox import PeerMessage

    from ._app_test_utils import FakeCoderAgent, wait_for_turn_idle

    app, _ = _mcp_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_verify(self: MCPService, server_id: str, **kwargs: Any) -> MCPVerificationResult:
        started.set()
        await release.wait()
        return MCPVerificationResult(server_id, False, MCP_FAILURE_MESSAGE_GENERIC)

    monkeypatch.setattr(MCPService, "verify_server", slow_verify)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)
        await pilot.press("v")
        await _wait_for_layout(pilot, started.is_set)
        try:
            assert app._mcp_operation_in_flight()
            assert not app._loop_ready_to_fire()
            assert app._mode_switch_blocked()
            assert not app._maybe_start_queued_message()
            for text in ("first peer message", "second peer message"):
                app._enqueue_peer_message(
                    PeerMessage.create(sender_session_id="fake-peer", sender_title="Fake peer", text=text)
                )
            await pilot.pause()
            assert len(app._queued_messages) == 2
            mode = app.interaction_mode
            await pilot.press("shift+tab")
            assert app.interaction_mode == mode
            await pilot.press("escape")
            assert app.screen is screen
            assert len(app._queued_messages) == 2
        finally:
            release.set()
            await _wait_for_layout(pilot, lambda: not app._mcp_operation_in_flight())
        await _wait_for_layout(pilot, lambda: isinstance(app.agent, FakeCoderAgent) and len(app.agent.messages) == 2)
        await wait_for_turn_idle(app, pilot)


@pytest.mark.asyncio
async def test_mcp_panel_clear_sign_in_forgets_tokens_and_cached_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.mcp.state import MCPOAuthTokenStore, MCPStatusStore

    app, settings_store = _mcp_app(tmp_path, monkeypatch)
    docs = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.invalid/mcp")
    MCPStatusStore(settings_store.root).update(
        "docs",
        MCPServerStatus.verified(
            fingerprint=server_fingerprint(docs),
            transport=docs.transport,
            source="global",
            tools=[MCPToolStatus(id="ping")],
            oauth=True,
        ),
    )
    MCPOAuthTokenStore(settings_store.root).set_tokens("docs", {"access_token": "a", "refresh_token": "r"})

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)
        await _wait_for_layout(pilot, lambda: "Connected" in _row(screen, "docs"))

        await pilot.press("c")
        await _confirm(pilot)
        await _wait_for_layout(
            pilot, lambda: "Not connected" in _row(screen, "docs") and not app._mcp_operation_in_flight()
        )

        assert MCPStatusStore(settings_store.root).get("docs") is None
        assert MCPOAuthTokenStore(settings_store.root).secret_values() == []


@pytest.mark.asyncio
async def test_mcp_text_forms_reconnect_enable_and_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)
    calls = _fake_verify(monkeypatch)

    def last_entry() -> str:
        return app.conversation_entries[-1].content

    async with app.run_test(size=(120, 40)):
        await app._command_mcp("reconnect all")

        # Disabled servers and servers waiting for trust are not retried.
        assert [server_id for server_id, _ in calls] == ["docs"]
        assert "✓ docs: Verified 1 tool(s)." in last_entry()

        calls.clear()
        await app._command_mcp("reconnect all")
        assert calls == []
        assert "already connected" in last_entry()

        await app._command_mcp("disable docs")
        assert mcp_ops.load_config(app).servers["docs"].enabled is False
        assert "Disabled docs" in last_entry()
        await app._command_mcp("enable docs")
        assert mcp_ops.load_config(app).servers["docs"].enabled is True

        await app._command_mcp("enable")
        assert last_entry().startswith("Usage: /mcp")
        await app._command_mcp("disable ghost")
        assert "MCP server not found: ghost" in last_entry()

        app._turn_active = True
        try:
            await app._command_mcp("disable docs")
        finally:
            app._turn_active = False
        assert last_entry() == mcp_ops.BLOCKED_MESSAGE
        assert mcp_ops.load_config(app).servers["docs"].enabled is True
        assert not app._mcp_operation_in_flight()


def test_mcp_is_a_tui_slash_command() -> None:
    from kolega_code.cli.slash_commands import TUI_COMMAND_ENTRIES

    entry = next(entry for entry in TUI_COMMAND_ENTRIES if entry.name == "mcp")
    assert "reconnect" in entry.description
    assert "/mcp" in KolegaCodeApp._tui_command_handlers(KolegaCodeApp.__new__(KolegaCodeApp))


def test_settings_status_block_labels_servers_waiting_for_trust() -> None:
    from kolega_code.cli.tui import settings_panel

    rows = [
        {
            "id": "repo-tools",
            "name": "Repo Tools",
            "scope": "project",
            "transport": "stdio",
            "enabled": True,
            "oauth": False,
            "status": "pending_trust",
            "tool_count": 0,
            "message": "",
        }
    ]
    sep = settings_panel._mcp_separator()

    content, tone = settings_panel._render_mcp_status_text([], rows)

    assert tone == "warning"
    assert f"pending approval{sep}project{sep}stdio" in content.plain
    assert "Trust the project to load this server." in content.plain


async def _press(pilot: Any, selector: str) -> None:
    button = pilot.app.screen.query_one(selector, Button)
    button.focus()
    await _wait_for_layout(pilot, lambda: button.has_focus)
    await pilot.press("enter")


@pytest.mark.asyncio
async def test_settings_saves_and_deletes_local_servers_in_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, settings_store = _mcp_app(tmp_path, monkeypatch)
    user_file = global_mcp_config_path(settings_store.root)
    original = user_file.read_bytes()

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings(category="mcp")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        scope = screen.query_one("#mcp_scope_select", Select)
        assert (scope.display, str(scope.value)) == (True, "user")

        screen.query_one("#mcp_name_input", Input).value = "My Local"
        screen.query_one("#mcp_transport_select", Select).value = "stdio"
        screen.query_one("#mcp_command_input", Input).value = "uvx"
        scope.value = "local"
        await app._handle_mcp_settings_button("mcp_save_server")

        config = mcp_ops.load_config(app)
        saved = next(server for server in config.servers.values() if server.name == "My Local")
        assert (saved.scope, saved.command) == ("local", "uvx")
        assert user_file.read_bytes() == original
        key = str(app.project_path.resolve())
        assert list(SettingsStore(settings_store.root).load().mcp_local_servers[key]) == [saved.id]
        assert scope.display is False

        screen.query_one("#mcp_server_select", Select).value = saved.id
        await _wait_for_layout(pilot, lambda: screen.mcp_server_id == saved.id)
        assert screen._mcp_read_only is False
        assert "local" in _text(screen.query_one("#mcp_source_hint", Static))

        await app._handle_mcp_settings_button("mcp_delete_server")
        assert SettingsStore(settings_store.root).load().mcp_local_servers == {}
        assert user_file.read_bytes() == original


@pytest.mark.asyncio
async def test_settings_keeps_shared_project_servers_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, settings_store = _mcp_app(tmp_path, monkeypatch)
    app.settings.trust_mcp_project(app.project_path)
    settings_store.save(app.settings)
    shared = app.project_path / ".mcp.json"
    original = shared.read_bytes()

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings(category="mcp")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#mcp_server_select", Select).value = "repo"
        await _wait_for_layout(pilot, lambda: screen.mcp_server_id == "repo")

        assert screen._mcp_read_only is True
        assert screen.query_one("#mcp_scope_select", Select).display is False
        assert "read-only" in _text(screen.query_one("#mcp_source_hint", Static))
        await app._handle_mcp_settings_button("mcp_save_server")
        assert "read-only" in str(screen.query_one("#mcp_status", Static).render()).lower()
        assert shared.read_bytes() == original


@pytest.mark.asyncio
async def test_startup_notice_points_at_the_panel_for_untrusted_project_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)
    # A real launch loads the project's config after the files exist; the fixture wrote
    # the shared file after building the launch config.
    assert app.config is not None
    app.config.mcp_config = mcp_ops.load_config(app)
    notices: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(app, "_notify_user", lambda *args, **kwargs: notices.append((args, kwargs)))

    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_for_layout(
            pilot, lambda: any(kwargs.get("title") == "Untrusted project MCP servers" for _, kwargs in notices)
        )

        notice = next(args for args, kwargs in notices if kwargs.get("title") == "Untrusted project MCP servers")
        kwargs = next(kwargs for _, kwargs in notices if kwargs.get("title") == "Untrusted project MCP servers")
        assert notice == (
            "This project defines 1 MCP server, but they are not trusted, so none are loaded. "
            "Run /mcp to review and trust them.",
        )
        assert kwargs["severity"] == "warning"
        assert app._mcp_notice_shown is True


@pytest.mark.asyncio
async def test_mcp_panel_closes_when_the_project_changes_and_never_stacks_on_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_panel(app, pilot)
        app.action_open_mcp()
        assert app.screen is screen

        app._dismiss_mcp_screen()
        await _wait_for_layout(pilot, lambda: app.screen is not screen)
        assert app._mcp_screen is None

        app.action_open_settings(category="mcp")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
        app.action_open_mcp()
        assert isinstance(app.screen, SettingsScreen)
        assert app._mcp_screen is None


@pytest.mark.asyncio
async def test_mcp_import_screen_offers_found_sources_and_writes_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app, _ = _mcp_app(tmp_path, monkeypatch)

    def found(name: str, url: str) -> ImportCandidate:
        return ImportCandidate("cursor", "Cursor", "user", name, server=MCPServerConfig(id=name, url=url))

    source = ImportSource(
        "cursor",
        "Cursor",
        (
            found("one", "https://one.invalid/mcp"),
            found("docs", "https://another.invalid/mcp"),
            ImportCandidate("cursor", "Cursor", "user", "socket", unsupported="unsupported WebSocket type"),
        ),
    )
    monkeypatch.setattr(import_screen_module, "discover_import_sources", lambda project: [source])
    shared = app.project_path / ".mcp.json"

    async with app.run_test(size=(120, 40)) as pilot:
        await app._command_mcp("import")

        def sources_listed() -> bool:
            screen = app.screen
            return (
                isinstance(screen, MCPImportScreen)
                and bool(screen.query("#mcp_import_sources"))
                and screen.query_one("#mcp_import_sources", SelectionList).option_count == 1
            )

        await _wait_for_layout(pilot, sources_listed)
        screen = app.screen
        assert isinstance(screen, MCPImportScreen)
        assert "Found 3 MCP servers in 1 tool" in _text(screen.query_one("#mcp_import_detail", Static))

        await _press(pilot, "#mcp_import_next")
        servers = screen.query_one("#mcp_import_servers", SelectionList)
        await _wait_for_layout(pilot, lambda: servers.display and servers.option_count == 3)

        def states() -> dict[str, tuple[bool, bool]]:
            listing = screen.query_one("#mcp_import_servers", SelectionList)
            selected = {str(value) for value in listing.selected}
            return {
                str(option.value): (str(option.value) in selected, bool(option.disabled))
                for option in (listing.get_option_at_index(index) for index in range(listing.option_count))
            }

        # New servers are selected; one already defined in the target scope and an
        # unsupported entry are listed but cannot be chosen.
        assert states() == {"cursor:0": (True, False), "cursor:1": (False, True), "cursor:2": (False, True)}
        assert "1 of 3 servers will be added" in _text(screen.query_one("#mcp_import_detail", Static))

        await _press(pilot, "#mcp_import_scope_toggle")
        await _wait_for_layout(pilot, lambda: states()["cursor:1"] == (True, False))
        assert "project" in str(screen.query_one("#mcp_import_scope_toggle", Button).label)

        # Textual drops a press while the button's short "active" animation is still running.
        await _wait_for_layout(pilot, lambda: not screen.query_one("#mcp_import_next", Button).has_class("-active"))
        await _press(pilot, "#mcp_import_next")
        await _wait_for_layout(pilot, lambda: app.screen is not screen and not app._mcp_operation_in_flight())

        written = json.loads(shared.read_text(encoding="utf-8"))["mcpServers"]
        assert set(written) == {"repo", "one", "docs"}
        assert written["repo"] == {"command": "npx", "args": ["repo-tools"]}
        assert not app.settings.is_mcp_project_trusted(app.project_path)
