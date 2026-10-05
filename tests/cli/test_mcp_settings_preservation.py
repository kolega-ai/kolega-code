"""Saving an existing MCP server must not reset settings absent from the editor."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, Select

from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.tui.settings_screen import SettingsScreen
from kolega_code.mcp.config import (
    MCPConfigFile,
    MCPOAuthConfig,
    MCPServerConfig,
    global_mcp_config_path,
    save_config_file,
)

from .test_tui_settings_screens import _configured_app


async def _wait_for_layout(pilot: Pilot, predicate: Callable[[], bool], *, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError("MCP settings form did not settle")


def _app_with_saved_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[KolegaCodeApp, Path]:
    # _configured_app installs fake agents: no server connection or OAuth flow.
    app, store = _configured_app(tmp_path, monkeypatch)
    path = global_mcp_config_path(store.root)
    save_config_file(
        path,
        MCPConfigFile(
            servers=[
                MCPServerConfig(
                    id="preserved-server",
                    name="Saved server",
                    url="https://mcp.example.invalid/original",
                    headers={"X-Fake-Key": "fake-header-secret"},
                    timeout_seconds=77,
                    sse_read_timeout_seconds=888,
                    oauth=MCPOAuthConfig(
                        enabled=True,
                        client_id="fake-client-id",
                        client_secret="fake-client-secret",
                        client_secret_env="FAKE_MCP_CLIENT_SECRET",
                        redirect_uri="http://127.0.0.1:33418/callback",
                        scope="fake.read fake.write",
                        token_endpoint_auth_method="client_secret_post",
                        timeout_seconds=619,
                        client_name="Synthetic MCP client",
                        client_uri="https://client.example.invalid/",
                        client_metadata_url="https://client.example.invalid/metadata.json",
                    ),
                )
            ]
        ),
    )
    return app, path


async def _open_saved_server(app: KolegaCodeApp, pilot: Pilot) -> SettingsScreen:
    app.action_open_settings(category="mcp")
    await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    screen.query_one("#mcp_server_select", Select).value = "preserved-server"
    await _wait_for_layout(
        pilot,
        lambda: (
            screen.query_one("#mcp_server_select", Select).value == "preserved-server"
            and screen.query_one("#mcp_name_input", Input).value == "Saved server"
            and screen.query_one("#mcp_url_input", Input).value == "https://mcp.example.invalid/original"
            and screen.query_one("#mcp_transport_select", Select).value == "streamable_http"
            and screen.query_one("#mcp_oauth_select", Select).value == "true"
            and screen.query_one("#mcp_oauth_client_secret_input", Input).value == "fake-client-secret"
            and screen.query_one("#mcp_oauth_client_secret_input", Input).display
        ),
    )
    return screen


def _saved_server(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["servers"]) == 1
    server = payload["servers"][0]
    assert server["id"] == "preserved-server"
    return server


def _assert_oauth_metadata(oauth: dict[str, Any]) -> None:
    # Independent expectations, not a comparison to the collector's output.
    assert oauth["timeout_seconds"] == 619
    assert oauth["client_name"] == "Synthetic MCP client"
    assert oauth["client_uri"] == "https://client.example.invalid/"
    assert oauth["client_metadata_url"] == "https://client.example.invalid/metadata.json"


@pytest.mark.asyncio
async def test_editing_saved_mcp_preserves_connection_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    app, path = _app_with_saved_server(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_saved_server(app, pilot)
        screen.query_one("#mcp_name_input", Input).value = "Renamed server"
        screen.query_one("#mcp_url_input", Input).value = "https://mcp.example.invalid/edited"
        await pilot.pause()
        await app._save_mcp_server_from_ui()

        saved = _saved_server(path)
        assert saved["name"] == "Renamed server"
        assert saved["url"] == "https://mcp.example.invalid/edited"
        assert saved["transport"] == "streamable_http"
        assert saved["headers"] == {"X-Fake-Key": "fake-header-secret"}
        assert (saved["timeout_seconds"], saved["sse_read_timeout_seconds"]) == (77, 888)


@pytest.mark.asyncio
@pytest.mark.parametrize("oauth_enabled", [True, False], ids=["oauth-enabled", "oauth-disabled"])
async def test_editing_saved_mcp_preserves_unexposed_oauth_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    oauth_enabled: bool,
) -> None:
    app, path = _app_with_saved_server(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_saved_server(app, pilot)
        screen.query_one("#mcp_oauth_select", Select).value = str(oauth_enabled).lower()
        await _wait_for_layout(
            pilot,
            lambda: screen.query_one("#mcp_oauth_client_secret_input", Input).display == oauth_enabled,
        )
        screen.query_one("#mcp_name_input", Input).value = "OAuth server edited"
        screen.query_one("#mcp_url_input", Input).value = "https://mcp.example.invalid/oauth-edited"
        await pilot.pause()
        await app._save_mcp_server_from_ui()

        saved = _saved_server(path)
        assert saved["name"] == "OAuth server edited"
        assert saved["url"] == "https://mcp.example.invalid/oauth-edited"
        oauth = saved["oauth"]
        assert oauth["enabled"] is oauth_enabled
        _assert_oauth_metadata(oauth)
        assert oauth["client_id"] == "fake-client-id"
        assert oauth["client_secret"] == "fake-client-secret"
        assert oauth["client_secret_env"] == "FAKE_MCP_CLIENT_SECRET"
        assert oauth["redirect_uri"] == "http://127.0.0.1:33418/callback"
        assert oauth["scope"] == "fake.read fake.write"
        assert oauth["token_endpoint_auth_method"] == "client_secret_post"


@pytest.mark.asyncio
async def test_switching_saved_mcp_to_stdio_preserves_unexposed_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    app, path = _app_with_saved_server(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_saved_server(app, pilot)
        screen.query_one("#mcp_transport_select", Select).value = "stdio"
        await _wait_for_layout(
            pilot,
            lambda: (
                screen.query_one("#mcp_command_input", Input).display
                and not screen.query_one("#mcp_oauth_select", Select).display
            ),
        )
        screen.query_one("#mcp_name_input", Input).value = "Local server edited"
        screen.query_one("#mcp_command_input", Input).value = "fake-mcp-command"
        await pilot.pause()
        await app._save_mcp_server_from_ui()

        saved = _saved_server(path)
        assert saved["name"] == "Local server edited"
        assert saved["transport"] == "stdio"
        assert saved["command"] == "fake-mcp-command"
        assert saved["oauth"]["enabled"] is False
        _assert_oauth_metadata(saved["oauth"])
        assert (saved["timeout_seconds"], saved["sse_read_timeout_seconds"]) == (77, 888)
