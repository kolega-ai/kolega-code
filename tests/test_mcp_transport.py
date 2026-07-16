from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

from kolega_code.mcp.config import LoadedMCPConfig, MCPOAuthConfig, MCPServerConfig, server_fingerprint
from kolega_code.mcp.service import (
    MCPService,
    MCP_FAILURE_MESSAGE_GENERIC,
    MCP_TOOL_FAILURE_MESSAGE_TIMEOUT,
    _is_github_copilot_api_url,
)
from kolega_code.mcp.state import MCPServerStatus, MCPOAuthTokenStore
from kolega_code.mcp.transport import open_mcp_session
from kolega_code.tools import ToolError


MCP_STDERR_SENTINEL = "raw stdio MCP child stderr should not reach terminal"


def _install_fake_stdio_session(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    import mcp.client.session
    import mcp.client.stdio

    errlogs: list[object] = []

    @asynccontextmanager
    async def fake_stdio_client(params, errlog=sys.stderr):
        errlogs.append(errlog)
        print(MCP_STDERR_SENTINEL, file=errlog)
        errlog.flush()
        yield object(), object()

    class FakeClientSession:
        def __init__(self, read_stream, write_stream, read_timeout_seconds=None):
            self.read_stream = read_stream
            self.write_stream = write_stream
            self.read_timeout_seconds = read_timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

        async def list_tools(self, params=None):
            tool = SimpleNamespace(
                name="ping",
                title="Ping",
                description="Ping test tool",
                input_schema={"type": "object", "properties": {}},
            )
            return SimpleNamespace(tools=[tool], next_cursor=None)

        async def call_tool(self, tool_name, arguments):
            assert tool_name == "ping"
            return SimpleNamespace(
                is_error=False,
                content=[SimpleNamespace(type="text", text="pong")],
                structured_content=None,
            )

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp.client.session, "ClientSession", FakeClientSession)
    return errlogs


def _stdio_server() -> MCPServerConfig:
    return MCPServerConfig(id="local-filesystem", transport="stdio", command="fake-mcp-server")


def _verified_service(tmp_path: Path, server: MCPServerConfig) -> MCPService:
    service = MCPService(
        LoadedMCPConfig(servers={server.id: server}),
        state_dir=tmp_path,
        project_path=tmp_path,
    )
    service.status_store.update(
        server.id,
        MCPServerStatus.verified(
            fingerprint=server_fingerprint(server),
            transport=server.transport,
            source=server.source,
            tools=[],
            oauth=server.oauth.enabled,
        ),
    )
    return service


def _install_fake_tool_result(monkeypatch: pytest.MonkeyPatch, result: object) -> list[tuple[str, dict]]:
    import kolega_code.mcp.service as service_module

    calls: list[tuple[str, dict]] = []

    class FakeSession:
        async def call_tool(self, tool_name, arguments):
            calls.append((tool_name, arguments))
            return result

    @asynccontextmanager
    async def fake_open_mcp_session(*args, **kwargs):
        yield FakeSession()

    monkeypatch.setattr(service_module, "open_mcp_session", fake_open_mcp_session)
    return calls


def test_github_copilot_api_url_matches_hostname_only() -> None:
    assert _is_github_copilot_api_url("https://api.githubcopilot.com/mcp/")
    assert _is_github_copilot_api_url("https://api.githubcopilot.com./mcp/")
    assert not _is_github_copilot_api_url("https://api.githubcopilot.com.evil.example/mcp/")
    assert not _is_github_copilot_api_url("https://evil.example/mcp/?next=api.githubcopilot.com")
    assert not _is_github_copilot_api_url("https://api.githubcopilot.com@evil.example/mcp/")
    assert not _is_github_copilot_api_url("not a url with api.githubcopilot.com")


@pytest.mark.asyncio
async def test_stdio_transport_passes_non_terminal_errlog_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    errlogs = _install_fake_stdio_session(monkeypatch)

    async with open_mcp_session(
        _stdio_server(),
        project_path=tmp_path,
        token_store=MCPOAuthTokenStore(tmp_path),
    ) as session:
        assert session is not None

    captured = capsys.readouterr()
    assert MCP_STDERR_SENTINEL not in captured.err
    assert errlogs
    assert errlogs[0] is not sys.stderr
    assert getattr(errlogs[0], "name", None) == os.devnull


@pytest.mark.asyncio
async def test_streamable_http_uses_first_two_streams_and_timedelta_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mcp.client.session
    import kolega_code.mcp.transport as transport_module

    read_stream = object()
    write_stream = object()
    get_session_id = object()
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_streamable_http_client(server, *, auth=None):
        captured["transport_server"] = server
        captured["auth"] = auth
        yield read_stream, write_stream, get_session_id

    class FakeClientSession:
        def __init__(self, read, write, read_timeout_seconds=None):
            captured["read"] = read
            captured["write"] = write
            captured["read_timeout_seconds"] = read_timeout_seconds
            captured["initialized"] = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            captured["initialized"] = True

    monkeypatch.setattr(transport_module, "_streamable_http_client", fake_streamable_http_client)
    monkeypatch.setattr(mcp.client.session, "ClientSession", FakeClientSession)
    server = MCPServerConfig(
        id="docs",
        transport="streamable_http",
        url="https://docs.example/mcp",
        timeout_seconds=12.5,
    )

    async with open_mcp_session(
        server,
        project_path=tmp_path,
        token_store=MCPOAuthTokenStore(tmp_path),
    ) as session:
        assert session is not None

    assert captured["transport_server"] is server
    assert captured["read"] is read_stream
    assert captured["write"] is write_stream
    assert captured["read_timeout_seconds"] == timedelta(seconds=12.5)
    assert captured["initialized"] is True


@pytest.mark.asyncio
async def test_verify_server_suppresses_stdio_child_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_stdio_session(monkeypatch)
    server = _stdio_server()
    service = MCPService(
        LoadedMCPConfig(servers={server.id: server}),
        state_dir=tmp_path,
        project_path=tmp_path,
    )

    result = await service.verify_server(server.id)

    captured = capsys.readouterr()
    assert result.ok is True
    assert result.tool_count == 1
    assert MCP_STDERR_SENTINEL not in captured.err


@pytest.mark.asyncio
async def test_call_tool_suppresses_stdio_child_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_stdio_session(monkeypatch)
    server = _stdio_server()
    service = MCPService(
        LoadedMCPConfig(servers={server.id: server}),
        state_dir=tmp_path,
        project_path=tmp_path,
    )
    verified = await service.verify_server(server.id)
    assert verified.ok is True
    capsys.readouterr()

    output = await service.call_tool(server.id, "ping", {})

    captured = capsys.readouterr()
    assert output == "pong"
    assert MCP_STDERR_SENTINEL not in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(
            content=[TextContent(type="text", text="pong")],
            structuredContent={"answer": 42},
            isError=False,
        ),
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="pong")],
            structured_content={"answer": 42},
            is_error=False,
        ),
    ],
    ids=["stable-camel-case", "snake-case-compatibility"],
)
async def test_call_tool_supports_structured_result_field_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    server = _stdio_server()
    service = _verified_service(tmp_path, server)
    calls = _install_fake_tool_result(monkeypatch, result)

    output = await service.call_tool(server.id, "ping", {})

    assert output == 'pong\n\nStructured content:\n{\n  "answer": 42\n}'
    assert calls == [("ping", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(
            content=[TextContent(type="text", text="remote tool failed")],
            isError=True,
        ),
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="remote tool failed")],
            structured_content=None,
            is_error=True,
        ),
    ],
    ids=["stable-camel-case", "snake-case-compatibility"],
)
async def test_call_tool_supports_error_result_field_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    server = _stdio_server()
    service = _verified_service(tmp_path, server)
    calls = _install_fake_tool_result(monkeypatch, result)

    with pytest.raises(ToolError, match="remote tool failed"):
        await service.call_tool(server.id, "ping", {})

    assert calls == [("ping", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("leaf_exception", "expected_message"),
    [
        (
            ValueError("too many values to unpack password=secret-token"),
            "MCP tool call failed (ValueError).",
        ),
        (
            TimeoutError("request timed out Authorization: Bearer secret-token"),
            MCP_TOOL_FAILURE_MESSAGE_TIMEOUT,
        ),
    ],
    ids=["unclassified-leaf-type", "classified-timeout"],
)
async def test_call_tool_sanitizes_nested_exception_groups_without_invalidating_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf_exception: Exception,
    expected_message: str,
) -> None:
    import kolega_code.mcp.service as service_module

    calls = 0

    @asynccontextmanager
    async def fake_open_mcp_session(*args, **kwargs):
        nonlocal calls
        calls += 1
        nested = ExceptionGroup("nested transport failure", [leaf_exception])
        raise ExceptionGroup("unhandled errors in a TaskGroup", [nested])
        yield  # pragma: no cover - required to make this an async context manager

    monkeypatch.setattr(service_module, "open_mcp_session", fake_open_mcp_session)
    server = MCPServerConfig(
        id="docs",
        transport="streamable_http",
        url="https://docs.example/mcp",
    )
    service = _verified_service(tmp_path, server)

    with pytest.raises(ToolError) as exc_info:
        await service.call_tool(server.id, "read_docs", {})

    message = str(exc_info.value)
    assert message == expected_message
    assert "TaskGroup" not in message
    assert "secret-token" not in message
    assert calls == 1
    assert service.is_verified(server) is True
    assert service.list_status_rows()[0]["status"] == "verified"


@pytest.mark.asyncio
async def test_verify_server_reports_exception_group_leaf_and_suppresses_sdk_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import kolega_code.mcp.service as service_module

    @asynccontextmanager
    async def fake_open_mcp_session(*args, **kwargs):
        try:
            raise RuntimeError("Registration failed: 404 404 page not found password=secret-token")
        except RuntimeError as exc:
            logging.getLogger("mcp.client.auth.oauth2").exception("OAuth flow error")
            raise ExceptionGroup("unhandled errors in a TaskGroup", [exc]) from exc
        yield  # pragma: no cover - required to make this an async context manager

    monkeypatch.setattr(service_module, "open_mcp_session", fake_open_mcp_session)
    caplog.set_level(logging.ERROR, logger="mcp.client.auth.oauth2")
    server = MCPServerConfig(
        id="github-copilot",
        transport="streamable_http",
        url="https://api.githubcopilot.com/mcp/",
        oauth=MCPOAuthConfig(enabled=True),
    )
    service = MCPService(
        LoadedMCPConfig(servers={server.id: server}),
        state_dir=tmp_path,
        project_path=tmp_path,
    )

    result = await service.verify_server(server.id, interactive_oauth=True)

    assert result.ok is False
    assert "GitHub remote MCP OAuth failed" in result.message
    assert "Registration failed" not in result.message
    assert "secret-token" not in result.message
    assert "TaskGroup" not in result.message
    assert service.list_status_rows()[0]["message"] == result.message
    assert "secret-token" not in service.list_status_rows()[0]["message"]
    assert "OAuth flow error" not in caplog.text


def test_list_status_rows_replaces_legacy_failed_status_messages(tmp_path: Path) -> None:
    server = MCPServerConfig(
        id="docs",
        transport="streamable_http",
        url="https://docs.example/mcp/",
        oauth=MCPOAuthConfig(enabled=True),
    )
    service = MCPService(
        LoadedMCPConfig(servers={server.id: server}),
        state_dir=tmp_path,
        project_path=tmp_path,
    )
    service.status_store.update(
        server.id,
        MCPServerStatus.failed(
            fingerprint=server_fingerprint(server),
            transport=server.transport,
            source=server.source,
            message="RuntimeError: Registration failed with password=legacy-secret",
            oauth=True,
        ),
    )

    rows = service.list_status_rows()

    assert rows[0]["status"] == "failed"
    assert rows[0]["message"] == MCP_FAILURE_MESSAGE_GENERIC
    assert "legacy-secret" not in rows[0]["message"]
