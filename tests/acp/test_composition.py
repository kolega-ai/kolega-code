from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer

from kolega_code.acp import agent_factory
from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.mcp import build_session_mcp_extension, merge_mcp_config, normalize_client_servers
from kolega_code.acp.server import AcpAgent
from kolega_code.agent import CoderAgent, PlanningAgent
from kolega_code.agent.tools import ToolCollection
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.hooks import HookEvent
from kolega_code.hooks.outcome import HookOutcome
from kolega_code.llm.models import Message, TextBlock
from kolega_code.mcp.config import (
    LoadedMCPConfig,
    MCPConfigFile,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    project_mcp_config_path,
    save_config_file,
    server_fingerprint,
)
from kolega_code.mcp.service import MCPService
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore, MCPToolStatus
from kolega_code.permissions import PermissionDecision, PermissionMode
from kolega_code.tools import ToolError

from tests.acp.test_server import _FakeConn

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.parametrize("retired", [False, True])
def test_run_acp_eof_joins_finite_cleanup_before_closing_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retired: bool
) -> None:
    import acp
    import acp.stdio

    from kolega_code.cli.acp import run_acp

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-composition-provider-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(ToolCollection, "initialize", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    monkeypatch.setattr(acp.stdio, "stdio_streams", AsyncMock(return_value=(object(), object())))
    records: dict[str, Any] = {}
    steps: list[str] = []

    async def transport_eof(server: AcpAgent, **kwargs: Any) -> None:
        server.on_connect(cast(Client, _FakeConn()))
        monkeypatch.setattr(server, "_schedule_open_updates", lambda _sid: None)
        opened = await server.new_session(str(tmp_path))
        session = server._sessions[opened.session_id]
        records.update(server=server, session=session)
        original_cleanup = session.agent.cleanup
        assert session.accounting is not None
        close_accounting = AsyncMock(wraps=session.accounting.aclose)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        records["accounting_close"] = close_accounting

        async def finite_cleanup() -> None:
            steps.append("cleanup-started")
            try:
                await asyncio.sleep(0.15)
                close_accounting.assert_not_awaited()
                assert server._factory._acp_sessions[session.session_id] is session
                await original_cleanup()
                steps.append("cleanup-completed")
            except asyncio.CancelledError:
                steps.append("cleanup-cancelled")
                raise

        monkeypatch.setattr(session.agent, "cleanup", finite_cleanup)
        if retired:
            await server.set_session_mode(session.session_id, "plan")
            assert session.cleanup_tasks

    monkeypatch.setattr(acp, "run_agent", transport_eof)
    assert run_acp(SimpleNamespace(debug=False, acp_log=None, permission_mode="ask")) == 0
    session = records["session"]
    server = records["server"]
    assert steps == ["cleanup-started", "cleanup-completed"]
    assert session.close_task.done() and not session.close_task.cancelled()
    assert not session.cleanup_tasks and not session.retired_agents
    assert session.session_id not in server._sessions
    assert session.session_id not in server._factory._acp_sessions
    records["accounting_close"].assert_awaited_once()


def _client(transport: str = "stdio", name: str = "client") -> Any:
    if transport == "stdio":
        return {
            "name": name,
            "command": "unused-mcp-command",
            "args": ["--credential", "fake-argument-secret"],
            "env": [{"name": "TOKEN", "value": "fake-env-secret"}],
        }
    return {
        "name": name,
        "type": transport,
        "url": "https://fake-user:fake-password@example.invalid/mcp?token=fake-query-secret",
        "headers": [{"name": "Authorization", "value": "Bearer fake-header-secret"}],
    }


def _verified(server: MCPServerConfig) -> MCPServerStatus:
    return MCPServerStatus.verified(
        fingerprint=server_fingerprint(server),
        transport=server.transport,
        source=server.source,
        tools=[MCPToolStatus(id="inspect.file", description="Inspect a file.")],
    )


def _context_events(path: Path) -> list[dict[str, Any]]:
    return [
        event
        for line in path.read_text().splitlines()
        if (event := json.loads(line))["type"].startswith("context.") or event["type"] == "turn.started"
    ]


@pytest.fixture
def configured_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentFactory:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-composition-provider-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(ToolCollection, "initialize", AsyncMock(return_value=[]))
    return AgentFactory()


@pytest.fixture
def discovery(monkeypatch: pytest.MonkeyPatch) -> list[tuple[MCPServerConfig, Path]]:
    calls: list[tuple[MCPServerConfig, Path]] = []

    @asynccontextmanager
    async def open_session(server: MCPServerConfig, **kwargs: Any) -> AsyncGenerator[Any, None]:
        calls.append((server, kwargs["project_path"]))
        yield SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(
                    tools=[SimpleNamespace(name="inspect.file", description="Inspect a file.", inputSchema={})]
                )
            ),
            call_tool=AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="inspected")])),
        )

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", open_session)
    return calls


@pytest.mark.parametrize("transport,model", [("stdio", McpServerStdio), ("http", HttpMcpServer), ("sse", SseMcpServer)])
@pytest.mark.parametrize("sdk_model", [False, True])
def test_client_transports_preserve_values_and_use_session_cwd(
    tmp_path: Path, transport: str, model: Any, sdk_model: bool
) -> None:
    payload = _client(transport)
    servers = normalize_client_servers([model.model_validate(payload) if sdk_model else payload], tmp_path)
    server = servers["client"]
    assert server.source == "acp"
    assert server.transport == {"stdio": "stdio", "http": "streamable_http", "sse": "sse"}[transport]
    if transport == "stdio":
        assert server.command == payload["command"]
        assert server.args == payload["args"]
        assert server.env == {"TOKEN": "fake-env-secret"}
        assert server.cwd == str(tmp_path)
    else:
        assert server.url == payload["url"]
        assert server.headers == {"Authorization": "Bearer fake-header-secret"}


@pytest.mark.parametrize(
    "payload",
    [
        {**_client(), "command": ""},
        {**_client(), "type": "fake-unknown-secret"},
        {**_client("http"), "headers": "fake-malformed-secret"},
        {**_client("http"), "url": None},
        {"name": "fake-incomplete-secret"},
    ],
)
def test_invalid_client_payloads_do_not_echo_any_input(tmp_path: Path, payload: dict[str, Any]) -> None:
    with pytest.raises(RequestError) as caught:
        normalize_client_servers([payload], tmp_path)
    assert caught.value.data == {"message": "Invalid ACP MCP server configuration."}
    assert caught.value.__suppress_context__
    assert "fake-" not in str(caught.value)


@pytest.mark.parametrize("trusted", [False, True])
def test_merge_honors_existing_trust_and_client_precedence(tmp_path: Path, trusted: bool) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    save_config_file(
        global_mcp_config_path(state),
        MCPConfigFile(
            servers=[MCPServerConfig(id=name, command="global", transport="stdio") for name in ("same", "global")]
        ),
    )
    save_config_file(
        project_mcp_config_path(project),
        MCPConfigFile(
            servers=[MCPServerConfig(id=name, command="project", transport="stdio") for name in ("same", "project")]
        ),
    )
    configured = load_mcp_config(project, state, project_trusted=trusted)
    clients = normalize_client_servers([_client(name="same"), _client(name="client")], project)
    merged = merge_mcp_config(configured, clients)
    assert merged.project_trusted is trusted
    assert set(merged.servers) == {"same", "global", "client"} | ({"project"} if trusted else set())
    assert merged.servers["same"] is clients["same"]
    assert configured.servers["same"].source == ("project" if trusted else "global")
    assert "client" not in configured.servers
    assert merge_mcp_config(
        LoadedMCPConfig(diagnostics=["fake-private-endpoint?token=fake-secret"]), {}
    ).diagnostics == ["MCP configuration warning; check the saved MCP configuration."]


@pytest.mark.asyncio
async def test_client_tools_discover_and_call_without_changing_disk_status(
    tmp_path: Path, discovery: list[Any]
) -> None:
    state = tmp_path / "state"
    saved = MCPServerConfig(id="global", command="unused", transport="stdio")
    status_store = MCPStatusStore(state)
    status_store.update(saved.id, _verified(saved))
    before = status_store.path.read_bytes()
    clients = normalize_client_servers([_client("http", "global"), _client("sse"), _client(name="stdio")], tmp_path)
    config = merge_mcp_config(LoadedMCPConfig(servers={"global": saved}), clients)
    extension = await build_session_mcp_extension(config, tmp_path, state)
    assert extension is not None
    try:
        names = {name: f"mcp__{server.id}__inspect_file" for name, server in clients.items()}
        assert set(extension.tools) == set(names.values())
        assert len(discovery) == 3
        assert extension.propagate_to_sub_agents is False
        assert await extension.tools[names["global"]](path="file") == "inspected"
        assert (
            await extension.tools[names["global"]](_server_id="stdio", _tool_id="unverified", path="file")
            == "inspected"
        )
        assert discovery[-1][0] is clients["global"]
        assert "server-side tool name is `inspect.file`" in extension.tool_descriptions[names["global"]]
        assert status_store.path.read_bytes() == before
        assert list(state.iterdir()) == [status_store.path]
    finally:
        assert extension.cleanup is not None
        await extension.cleanup()


@pytest.mark.asyncio
async def test_discovery_errors_are_safe_nonfatal_and_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @asynccontextmanager
    async def fail(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        logging.getLogger("httpx").info("HTTP Request: %s", _client("http")["url"])
        if args:
            raise RuntimeError(json.dumps(_client("http")))
        yield None

    cleanup = AsyncMock()
    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", fail)
    monkeypatch.setattr(MCPService, "cleanup", cleanup)
    clients = normalize_client_servers([_client("http")], tmp_path)
    with caplog.at_level(logging.DEBUG):
        extension = await build_session_mcp_extension(merge_mcp_config(None, clients), tmp_path, tmp_path / "state")
    assert extension is None
    assert "discovery failed" in caplog.text
    assert "fake-" not in caplog.text
    logging.getLogger("httpx").warning("unrelated HTTP request remains visible")
    assert "unrelated HTTP request remains visible" in caplog.text
    assert not (tmp_path / "state").exists()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_invocation_transport_failures_do_not_expose_client_secrets(
    tmp_path: Path,
    discovery: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients = normalize_client_servers([_client("http")], tmp_path)
    extension = await build_session_mcp_extension(merge_mcp_config(None, clients), tmp_path, tmp_path / "state")
    assert extension is not None

    @asynccontextmanager
    async def fail(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        logging.getLogger("httpx").warning("HTTP Request: %s", _client("http")["url"])
        if args:
            raise RuntimeError(json.dumps(_client("http")))
        yield None

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", fail)
    try:
        with caplog.at_level(logging.DEBUG), pytest.raises(ToolError) as caught:
            await extension.tools[f"mcp__{clients['client'].id}__inspect_file"]()
        assert "fake-" not in str(caught.value)
        assert "fake-" not in caplog.text
        assert not (tmp_path / "state").exists()
    finally:
        assert extension.cleanup is not None
        await extension.cleanup()


def _write_features(project: Path, state: Path) -> None:
    skill = project / ".agents/skills/demo/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Follow the demo workflow\n---\nDemo workflow steps.\n")
    for name, mode in (("builder", "build"), ("planner", "plan"), ("shared", "all")):
        definition = project / ".kolega/agents" / f"{name}.md"
        definition.parent.mkdir(parents=True, exist_ok=True)
        definition.write_text(
            f"---\nname: {name}\ndescription: Demo agent\nmode: {mode}\n---\nFollow the demo steps.\n"
        )
    for path in (state / "hooks.json", project / ".kolega/hooks.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "hooks": {
                        event.value: [{"hooks": [{"type": "command", "command": "unused-test-hook"}]}]
                        for event in (HookEvent.SESSION_START, HookEvent.SESSION_END, HookEvent.PRE_TOOL_USE)
                    },
                }
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted", [False, True])
async def test_factory_shared_composition_rebuild_restore_and_saved_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_factory: AgentFactory,
    discovery: list[Any],
    trusted: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SettingsStore()
    settings = CliSettings()
    if trusted:
        settings.trust_hook_project(project)
        settings.trust_mcp_project(project)
    store.save(settings)
    settings_before = store.path.read_bytes()
    _write_features(project, store.root)
    for path, name in ((global_mcp_config_path(store.root), "global"), (project_mcp_config_path(project), "project")):
        server = MCPServerConfig(id=name, command="unused", transport="stdio")
        save_config_file(path, MCPConfigFile(servers=[server]))
        MCPStatusStore(store.root).update(server.id, _verified(server))
    hook_events: list[tuple[HookEvent, list[str]]] = []

    async def hook(event: Any, spec: Any, caps: Any) -> HookOutcome:
        hook_events.append((event.name, [spec.scope]))
        return HookOutcome.empty()

    monkeypatch.setattr("kolega_code.hooks.dispatcher.run_hook", hook)
    callback = AsyncMock(return_value=PermissionDecision(allowed=False))
    factory = configured_factory
    prepared = Mock(wraps=agent_factory.prepare_composition)
    constructed = Mock(wraps=agent_factory.construct_agent)
    monkeypatch.setattr(agent_factory, "prepare_composition", prepared)
    monkeypatch.setattr(agent_factory, "construct_agent", constructed)
    session = await factory.open_session(str(project), mcp_servers=[_client("http")], permission_callback=callback)
    first = session.agent
    memory = first.memory_manager
    try:
        assert prepared.call_count == constructed.call_count == 1
        assert isinstance(first, CoderAgent)
        assert first.agent_mode == agent_factory.ACP_AGENT_MODE
        assert first.permission_mode == PermissionMode.ASK
        assert first.permission_callback is callback
        assert first._owns_memory_manager and memory is not None
        assert first.custom_agent_catalog.names() == ["builder", "shared"]
        assert "cli-agent-skills" in {extension.name for extension in first.tool_extensions}
        mcp_tools = next(extension for extension in first.tool_extensions if extension.name == "mcp")
        client_name = f"mcp__{session.client_mcp_servers['client'].id}__inspect_file"
        assert set(mcp_tools.tools) == {client_name, "mcp__global__inspect_file"} | (
            {"mcp__project__inspect_file"} if trusted else set()
        )
        expected_scopes = ["global", "project"] if trusted else ["global"]
        assert [
            scope for event, scopes in hook_events if event == HookEvent.SESSION_START for scope in scopes
        ] == expected_scopes
        await first.fire_hook(HookEvent.PRE_TOOL_USE, {"tool_input": {}}, target="read")
        assert [
            scope for event, scopes in hook_events if event == HookEvent.PRE_TOOL_USE for scope in scopes
        ] == expected_scopes
        assert (await first.permission_callback(Mock())).allowed is False

        session.question_state["elicit"] = AsyncMock(return_value="One")
        task_extension = next(
            extension for extension in first.tool_extensions if extension.name == "acp-shared-task-list"
        )
        await task_extension.tools["update_task_list"]("- [ ] demo")
        skill_extension = next(extension for extension in first.tool_extensions if extension.name == "cli-agent-skills")
        activated = await skill_extension.tools["skill"](name="demo")
        assert "Demo workflow steps." in activated
        message = Message(role="user", content=[TextBlock(f"remember this\n{activated}")])
        first.history.append(message)
        assert first.session_recorder is not None
        first.session_recorder.start_turn(message)
        expected_history = first.dump_message_history()
        expected_compaction = {
            "summary": "Compacted demo context.",
            "compacted_through": 1,
            "compacted_history_length": len(first.history),
        }
        first.restore_compaction_state(expected_compaction)
        first.session_recorder.record_compaction(expected_compaction)
        await factory.set_permission_mode(session, PermissionMode.AUTO)
        await factory.set_interaction_mode(session, "plan")
        assert isinstance(session.agent, PlanningAgent)
        assert first.memory_manager is None
        assert session.agent.permission_mode == PermissionMode.AUTO
        assert session.agent.permission_callback is callback
        assert session.agent.custom_agent_catalog.names() == ["planner", "shared"]
        assert session.agent.dump_message_history() == expected_history
        assert session.agent.dump_compaction_state() == expected_compaction
        skill_extension = next(
            extension for extension in session.agent.tool_extensions if extension.name == "cli-agent-skills"
        )
        assert "already active" in await skill_extension.tools["skill"](name="demo")
        assert session.agent.session_recorder is not None
        tasks = next(
            extension for extension in session.agent.tool_extensions if extension.name == "acp-shared-task-list"
        )
        assert set(tasks.tools) == {"get_task_list"}
        assert await tasks.tools["get_task_list"]() == "- [ ] demo"
        question = next(
            extension for extension in session.agent.tool_extensions if extension.name == "acp-user-questions"
        )
        assert json.loads(
            await question.tools["ask_user_choice"](
                questions=[{"question": "Which?", "header": "Choice", "options": [{"label": "One"}, {"label": "Two"}]}]
            )
        ) == {"Choice": "One"}
        await factory.apply_model(session, "anthropic", session.config.long_context_config.model)
        assert session.client_mcp_servers["client"].headers == {"Authorization": "Bearer fake-header-secret"}
        assert session.config.mcp_config.servers["client"] is session.client_mcp_servers["client"]
        assert len([event for event, _ in hook_events if event == HookEvent.SESSION_START]) == len(expected_scopes)
        assert "fake-header-secret" not in repr(session)
        factory.persist(session)
    finally:
        await factory.close_session(session.session_id)
    journal_path = factory._store.journal(session.session_id).events_path
    context_before = _context_events(journal_path)
    assert len([event for event, _ in hook_events if event == HookEvent.SESSION_END]) == len(expected_scopes)
    assert not session.client_mcp_servers
    assert store.path.read_bytes() == settings_before
    for path in store.root.rglob("*.json*"):
        assert b"fake-header-secret" not in path.read_bytes(), path
        assert b"fake-query-secret" not in path.read_bytes(), path
        assert b"fake-password" not in path.read_bytes(), path
    loaded = await factory.load_session(str(tmp_path / "other"), session.session_id, mcp_servers=[_client("sse")])
    assert loaded is not None
    try:
        assert loaded.agent.dump_message_history() == expected_history
        assert loaded.agent.dump_compaction_state() == expected_compaction
        assert loaded.agent.permission_mode == PermissionMode.AUTO
        assert isinstance(loaded.agent, PlanningAgent)
        assert loaded.record.task_list_markdown == "- [ ] demo"
        assert loaded.client_mcp_servers["client"].transport == "sse"
        assert loaded.agent.project_path == project
        assert _context_events(journal_path) == context_before
    finally:
        await factory.close_session(loaded.session_id)


@pytest.mark.asyncio
async def test_new_and_load_forward_transport_payloads_and_advertise_capabilities(
    tmp_path: Path, configured_factory: AgentFactory, discovery: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = configured_factory
    server = AcpAgent(factory)
    server.on_connect(cast(Client, _FakeConn()))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _session_id: None)
    initialized = await server.initialize(1)
    assert initialized.agent_capabilities is not None
    assert initialized.agent_capabilities.mcp_capabilities is not None
    assert initialized.agent_capabilities.mcp_capabilities.http
    assert initialized.agent_capabilities.mcp_capabilities.sse
    opened = await server.new_session(str(tmp_path), mcp_servers=[McpServerStdio.model_validate(_client())])
    session = server._sessions[opened.session_id]
    try:
        assert discovery[-1][0].transport == "stdio"
        await server.load_session(
            str(tmp_path), opened.session_id, mcp_servers=[HttpMcpServer.model_validate(_client("http"))]
        )
        assert session.client_mcp_servers["client"].transport == "streamable_http"
        await server.load_session(str(tmp_path), opened.session_id, mcp_servers=[])
        assert not session.client_mcp_servers
        assert not session.config.mcp_config.servers
        await server.close_session(opened.session_id)
        await server.load_session(
            str(tmp_path), opened.session_id, mcp_servers=[SseMcpServer.model_validate(_client("sse"))]
        )
        assert server._sessions[opened.session_id].client_mcp_servers["client"].transport == "sse"
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_failed_rebuild_and_cancelled_initialization_cleanup_without_losing_live_session(
    tmp_path: Path, configured_factory: AgentFactory, discovery: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = configured_factory
    session = await factory.open_session(str(tmp_path), mcp_servers=[_client()])
    old = session.agent
    old_config = session.config
    old_clients = dict(session.client_mcp_servers)
    generations: list[Any] = []
    construct = agent_factory.construct_agent

    def capture(*args: Any) -> Any:
        agent, bundle = construct(*args)
        agent.cleanup = AsyncMock(wraps=agent.cleanup)
        generations.append(agent)
        return agent, bundle

    monkeypatch.setattr(agent_factory, "construct_agent", capture)
    failure = AsyncMock(side_effect=RuntimeError("test initialization failure"))
    monkeypatch.setattr(ToolCollection, "initialize", failure)
    try:
        with pytest.raises(RuntimeError, match="test initialization failure"):
            await factory.set_interaction_mode(session, "plan")
        assert session.agent is old
        assert session.config is old_config
        assert session.record.interaction_mode != "plan"
        with pytest.raises(RuntimeError, match="test initialization failure"):
            await factory.update_client_servers(session, [_client("http")])
        assert session.client_mcp_servers == old_clients
        assert session.agent is old
        monkeypatch.setattr(ToolCollection, "initialize", AsyncMock(side_effect=asyncio.CancelledError))
        with pytest.raises(asyncio.CancelledError):
            await factory.open_session(str(tmp_path), mcp_servers=[_client()])
        assert list(factory._acp_sessions) == [session.session_id]
        assert len(generations) == 3
        for generation in generations:
            generation.cleanup.assert_awaited_once()
            assert generation.memory_manager is None
    finally:
        await factory.close_session(session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["agent", "mcp"])
async def test_partial_construction_cleanup_survives_repeated_cancellation(
    tmp_path: Path, configured_factory: AgentFactory, monkeypatch: pytest.MonkeyPatch, partial: str
) -> None:
    initialized = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        cleaning.set()
        await release.wait()
        cleaned.set()

    if partial == "agent":
        construct = agent_factory.construct_agent

        def capture(*args: Any) -> Any:
            agent, bundle = construct(*args)
            original_cleanup = agent.cleanup

            async def agent_cleanup() -> None:
                await cleanup()
                await original_cleanup()

            agent.cleanup = agent_cleanup
            return agent, bundle

        async def initialize(*args: Any, **kwargs: Any) -> Any:
            initialized.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(agent_factory, "construct_agent", capture)
        monkeypatch.setattr(ToolCollection, "initialize", initialize)
    else:
        monkeypatch.setattr(
            agent_factory, "build_session_mcp_extension", AsyncMock(return_value=SimpleNamespace(cleanup=cleanup))
        )
        monkeypatch.setattr(agent_factory, "construct_agent", Mock(side_effect=RuntimeError("construction failed")))

    task = asyncio.create_task(configured_factory.open_session(str(tmp_path)))
    try:
        if partial == "agent":
            await asyncio.wait_for(initialized.wait(), timeout=5)
            task.cancel()
        await asyncio.wait_for(cleaning.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not configured_factory._acp_sessions
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert cleaned.is_set()
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_editor_configuration_retains_identity_but_changed_server_rotates(
    tmp_path: Path, configured_factory: AgentFactory, discovery: list[Any]
) -> None:
    factory = configured_factory
    session = await factory.open_session(str(tmp_path), mcp_servers=[_client()])
    try:
        original = session.client_mcp_servers["client"]
        agent = session.agent
        await factory.update_client_servers(session, [_client()])
        assert session.agent is agent
        assert session.client_mcp_servers["client"] is original
        async with session.execution_lock:
            with pytest.raises(RequestError):
                await factory.update_client_servers(session, [_client()])
        assert session.agent is agent
        assert session.client_mcp_servers["client"] is original
        await factory.update_client_servers(session, [_client("http")])
        assert session.client_mcp_servers["client"].id != original.id
    finally:
        await factory.close_session(session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", ["stdio", "http"])
async def test_editor_configuration_does_not_succeed_during_pending_rebuild(
    tmp_path: Path,
    configured_factory: AgentFactory,
    discovery: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    factory = configured_factory
    session = await factory.open_session(str(tmp_path), mcp_servers=[_client()])
    original = session.client_mcp_servers["client"]
    constructing = asyncio.Event()
    release = asyncio.Event()
    construct = factory._construct_agent

    async def blocked_construct(*args: Any, **kwargs: Any) -> Any:
        constructing.set()
        await release.wait()
        return await construct(*args, **kwargs)

    monkeypatch.setattr(factory, "_construct_agent", blocked_construct)
    update = asyncio.create_task(factory.update_client_servers(session, [_client("http")]))
    try:
        await asyncio.wait_for(constructing.wait(), timeout=5)
        assert session.execution_lock.locked()
        assert session.client_mcp_servers["client"] is original
        with pytest.raises(RequestError) as error:
            await factory.update_client_servers(session, [_client(requested)])
        assert error.value.data == {"message": "MCP configuration applies between turns only."}
        release.set()
        await asyncio.wait_for(update, timeout=5)
        assert session.client_mcp_servers["client"].transport == "streamable_http"
        await factory.update_client_servers(session, [_client(requested)])
        assert session.client_mcp_servers["client"].transport == (
            "stdio" if requested == "stdio" else "streamable_http"
        )
    finally:
        release.set()
        await asyncio.gather(update, return_exceptions=True)
        await factory.close_session(session.session_id)


@pytest.mark.asyncio
async def test_editor_configuration_rechecks_after_lock_handoff(
    tmp_path: Path, configured_factory: AgentFactory, discovery: list[Any]
) -> None:
    factory = configured_factory
    session = await factory.open_session(str(tmp_path), mcp_servers=[_client()])
    waiting = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocker() -> None:
        waiting.set()
        async with session.execution_lock:
            entered.set()
            await release.wait()

    await session.execution_lock.acquire()
    holder = asyncio.create_task(blocker())
    await asyncio.wait_for(waiting.wait(), 5)
    first = asyncio.create_task(factory.update_client_servers(session, [_client("http")]))
    second = asyncio.create_task(factory.update_client_servers(session, [_client()]))
    session.execution_lock.release()
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not first.done()
        assert not second.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        assert session.client_mcp_servers["client"].transport == "stdio"
        assert session.config.mcp_config.servers["client"] is session.client_mcp_servers["client"]
    finally:
        release.set()
        await asyncio.gather(holder, first, second, return_exceptions=True)
        await factory.close_session(session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["mode", "client"])
async def test_cancellation_during_retired_cleanup_keeps_adopted_state_consistent(
    tmp_path: Path, configured_factory: AgentFactory, discovery: list[Any], change: str
) -> None:
    factory = configured_factory
    session = await factory.open_session(str(tmp_path))
    old = session.agent
    cleanup = old.cleanup
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    calls = 0

    async def blocked_cleanup() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        await cleanup()
        finished.set()

    old.cleanup = blocked_cleanup
    try:
        operation = (
            factory.set_interaction_mode(session, "plan")
            if change == "mode"
            else factory.update_client_servers(session, [_client()])
        )
        task = asyncio.create_task(operation)
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert session.agent is not old
        assert old not in session.retired_agents
        assert finished.is_set()
        assert calls == 1
        if change == "mode":
            assert isinstance(session.agent, PlanningAgent)
            assert session.record.interaction_mode == "plan"
            assert factory._store.load(session.session_id).interaction_mode == "plan"
        else:
            assert session.client_mcp_servers["client"] is session.config.mcp_config.servers["client"]
    finally:
        release.set()
        await factory.close_session(session.session_id)
    assert calls == 1


@pytest.mark.asyncio
async def test_cancelled_rebuild_finishes_cleanup_before_concurrent_close(
    tmp_path: Path, configured_factory: AgentFactory, discovery: list[Any]
) -> None:
    factory = configured_factory
    session = await factory.open_session(str(tmp_path))
    old = session.agent
    original_cleanup = old.cleanup
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        await original_cleanup()

    async def close() -> None:
        async with session.execution_lock:
            await factory.close_session(session.session_id)

    old.cleanup = cleanup
    rebuild = asyncio.create_task(factory.update_client_servers(session, [_client()]))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        rebuild.cancel()
        await asyncio.sleep(0)
        closing = asyncio.create_task(close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(rebuild, timeout=5)
        await asyncio.wait_for(closing, timeout=5)
        assert calls == 1
        assert not session.retired_agents
    finally:
        release.set()
        await asyncio.gather(rebuild, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await factory.close_session(session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("normal_cleanup_delay", [0, 0.05], ids=["normal-close", "delayed-close"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_stalled_mode_cleanup_releases_request_lock_without_releasing_ownership(
    tmp_path: Path,
    configured_factory: AgentFactory,
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
    normal_cleanup_delay: float,
) -> None:
    original_cleanup_timeout = agent_factory.CLEANUP_TIMEOUT
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02, raising=False)
    factory = configured_factory
    server = AcpAgent(factory)
    server.on_connect(cast(Client, _FakeConn()))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _session_id: None)
    opened = await server.new_session(str(tmp_path))
    session = server._sessions[opened.session_id]
    old = session.agent
    original_cleanup = old.cleanup
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    cleanup_tasks: list[asyncio.Task[Any]] = []

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        cleanup_task = asyncio.current_task()
        assert cleanup_task is not None
        cleanup_tasks.append(cleanup_task)
        started.set()
        await release.wait()
        await original_cleanup()

    old.cleanup = cleanup
    switch = asyncio.create_task(server.set_session_mode(session.session_id, "plan"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        adopted = session.agent
        assert adopted is not old
        normal_cleanup = adopted.cleanup

        async def cleanup_adopted() -> None:
            await asyncio.sleep(normal_cleanup_delay)
            await normal_cleanup()

        adopted_cleanup = AsyncMock(wraps=cleanup_adopted)
        monkeypatch.setattr(adopted, "cleanup", adopted_cleanup)
        assert session.accounting is not None
        close_accounting = AsyncMock(wraps=session.accounting.aclose)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        if cancelled:
            switch.cancel()
            await asyncio.sleep(0)
            switch.cancel()
        done, _ = await asyncio.wait({switch}, timeout=5)
        assert switch in done, "stalled cleanup still holds the request lock"
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await switch
        else:
            await switch
        assert not session.execution_lock.locked()
        assert len(cleanup_tasks) == 1
        cleanup_task = cleanup_tasks[0]
        assert not cleanup_task.done()
        assert session.cleanup_tasks[id(old)] is cleanup_task
        assert any(agent is old for agent in session.retired_agents)
        assert not session.idle()
        assert isinstance(adopted, PlanningAgent)
        assert session.record.interaction_mode == "plan"
        assert factory._store.load(session.session_id).interaction_mode == "plan"
        for operation in (
            lambda: factory.close_session(session.session_id),
            lambda: server.load_session(str(tmp_path), session.session_id),
            lambda: factory.load_session(str(tmp_path), session.session_id),
            lambda: factory.open_session(str(tmp_path), session_id=session.session_id),
            lambda: server.set_session_mode(session.session_id, "build"),
            lambda: factory.set_interaction_mode(session, "build"),
            lambda: factory.update_client_servers(session, [_client()]),
            lambda: factory.apply_model(session, "anthropic", session.config.long_context_config.model),
            lambda: server.prompt(session.session_id, []),
        ):
            with pytest.raises(RequestError):
                await operation()
        recovered = asyncio.Event()

        async def recover() -> None:
            recovered.set()

        jobs = SimpleNamespace(recover=recover, pending_count=0, deliver=AsyncMock())
        session.background_wake.set()
        worker = asyncio.create_task(server._background_worker(session, jobs))
        try:
            await asyncio.wait_for(recovered.wait(), 5)
            assert not session.background_wake.is_set()
            jobs.deliver.assert_not_awaited()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        assert server._sessions[session.session_id] is session
        assert factory._acp_sessions[session.session_id] is session
        assert not session.closed
        assert session.config is not None
        adopted_cleanup.assert_not_awaited()
        close_accounting.assert_not_awaited()
        assert calls == 1
        assert session.cleanup_tasks[id(old)] is cleanup_task
        release.set()
        await asyncio.wait_for(asyncio.shield(cleanup_task), 5)
        assert not session.cleanup_tasks
        assert old not in session.retired_agents
        assert session.idle()
        assert await factory.load_session(str(tmp_path), session.session_id) is session
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", original_cleanup_timeout)
        await asyncio.wait_for(server.close_session(session.session_id), 10)
        adopted_cleanup.assert_awaited_once()
        close_accounting.assert_awaited_once()
        assert calls == 1
    finally:
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", original_cleanup_timeout)
        release.set()
        await asyncio.gather(switch, return_exceptions=True)
        await asyncio.wait_for(server.close_session(session.session_id), 10)


@pytest.mark.asyncio
async def test_queued_close_rechecks_cleanup_before_removing_session(
    tmp_path: Path, configured_factory: AgentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_cleanup_timeout = agent_factory.CLEANUP_TIMEOUT
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    factory = configured_factory
    session = await factory.open_session(str(tmp_path))
    server = AcpAgent(factory)
    server._sessions[session.session_id] = session
    old = session.agent
    cleanup = old.cleanup
    release, constructing, finish_construct = asyncio.Event(), asyncio.Event(), asyncio.Event()
    construct = factory._construct_agent

    async def blocked_construct(*args: Any, **kwargs: Any) -> Any:
        constructing.set()
        await finish_construct.wait()
        return await construct(*args, **kwargs)

    async def blocked_cleanup() -> None:
        await release.wait()
        await cleanup()

    monkeypatch.setattr(old, "cleanup", blocked_cleanup)
    monkeypatch.setattr(factory, "_construct_agent", blocked_construct)
    rebuild = asyncio.create_task(server.set_session_mode(session.session_id, "plan"))
    await asyncio.wait_for(constructing.wait(), 5)
    closing = asyncio.create_task(server.close_session(session.session_id))
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        assert session.closed
        finish_construct.set()
        await asyncio.wait_for(rebuild, 5)
        assert session.cleanup_tasks
        with pytest.raises(RequestError):
            await asyncio.wait_for(closing, 5)
        assert server._sessions[session.session_id] is session
        assert factory._acp_sessions[session.session_id] is session
        assert session.closed
    finally:
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", original_cleanup_timeout)
        release.set()
        finish_construct.set()
        await asyncio.gather(rebuild, closing, *session.cleanup_tasks.values(), return_exceptions=True)
        await server.close_session(session.session_id)


@pytest.mark.asyncio
async def test_start_hook_context_is_recorded_and_restored(
    tmp_path: Path, configured_factory: AgentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SettingsStore()
    _write_features(project, store.root)

    async def hook(event: Any, spec: Any, caps: Any) -> HookOutcome:
        return (
            HookOutcome(additional_context="startup context marker")
            if event.name == HookEvent.SESSION_START
            else HookOutcome.empty()
        )

    monkeypatch.setattr("kolega_code.hooks.dispatcher.run_hook", hook)
    factory = configured_factory
    session = await factory.open_session(str(project))
    session_id = session.session_id
    try:
        assert "startup context marker" in str(session.agent.dump_message_history())
        factory.persist(session)
    finally:
        await factory.close_session(session_id)
    assert any(
        "startup context marker" in json.dumps(event)
        for event in _context_events(factory._store.journal(session_id).events_path)
    )
    resumed = await factory.load_session(str(project), session_id)
    assert resumed is not None
    try:
        assert str(resumed.agent.dump_message_history()).count("startup context marker") == 2
    finally:
        await factory.close_session(session_id)


@pytest.mark.asyncio
async def test_discovery_cancellation_reclaims_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup = AsyncMock()
    monkeypatch.setattr(MCPService, "cleanup", cleanup)
    monkeypatch.setattr(MCPService, "verify_server", AsyncMock(side_effect=asyncio.CancelledError))
    clients = normalize_client_servers([_client()], tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await build_session_mcp_extension(merge_mcp_config(None, clients), tmp_path, tmp_path / "state")
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_colliding_names_keep_distinct_pinned_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = {
        server_id: MCPServerConfig(id=server_id, command="unused", transport="stdio") for server_id in ("a", "a__b")
    }
    store = MCPStatusStore(tmp_path / "state")
    targets = {("a", "b__c"), ("a__b", "c")}
    for server_id, tool_id in targets:
        server = servers[server_id]
        store.update(
            server_id,
            MCPServerStatus.verified(
                fingerprint=server_fingerprint(server),
                transport=server.transport,
                source=server.source,
                tools=[MCPToolStatus(id=tool_id)],
            ),
        )
    call = AsyncMock(return_value="done")
    monkeypatch.setattr(MCPService, "call_tool", call)
    extension = await build_session_mcp_extension(LoadedMCPConfig(servers=servers), tmp_path, tmp_path / "state")
    assert extension is not None
    try:
        assert len(extension.tools) == 2
        for name, callback in extension.tools.items():
            assert len(name) <= 64
            assert await callback(_server_id="untrusted", _tool_id="unverified") == "done"
        assert {invocation.args[:2] for invocation in call.await_args_list} == targets
        assert all(
            invocation.args[2] == {"_server_id": "untrusted", "_tool_id": "unverified"}
            for invocation in call.await_args_list
        )
    finally:
        assert extension.cleanup is not None
        await extension.cleanup()


def test_acp_composition_is_headless(tmp_path: Path, configured_factory: AgentFactory) -> None:
    program = textwrap.dedent(
        """
        import asyncio
        import sys

        class RejectUI:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"textual", "rich"}:
                    raise ImportError(f"{fullname} is blocked")
                return None

        sys.meta_path.insert(0, RejectUI())
        from kolega_code.acp.agent_factory import AgentFactory
        from kolega_code.agent.tools import ToolCollection

        async def initialize(self):
            return []
        ToolCollection.initialize = initialize

        async def run():
            factory = AgentFactory()
            session = await factory.open_session(sys.argv[1])
            assert session.agent.memory_manager is not None
            await factory.close_session(session.session_id)
            assert not any(name.split(".")[0] in {"textual", "rich"} for name in sys.modules)
        asyncio.run(run())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
