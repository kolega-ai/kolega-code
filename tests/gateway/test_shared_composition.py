from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kolega_code.agent import CoderAgent, PromptExtension, ToolExtension
from kolega_code.agent.custom_agents import CustomAgentCatalog
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.gateway import sessions as gateway_sessions
from kolega_code.gateway.adapters.base import ChatRef
from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.control_relay import ControlRelay
from kolega_code.gateway.daemon import GatewayShutdownIncomplete
from kolega_code.gateway.event_router import EventRouter
from kolega_code.gateway.sessions import AgentTurnHandler
from kolega_code.hooks import HookEvent, HookOutcome, LifecycleEvent, load_hook_config
from kolega_code.hooks import dispatcher as hook_dispatcher_module
from kolega_code.hooks.dispatcher import NO_OP_DISPATCHER
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, TextBlock
from kolega_code.mcp.config import (
    MCPConfigFile,
    MCPServerConfig,
    global_mcp_config_path,
    project_mcp_config_path,
    save_config_file,
    server_fingerprint,
)
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore, MCPToolStatus
from kolega_code.mcp.tools import build_mcp_tool_extension
from kolega_code.permissions import (
    PermissionMode,
    ProjectPermissionStore,
    allow_rule_options,
    permission_request_for_tool,
)
from kolega_code.session import composition

from .test_sessions import RecordingAdapter, ScriptedLLM

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

CHAT = ChatRef(channel="recording", chat_id="42")


@pytest.fixture
def startup_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[EventRouter], list[ControlRelay]]:
    routers: list[EventRouter] = []
    relays: list[ControlRelay] = []
    router_start, relay_start = EventRouter.start, ControlRelay.start

    def start_router(router: EventRouter) -> None:
        routers.append(router)
        router_start(router)

    def start_relay(relay: ControlRelay) -> None:
        relays.append(relay)
        relay_start(relay)

    monkeypatch.setattr(EventRouter, "start", start_router)
    monkeypatch.setattr(ControlRelay, "start", start_relay)
    return routers, relays


class _RecorderAwareAgent:
    agent_name = "coder"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = kwargs["config"]
        self.permission_mode = kwargs["permission_mode"]
        self.permission_callback = kwargs["permission_callback"]
        self.session_recorder = kwargs["session_recorder"]
        self.hook_dispatcher = kwargs.get("hook_dispatcher") or NO_OP_DISPATCHER
        self.tool_collection = AsyncMock()
        self.tool_collection.initialize.return_value = []
        self.compaction: dict[str, Any] = {}
        self.restore_recorders: list[Any] = []
        self.cleanup_calls = 0
        initial = Message(role="user", content=[TextBlock("constructor context")])
        self.history = [initial]
        if self.session_recorder is not None:
            self.session_recorder.record_context_message(initial, actor="user")

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode

    def set_permission_callback(self, callback: Any) -> None:
        self.permission_callback = callback

    def dump_message_history(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.history]

    def restore_message_history(self, history: list[dict[str, Any]]) -> None:
        self.restore_recorders.append(self.session_recorder)
        self.history = [Message.from_dict(item) for item in history]

    def dump_compaction_state(self) -> dict[str, Any]:
        return self.compaction

    def restore_compaction_state(self, data: dict[str, Any]) -> None:
        self.restore_recorders.append(self.session_recorder)
        self.compaction = data

    async def process_message_stream(self, text: str) -> AsyncIterator[dict[str, Any]]:
        user = Message(role="user", content=[TextBlock(text)])
        assistant = Message(role="assistant", content=[TextBlock("done")], stop_reason="end_turn")
        assert self.session_recorder is not None
        self.session_recorder.start_turn(user)
        self.session_recorder.record_assistant(assistant)
        self.session_recorder.finish_turn("completed")
        self.history.extend([user, assistant])
        yield {"type": "response", "content": "done", "complete": True}

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest_asyncio.fixture
async def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AgentTurnHandler]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(gateway_sessions, "CoderAgent", _RecorderAwareAgent)
    monkeypatch.setattr(composition, "CoderAgent", _RecorderAwareAgent)
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    settings_store = SettingsStore(state)
    settings_store.save(
        CliSettings(
            active_provider="anthropic",
            active_model="claude-haiku-4-5-20251001",
            api_keys={"anthropic": "fake-gateway-composition-key"},
            skills_enabled=True,
        )
    )
    handler = AgentTurnHandler(
        config=GatewayConfig(adapter="recording", project_path=project, state_dir=state, session_idle_ttl_seconds=None),
        adapter=RecordingAdapter(),
        store=SessionStore(state),
        settings_store=settings_store,
    )
    assert handler._agent_builder is None
    try:
        yield handler
    finally:
        if handler.status()["failed_cleanups"]:
            with pytest.raises(GatewayShutdownIncomplete):
                await handler.shutdown()
        else:
            await handler.shutdown()


def _write_skill(root: Path, name: str, body: str) -> None:
    path = root / ".agents" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: Discover {name}.\n---\n{body}\n", encoding="utf-8")


def _write_agent(root: Path, name: str, mode: str, body: str) -> None:
    path = root / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: Delegate to {name}.\nmode: {mode}\n---\n{body}\n", encoding="utf-8"
    )


def _skill_tool(agent: _RecorderAwareAgent) -> ToolExtension:
    extensions = [extension for extension in agent.kwargs["tool_extensions"] if "skill" in extension.tools]
    assert len(extensions) == 1, "gateway default composition must expose the discovered skill tool"
    return extensions[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
async def test_default_builder_discovers_project_and_user_skills(gateway: AgentTurnHandler, enabled: bool) -> None:
    _write_skill(gateway._config.project_path, "gateway-project", "Project workflow.")
    _write_skill(Path.home(), "gateway-user", "User workflow.")
    gateway._load_settings().skills_enabled = enabled
    entry = await gateway._ensure_session(CHAT)
    agent = entry.runtime.agent
    assert isinstance(agent, _RecorderAwareAgent)
    prompts: list[PromptExtension] = agent.kwargs["prompt_extensions"]
    tools: list[ToolExtension] = agent.kwargs["tool_extensions"]
    if not enabled:
        assert not any("skill" in extension.tools for extension in tools)
        assert not any(extension.id == "cli-agent-skills" for extension in prompts)
        return
    skill = _skill_tool(agent)
    catalog = next(extension for extension in prompts if extension.id == "cli-agent-skills")
    assert "gateway-project" in catalog.markdown
    assert "gateway-user" in catalog.markdown
    assert "Project workflow." in await skill.tools["skill"](name="gateway-project")
    assert "User workflow." in await skill.tools["skill"](name="gateway-user")


@pytest.mark.asyncio
async def test_skill_activation_reads_live_history_after_replacement_and_rebuild(gateway: AgentTurnHandler) -> None:
    _write_skill(gateway._config.project_path, "gateway-live", "Follow the live workflow.")
    entry = await gateway._ensure_session(CHAT)
    agent = entry.runtime.agent
    assert isinstance(agent, _RecorderAwareAgent)
    activate = _skill_tool(agent).tools["skill"]
    output = await activate(name="gateway-live")
    assert "Follow the live workflow." in output
    assert "already active" not in await activate(name="gateway-live")
    agent.history = [Message(role="user", content=[TextBlock(output)])]
    assert "already active" in await activate(name="gateway-live")
    gateway._persist(entry)
    rebuilt = await entry.runtime.rebuild()
    assert isinstance(rebuilt, _RecorderAwareAgent)
    assert rebuilt is not agent
    activate_rebuilt = _skill_tool(rebuilt).tools["skill"]
    assert "already active" in await activate_rebuilt(name="gateway-live")
    rebuilt.history = []
    assert "Follow the live workflow." in await activate_rebuilt(name="gateway-live")


@pytest.mark.asyncio
async def test_default_builder_discovers_custom_agents_for_build_only(gateway: AgentTurnHandler) -> None:
    user_agents = gateway._config.state_dir / "agents"
    project_agents = gateway._config.project_path / ".kolega" / "agents"
    _write_agent(user_agents, "gateway-user", "build", "User agent.")
    _write_agent(user_agents, "gateway-override", "all", "Overridden user agent.")
    _write_agent(project_agents, "gateway-override", "build", "Project wins.")
    _write_agent(project_agents, "gateway-build", "build", "Build agent.")
    _write_agent(project_agents, "gateway-plan", "plan", "Plan agent.")
    _write_agent(project_agents, "gateway-all", "all", "Shared agent.")
    entry = await gateway._ensure_session(CHAT)
    agent = entry.runtime.agent
    assert isinstance(agent, _RecorderAwareAgent)
    catalog = agent.kwargs.get("custom_agent_catalog")
    assert isinstance(catalog, CustomAgentCatalog), "gateway default composition must pass discovered custom agents"
    assert set(catalog.names()) == {"gateway-user", "gateway-override", "gateway-build", "gateway-all"}
    override = catalog.get("gateway-override")
    assert override is not None
    assert override.scope == "project"
    assert override.prompt == "Project wins."
    assert catalog.get("gateway-plan") is None
    assert any("overrides user definition" in diagnostic.message for diagnostic in catalog.diagnostics)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_mode", [PermissionMode.ASK, PermissionMode.AUTO])
async def test_default_builder_keeps_runtime_permissions_on_build_and_rebuild(
    gateway: AgentTurnHandler, initial_mode: PermissionMode
) -> None:
    gateway._config = replace(gateway._config, permission_mode=initial_mode.value)
    entry = await gateway._ensure_session(CHAT)
    request = permission_request_for_tool("exec_command", {"command": "npm run test"})
    assert request is not None
    ProjectPermissionStore(gateway._config.project_path).add_rule(allow_rule_options(request)[0].rule)
    changed_mode = PermissionMode.AUTO if initial_mode is PermissionMode.ASK else PermissionMode.ASK
    for expected in (initial_mode, changed_mode):
        if expected != entry.runtime.permission_mode:
            entry.runtime.set_permission_mode(expected)
            gateway._persist(entry)
            await entry.runtime.rebuild()
        agent = entry.runtime.agent
        assert isinstance(agent, _RecorderAwareAgent)
        assert agent.kwargs["permission_mode"] is expected
        assert agent.permission_mode is expected
        callback = agent.kwargs["permission_callback"]
        assert callback == entry.runtime.permission_callback
        assert callback.__self__ is entry.runtime
        assert agent.permission_callback == callback
        assert (await callback(request)).allowed
        assert entry.runtime.control.pending() == []
        assert agent.kwargs["connection_manager"].inner is entry.manager
        assert agent.kwargs["project_path"] == gateway._config.project_path
        assert agent.kwargs["workspace_id"] == entry.record.workspace_id
        assert agent.kwargs["thread_id"] == entry.record.thread_id
        assert agent.kwargs["agent_mode"] == AgentMode.CLI


@pytest.mark.asyncio
async def test_resume_and_rebuild_restore_with_recorder_detached_then_record_next_turn(
    gateway: AgentTurnHandler,
) -> None:
    entry = await gateway._ensure_session(CHAT)
    first = entry.runtime.agent
    assert isinstance(first, _RecorderAwareAgent)
    recorder = first.session_recorder
    assert recorder is not None
    assert first.kwargs["session_recorder"] is recorder
    assert first.restore_recorders == []
    async for _ in entry.runtime.send_message("before rebuild"):
        pass
    first.compaction = {"summary": "retained", "compacted_through": 1, "compacted_history_length": 1}
    recorder.record_compaction(first.compaction)
    gateway._persist(entry)
    expected_history = first.dump_message_history()
    recorded_history = gateway._store.load(entry.session_id).history
    assert entry.usage_presentation is not None
    await entry.usage_presentation.flush()
    events = gateway._store.journal(entry.session_id).read_events()
    rebuilt = await entry.runtime.rebuild()
    assert isinstance(rebuilt, _RecorderAwareAgent)
    assert first.cleanup_calls == 1
    assert rebuilt.kwargs["session_recorder"] is None
    assert rebuilt.restore_recorders == [None, None]
    assert rebuilt.session_recorder is recorder
    assert rebuilt.dump_message_history() == expected_history
    assert rebuilt.compaction == first.compaction
    assert gateway._store.load(entry.session_id).history == recorded_history
    assert gateway._store.journal(entry.session_id).read_events() == events
    entry.runtime.set_permission_mode(PermissionMode.AUTO)
    gateway._persist(entry)
    await gateway.shutdown()
    events = gateway._store.journal(entry.session_id).read_events()

    resumed_handler = AgentTurnHandler(
        config=replace(gateway._config, permission_mode="ask"),
        adapter=RecordingAdapter(),
        store=gateway._store,
        settings_store=gateway._settings_store,
    )
    try:
        resumed_entry = await resumed_handler._ensure_session(CHAT)
        resumed = resumed_entry.runtime.agent
        assert isinstance(resumed, _RecorderAwareAgent)
        assert resumed_entry.session_id == entry.session_id
        assert resumed.kwargs["session_recorder"] is None
        assert resumed.restore_recorders == [None, None]
        assert resumed.session_recorder is not None
        assert resumed.dump_message_history() == expected_history
        assert resumed.compaction == first.compaction
        assert resumed.kwargs["permission_mode"] is PermissionMode.AUTO
        assert resumed_entry.runtime.permission_mode is PermissionMode.AUTO
        restored_events = gateway._store.journal(entry.session_id).read_events()
        assert restored_events[:-1] == events
        assert restored_events[-1].event_type == "ui.usage_updated"
        previous_usage = next(event for event in reversed(events) if event.event_type == "ui.usage_updated")
        assert restored_events[-1].payload["event"]["content"] == previous_usage.payload["event"]["content"]
        async for _ in resumed_entry.runtime.send_message("after resume"):
            pass
        persisted = gateway._store.load(entry.session_id)
        assert [Message.from_dict(item).to_dict() for item in persisted.history] == resumed.dump_message_history()
        assert len(resumed.session_recorder.list_rewindable_turns()) == 2
    finally:
        await resumed_handler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["global", "project", "both"])
@pytest.mark.parametrize("trusted", [False, True], ids=["untrusted", "trusted"])
async def test_mcp_and_hooks_never_activate_even_when_configured_and_trusted(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, scope: str, trusted: bool
) -> None:
    project, state = gateway._config.project_path, gateway._config.state_dir
    if trusted:
        settings = gateway._load_settings()
        settings.trust_mcp_project(project)
        settings.trust_hook_project(project)
        gateway._settings_store.save(settings)
    scopes = {"global", "project"} if scope == "both" else {scope}
    for source in scopes:
        server = MCPServerConfig(id=f"gateway-{source}", transport="stdio", command="unused-gateway-mcp-command")
        path = global_mcp_config_path(state) if source == "global" else project_mcp_config_path(project)
        save_config_file(path, MCPConfigFile(servers=[server]))
        MCPStatusStore(state).update(
            server.id,
            MCPServerStatus.verified(
                fingerprint=server_fingerprint(server),
                transport=server.transport,
                source=source,
                tools=[MCPToolStatus(id="inspect")],
            ),
        )
        hooks_path = state / "hooks.json" if source == "global" else project / ".kolega" / "hooks.json"
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "hooks": {
                        event.value: [{"hooks": [{"type": "command", "command": "unused-gateway-hook-command"}]}]
                        for event in HookEvent
                    },
                }
            ),
            encoding="utf-8",
        )
    eligible = scopes if trusted else scopes - {"project"}
    available_mcp = build_mcp_tool_extension(project, state, project_trusted=trusted)
    assert set(available_mcp.tools if available_mcp else {}) == {
        f"mcp__gateway-{source}__inspect" for source in eligible
    }
    available_hooks = load_hook_config(project, state, project_trusted=trusted)
    assert {spec.scope for spec in available_hooks.specs_for(HookEvent.SESSION_START)} == eligible
    run_hook = AsyncMock(return_value=HookOutcome.empty())
    monkeypatch.setattr(hook_dispatcher_module, "run_hook", run_hook)
    entry = await gateway._ensure_session(CHAT)
    for rebuild in (False, True):
        if rebuild:
            gateway._persist(entry)
            await entry.runtime.rebuild()
        agent = entry.runtime.agent
        assert isinstance(agent, _RecorderAwareAgent)
        tools: list[ToolExtension] = agent.kwargs["tool_extensions"]
        assert not any(extension.name == "mcp" for extension in tools)
        assert not any(name.startswith("mcp__") for extension in tools for name in extension.tools)
        assert not agent.hook_dispatcher.is_active
        for event in HookEvent:
            await agent.hook_dispatcher.dispatch(LifecycleEvent(event))
        run_hook.assert_not_awaited()
    saved = gateway._settings_store.load()
    assert saved.is_mcp_project_trusted(project) is trusted
    assert saved.is_hook_project_trusted(project) is trusted


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["initialize", "cancel", "history", "compaction"])
@pytest.mark.parametrize("cleanup_failure", [None, RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("late_cancellations", [0, 3])
async def test_failed_startup_drains_agent_cleanup_despite_repeated_cancellation(
    gateway: AgentTurnHandler,
    monkeypatch: pytest.MonkeyPatch,
    startup_resources: tuple[list[EventRouter], list[ControlRelay]],
    failure_stage: str,
    cleanup_failure: type[BaseException] | None,
    late_cancellations: int,
) -> None:
    failure: BaseException = RuntimeError(failure_stage)
    initialize_started = asyncio.Event()
    cleanup_started, cleanup_release, cleanup_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    agents: list[_RecorderAwareAgent] = []

    async def cancelled_initialize() -> list[str]:
        nonlocal failure
        initialize_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            failure = exc
            raise
        return []

    class FailingAgent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            if failure_stage == "initialize":
                self.tool_collection.initialize.side_effect = failure
            elif failure_stage == "cancel":
                self.tool_collection.initialize.side_effect = cancelled_initialize

        def restore_message_history(self, history: list[dict[str, Any]]) -> None:
            if failure_stage == "history":
                raise failure
            super().restore_message_history(history)

        def restore_compaction_state(self, data: dict[str, Any]) -> None:
            if failure_stage == "compaction":
                raise failure
            super().restore_compaction_state(data)

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_finished.set()
            if cleanup_failure is not None:
                raise cleanup_failure("secondary cleanup failure")

    monkeypatch.setattr(gateway_sessions, "CoderAgent", FailingAgent)
    if failure_stage in {"history", "compaction"}:
        record = gateway._store.create(gateway._config.project_path, AgentMode.CLI.value, {})
        gateway._chat_sessions[CHAT.key] = record.session_id
    task = asyncio.create_task(gateway._ensure_session(CHAT))
    request_cancelled = False
    try:
        if failure_stage == "cancel":
            await asyncio.wait_for(initialize_started.wait(), timeout=10)
            request_cancelled = task.cancel("original startup cancellation")
        await asyncio.wait_for(cleanup_started.wait(), timeout=10)
        owner = gateway._operations[CHAT.key]
        owned_task = owner.task
        assert owned_task is not None
        for attempt in range(late_cancellations):
            request_cancelled = task.cancel(f"late cancellation {attempt}") or request_cancelled
            await asyncio.sleep(0)
            assert not owned_task.done(), "the retained owner must outlive request cancellation"
            assert not cleanup_finished.is_set()
        cleanup_release.set()
        done, _ = await asyncio.wait({task, owned_task}, timeout=10)
        assert done == {task, owned_task}
        with pytest.raises(type(failure)) as raised:
            owned_task.result()
        if not isinstance(failure, asyncio.CancelledError):
            assert raised.value is failure
        if request_cancelled:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(type(failure)) as request_failure:
                await task
            assert request_failure.value is failure
        assert cleanup_finished.is_set()
        assert agents[0].cleanup_calls == 1
        if failure_stage in {"history", "compaction"}:
            assert agents[0].session_recorder is None
        routers, relays = startup_resources
        assert len(routers) == len(relays) == 1
        assert routers[0]._task is None
        assert relays[0]._pump_task is None
        assert relays[0]._runtime.control.controller is None
        assert gateway._registry.get(CHAT) is None
        assert owner.cleanup_failed is (cleanup_failure is not None)
        if cleanup_failure is not None:
            assert gateway._operations[CHAT.key] is owner
            for _ in range(2):
                with pytest.raises(GatewayShutdownIncomplete):
                    await gateway.shutdown()
            assert agents[0].cleanup_calls == 1
    finally:
        cleanup_release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_stage", ["relay", "router"])
async def test_failed_startup_drains_transport_teardown_despite_repeated_cancellation(
    gateway: AgentTurnHandler,
    monkeypatch: pytest.MonkeyPatch,
    startup_resources: tuple[list[EventRouter], list[ControlRelay]],
    stop_stage: str,
) -> None:
    failure = RuntimeError("initialize failed")
    stop_started, stop_release = asyncio.Event(), asyncio.Event()
    stopped: list[str] = []

    class FailingAgent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.tool_collection.initialize.side_effect = failure

    original_relay_stop, original_router_stop = ControlRelay.stop, EventRouter.stop

    async def stop_relay(relay: ControlRelay) -> None:
        if stop_stage == "relay":
            stop_started.set()
            await stop_release.wait()
        await original_relay_stop(relay)
        stopped.append("relay")

    async def stop_router(router: EventRouter) -> None:
        if stop_stage == "router":
            stop_started.set()
            await stop_release.wait()
        await original_router_stop(router)
        stopped.append("router")

    monkeypatch.setattr(gateway_sessions, "CoderAgent", FailingAgent)
    monkeypatch.setattr(ControlRelay, "stop", stop_relay)
    monkeypatch.setattr(EventRouter, "stop", stop_router)
    task = asyncio.create_task(gateway._ensure_session(CHAT))
    request_cancelled = False
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=10)
        owner = gateway._operations[CHAT.key]
        owned_task = owner.task
        assert owned_task is not None
        for attempt in range(3):
            request_cancelled = task.cancel(f"late cancellation {attempt}") or request_cancelled
            await asyncio.sleep(0)
            assert not owned_task.done(), "transport cleanup must outlive request cancellation"
        stop_release.set()
        done, _ = await asyncio.wait({task, owned_task}, timeout=10)
        assert done == {task, owned_task}
        with pytest.raises(asyncio.CancelledError if request_cancelled else RuntimeError):
            await task
        with pytest.raises(RuntimeError) as raised:
            owned_task.result()
        assert raised.value is failure
        assert set(stopped) == {"relay", "router"}
        routers, relays = startup_resources
        assert routers[0]._task is None
        assert relays[0]._pump_task is None
        assert relays[0]._runtime.control.controller is None
        assert gateway._registry.get(CHAT) is None
    finally:
        stop_release.set()
        await asyncio.gather(task, return_exceptions=True)
        routers, relays = startup_resources
        await asyncio.gather(
            *(original_relay_stop(relay) for relay in relays),
            *(original_router_stop(router) for router in routers),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_real_default_builder_build_rebuild_and_resume(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gateway_sessions, "CoderAgent", CoderAgent)
    monkeypatch.setattr(composition, "CoderAgent", CoderAgent)
    gateway._load_settings().lsp_enabled = False
    llm = ScriptedLLM("real default builder response")
    monkeypatch.setattr(LLMClient, "stream", llm.stream)
    monkeypatch.setattr(LLMClient, "generate", llm.generate)
    monkeypatch.setattr(LLMClient, "count_tokens", llm.count_tokens)
    _write_skill(gateway._config.project_path, "real-gateway", "Real construction workflow.")
    entry = await gateway._ensure_session(CHAT)
    agent = entry.runtime.agent
    assert agent is not None
    assert isinstance(agent, CoderAgent)
    assert agent.tool_collection is not None
    assert "skill" in agent.tool_collection.extension_callbacks
    assert "ask_user_choice" in agent.tool_collection.extension_callbacks
    assert not any(name.startswith("mcp__") for name in agent.tool_collection.extension_callbacks)
    assert agent.hook_dispatcher is NO_OP_DISPATCHER
    assert agent.tool_collection.lsp_manager is None
    chunks = [chunk async for chunk in entry.runtime.send_message("real builder smoke")]
    assert any(chunk.get("content") == "real default builder response" for chunk in chunks)
    llm.stream.assert_awaited_once()
    gateway._persist(entry)
    history = agent.dump_message_history()
    rebuilt = await entry.runtime.rebuild()
    assert isinstance(rebuilt, CoderAgent)
    assert rebuilt is not agent
    assert rebuilt.dump_message_history() == history
    assert rebuilt.session_recorder is agent.session_recorder
    await gateway.shutdown()

    resumed_handler = AgentTurnHandler(
        config=gateway._config,
        adapter=RecordingAdapter(),
        store=gateway._store,
        settings_store=gateway._settings_store,
    )
    resumed_handler._load_settings().lsp_enabled = False
    try:
        resumed_entry = await resumed_handler._ensure_session(CHAT)
        resumed = resumed_entry.runtime.agent
        assert resumed is not None
        assert isinstance(resumed, CoderAgent)
        assert resumed_entry.session_id == entry.session_id
        assert resumed.dump_message_history() == history
        assert resumed.session_recorder is not None
        assert resumed.hook_dispatcher is NO_OP_DISPATCHER
    finally:
        await resumed_handler.shutdown()
