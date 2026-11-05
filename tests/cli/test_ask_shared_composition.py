from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli import main as main_module
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.extensions import ExtensionSelection, KolegaExtensionBundle
from kolega_code.hooks import HookEvent
from kolega_code.llm.models import Message, TextBlock
from kolega_code.permissions import PermissionMode
from kolega_code.session import composition

from ._app_test_utils import FakeCoderAgent
from .test_main_worktree_resume import _repository_with_worktree

pytestmark = pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")


class _AskAgent(FakeCoderAgent):
    agent_name = "coder"
    instances: list[_AskAgent] = []
    events: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.restore_recorders: list[Any] = []
        self.compaction: dict[str, Any] = {}
        self.instances.append(self)
        self.events.append("construct")
        context = Message(role="user", content=[TextBlock("constructor context")])
        self.history.append(context)
        if self.session_recorder is not None:
            self.session_recorder.record_context_message(context)

    def restore_message_history(self, history: list[dict[str, Any]]) -> None:
        self.restore_recorders.append(self.session_recorder)
        self.events.append("restore-history")
        super().restore_message_history(history)

    def restore_compaction_state(self, data: dict[str, Any]) -> None:
        self.restore_recorders.append(self.session_recorder)
        self.compaction = data
        self.events.append("restore-compaction")

    async def fire_hook(self, event: HookEvent, payload: dict[str, Any]) -> Any:
        if event == HookEvent.SESSION_START:
            assert self.session_recorder is not None
            self.events.append("session-start")
        return SimpleNamespace(additional_context=None)

    async def cleanup(self) -> None:
        self.events.append("agent-cleanup")
        await super().cleanup()


@pytest.fixture
def ask_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _AskAgent.instances = []
    _AskAgent.events = []
    monkeypatch.setattr(main_module, "CoderAgent", _AskAgent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-ask-test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    project = tmp_path / "project"
    project.mkdir()
    return project


def _write_skill(project: Path) -> None:
    skill = project / ".agents" / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Demo workflow.\n---\nFollow demo steps.\n")


@pytest.mark.parametrize("prompt", ["/skills", "/demo"])
def test_informational_skills_bypass_config_and_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_skill(tmp_path)
    forbidden = Mock(side_effect=AssertionError("informational skills must not build an agent"))
    monkeypatch.setattr(main_module, "build_agent_config", forbidden)
    monkeypatch.setattr(main_module, "prepare_composition", forbidden)
    monkeypatch.setattr(main_module, "construct_agent", forbidden)

    assert main_module.main(["ask", prompt, "--project", str(tmp_path)]) == 0

    forbidden.assert_not_called()
    assert "demo" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_enabled", [False, True])
async def test_real_ask_agent_owns_memory_and_does_not_gain_question_tools(
    ask_setup: Path, monkeypatch: pytest.MonkeyPatch, memory_enabled: bool
) -> None:
    instances: list[CoderAgent] = []
    managers: list[Any] = []

    class RecordingAgent(CoderAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            instances.append(self)
            assert self._owns_memory_manager
            assert self.memory_manager is not None
            managers.append(self.memory_manager)
            assert self.memory_manager._inert is not memory_enabled
            assert self.tool_collection is not None
            assert "ask_user_choice" not in {tool.name for tool in self.tool_collection.get_tool_list()}
            yield {"type": "response", "content": "done", "complete": True}

    monkeypatch.setattr(main_module, "CoderAgent", RecordingAgent)
    argv = ["ask", "inspect", "--project", str(ask_setup)]
    if not memory_enabled:
        argv.append("--no-memory-tools")

    assert await main_module._run_ask(main_module.parse_args(argv)) == 0

    assert len(instances) == 1
    assert managers[0]._closed
    assert instances[0].memory_manager is None


@pytest.mark.asyncio
@pytest.mark.parametrize("linked", [False, True], ids=["launch", "active-worktree"])
@pytest.mark.parametrize("trusted", [False, True], ids=["untrusted", "trusted"])
@pytest.mark.parametrize("memory_enabled", [False, True])
async def test_shared_composition_preserves_ask_paths_capabilities_and_live_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ask_setup: Path,
    linked: bool,
    trusted: bool,
    memory_enabled: bool,
) -> None:
    launch, worktree = _repository_with_worktree(tmp_path)
    active = worktree if linked else launch
    _write_skill(active)
    agent_dir = active / ".kolega" / "agents"
    agent_dir.mkdir(parents=True)
    for name, mode in (("builder", "build"), ("planner", "plan"), ("shared", "all")):
        (agent_dir / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: {name} specialist\nmode: {mode}\n---\nWork carefully.\n"
        )
    state = tmp_path / "state"
    store = SessionStore(state)
    session = store.create(launch, "code", {})
    if linked:
        session.active_project_path = str(active)
        store.save(session)
    settings = CliSettings()
    if trusted:
        settings.trust_mcp_project(launch)
    else:
        settings.trust_mcp_project(worktree)
    SettingsStore(state).save(settings)
    catalog = main_module.discover_skills(active)
    discover = Mock(return_value=catalog)
    monkeypatch.setattr(main_module, "discover_skills", discover)
    monkeypatch.setattr(
        composition, "discover_skills", Mock(side_effect=AssertionError("ask skills must only be discovered once"))
    )
    mcp_extension = ToolExtension(name="ask-mcp", tools={})
    mcp = Mock(return_value=mcp_extension)
    monkeypatch.setattr(main_module, "build_mcp_tool_extension", mcp)
    load_mcp = Mock(wraps=main_module.load_mcp_config)
    monkeypatch.setattr(main_module, "load_mcp_config", load_mcp)
    warnings = Mock()
    monkeypatch.setattr(main_module, "_print_mcp_tool_name_warnings", warnings)
    hooks = Mock(wraps=main_module.load_hook_config)
    monkeypatch.setattr(main_module, "load_hook_config", hooks)
    prepare = Mock(wraps=composition.prepare_composition)
    construct = Mock(wraps=composition.construct_agent)
    monkeypatch.setattr(main_module, "prepare_composition", prepare)
    monkeypatch.setattr(main_module, "construct_agent", construct)
    argv = ["ask", "inspect", "--project", str(launch), "--session", session.session_id]
    if not memory_enabled:
        argv.append("--no-memory-tools")

    assert await main_module._run_ask(main_module.parse_args(argv)) == 0

    prepare.assert_called_once()
    construct.assert_called_once()
    request, host = prepare.call_args.args
    agent = _AskAgent.instances[0]
    discover.assert_called_once_with(active)
    assert request.preloaded_skills is discover.return_value
    assert request.project_path == launch
    assert request.active_project_path == active
    assert request.agent_mode is AgentMode.ASK
    assert host.questions is None
    assert agent.kwargs["project_path"] == active
    assert agent.kwargs["memory_manager"] is None
    assert agent.kwargs["memory_project_path"] == launch
    assert agent.kwargs["memory_enabled"] is memory_enabled
    assert agent.kwargs["permission_mode"] is PermissionMode.AUTO
    assert agent.kwargs["browser_manager"] is not None
    assert agent.kwargs["usage_ledger"].observer is not None
    assert agent.kwargs["hook_dispatcher"] is host.hook_dispatcher()
    assert hooks.call_args.args == (launch, state)
    assert hooks.call_args.kwargs == {"project_trusted": False}
    assert agent.kwargs["custom_agent_catalog"].names() == ["builder", "shared"]
    cached = request.config.mcp_config
    active_config = mcp.call_args.kwargs["loaded_config"]
    mcp.assert_called_once_with(active, state, project_trusted=trusted, loaded_config=active_config)
    if linked:
        load_mcp.assert_called_once_with(active, state, project_trusted=trusted)
        assert active_config is not cached
    else:
        load_mcp.assert_not_called()
        assert active_config is cached
    warnings.assert_called_once_with(active_config, state, active)
    assert [ext.id for ext in agent.kwargs["prompt_extensions"]] == ["cli-agent-skills"]
    assert [ext.name for ext in agent.kwargs["tool_extensions"]] == ["cli-agent-skills", "ask-mcp"]
    skill = agent.kwargs["tool_extensions"][0].tools["skill"]
    content = await skill(name="demo")
    assert "Follow demo steps." in content
    agent.history.append(Message(role="user", content=[TextBlock(content)]))
    assert "already active" in await skill(name="demo")
    assert agent.cleanup_calls == 1


@pytest.mark.asyncio
async def test_resume_restores_with_recorder_detached_and_binds_after_restoration(
    ask_setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    store = SessionStore(state)
    history = [Message(role="user", content=[TextBlock("saved context")]).to_dict()]
    compaction = {"summary": "saved summary", "compacted_through": 1, "compacted_history_length": 1}
    session = store.create(ask_setup, "code", {})
    recorder = store.recorder(session.session_id)
    recorder.record_context_message(Message.from_dict(history[0]))
    recorder.record_compaction(compaction)
    store.release_session_locks()
    binds: list[_AskAgent] = []

    def bind(agent: Any) -> None:
        assert [message.to_dict() for message in agent.history] == history
        assert agent.compaction == compaction
        assert agent.session_recorder is not None
        binds.append(agent)
        _AskAgent.events.append("bind")

    selection = ExtensionSelection(
        "fake:factory",
        lambda host, path: KolegaExtensionBundle(
            bind_agent=bind, cleanup=lambda: _AskAgent.events.append("bundle-cleanup")
        ),
        None,
    )
    monkeypatch.setattr(main_module, "_resolve_extension_selection_from_args", lambda args: selection)
    argv = ["ask", "next", "--project", str(ask_setup), "--session", session.session_id]

    assert await main_module._run_ask(main_module.parse_args(argv)) == 0

    agent = _AskAgent.instances[0]
    assert binds == [agent]
    assert agent.kwargs["session_recorder"] is None
    assert agent.restore_recorders == [None, None]
    assert _AskAgent.events == [
        "construct",
        "restore-history",
        "restore-compaction",
        "bind",
        "session-start",
        "agent-cleanup",
        "bundle-cleanup",
    ]
    restored = store.load(session.session_id)
    texts = [Message.from_dict(item).get_text_content() for item in restored.history]
    assert texts == ["saved context", "next", "done"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["factory", "constructor", "bind", "cancel"])
async def test_shared_builder_failures_close_sink_record_terminal_and_release_owned_bundle(
    ask_setup: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    sink_close = Mock()
    original_close = main_module.SessionUsageSink.wait_closed

    async def close(sink: Any) -> Any:
        _AskAgent.events.append("sink-close")
        sink_close()
        return await original_close(sink)

    monkeypatch.setattr(main_module.SessionUsageSink, "wait_closed", close)

    class FailingAgent(_AskAgent):
        def __init__(self, **kwargs: Any) -> None:
            if failure == "constructor":
                raise ValueError("construction refused")
            super().__init__(**kwargs)

    def bind(agent: Any) -> None:
        _AskAgent.events.append("bind")
        if failure == "bind":
            raise RuntimeError("bind refused")
        if failure == "cancel":
            raise asyncio.CancelledError()

    def factory(host: Any, path: Any) -> KolegaExtensionBundle:
        if failure == "factory":
            raise RuntimeError("factory refused")
        return KolegaExtensionBundle(
            prompt_extensions=[PromptExtension(id="selected", title="Selected", markdown="body")],
            bind_agent=bind,
            cleanup=lambda: _AskAgent.events.append("bundle-cleanup"),
        )

    monkeypatch.setattr(main_module, "CoderAgent", FailingAgent)
    monkeypatch.setattr(
        main_module,
        "_resolve_extension_selection_from_args",
        lambda args: ExtensionSelection("fake:factory", factory, None),
    )
    argv = ["ask", "next", "--project", str(ask_setup), "--session", "failure-test"]
    args = main_module.parse_args(argv)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await main_module._run_ask(args)
    else:
        assert await main_module._run_ask(args) == 2

    sink_close.assert_called_once_with()
    assert _AskAgent.events.count("bundle-cleanup") == int(failure != "factory")
    assert _AskAgent.events.count("agent-cleanup") == int(failure in ("bind", "cancel"))
    assert _AskAgent.events[-1] == "sink-close"
    if failure in ("bind", "cancel"):
        assert _AskAgent.events.index("agent-cleanup") < _AskAgent.events.index("bundle-cleanup")
    store = SessionStore(tmp_path / "state")
    events = store.journal("failure-test").read_events()
    terminals = [event for event in events if event.event_type.startswith("run.")]
    assert len(terminals) == 1
    assert terminals[0].event_type == ("run.cancelled" if failure == "cancel" else "run.failed")
    if failure != "cancel":
        assert terminals[0].payload["error"]["code"] == "extension_error"
    assert store.claim_for_resume("failure-test", ask_setup).session_id == "failure-test"
    store.release_session_locks()
