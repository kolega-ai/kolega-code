import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.llm.models import Message, TextBlock
from kolega_code.mcp.config import (
    MCPConfigFile,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    project_mcp_config_path,
    save_config_file,
    server_fingerprint,
)
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore, MCPToolStatus
from kolega_code.permissions import ProjectPermissionStore, allow_rule_options, permission_request_for_tool

from ._app_test_utils import (
    FakeCoderAgent,
    build_test_config,
    extension_by_name,
    install_fake_agents,
    question_payload,
    wait_for_question_prompt,
)
from .test_app_worktree_switch import _make_worktrees

pytestmark = pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")


class _FakePlanningAgent(FakeCoderAgent):
    pass


@pytest.mark.asyncio
async def test_question_extension_uses_live_tui_normalizer_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    extension = app._planning_question_tool_extension()
    normalize = Mock(return_value=[("Changed?", "Custom", ["One", "Two"], ["", ""])])
    ask = AsyncMock(return_value="Two")
    monkeypatch.setattr(app, "_normalize_choice_questions", normalize)
    monkeypatch.setattr(app, "_ask_user_choice", ask)
    payload = question_payload("Original?", ["A", "B"])
    assert json.loads(await extension.tools["ask_user_choice"](questions=payload)) == {"Custom": "Two"}
    normalize.assert_called_once_with(payload)
    ask.assert_awaited_once_with("Changed?", ["One", "Two"], ["", ""])


@pytest.mark.asyncio
async def test_mode_round_trip_keeps_live_task_list_and_question_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=_FakePlanningAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        for mode in ("build", "plan", "build"):
            await app._set_interaction_mode(mode)
            agent = app.agent
            assert isinstance(agent, _FakePlanningAgent if mode == "plan" else FakeCoderAgent)
            tools = agent.kwargs["tool_extensions"]
            prompts = agent.kwargs["prompt_extensions"]
            task_name = "cli-shared-task-list-readonly" if mode == "plan" else "cli-shared-task-list"
            task_extension = extension_by_name(tools, task_name)
            assert task_extension.propagate_to_sub_agents is False
            assert extension_by_name(prompts, task_name).propagate_to_sub_agents is False
            if mode == "build":
                assert set(task_extension.tools) == {"get_task_list", "update_task_list"}
                await task_extension.tools["update_task_list"]("  - [x] inspect\n- [ ] implement  ")
            else:
                assert set(task_extension.tools) == {"get_task_list"}
                assert task_extension.tool_groups["planning_tools"] == ["get_task_list"]
            assert await task_extension.tools["get_task_list"]() == "- [x] inspect\n- [ ] implement"
            assert store.load(session.session_id).task_list_markdown == "- [x] inspect\n- [ ] implement"

            question_extension = extension_by_name(tools, "cli-planning-questions")
            assert question_extension.propagate_to_sub_agents is False
            group = "planning_tools" if mode == "plan" else "coder_agent_tools"
            assert question_extension.tool_groups[group] == ["ask_user_choice"]
            prompt_ids = {extension.id for extension in prompts}
            question_prompt_id = "cli-planning-questions" if mode == "plan" else "cli-build-questions"
            assert prompt_ids & {"cli-planning-questions", "cli-build-questions"} == {question_prompt_id}
            assert extension_by_name(prompts, question_prompt_id).propagate_to_sub_agents is False
            answer = asyncio.create_task(
                question_extension.tools["ask_user_choice"](
                    questions=question_payload("Which scope?", ["Small", "Full"], header="Scope")
                )
            )
            try:
                await wait_for_question_prompt(app, pilot)
                await app._answer_question_option(1)
                assert json.loads(await asyncio.wait_for(answer, timeout=10)) == {"Scope": "Full"}
                assert app.control_channel.pending() == []
            finally:
                if not answer.done():
                    answer.cancel()
                await asyncio.gather(answer, return_exceptions=True)


class _RecorderAwareAgent(FakeCoderAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.compaction: dict[str, Any] = {}
        self.restore_recorders: list[Any] = []
        initial = Message(role="user", content=[TextBlock("constructor context")])
        self.history.append(initial)
        if self.session_recorder is not None:
            self.session_recorder.record_context_message(initial, actor="user")

    def restore_message_history(self, history: list[dict[str, Any]]) -> None:
        self.restore_recorders.append(self.session_recorder)
        super().restore_message_history(history)

    def dump_compaction_state(self) -> dict[str, Any]:
        return self.compaction

    def restore_compaction_state(self, data: dict[str, Any]) -> None:
        self.restore_recorders.append(self.session_recorder)
        self.compaction = data


@pytest.mark.asyncio
async def test_rebuild_restores_history_with_recorder_detached_then_records_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, coder_cls=_RecorderAwareAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        first = app.agent
        assert isinstance(first, _RecorderAwareAgent)
        recorder = app._session_recorder
        assert first.kwargs["session_recorder"] is recorder
        assert first.restore_recorders == []
        async for _ in first.process_message_stream("before rebuild"):
            pass
        first.compaction = {"summary": "retained", "compacted_through": 1, "compacted_history_length": 1}
        expected_history = first.dump_message_history()
        recorded_history = store.load(session.session_id).history
        assert [Message.from_dict(item).to_dict() for item in recorded_history] == expected_history

        await app._build_agent(config, rebuild=True, restore_transcript=False)

        rebuilt = app.agent
        assert isinstance(rebuilt, _RecorderAwareAgent)
        assert rebuilt is not first
        assert first.cleanup_calls == 1
        assert rebuilt.kwargs["session_recorder"] is None
        assert rebuilt.restore_recorders == [None, None]
        assert rebuilt.dump_message_history() == expected_history
        assert rebuilt.compaction == first.compaction
        assert rebuilt.session_recorder is recorder
        assert app.session_runtime.agent is rebuilt
        assert store.load(session.session_id).history == recorded_history

        async for _ in rebuilt.process_message_stream("after rebuild"):
            pass
        assert [
            Message.from_dict(item).to_dict() for item in store.load(session.session_id).history
        ] == rebuilt.dump_message_history()
        assert len(recorder.list_rewindable_turns()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("linked_workspace", [False, True], ids=["launch", "linked"])
@pytest.mark.parametrize("launch_trusted", [False, True], ids=["untrusted", "trusted"])
async def test_mcp_uses_active_workspace_config_but_launch_project_trust_and_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_workspace: bool,
    launch_trusted: bool,
) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    launch, linked = _make_worktrees(tmp_path)
    active = linked if linked_workspace else launch
    store = SessionStore(tmp_path / "state")
    settings_store = SettingsStore(store.root)
    settings = CliSettings()
    settings.trust_mcp_project(launch if launch_trusted else linked)
    settings_store.save(settings)
    for server_id, config_path in (
        ("global", global_mcp_config_path(store.root)),
        ("launch", project_mcp_config_path(launch)),
        ("linked", project_mcp_config_path(linked)),
    ):
        server = MCPServerConfig(id=server_id, transport="stdio", command="unused-mcp-test-command")
        save_config_file(config_path, MCPConfigFile(servers=[server]))
        MCPStatusStore(store.root).update(
            server.id,
            MCPServerStatus.verified(
                fingerprint=server_fingerprint(server),
                transport=server.transport,
                source="global" if server_id == "global" else "project",
                tools=[MCPToolStatus(id="inspect")],
            ),
        )
    config = build_test_config(launch)
    config.mcp_config = load_mcp_config(launch, store.root, project_trusted=launch_trusted)
    session = store.create(launch, "code", config_summary(config))
    if linked_workspace:
        session.active_project_path = str(linked)
        store.save(session)
    app = KolegaCodeApp(
        project_path=launch,
        config=config,
        mode="code",
        store=store,
        session=session,
        settings_store=settings_store,
    )
    request = permission_request_for_tool("exec_command", {"command": "npm run test"})
    assert request is not None
    ProjectPermissionStore(launch).add_rule(allow_rule_options(request)[0].rule)

    async with app.run_test():
        agent = app.agent
        assert isinstance(agent, FakeCoderAgent)
        assert agent.kwargs["project_path"] == active
        mcp_extension = extension_by_name(agent.kwargs["tool_extensions"], "mcp")
        expected = {"mcp__global__inspect"}
        if launch_trusted:
            expected.add(f"mcp__{'linked' if linked_workspace else 'launch'}__inspect")
        assert set(mcp_extension.tools) == expected
        assert mcp_extension.propagate_to_sub_agents is False
        callback = agent.kwargs["permission_callback"]
        assert callback == app.session_runtime.permission_callback
        assert callback.__self__ is app.session_runtime
        assert app.session_runtime.project_path == launch
        assert (await callback(request)).allowed
        assert app.control_channel.pending() == []
        assert app._pending_approval is None


def test_composition_import_does_not_import_textual() -> None:
    program = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class RejectTextual(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "textual" or fullname.startswith("textual."):
                    raise AssertionError(f"UI-free composition tried to import {fullname}")
                return None

        sys.meta_path.insert(0, RejectTextual())
        import kolega_code.session.composition
        loaded = [name for name in sys.modules if name == "textual" or name.startswith("textual.")]
        assert not loaded, loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
