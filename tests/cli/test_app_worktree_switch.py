import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import Mock

import pytest

from kolega_code.agent.tools import ToolExtension
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore, SessionStoreError
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.tui.state import PendingApproval
from kolega_code.permissions import PERMISSIONS_RELATIVE_PATH, allow_rule_options, permission_request_for_tool
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.tools import ToolError

from ._app_test_utils import FakeCoderAgent, build_test_config, extension_by_name, install_fake_agents


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_skill(root: Path, name: str) -> None:
    skill_dir = root / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill from {root.name}.\n---\n\nFollow {name} instructions.\n",
        encoding="utf-8",
    )


def _write_custom_agent(root: Path, name: str) -> None:
    agent_dir = root / ".kolega" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Agent from {root.name}.\nmode: all\n---\n\nHandle {name} tasks.\n",
        encoding="utf-8",
    )


def _make_worktrees(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    linked = tmp_path / "linked"
    main.mkdir()
    _git(main, "init", "-b", "main")
    _git(main, "config", "user.email", "test@example.com")
    _git(main, "config", "user.name", "Test User")
    (main / "tracked.txt").write_text("main\n", encoding="utf-8")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "initial")
    _git(main, "worktree", "add", "-b", "feature/switch-test", str(linked))

    _write_skill(main, "main-only-skill")
    _write_custom_agent(main, "main-only-helper")
    _write_skill(linked, "linked-only-skill")
    _write_custom_agent(linked, "linked-only-helper")
    (linked / "linked-only.txt").write_text("linked\n", encoding="utf-8")
    (linked / "nested").mkdir()
    return main.resolve(), linked.resolve()


class RecordingFakeCoderAgent(FakeCoderAgent):
    """Fake agent recording every constructed instance on its class.

    ``_build_app`` mints a fresh subclass per call so the ``instances`` list
    never leaks across tests; callers can assert a rebuild happened by
    checking ``len(agent_cls.instances)``.
    """

    instances: ClassVar[list["RecordingFakeCoderAgent"]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        type(self).instances.append(self)


def _build_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume_in_linked: bool = False,
    remove_linked_before_resume: bool = False,
    fail_fallback_projection_once: bool = False,
) -> tuple[Any, Path, Path, SessionStore, type[RecordingFakeCoderAgent]]:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    main, linked = _make_worktrees(tmp_path)

    class _SessionRecordingAgent(RecordingFakeCoderAgent):
        instances: ClassVar[list[RecordingFakeCoderAgent]] = []

    install_fake_agents(monkeypatch, coder_cls=_SessionRecordingAgent)
    config = build_test_config(main)
    store = SessionStore(tmp_path / "state")
    session = store.create(main, "code", config_summary(config))
    if resume_in_linked:
        session.active_project_path = str(linked)
        store.save(session)
    if remove_linked_before_resume:
        _git(linked, "add", ".")
        _git(linked, "commit", "-m", "make linked worktree removable")
        _git(main, "worktree", "remove", str(linked))
    if fail_fallback_projection_once:
        original_write_metadata = store._write_metadata
        failed = False

        def fail_once(metadata: dict[str, Any]) -> None:
            nonlocal failed
            if metadata.get("active_project_path") is None and not failed:
                failed = True
                raise OSError("projection unavailable")
            original_write_metadata(metadata)

        monkeypatch.setattr(store, "_write_metadata", fail_once)
    app = KolegaCodeApp(project_path=main, config=config, mode="code", store=store, session=session)
    return app, main, linked, store, _SessionRecordingAgent


def _workspace_events(store: SessionStore, session_id: str) -> list[Any]:
    return [
        event for event in store.journal(session_id).read_events() if event.event_type == "session.workspace_switched"
    ]


def _assert_agent_root(agent: FakeCoderAgent, root: Path) -> None:
    assert agent.kwargs["project_path"] == root


def _switch_tool(app: Any):
    return extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]


@pytest.mark.asyncio
async def test_top_level_agent_exposes_exclusive_non_propagating_switch_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _main, _linked, _store, agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, agent_cls)
        extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control")
        prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-worktree-control")

        assert set(extension.tools) == {"switch_worktree"}
        assert extension.propagate_to_sub_agents is False
        assert extension.exclusive_tools == frozenset({"switch_worktree"})
        assert extension.tool_groups["planning_tools"] == ["switch_worktree"]
        assert prompt.propagate_to_sub_agents is False


@pytest.mark.asyncio
async def test_tui_resume_constructs_agent_and_changes_tracker_in_persisted_active_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_roots: list[Path] = []
    monkeypatch.setattr(
        agent_runtime_module,
        "build_mcp_tool_extension",
        lambda project_path, *_args, **_kwargs: mcp_roots.append(Path(project_path).resolve()) or None,
    )
    app, main, linked, store, agent_cls = _build_app(tmp_path, monkeypatch, resume_in_linked=True)

    async with app.run_test():
        assert isinstance(app.agent, agent_cls)
        assert app.project_path == main
        assert app.active_project_path == linked
        assert app.session.project_path == str(main)
        assert app.session.active_project_path == str(linked)
        assert store.load(app.session.session_id).active_project_path == str(linked)
        _assert_agent_root(app.agent, linked)
        assert app._session_diff_tracker is not None
        assert app._session_diff_tracker.project_path == linked
        assert app._session_diff_tracker.checkpoints()[0].checkpoint_id == 0
        assert mcp_roots == [linked]


def test_tui_resume_warns_and_persists_fallback_when_active_worktree_was_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, _linked, store, _agent_cls = _build_app(
        tmp_path,
        monkeypatch,
        resume_in_linked=True,
        remove_linked_before_resume=True,
    )

    assert app.project_path == main
    assert app.active_project_path == main
    assert app.session.active_project_path is None
    assert store.load(app.session.session_id).active_project_path is None
    assert "Saved active worktree is unavailable" in app._startup_workspace_warning
    assert f"`{main}`" in app._startup_workspace_warning


def test_tui_resume_uses_canonical_fallback_when_metadata_projection_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, _linked, store, _agent_cls = _build_app(
        tmp_path,
        monkeypatch,
        resume_in_linked=True,
        remove_linked_before_resume=True,
        fail_fallback_projection_once=True,
    )

    assert app.active_project_path == main
    assert app.session.active_project_path is None
    assert store.load(app.session.session_id).active_project_path is None
    assert "Saved active worktree is unavailable" in app._startup_workspace_warning


def test_tui_resume_fails_without_clearing_metadata_when_launch_and_active_worktrees_are_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    main, linked = _make_worktrees(tmp_path)
    install_fake_agents(monkeypatch, coder_cls=FakeCoderAgent)
    config = build_test_config(main)
    store = SessionStore(tmp_path / "state")
    session = store.create(main, "code", config_summary(config))
    session.active_project_path = str(linked)
    store.save(session)

    _git(linked, "add", ".")
    _git(linked, "commit", "-m", "make linked worktree removable")
    _git(main, "worktree", "remove", str(linked))
    (main / ".git").rename(tmp_path / "unregistered-git-dir")

    with pytest.raises(SessionStoreError, match="cannot resume"):
        KolegaCodeApp(project_path=main, config=config, mode="code", store=store, session=session)

    assert store.load(session.session_id).active_project_path == str(linked)


@pytest.mark.asyncio
async def test_switch_worktree_same_root_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, main, _linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        first_agent = app.agent
        tracker = app._session_diff_tracker
        generation = app._session_diff_generation
        switch = _switch_tool(app)

        result = await switch(str(main))

        assert "already active" in result
        assert "baseline was not reset" in result
        assert app._session_diff_tracker is tracker
        assert app._session_diff_generation == generation
        assert app.session.active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        assert app._pending_workspace_switch is None
        assert app.agent is first_agent
        _assert_agent_root(app.agent, main)


@pytest.mark.asyncio
async def test_switch_worktree_is_blocked_by_running_local_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        tracker = app._session_diff_tracker
        terminal = LocalTerminalManager(
            app.agent.kwargs["workspace_id"],
            app.agent.kwargs["thread_id"],
            app.agent.kwargs["connection_manager"],
            default_workdir=main,
        )
        app.agent.terminal_manager = terminal
        terminal.sessions["running"] = cast(Any, SimpleNamespace(running=True))
        switch = _switch_tool(app)
        try:
            with pytest.raises(ToolError, match="terminal sessions are running"):
                await switch(str(linked))
        finally:
            terminal.sessions.pop("running", None)

        assert app.active_project_path == main
        assert app._session_diff_tracker is tracker
        assert app.session.active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        _assert_agent_root(app.agent, main)


@pytest.mark.asyncio
async def test_invalid_worktree_target_leaves_workspace_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, _linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        first_agent = app.agent
        tracker = app._session_diff_tracker
        skills = app.skill_catalog
        custom_agents = app.custom_agent_catalog
        file_index = app.file_index
        switch = _switch_tool(app)

        with pytest.raises(ToolError, match="No registered worktree matches"):
            await switch("missing-worktree")

        assert app.active_project_path == main
        assert app._session_diff_tracker is tracker
        assert app.skill_catalog is skills
        assert app.custom_agent_catalog is custom_agents
        assert app.file_index is file_index
        assert app.session.active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        assert app._pending_workspace_switch is None
        assert app.agent is first_agent
        _assert_agent_root(app.agent, main)


@pytest.mark.asyncio
async def test_switch_commits_boundary_and_defers_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, main, linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        first_agent = app.agent
        tracker = app._session_diff_tracker
        switch = _switch_tool(app)

        result = await switch(str(linked))

        assert result.startswith("Committed a switch")
        pending = app._pending_workspace_switch
        assert pending is not None
        assert pending.new_root == linked
        assert pending.old_root == main
        assert pending.branch == "feature/switch-test"
        assert pending.previous_active_metadata is None

        assert store.load(app.session.session_id).active_project_path == str(linked)
        events = _workspace_events(store, app.session.session_id)
        assert len(events) == 1
        assert events[0].payload == {
            "old_root": str(main),
            "new_root": str(linked),
            "old_label": main.name,
            "new_label": linked.name,
            "old_branch": "main",
            "new_branch": "feature/switch-test",
        }

        # Nothing about the live workspace moves until the turn ends.
        assert app.active_project_path == main
        assert app.agent is first_agent
        assert app._session_diff_tracker is tracker


@pytest.mark.asyncio
async def test_apply_pending_switch_rebuilds_workspace_and_returns_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_calls: list[Path] = []

    async def linked_mcp_tool() -> str:
        return "linked MCP"

    linked_mcp_extension = ToolExtension(name="mcp", tools={"linked_mcp_tool": linked_mcp_tool})

    app, main, linked, store, agent_cls = _build_app(tmp_path, monkeypatch)

    def fake_build_mcp(project_path, *_args, **_kwargs):
        root = Path(project_path).resolve()
        mcp_calls.append(root)
        return linked_mcp_extension if root == linked else None

    monkeypatch.setattr(agent_runtime_module, "build_mcp_tool_extension", fake_build_mcp)

    async with app.run_test():
        first_agent = app.agent
        tracker_before = app._session_diff_tracker
        (main / "same-name.txt").write_text("launch root\n", encoding="utf-8")
        (linked / "same-name.txt").write_text("active root\n", encoding="utf-8")

        switch = _switch_tool(app)
        result = await switch(str(linked))
        assert result.startswith("Committed a switch")

        first_agent.append_user_message("before switch")

        continuation = await app._apply_pending_workspace_switch()

        assert continuation is not None
        assert continuation.startswith(f"The active workspace is now `{linked}`")

        assert len(agent_cls.instances) == 2
        assert app.agent is agent_cls.instances[1]
        assert app.agent is not first_agent
        assert app.agent.kwargs["project_path"] == linked

        # Message history survived the rebuild via the session projection.
        assert any(getattr(message, "content", None) == "before switch" for message in app.agent.history)

        assert app.active_project_path == linked
        # Permission matching stays launch-scoped, like trust.
        assert app.session_runtime.project_path == main

        assert "linked-only-skill" in app.skill_catalog.skills
        assert "main-only-skill" not in app.skill_catalog.skills
        assert "linked-only-helper" in app.custom_agent_catalog.agents
        assert "main-only-helper" not in app.custom_agent_catalog.agents
        assert app.agent.kwargs["custom_agent_catalog"] is app.custom_agent_catalog

        assert app.file_index.project_path == linked
        assert "linked-only.txt" in {entry.path for entry in app.file_index.entries()}

        assert app._session_diff_tracker is not tracker_before
        assert app._session_diff_tracker is not None
        assert app._session_diff_tracker.project_path == linked
        checkpoints = app._session_diff_tracker.checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint_id == 0
        assert checkpoints[0].label == 'Workspace switched · "feature/switch-test"'
        assert app._session_file_changes == []
        assert app._pending_workspace_switch is None

        meta = app._meta_content()
        assert str(linked) in meta
        assert str(main) not in meta

        assert any(
            entry.kind == "system" and "Workspace switched to" in entry.content for entry in app.conversation_entries
        )

        assert linked in mcp_calls
        mcp_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "mcp")
        assert set(mcp_extension.tools) == {"linked_mcp_tool"}

        mention_attachments = app._build_mention_attachments("@same-name.txt")
        assert mention_attachments is not None
        assert mention_attachments[0]["content"] == "active root\n"

        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        app._pending_approval = PendingApproval(
            request=request,
            request_id="switched-root-permission",
            rule_options=allow_rule_options(request),
        )
        monkeypatch.setattr(app, "_answer_permission_request", Mock())
        await app._answer_approval_option(2)
        # Saved rules persist in the enclosing launch checkout, so they
        # survive the worktree and apply to every workspace in the session.
        assert (main / PERMISSIONS_RELATIVE_PATH).is_file()
        assert not (linked / PERMISSIONS_RELATIVE_PATH).exists()


@pytest.mark.asyncio
async def test_process_message_runs_continuation_turn_in_new_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store, agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        first_agent = app.agent
        switch = _switch_tool(app)
        await switch(str(linked))

        await app._process_message("original task")

        assert first_agent.messages == ["original task"]
        assert len(agent_cls.instances) == 2
        assert app.agent is agent_cls.instances[1]
        assert app.agent is not first_agent
        assert len(app.agent.messages) == 1
        assert "The active workspace is now" in app.agent.messages[0]
        assert app.active_project_path == linked


@pytest.mark.asyncio
async def test_second_switch_before_turn_end_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, main, linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        switch = _switch_tool(app)
        first_result = await switch(str(linked))
        assert first_result.startswith("Committed a switch")

        with pytest.raises(ToolError, match="already committed"):
            await switch(str(main))

        assert len(_workspace_events(store, app.session.session_id)) == 1


@pytest.mark.asyncio
async def test_persist_failure_leaves_workspace_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, main, linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        first_agent = app.agent

        async def fail_save(**_kwargs: Any) -> None:
            raise OSError("metadata unavailable")

        monkeypatch.setattr(app, "_save_workspace_switch_async", fail_save)
        switch = _switch_tool(app)

        with pytest.raises(ToolError, match="Could not persist.*metadata unavailable"):
            await switch(str(linked))

        assert app._pending_workspace_switch is None
        assert app.session.active_project_path is None
        assert store.load(app.session.session_id).active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        assert app.agent is first_agent


@pytest.mark.asyncio
async def test_workspace_event_failure_does_not_publish_phantom_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        recorder = app._session_recorder
        record_switch = Mock(side_effect=OSError("workspace event unavailable"))
        monkeypatch.setattr(recorder, "record_workspace_switched", record_switch)

        switch = _switch_tool(app)
        with pytest.raises(ToolError, match="Could not persist"):
            await switch(str(linked))

        assert _workspace_events(store, app.session.session_id) == []
        assert not any(
            event.event_type == "session.metadata_updated"
            and event.payload.get("patch", {}).get("active_project_path") == str(linked)
            for event in store.journal(app.session.session_id).read_events()
        )
        assert store.load(app.session.session_id).active_project_path is None
        assert app._pending_workspace_switch is None
        record_switch.assert_called_once()


@pytest.mark.asyncio
async def test_rebuild_failure_falls_back_to_previous_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store, agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        first_agent = app.agent
        switch = _switch_tool(app)
        await switch(str(linked))

        original_build_agent = app._build_agent

        async def flaky_build_agent(config, rebuild=False, *, restore_transcript=True, preserve_queued=False):
            if app.active_project_path == linked:
                raise RuntimeError("rebuild failed")
            return await original_build_agent(
                config, rebuild=rebuild, restore_transcript=restore_transcript, preserve_queued=preserve_queued
            )

        monkeypatch.setattr(app, "_build_agent", flaky_build_agent)

        result = await app._apply_pending_workspace_switch()

        assert result is None
        assert app.active_project_path == main
        assert len(agent_cls.instances) == 2
        assert app.agent is agent_cls.instances[1]
        assert app.agent is not first_agent
        assert app.agent.kwargs["project_path"] == main

        events = _workspace_events(store, app.session.session_id)
        assert len(events) == 2
        assert events[0].payload == {
            "old_root": str(main),
            "new_root": str(linked),
            "old_label": main.name,
            "new_label": linked.name,
            "old_branch": "main",
            "new_branch": "feature/switch-test",
        }
        assert events[1].payload == {
            "old_root": str(linked),
            "new_root": str(main),
            "old_label": linked.name,
            "new_label": main.name,
            "old_branch": "feature/switch-test",
            "new_branch": "main",
        }

        assert store.load(app.session.session_id).active_project_path is None
        assert any(entry.kind == "system" and "continuing in" in entry.content for entry in app.conversation_entries)
        assert app._pending_workspace_switch is None


@pytest.mark.asyncio
async def test_rebuild_double_failure_disables_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli.tui.widgets import ChatComposer

    app, main, linked, store, _agent_cls = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        switch = _switch_tool(app)
        await switch(str(linked))

        async def always_fail_build_agent(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("rebuild failed")

        monkeypatch.setattr(app, "_build_agent", always_fail_build_agent)

        result = await app._apply_pending_workspace_switch()

        assert result is None
        assert any(
            entry.kind == "system" and "Restart or resume the session" in entry.content
            for entry in app.conversation_entries
        )
        assert store.load(app.session.session_id).active_project_path is None
        assert app.query_one("#composer", ChatComposer).disabled is True
