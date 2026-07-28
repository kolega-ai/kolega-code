import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.tools import ToolCollection, ToolExtension
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore, SessionStoreError
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.tui.state import PendingApproval
from kolega_code.permissions import PERMISSIONS_RELATIVE_PATH, allow_rule_options, permission_request_for_tool
from kolega_code.services.file_system import LocalFileSystem
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.tools import ToolError
from kolega_code.utils import images as image_utils

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


class SwitchingFakeAgent(FakeCoderAgent):
    """Model-free agent using the production workspace-switching stack."""

    switch_project_path = BaseAgent.switch_project_path
    replace_workspace_extensions = BaseAgent.replace_workspace_extensions

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        root = Path(kwargs["project_path"]).resolve()
        self.agent_name = "coder"
        self.project_path = root
        self.prompt_extensions = list(kwargs["prompt_extensions"] or [])
        self.tool_extensions = list(kwargs["tool_extensions"] or [])
        self.custom_agent_catalog = kwargs["custom_agent_catalog"]
        self.filesystem = LocalFileSystem(root)
        self.terminal_manager = LocalTerminalManager(
            kwargs["workspace_id"],
            kwargs["thread_id"],
            kwargs["connection_manager"],
            default_workdir=root,
        )
        self.context = SimpleNamespace(
            workspace=SimpleNamespace(project_path=root),
            prompt_extensions=self.prompt_extensions,
            tool_extensions=self.tool_extensions,
            services=SimpleNamespace(
                filesystem=self.filesystem,
                terminal_manager=self.terminal_manager,
            ),
        )
        self.system_prompt_roots: list[Path] = []
        self.tool_collection = ToolCollection(
            project_path=root,
            workspace_id=kwargs["workspace_id"],
            thread_id=kwargs["thread_id"],
            connection_manager=kwargs["connection_manager"],
            config=kwargs["config"],
            caller=self,
            filesystem=self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=kwargs["browser_manager"],
            tool_extensions=kwargs["tool_extensions"],
        )

    def _initialize_system_prompt(self) -> None:
        self.system_prompt_roots.append(Path(self.project_path).resolve())

    async def cleanup(self) -> None:
        await self.tool_collection.cleanup()


def _build_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume_in_linked: bool = False,
    remove_linked_before_resume: bool = False,
    fail_fallback_projection_once: bool = False,
) -> tuple[Any, Path, Path, SessionStore]:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    main, linked = _make_worktrees(tmp_path)
    install_fake_agents(monkeypatch, coder_cls=SwitchingFakeAgent)
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
    return app, main, linked, store


def _workspace_events(store: SessionStore, session_id: str) -> list[Any]:
    return [
        event for event in store.journal(session_id).read_events() if event.event_type == "session.workspace_switched"
    ]


def _assert_agent_root(agent: SwitchingFakeAgent, root: Path) -> None:
    collection = agent.tool_collection
    assert isinstance(collection.filesystem, LocalFileSystem)
    assert isinstance(collection.terminal_manager, LocalTerminalManager)
    assert agent.project_path == root
    assert agent.context.workspace.project_path == root
    assert agent.filesystem.root_path == root
    assert collection.project_path == root
    assert collection.filesystem.root_path == root
    assert Path(agent.terminal_manager.default_workdir) == root
    assert Path(collection.terminal_manager.default_workdir) == root
    assert collection.read_file_tool.project_path == root
    assert collection.edit_tool.project_path == root
    assert collection.terminal_tool.project_path == root
    assert collection.agent_tool.project_path == root
    assert collection.workflow_tool.project_path == root
    assert collection.snapshot_service.project_path == root


@pytest.mark.asyncio
async def test_top_level_agent_exposes_exclusive_non_propagating_switch_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _main, _linked, _store = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control")
        prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-worktree-control")

        assert set(extension.tools) == {"switch_worktree"}
        assert extension.propagate_to_sub_agents is False
        assert extension.exclusive_tools == frozenset({"switch_worktree"})
        assert extension.tool_groups["planning_tools"] == ["switch_worktree"]
        assert prompt.propagate_to_sub_agents is False
        assert app.agent.tool_collection.exclusive_tools == frozenset({"switch_worktree"})


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
    app, main, linked, store = _build_app(tmp_path, monkeypatch, resume_in_linked=True)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
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
    app, main, _linked, store = _build_app(
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
    app, main, _linked, store = _build_app(
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
    install_fake_agents(monkeypatch, coder_cls=SwitchingFakeAgent)
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
async def test_switch_worktree_reroots_complete_workspace_and_records_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store = _build_app(tmp_path, monkeypatch)
    mcp_cleanup = AsyncMock()

    async def linked_mcp_tool() -> str:
        return "linked MCP"

    linked_mcp_extension = ToolExtension(
        name="mcp",
        tools={"linked_mcp_tool": linked_mcp_tool},
        cleanup=mcp_cleanup,
    )
    monkeypatch.setattr(
        "kolega_code.cli.app.build_mcp_tool_extension",
        lambda project_path, *_args, **_kwargs: (
            linked_mcp_extension if Path(project_path).resolve() == linked else None
        ),
    )

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        old_tracker = app._session_diff_tracker
        app._session_file_changes = [SimpleNamespace(path="old-root.txt")]
        (main / "same-name.txt").write_text("launch root\n", encoding="utf-8")
        (linked / "same-name.txt").write_text("active root\n", encoding="utf-8")
        switch = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]

        result = await switch(str(linked / "nested"))

        assert "Switched the complete active workspace" in result
        assert app.project_path == main
        assert app.active_project_path == linked
        assert str(linked) in app._meta_content()
        assert str(main) not in app._meta_content()
        assert app.session_runtime.project_path == linked
        _assert_agent_root(app.agent, linked)
        assert app.agent.system_prompt_roots[-1] == linked
        assert app.agent.filesystem.read_text("linked-only.txt") == "linked\n"
        mention_attachments = app._build_mention_attachments("@same-name.txt")
        assert mention_attachments is not None
        assert mention_attachments[0]["content"] == "active root\n"

        attached_paths: list[Path] = []
        monkeypatch.setattr(
            image_utils,
            "encode_image_file",
            lambda path: attached_paths.append(Path(path)) or None,
        )
        await app._command_attach("same-name.txt")
        assert attached_paths == [linked / "same-name.txt"]

        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        app._pending_approval = PendingApproval(
            request=request,
            request_id="switched-root-permission",
            rule_options=allow_rule_options(request),
        )
        monkeypatch.setattr(app, "_answer_permission_request", Mock())
        await app._answer_approval_option(2)
        assert (linked / PERMISSIONS_RELATIVE_PATH).is_file()
        assert not (main / PERMISSIONS_RELATIVE_PATH).exists()

        assert app._session_diff_tracker is not old_tracker
        assert app._session_diff_tracker is not None
        assert app._session_diff_tracker.project_path == linked
        checkpoints = app._session_diff_tracker.checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint_id == 0
        assert checkpoints[0].label == 'Workspace switched · "feature/switch-test"'
        assert app._session_file_changes == []

        assert app.file_index.project_path == linked
        assert "linked-only.txt" in {entry.path for entry in app.file_index.entries()}
        assert "linked-only-skill" in app.skill_catalog.skills
        assert "main-only-skill" not in app.skill_catalog.skills
        assert "linked-only-helper" in app.custom_agent_catalog.agents
        assert "main-only-helper" not in app.custom_agent_catalog.agents
        assert app.agent.custom_agent_catalog is app.custom_agent_catalog

        skill_extension = extension_by_name(app.agent.tool_extensions, "cli-agent-skills")
        skill_listing = await skill_extension.tools["list_skills"]()
        assert "linked-only-skill" in skill_listing
        assert "main-only-skill" not in skill_listing
        assert app.agent.tool_collection.has_tool("linked_mcp_tool")
        assert await app.agent.tool_collection.call("linked_mcp_tool") == "linked MCP"
        mcp_cleanup.assert_not_awaited()

        assert app.session.active_project_path == str(linked)
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


@pytest.mark.asyncio
async def test_switch_worktree_same_root_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, main, _linked, store = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        tracker = app._session_diff_tracker
        generation = app._session_diff_generation
        switch = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]

        result = await switch(str(main))

        assert "already active" in result
        assert "baseline was not reset" in result
        assert app._session_diff_tracker is tracker
        assert app._session_diff_generation == generation
        assert app.session.active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        _assert_agent_root(app.agent, main)


@pytest.mark.asyncio
async def test_switch_worktree_is_blocked_by_running_local_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        tracker = app._session_diff_tracker
        terminal = app.agent.terminal_manager
        terminal.sessions["running"] = cast(Any, SimpleNamespace(running=True))
        switch = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]
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
    app, main, _linked, store = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        tracker = app._session_diff_tracker
        skills = app.skill_catalog
        custom_agents = app.custom_agent_catalog
        file_index = app.file_index
        switch = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]

        with pytest.raises(ToolError, match="No registered worktree matches"):
            await switch("missing-worktree")

        assert app.active_project_path == main
        assert app._session_diff_tracker is tracker
        assert app.skill_catalog is skills
        assert app.custom_agent_catalog is custom_agents
        assert app.file_index is file_index
        assert app.session.active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        _assert_agent_root(app.agent, main)


@pytest.mark.asyncio
async def test_session_save_failure_rolls_back_workspace_and_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store = _build_app(tmp_path, monkeypatch)
    replacement_cleanup = AsyncMock()
    replacement_mcp = ToolExtension(
        name="mcp",
        tools={"replacement_mcp": AsyncMock(return_value="unused")},
        cleanup=replacement_cleanup,
    )
    monkeypatch.setattr(
        "kolega_code.cli.app.build_mcp_tool_extension",
        lambda project_path, *_args, **_kwargs: replacement_mcp if Path(project_path).resolve() == linked else None,
    )

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        tracker = app._session_diff_tracker
        skills = app.skill_catalog
        custom_agents = app.custom_agent_catalog
        file_index = app.file_index
        pending_previews = {"old-call": [SimpleNamespace(path="old-root.txt")]}
        timer = Mock()
        app._pending_edit_previews = pending_previews
        app._session_diff_dirty = True
        app._session_diff_timer = timer
        save_calls = 0

        async def fail_switch_save(**_kwargs: Any) -> None:
            nonlocal save_calls
            save_calls += 1
            raise OSError("metadata unavailable")

        switch = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]
        with monkeypatch.context() as save_patch:
            save_patch.setattr(app, "_save_workspace_switch_async", fail_switch_save)
            with pytest.raises(ToolError, match="previous workspace was restored.*metadata unavailable"):
                await switch(str(linked))

        assert save_calls == 1
        assert app.active_project_path == main
        assert app.session_runtime.project_path == main
        assert app._session_diff_tracker is tracker
        assert app.skill_catalog is skills
        assert app.custom_agent_catalog is custom_agents
        assert app.file_index is file_index
        assert app._pending_edit_previews is pending_previews
        assert app._session_diff_dirty is True
        assert app._session_diff_timer is timer
        timer.pause.assert_not_called()
        assert app.session.active_project_path is None
        assert store.load(app.session.session_id).active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        _assert_agent_root(app.agent, main)
        assert linked in app.agent.system_prompt_roots
        assert app.agent.system_prompt_roots[-1] == main
        assert app.agent.tool_collection.has_tool("replacement_mcp") is False
        replacement_cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspace_event_failure_rolls_back_without_phantom_metadata_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        tracker = app._session_diff_tracker
        recorder = app._session_recorder
        record_switch = Mock(side_effect=OSError("workspace event unavailable"))
        monkeypatch.setattr(recorder, "record_workspace_switched", record_switch)

        switch = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-worktree-control").tools["switch_worktree"]
        with pytest.raises(ToolError, match="previous workspace was restored.*workspace event unavailable"):
            await switch(str(linked))

        assert app.active_project_path == main
        assert app._session_diff_tracker is tracker
        assert app.session.active_project_path is None
        assert store.load(app.session.session_id).active_project_path is None
        assert _workspace_events(store, app.session.session_id) == []
        assert not any(
            event.event_type == "session.metadata_updated"
            and event.payload.get("patch", {}).get("active_project_path") == str(linked)
            for event in store.journal(app.session.session_id).read_events()
        )
        record_switch.assert_called_once()
        _assert_agent_root(app.agent, main)


@pytest.mark.asyncio
async def test_rollback_failure_disables_tools_and_reports_unrestored_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, main, linked, store = _build_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert isinstance(app.agent, SwitchingFakeAgent)
        agent = app.agent
        original_switch = agent.switch_project_path

        async def fail_old_root_rollback(new_root: str | Path) -> None:
            candidate = Path(new_root).resolve()
            if candidate == main and agent.project_path == linked:
                raise RuntimeError("old checkout unavailable")
            await original_switch(candidate)

        async def fail_switch_save(**_kwargs: Any) -> None:
            raise OSError("journal unavailable")

        monkeypatch.setattr(agent, "switch_project_path", fail_old_root_rollback)
        monkeypatch.setattr(app, "_save_workspace_switch_async", fail_switch_save)
        switch = extension_by_name(agent.tool_extensions, "cli-worktree-control").tools["switch_worktree"]

        with pytest.raises(
            ToolError,
            match="restoring the previous workspace also failed.*tools are disabled.*journal unavailable.*old checkout",
        ):
            await switch(str(linked))

        assert agent.project_path == linked
        assert app.active_project_path == linked
        assert app.session_runtime.project_path == linked
        assert app._session_diff_tracker is None
        assert agent.tool_collection._workspace_disabled_reason is not None
        assert "rollback failed" in agent.tool_collection._workspace_disabled_reason
        assert agent.tool_collection.get_tool_list() == []
        assert agent.tool_collection.has_tool("read_file") is False

        # Durable session metadata still names the last committed workspace;
        # callers must restart/resume before any tools can run again.
        assert app.session.active_project_path is None
        assert store.load(app.session.session_id).active_project_path is None
