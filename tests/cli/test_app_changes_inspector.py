import subprocess
from pathlib import Path

import pytest
from textual.widgets import TextArea

from kolega_code.events import AgentEvent
from kolega_code.cli.tui.widgets import ChatComposer

from ._app_test_utils import _build_sub_agent_test_app, _sub_agent_event, renderable_text


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _init_git_project(project: Path) -> None:
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test User")
    (project / "src").mkdir(exist_ok=True)
    (project / "src" / "a.py").write_text("old a\n", encoding="utf-8")
    (project / "src" / "b.py").write_text("old b\n", encoding="utf-8")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "initial")


def _binding_keys(bindings) -> set[str]:
    keys: set[str] = set()
    for binding in bindings:
        for key in str(getattr(binding, "key", "")).split(","):
            if key.strip():
                keys.add(key.strip())
    return keys


def _preview(path: str, *, tool_call_id: str = "t1", tool_name: str = "search_and_replace") -> dict:
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "kind": "diff",
        "path": path,
        "language": "python",
        "lines": [["meta", "@@ -1 +1 @@"], ["del", "-old"], ["add", "+new"]],
        "more": 0,
        "adds": 1,
        "dels": 1,
    }


def _file_edit_preview_event(path: str, *, tool_call_id: str = "t1", sub_agent_info: dict | None = None) -> AgentEvent:
    return AgentEvent(
        event_type="file_edit_preview",
        sender="coder",
        content=_preview(path, tool_call_id=tool_call_id),
        sub_agent_info=sub_agent_info,
    )


def test_changes_hotkey_uses_ctrl_r_without_known_composer_conflict() -> None:
    from kolega_code.cli.app import KolegaCodeApp

    app_bindings = {binding.action: binding for binding in KolegaCodeApp.BINDINGS}
    assert app_bindings["open_changes"].key == "ctrl+r"
    assert app_bindings["open_changes"].key_display == "Ctrl+R"

    assert "ctrl+r" not in _binding_keys(TextArea.BINDINGS)
    assert "ctrl+r" not in _binding_keys(ChatComposer.BINDINGS)
    # Keep Ctrl+D free for the composer/TextArea delete-right behavior.
    assert "ctrl+d" in _binding_keys(TextArea.BINDINGS)


@pytest.mark.asyncio
async def test_main_agent_preview_is_recorded_and_still_attaches_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "tool_call_id": "t1",
                    "tool_name": "search_and_replace",
                    "text": "Editing src/a.py",
                },
            )
        )
        app._render_event(_file_edit_preview_event("src/a.py", tool_call_id="t1"))

        assert len(app._session_file_changes) == 1
        change = app._session_file_changes[0]
        assert change.path == "src/a.py"
        assert change.tool_call_id == "t1"
        assert change.source_label == "Agent"

        entry = app._tool_entries["t1"]
        assert entry.edit_preview is not None
        assert entry.edit_preview["path"] == "src/a.py"


@pytest.mark.asyncio
async def test_sub_agent_preview_is_recorded_and_attached_to_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _sub_agent_event(
                message_type="tool_call",
                text="Editing",
                tool_description="search_and_replace",
                tool_call_id="t1",
            )
        )
        sub_info = {
            "agent_id": "agent-1",
            "agent_name": "general-agent",
            "task": "inspect sessions",
            "parent_tool_call_id": "tc-1",
            "conversation_id": None,
            "depth": 1,
        }
        app._render_event(_file_edit_preview_event("src/sub.py", tool_call_id="t1", sub_agent_info=sub_info))

        assert len(app._session_file_changes) == 1
        change = app._session_file_changes[0]
        assert change.path == "src/sub.py"
        assert change.source_label == "Sub-agent general-agent #1"

        activity = app._sub_agent_activities["agent-1"]
        step = activity.tool_steps["t1"]
        assert step.edit_preview is not None
        assert step.edit_preview["path"] == "src/sub.py"


@pytest.mark.asyncio
async def test_changes_inspector_opens_and_renders_git_shell_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.changes_screen import ChangesInspectorScreen

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        (app.project_path / "src" / "b.py").unlink()

        app.action_open_changes()
        await pilot.pause()

        assert isinstance(app._changes_inspector, ChangesInspectorScreen)
        screen = app._changes_inspector
        assert len(screen._rows) == 2
        assert {change.path: change.status for change in app._session_diff_files} == {
            "src/a.py": "modified",
            "src/b.py": "deleted",
        }
        assert screen._selected_path in {"src/a.py", "src/b.py"}
        assert "net" in screen._preview_widgets


@pytest.mark.asyncio
async def test_changes_inspector_header_owns_path_and_counts_body_is_just_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import Static

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        app.action_open_changes("src/a.py")
        await pilot.pause()

        screen = app._changes_inspector
        assert screen is not None
        change = screen._diff_for_path("src/a.py")
        assert change is not None

        body_text = renderable_text(screen._net_diff_renderable(change))
        assert "src/a.py" not in body_text
        assert "+1 -1" not in body_text
        assert "-old a" in body_text
        assert "+new a" in body_text

        header_text = screen.query_one("#changes_header", Static).render()
        spans = {str(span.style) for span in getattr(header_text, "spans", [])}
        assert any("green" in style for style in spans)
        assert any("red" in style for style in spans)


@pytest.mark.asyncio
async def test_git_project_opens_empty_changes_inspector_when_no_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli import messages as cli_messages
    from textual.widgets import Static

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        app.action_open_changes()
        await pilot.pause()

        screen = app._changes_inspector
        assert screen is not None
        assert screen._rows == {}
        assert cli_messages.CHANGES_INSPECTOR_EMPTY in str(screen.query_one("#changes_header", Static).render())
        assert cli_messages.CHANGES_INSPECTOR_EMPTY in str(screen.query_one(".changes-empty", Static).render())


@pytest.mark.asyncio
async def test_non_git_project_disables_changes_inspector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        assert app._session_diff_tracker is None
        assert app.check_action("open_changes", ()) is False
        app.action_open_changes()
        assert app._changes_inspector is None


@pytest.mark.asyncio
async def test_copy_selected_changes_includes_net_diff_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", lambda text: copied.append(text))

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        app._render_event(_file_edit_preview_event("src/a.py", tool_call_id="a1"))
        app.action_open_changes("src/a.py")
        await pilot.pause()

        screen = app._changes_inspector
        assert screen is not None
        screen.action_copy_changes()

        assert copied
        text = copied[0]
        assert "Changes for src/a.py" in text
        assert "Status: modified" in text
        assert "+1 -1" in text
        assert "-old a" in text
        assert "+new a" in text
        assert "Captured edit events:" not in text
        assert "#1 Agent · search_and_replace · a1" not in text
