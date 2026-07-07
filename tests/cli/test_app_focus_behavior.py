from pathlib import Path
import asyncio

import pytest

from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

from ._app_test_utils import FakeCoderAgent, build_test_config, install_fake_agents


def _build_focus_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, coder_cls=FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def _wait_for_focus(pilot, app, widget) -> None:
    for _ in range(5):
        await pilot.pause()
        if app.focused is widget:
            return


@pytest.mark.asyncio
async def test_app_focus_refocuses_idle_chat_composer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        conversation = app.query_one("#conversation")

        app.screen.set_focus(conversation)
        await pilot.pause()
        assert app.focused is conversation

        app.on_app_focus(events.AppFocus())
        await _wait_for_focus(pilot, app, composer)

        assert composer.disabled is False
        assert app.focused is composer


@pytest.mark.asyncio
async def test_app_focus_keeps_active_approval_prompt_focused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        request = permission_request_for_tool("exec_command", {"command": "npm test"})
        assert request is not None
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request,
            future=future,
            rule_options=allow_rule_options(request),
        )
        app._set_approval_actions_visible(True)
        app._set_chat_enabled(False)

        approval_actions = app.query_one("#approval_actions", ActionList)
        assert app.focused is approval_actions

        app.on_app_focus(events.AppFocus())
        await _wait_for_focus(pilot, app, approval_actions)

        assert app.focused is approval_actions

        app._pending_approval = None
        app._set_approval_actions_visible(False)


@pytest.mark.asyncio
async def test_app_focus_preserves_question_free_form_composer_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        composer = app.query_one("#composer", ChatComposer)
        app.screen.set_focus(composer)
        await pilot.pause()
        assert app.focused is composer

        app.on_app_focus(events.AppFocus())
        await _wait_for_focus(pilot, app, composer)

        assert composer.disabled is False
        assert app.focused is composer

        app._pending_question = None
        app._set_question_actions_visible(False)


@pytest.mark.asyncio
async def test_app_focus_heals_question_drift_to_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        question_actions = app.query_one("#question_actions", ActionList)
        assert app.focused is question_actions

        app.screen.set_focus(app.query_one("#conversation"))
        await pilot.pause()
        app.on_app_focus(events.AppFocus())
        await _wait_for_focus(pilot, app, question_actions)

        assert app.focused is question_actions

        app._pending_question = None
        app._set_question_actions_visible(False)


@pytest.mark.asyncio
async def test_question_composer_top_line_up_returns_focus_to_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        composer = app.query_one("#composer", ChatComposer)
        question_actions = app.query_one("#question_actions", ActionList)
        app.screen.set_focus(composer)
        await pilot.pause()
        assert app.focused is composer
        assert composer.cursor_location[0] == 0

        await pilot.press("up")
        assert app.focused is question_actions
        assert question_actions.highlighted == 1

        await pilot.press("enter")
        assert await future == "B"
        assert app._pending_question is None


@pytest.mark.asyncio
async def test_question_composer_multiline_up_keeps_editor_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        composer = app.query_one("#composer", ChatComposer)
        question_actions = app.query_one("#question_actions", ActionList)
        composer.load_text("one\ntwo")
        composer.move_cursor((1, 1), record_width=False)
        app.screen.set_focus(composer)
        await pilot.pause()
        assert app.focused is composer

        await pilot.press("up")

        assert app.focused is composer
        assert app.focused is not question_actions
        assert composer.cursor_location[0] == 0

        app._pending_question = None
        app._set_question_actions_visible(False)


@pytest.mark.asyncio
async def test_question_composer_dropdown_up_keeps_completion_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.file_index import IndexEntry
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer, CompletionDropdown, CompletionItem

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        composer = app.query_one("#composer", ChatComposer)
        question_actions = app.query_one("#question_actions", ActionList)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        app.screen.set_focus(composer)
        await pilot.pause()
        assert app.focused is composer
        dropdown.open_with(
            [
                CompletionItem(prompt="alpha.py", value=IndexEntry(path="alpha.py", is_dir=False)),
                CompletionItem(prompt="beta.py", value=IndexEntry(path="beta.py", is_dir=False)),
            ]
        )
        dropdown.highlighted = 1
        assert dropdown.is_open is True

        await pilot.press("up")

        assert app.focused is composer
        assert app.focused is not question_actions
        assert dropdown.highlighted == 0

        app._pending_question = None
        app._set_question_actions_visible(False)


@pytest.mark.asyncio
async def test_question_actions_bottom_down_focuses_composer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    app = _build_focus_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B", "C"], future=future)
        app._show_question_options("Choose?", ["A", "B", "C"])
        app._set_chat_enabled(True)

        composer = app.query_one("#composer", ChatComposer)
        question_actions = app.query_one("#question_actions", ActionList)
        assert app.focused is question_actions
        question_actions.highlighted = question_actions.option_count - 1

        await pilot.press("down")

        assert composer.disabled is False
        assert app.focused is composer
        assert question_actions.highlighted == question_actions.option_count - 1

        app._pending_question = None
        app._set_question_actions_visible(False)
