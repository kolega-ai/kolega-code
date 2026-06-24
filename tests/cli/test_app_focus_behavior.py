from pathlib import Path
import asyncio

import pytest

from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

from ._app_test_utils import MinimalFakeCoderAgent, build_test_config, install_fake_agents


def _build_focus_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, coder_cls=MinimalFakeCoderAgent)

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
