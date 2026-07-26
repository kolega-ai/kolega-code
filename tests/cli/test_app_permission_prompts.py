# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli import messages
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.tui import state as tui_state

from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
)
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.events import AgentEvent
from kolega_code.session.runtime import serialize_permission_request
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import (
    FakeCoderAgent,
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    install_fake_agents,
    question_payload,
    renderable_text,
)


def _build_permission_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, coder_cls=FakeCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def _wait_for_layout(pilot, predicate, *, timeout: float = 6.0) -> None:
    """Poll until ``predicate()`` is truthy or time out.

    Showing a prompt panel takes two layout passes: the first sizes the header's
    scroll container, and only the second gives its scrollbar a region.
    ``pilot.pause()``'s idle heuristic can return before that second pass on a
    busy CI runner, so geometry assertions must wait for the region to materialize.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"layout did not settle within {timeout}s")


DEFAULT_DENIAL = {"allowed": False, "reason": "nobody answered"}


async def _open_permission_request(app, request) -> tuple[str, "asyncio.Task[dict]"]:
    """Start a real control-channel request and return its id and the agent's wait.

    Driving the actual channel rather than a fabricated future is what makes these
    tests exercise the path every frontend uses: the terminal UI answers a prompt
    by responding to a request id, exactly as a browser client would.
    """
    task = asyncio.create_task(
        app.control_channel.request(
            "permission",
            {"request": serialize_permission_request(request), "rule_options": []},
            default=DEFAULT_DENIAL,
        )
    )
    for _ in range(400):
        if app.control_channel.pending():
            break
        await asyncio.sleep(0.005)
    pending = app.control_channel.pending()
    assert pending, "the control channel never registered the request"
    return pending[0].request_id, task


async def _open_question_request(app, question: str, options: list[str]) -> tuple[str, "asyncio.Task[dict]"]:
    """Start a real question request on the control channel and return its id.

    Used by tests that run without a pilot and so cannot wait for the announcement
    event; the answer path being asserted is still the real one.
    """
    task = asyncio.create_task(
        app.control_channel.request(
            "question",
            {"question": question, "options": list(options), "descriptions": []},
            default={"answer": None},
        )
    )
    for _ in range(400):
        if app.control_channel.pending():
            break
        await asyncio.sleep(0.005)
    pending = app.control_channel.pending()
    assert pending, "the control channel never registered the question"
    return pending[0].request_id, task


@pytest.mark.asyncio
async def test_textual_app_permission_approval_actions_show_rule_labels_without_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        request_id = "req-layout"
        app._pending_approval = PendingApproval(
            request=request,
            request_id=request_id,
            rule_options=allow_rule_options(request),
        )

        app._set_approval_actions_visible(True)

        approval_actions = app.query_one("#approval_actions", ActionList)
        # The options must be focused synchronously (no pilot.pause() above) so arrow
        # keys + Enter work without a click. A deferred Widget.focus() would not have run
        # yet here, and in a real terminal it races the refresh loop and loses focus.
        assert app.focused is approval_actions
        prompts = [
            approval_actions.get_option(f"approval_option_{index}").prompt
            for index in range(approval_actions.option_count)
        ]

        assert prompts == [
            "1. Allow once",
            "2. Deny",
            "3. Always allow this exact command",
            "4. Always allow commands starting with `npm run`",
            "5. Always allow `npm` commands",
        ]
        assert all(" — " not in str(prompt) for prompt in prompts)
        assert all("Allow commands whose" not in str(prompt) for prompt in prompts)

        app._pending_approval = None
        app._set_approval_actions_visible(False)


@pytest.mark.asyncio
async def test_textual_app_long_permission_command_keeps_approval_actions_visible_and_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer, PromptPanel
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test(size=(100, 40)) as pilot:
        for index in range(30):
            app._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="agent",
                    content=f"transcript entry {index}\nmore content\nmore content",
                    complete=True,
                )
            )
        long_command = 'python -c "' + "print('approval layout') ; " * 80 + '"'
        request = permission_request_for_tool("exec_command", {"command": long_command})
        assert request is not None
        request_id, decision_task = await _open_permission_request(app, request)
        app._pending_approval = PendingApproval(
            request=request,
            request_id=request_id,
            rule_options=allow_rule_options(request),
        )

        app._set_approval_actions_visible(True)
        await pilot.pause()

        approval_prompt = app.query_one("#approval_prompt", PromptPanel)
        approval_header = approval_prompt.query_one(".prompt-header-scroll")
        approval_actions = app.query_one("#approval_actions", ActionList)
        conversation = app.query_one("#conversation")
        composer = app.query_one("#composer", ChatComposer)
        await _wait_for_layout(
            pilot,
            lambda: (
                approval_header.vertical_scrollbar.region.width > 0 and conversation.vertical_scrollbar.region.width > 0
            ),
        )
        assert approval_prompt.display is True
        assert approval_actions.display is True
        assert approval_actions.option_count == 5
        assert app.focused is approval_actions
        assert approval_prompt.region.x == composer.region.x
        assert approval_prompt.region.width == composer.region.width
        assert approval_header.region.x == composer.region.x
        assert approval_header.region.width == composer.region.width
        assert approval_actions.region.x == composer.region.x
        assert approval_actions.region.width == composer.region.width
        composer_right_edge = composer.region.x + composer.region.width
        assert approval_header.vertical_scrollbar.display is True
        assert approval_header.vertical_scrollbar.region.x + approval_header.vertical_scrollbar.region.width == (
            composer_right_edge
        )
        assert conversation.vertical_scrollbar.display is True
        assert conversation.vertical_scrollbar.region.x + conversation.vertical_scrollbar.region.width == (
            composer_right_edge
        )
        assert approval_actions.region.y + approval_actions.region.height <= (
            approval_prompt.region.y + approval_prompt.region.height
        )
        assert approval_actions.region.y + approval_actions.region.height <= app.size.height

        await pilot.press("1")
        response = await asyncio.wait_for(decision_task, timeout=1)

        assert response["allowed"] is True
        assert app._pending_approval is None
        assert approval_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_long_question_keeps_actions_visible_and_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer, PromptPanel

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test(size=(100, 40)) as pilot:
        for index in range(30):
            app._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="agent",
                    content=f"transcript entry {index}\nmore content\nmore content",
                    complete=True,
                )
            )
        long_question = "Which migration path should we use? " + "Consider all edge cases and rollout steps. " * 80
        options = ["Keep current path", "Use bounded prompt header", "Defer the decision"]
        question_task = asyncio.create_task(
            app._ask_user_choice(
                long_question,
                options,
                ["Least change", "Fixes the layout", "Needs follow-up"],
            )
        )
        for _ in range(200):
            if app._pending_question is not None:
                break
            await pilot.pause()

        # _begin_question already rendered the prompt from the announcement, so
        # there is nothing for the test to show by hand.
        assert app._pending_question is not None
        assert app._pending_question.descriptions == ["Least change", "Fixes the layout", "Needs follow-up"]
        await pilot.pause()

        question_prompt = app.query_one("#question_prompt", PromptPanel)
        question_header = question_prompt.query_one(".prompt-header-scroll")
        question_actions = app.query_one("#question_actions", ActionList)
        conversation = app.query_one("#conversation")
        composer = app.query_one("#composer", ChatComposer)
        await _wait_for_layout(
            pilot,
            lambda: (
                question_header.vertical_scrollbar.region.width > 0 and conversation.vertical_scrollbar.region.width > 0
            ),
        )
        assert question_prompt.display is True
        assert question_actions.display is True
        assert question_actions.option_count == 3
        assert app.focused is question_actions
        assert question_prompt.region.x == composer.region.x
        assert question_prompt.region.width == composer.region.width
        assert question_header.region.x == composer.region.x
        assert question_header.region.width == composer.region.width
        assert question_actions.region.x == composer.region.x
        assert question_actions.region.width == composer.region.width
        composer_right_edge = composer.region.x + composer.region.width
        assert question_header.vertical_scrollbar.display is True
        assert question_header.vertical_scrollbar.region.x + question_header.vertical_scrollbar.region.width == (
            composer_right_edge
        )
        assert conversation.vertical_scrollbar.display is True
        assert conversation.vertical_scrollbar.region.x + conversation.vertical_scrollbar.region.width == (
            composer_right_edge
        )
        assert question_actions.region.y + question_actions.region.height <= (
            question_prompt.region.y + question_prompt.region.height
        )
        assert question_actions.region.y + question_actions.region.height <= app.size.height

        await pilot.press("2")
        answer = await asyncio.wait_for(question_task, timeout=1)

        assert answer == "Use bounded prompt header"
        assert app._pending_question is None
        assert question_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_approval_answer_reenables_active_turn_composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        request_id, decision_task = await _open_permission_request(app, request)
        app._turn_active = True
        app._pending_approval = PendingApproval(
            request=request,
            request_id=request_id,
            rule_options=allow_rule_options(request),
        )
        app._set_composer_status(messages.APPROVAL_PLACEHOLDER)
        app._set_chat_enabled(False)
        app._refresh_input_area_visibility()

        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled is True
        assert composer.display is True

        await app._answer_approval_option(0)
        response = await asyncio.wait_for(decision_task, timeout=1)

        assert response["allowed"] is True
        assert app._pending_approval is None
        assert composer.display is True
        assert composer.disabled is False
        assert composer.placeholder == messages.QUEUE_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_queued_messages_hide_during_permission_and_restore_after_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.permissions import permission_request_for_tool

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        app._set_chat_enabled(True)
        app._set_composer_status(messages.QUEUE_PLACEHOLDER)
        app._queue_user_message("second")
        app._queue_user_message("third")

        queued_panel = app.query_one("#queued_messages")
        assert queued_panel.display is True
        assert [item.text for item in app._queued_messages] == ["second", "third"]

        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        request_id, permission_task = await _open_permission_request(app, request)
        # Drive the real path: the prompt appears because the channel announced it.
        app._begin_approval(request_id, request)
        await pilot.pause()

        composer = app.query_one("#composer", ChatComposer)
        assert app._pending_approval is not None
        assert queued_panel.display is False
        assert composer.display is True
        assert composer.disabled is True
        assert [item.text for item in app._queued_messages] == ["second", "third"]

        await app._answer_approval_option(0)
        response = await asyncio.wait_for(permission_task, timeout=1)
        await pilot.pause()

        assert response["allowed"] is True
        assert app._pending_approval is None
        assert queued_panel.display is True
        assert composer.display is True
        assert composer.disabled is False
        assert [item.text for item in app._queued_messages] == ["second", "third"]
        queued_text = renderable_text(queued_panel.render())
        assert "second" in queued_text
        assert "third" in queued_text


@pytest.mark.asyncio
async def test_textual_app_queued_messages_do_not_drain_while_permission_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        request_id = "req-layout"
        app._queue_user_message("second")
        app._queue_user_message("third")
        app._pending_approval = PendingApproval(
            request=request,
            request_id=request_id,
            rule_options=allow_rule_options(request),
        )
        app._refresh_input_area_visibility()

        assert app._maybe_start_queued_message() is False

        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert app.agent_worker is None
        assert app.query_one("#queued_messages").display is False


@pytest.mark.asyncio
async def test_textual_app_question_answer_reenables_active_turn_composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._turn_active = True
        request_id, question_task = await _open_question_request(app, "Which path?", ["A", "B"])
        app._pending_question = PendingQuestion(
            question="Which path?",
            options=["A", "B"],
            request_id=request_id,
        )
        app._set_composer_status(messages.QUESTION_PLACEHOLDER)
        app._set_chat_enabled(True)

        await app._answer_pending_question("A")
        response = await asyncio.wait_for(question_task, timeout=1)

        composer = app.query_one("#composer", ChatComposer)
        assert response["answer"] == "A"
        assert app._pending_question is None
        assert composer.display is True
        assert composer.disabled is False
        assert composer.placeholder == messages.QUEUE_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_queued_messages_hide_during_question_and_restore_after_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, PromptPanel

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        app._set_chat_enabled(True)
        app._set_composer_status(messages.QUEUE_PLACEHOLDER)
        app._queue_user_message("second")
        app._queue_user_message("third")

        queued_panel = app.query_one("#queued_messages")
        assert queued_panel.display is True
        assert [item.text for item in app._queued_messages] == ["second", "third"]

        question_task = asyncio.create_task(app._ask_user_choice("Which path?", ["A", "B"]))
        await pilot.pause()

        composer = app.query_one("#composer", ChatComposer)
        question_prompt = app.query_one("#question_prompt", PromptPanel)
        assert app._pending_question is not None
        assert question_prompt.display is True
        assert queued_panel.display is False
        assert composer.display is True
        assert composer.disabled is False
        assert composer.placeholder == messages.QUESTION_PLACEHOLDER
        assert [item.text for item in app._queued_messages] == ["second", "third"]

        composer.load_text("custom answer")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        answer = await asyncio.wait_for(question_task, timeout=1)
        await pilot.pause()

        assert answer == "custom answer"
        assert app._pending_question is None
        assert question_prompt.display is False
        assert queued_panel.display is True
        assert composer.display is True
        assert composer.disabled is False
        assert composer.placeholder == messages.QUEUE_PLACEHOLDER
        assert [item.text for item in app._queued_messages] == ["second", "third"]
        queued_text = renderable_text(queued_panel.render())
        assert "second" in queued_text
        assert "third" in queued_text


@pytest.mark.asyncio
async def test_textual_app_queued_messages_hide_during_question_and_restore_after_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        request_id, question_task = await _open_question_request(app, "Which path?", ["A", "B"])
        app._turn_active = True
        app._queue_user_message("second")
        app._queue_user_message("third")
        app._pending_question = PendingQuestion(question="Which path?", options=["A", "B"], request_id=request_id)
        app._refresh_input_area_visibility()

        queued_panel = app.query_one("#queued_messages")
        composer = app.query_one("#composer", ChatComposer)
        assert queued_panel.display is False
        assert composer.display is True

        app._cancel_pending_question()

        # Cancelling settles the request with a non-answer, so the tool is told
        # nobody chose rather than being left waiting.
        response = await asyncio.wait_for(question_task, timeout=1)
        assert response["answer"] == ""
        assert app._pending_question is None
        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert queued_panel.display is True
        assert composer.display is True


@pytest.mark.asyncio
async def test_textual_app_queued_messages_do_not_drain_while_question_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingQuestion

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._queue_user_message("second")
        app._queue_user_message("third")
        app._pending_question = PendingQuestion(question="Which path?", options=["A", "B"], request_id="req-test")
        app._refresh_input_area_visibility()

        assert app._maybe_start_queued_message() is False

        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert app.agent_worker is None
        assert app.query_one("#queued_messages").display is False


@pytest.mark.asyncio
async def test_textual_app_queued_messages_hide_during_plan_decision_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._queue_user_message("second")
        app._queue_user_message("third")
        queued_panel = app.query_one("#queued_messages")
        assert queued_panel.display is True

        await app._show_plan_for_decision("# Plan\n\nDo the thing.", notification="Plan ready")
        await pilot.pause()

        plan_actions = app.query_one("#plan_actions", ActionList)
        composer = app.query_one("#composer", ChatComposer)
        assert app._plan_decision_active is True
        assert plan_actions.display is True
        assert plan_actions.region.x == composer.region.x
        assert plan_actions.region.width == composer.region.width
        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert queued_panel.display is False
        assert composer.display is True
        assert composer.disabled is True
        assert composer.placeholder == messages.PLAN_READY_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_queued_messages_restore_after_discussing_plan_further(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        plan = "# Plan\n\nDo the thing."
        app._latest_plan = plan
        app._queue_user_message("second")
        app._queue_user_message("third")
        await app._show_plan_for_decision(plan, notification="Plan ready")
        queued_panel = app.query_one("#queued_messages")
        assert queued_panel.display is False

        await app._discuss_pending_plan()
        await pilot.pause()

        composer = app.query_one("#composer", ChatComposer)
        assert app._plan_decision_active is False
        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert queued_panel.display is True
        assert composer.display is True
        assert composer.disabled is False


@pytest.mark.asyncio
async def test_textual_app_queued_messages_do_not_drain_while_plan_decision_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    app = _build_permission_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._queue_user_message("second")
        app._queue_user_message("third")
        app._plan_decision_active = True
        app._refresh_input_area_visibility()

        assert app._maybe_start_queued_message() is False

        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert app.agent_worker is None
        assert app.query_one("#queued_messages").display is False
