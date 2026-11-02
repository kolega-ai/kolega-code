# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

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
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.hooks.events import HookEvent
from kolega_code.permissions import (
    PERMISSIONS_RELATIVE_PATH,
    PermissionMode,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
    permission_request_for_tool,
)
from kolega_code.cli.tui.constants import TUI_CLIENT_ID
from kolega_code.session.runtime import deserialize_permission_rule_options
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
    wait_for_layout as _wait_for_layout,
    wait_for_turn_idle,
)


def _build_permission_test_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, coder_cls: type[FakeCoderAgent] = FakeCoderAgent
):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, coder_cls=coder_cls)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


# Generous cap for awaiting a decision/question task to resolve after we answer it.
# The happy path settles in milliseconds; this wait_for only guards against a true
# hang, so it must be loose enough never to fire under CI load (parallel workers +
# coverage), where a tight 1s occasionally lost the event-loop scheduling race.
_ANSWER_TIMEOUT_S = 5.0


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
            {
                "request": serialize_permission_request(request),
                "rule_options": [
                    {"label": option.label, "description": option.description, "rule": option.rule.to_dict()}
                    for option in allow_rule_options(request)
                ],
            },
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


def _control_event(request: Any) -> AgentEvent:
    return AgentEvent(
        sender="system",
        event_type=KnownEventType.CONTROL_REQUESTED,
        session_id="test",
        content=request.describe(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("save_fails", [False, True])
async def test_permission_delivery_notifies_once_and_runtime_owns_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_fails: bool
) -> None:
    from kolega_code.cli.tui.widgets import PromptPanel

    app = _build_permission_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        rule_store = ProjectPermissionStore(app.session_runtime.project_path)
        saves: list[Path] = []
        original_save = ProjectPermissionStore.save

        def save(store: ProjectPermissionStore, rules: Any) -> None:
            saves.append(store.path)
            if save_fails:
                raise PermissionStoreError("cannot save test rules")
            original_save(store, rules)

        monkeypatch.setattr(ProjectPermissionStore, "save", save)

        rendered: list[tuple[bool, bool, bool]] = []

        async def fire_hook(_event: HookEvent, _payload: dict[str, Any], *, target: str) -> None:
            rendered.append(
                (
                    app.query_one("#approval_prompt", PromptPanel).display,
                    app._pending_approval is not None,
                    rule_store.path.exists(),
                )
            )

        hook = AsyncMock(side_effect=fire_hook)
        monkeypatch.setattr(app.agent, "fire_hook", hook, raising=False)
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        decision_task = asyncio.create_task(app.session_runtime.permission_callback(request))
        await _wait_for_layout(pilot, lambda: hook.await_count == 1)
        assert rendered == [(True, True, False)]
        assert hook.await_args is not None
        assert hook.await_args.args == (
            HookEvent.NOTIFICATION,
            {
                "notification_type": "permission_prompt",
                "message": "Permission requested for exec_command",
                "tool_name": "exec_command",
            },
        )
        assert hook.await_args.kwargs == {"target": "permission_prompt"}
        pending = app.control_channel.pending()[0]
        canonical = pending.payload["rule_options"]
        assert app._pending_approval is not None
        assert [option.rule.to_dict() for option in app._pending_approval.rule_options] == [
            option["rule"] for option in canonical
        ]
        app._handle_control_request(_control_event(pending))
        assert await app.session_runtime.prompt_delivered(pending.request_id) is False
        hook.assert_awaited_once()
        assert saves == []
        await app._answer_approval_option(2)
        decision = await asyncio.wait_for(decision_task, timeout=_ANSWER_TIMEOUT_S)
        assert decision.allowed
        assert saves == [rule_store.path]
        if save_fails:
            assert decision.rule is None
            assert decision.reason == "Allowed once because the rule could not be saved."
            assert not rule_store.path.exists()
        else:
            assert decision.rule is not None
            assert decision.rule.to_dict() == canonical[0]["rule"]
            assert rule_store.load() == [decision.rule]
            assert (await app.session_runtime.permission_callback(request)).allowed
        hook.assert_awaited_once()


@pytest.mark.asyncio
async def test_question_delivery_notifies_once_after_panel_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.widgets import PromptPanel

    app = _build_permission_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        rendered: list[tuple[bool, bool]] = []

        async def fire_hook(_event: HookEvent, _payload: dict[str, Any], *, target: str) -> None:
            rendered.append((app.query_one("#question_prompt", PromptPanel).display, app._pending_question is not None))

        hook = AsyncMock(side_effect=fire_hook)
        monkeypatch.setattr(app.agent, "fire_hook", hook, raising=False)
        task = asyncio.create_task(app._ask_user_choice("Which approach?", ["red", "green"]))
        await _wait_for_layout(pilot, lambda: hook.await_count == 1)
        assert rendered == [(True, True)]
        assert hook.await_args is not None
        assert hook.await_args.args == (
            HookEvent.NOTIFICATION,
            {"notification_type": "question", "message": "Which approach?"},
        )
        assert hook.await_args.kwargs == {"target": "question"}
        pending = app.control_channel.pending()[0]
        app._handle_control_request(_control_event(pending))
        await app._answer_question_option(1)
        assert await asyncio.wait_for(task, timeout=_ANSWER_TIMEOUT_S) == "green"
        assert await app.session_runtime.prompt_delivered(pending.request_id) is False
        hook.assert_awaited_once()
        assert not (app.project_path / PERMISSIONS_RELATIVE_PATH).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["saved", "auto", "no_controller"])
async def test_permission_policy_without_human_prompt_does_not_notify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    app = _build_permission_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        hook = AsyncMock()
        monkeypatch.setattr(app.agent, "fire_hook", hook, raising=False)
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        if policy == "saved":
            ProjectPermissionStore(app.project_path).add_rule(allow_rule_options(request)[0].rule)
        elif policy == "auto":
            app.session_runtime.permission_mode = PermissionMode.AUTO
        else:
            app.control_channel.release(TUI_CLIENT_ID)
        decision = await app.session_runtime.permission_callback(request)
        assert decision.allowed is (policy != "no_controller")
        await pilot.pause()
        assert app._pending_approval is None
        assert app.control_channel.pending() == []
        hook.assert_not_awaited()
        if policy != "saved":
            assert not (app.project_path / PERMISSIONS_RELATIVE_PATH).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
@pytest.mark.parametrize("failure", ["missing_panel", "render_exception"])
async def test_prompt_render_failure_does_not_notify_or_leave_pending_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, failure: str
) -> None:
    from kolega_code.cli.tui.widgets import PromptPanel

    app = _build_permission_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        hook = AsyncMock()
        monkeypatch.setattr(app.agent, "fire_hook", hook, raising=False)
        panel = app.query_one("#approval_prompt" if kind == "permission" else "#question_prompt", PromptPanel)
        if failure == "missing_panel":
            await panel.remove()
        else:

            def fail_render(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("cannot render prompt")

            monkeypatch.setattr(panel, "prompt", fail_render)
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        task = asyncio.create_task(
            app.session_runtime.permission_callback(request)
            if kind == "permission"
            else app.control_channel.request(
                "question", {"question": "Pick", "options": ["a", "b"]}, default={"answer": None}
            )
        )
        await _wait_for_layout(pilot, lambda: bool(app.control_channel.pending()))
        pending = app.control_channel.pending()[0]
        app._handle_control_request(_control_event(pending))
        assert pending.delivered is False
        assert app._pending_approval is None
        assert app._pending_question is None
        hook.assert_not_awaited()
        app.control_channel.release(TUI_CLIENT_ID)
        await asyncio.wait_for(task, timeout=_ANSWER_TIMEOUT_S)
        assert not (app.project_path / PERMISSIONS_RELATIVE_PATH).exists()


@pytest.mark.asyncio
async def test_stale_permission_announcement_and_response_do_not_notify_or_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_permission_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        hook = AsyncMock()
        monkeypatch.setattr(app.agent, "fire_hook", hook, raising=False)
        handle = app._handle_control_request
        events: list[AgentEvent] = []
        monkeypatch.setattr(app, "_handle_control_request", events.append)
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        task = asyncio.create_task(app.session_runtime.permission_callback(request))
        await _wait_for_layout(pilot, lambda: bool(events))
        pending = app.control_channel.pending()[0]
        app.control_channel.release(TUI_CLIENT_ID)
        assert not (await asyncio.wait_for(task, timeout=_ANSWER_TIMEOUT_S)).allowed
        handle(events[0])
        assert app._pending_approval is None
        app._pending_approval = tui_state.PendingApproval(
            request=request,
            request_id=pending.request_id,
            rule_options=deserialize_permission_rule_options(pending.payload["rule_options"]),
        )
        await app._answer_approval_option(2)
        hook.assert_not_awaited()
        assert app._pending_approval is None
        assert not (app.project_path / PERMISSIONS_RELATIVE_PATH).exists()


@pytest.mark.asyncio
async def test_tui_rebuild_preserves_runtime_permission_mode_and_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ModeAwareAgent(FakeCoderAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.permission_mode = kwargs["permission_mode"]
            self.permission_callback = kwargs["permission_callback"]

        def set_permission_mode(self, mode: PermissionMode) -> None:
            self.permission_mode = mode

        def set_permission_callback(self, callback: Any) -> None:
            self.permission_callback = callback

    app = _build_permission_test_app(tmp_path, monkeypatch, coder_cls=ModeAwareAgent)
    async with app.run_test():
        first = app.agent
        await app._set_permission_mode(PermissionMode.AUTO)
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        rebuilt = app.agent
        assert isinstance(rebuilt, ModeAwareAgent)
        assert rebuilt is not first
        assert rebuilt.kwargs["permission_mode"] == PermissionMode.AUTO
        assert rebuilt.permission_mode == app.session_runtime.permission_mode == PermissionMode.AUTO
        assert rebuilt.permission_callback == app.session_runtime.permission_callback
        hook = AsyncMock()
        monkeypatch.setattr(rebuilt, "fire_hook", hook, raising=False)
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        assert (await rebuilt.permission_callback(request)).allowed
        hook.assert_not_awaited()
        assert app.control_channel.pending() == []
        assert not (app.project_path / PERMISSIONS_RELATIVE_PATH).exists()


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
        response = await asyncio.wait_for(decision_task, timeout=_ANSWER_TIMEOUT_S)

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
        answer = await asyncio.wait_for(question_task, timeout=_ANSWER_TIMEOUT_S)

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
        response = await asyncio.wait_for(decision_task, timeout=_ANSWER_TIMEOUT_S)

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
    from kolega_code.permissions import allow_rule_options, permission_request_for_tool

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
        _request_id, permission_task = await _open_permission_request(app, request)
        await _wait_for_layout(pilot, lambda: app._pending_approval is not None)

        composer = app.query_one("#composer", ChatComposer)
        assert app._pending_approval is not None
        assert queued_panel.display is False
        assert composer.display is True
        assert composer.disabled is True
        assert [item.text for item in app._queued_messages] == ["second", "third"]

        await app._answer_approval_option(0)
        response = await asyncio.wait_for(permission_task, timeout=_ANSWER_TIMEOUT_S)
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
        response = await asyncio.wait_for(question_task, timeout=_ANSWER_TIMEOUT_S)

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
        answer = await asyncio.wait_for(question_task, timeout=_ANSWER_TIMEOUT_S)
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
        response = await asyncio.wait_for(question_task, timeout=_ANSWER_TIMEOUT_S)
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


@pytest.mark.asyncio
async def test_cancelling_with_a_permission_prompt_open_restores_the_queued_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping a turn mid-prompt must not swallow what you queued behind it.

    Cancelling has to dismiss the outstanding approval *and* hand the queued
    text back, or the message is neither answered nor visible anywhere: the
    prompt vanishes with the turn and the queue drains into nothing.
    """
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.permissions import allow_rule_options, permission_request_for_tool

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    started = asyncio.Event()

    async def _never_finishes(*_args, **_kwargs):
        started.set()
        await asyncio.sleep(3600)
        yield {}  # pragma: no cover - unreachable, keeps this an async generator

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        monkeypatch.setattr(app, "_agent_turn_stream", _never_finishes)

        composer.load_text("run the thing")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await asyncio.wait_for(started.wait(), timeout=5)

        composer.load_text("and then check the logs")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert [item.text for item in app._queued_messages] == ["and then check the logs"]

        request = permission_request_for_tool("exec_command", {"command": "rm -rf build"})
        assert request is not None
        app._pending_approval = PendingApproval(
            request=request, request_id="req-cancel", rule_options=allow_rule_options(request)
        )

        app.action_cancel_generation()
        await wait_for_turn_idle(app, pilot)

        assert app.agent_worker is None, "cancelling with a prompt open must not hang the turn"
        assert app._pending_approval is None, "the prompt has to go with the turn it belonged to"
        assert app._queued_messages == []
        assert composer.text == "and then check the logs", "the queued follow-up was lost on cancel"
