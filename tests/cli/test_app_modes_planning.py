# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module

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
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_textual_app_context_usage_updates_status_without_raw_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 123456,
                    "max_tokens": 200000,
                    "usage_percentage": 61.7,
                    "alert_level": "info",
                    "message": "Context is getting large.",
                    "compression_threshold": 80.0,
                },
            )
        )
        dashboard = str(app.query_one("#status_dashboard", Static).render())

        assert "61.7%" in dashboard
        assert "123,456 / 200,000" in dashboard
        assert "Compresses at 80%" in dashboard
        assert "Context is getting large." in dashboard
        assert "input_tokens" not in dashboard
        assert composer.placeholder == COMPOSER_PLACEHOLDER

        app._render_event(AgentEvent(event_type="status_update", sender="coder", content={"input_tokens": 5}))
        assert composer.placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_status_dashboard_tracks_interaction_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakeAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._set_interaction_mode("plan")
        dashboard = str(app.query_one("#status_dashboard", Static).render())
        assert "Plan" in dashboard

        await app._set_interaction_mode("build")
        dashboard = str(app.query_one("#status_dashboard", Static).render())
        assert "Build" in dashboard


@pytest.mark.asyncio
async def test_textual_app_mode_switch_rebuild_skips_transcript_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    history = [{"role": "user", "content": [{"type": "text", "text": "keep me"}]}]
    compaction = {"summary": "summary", "compacted_through": 1, "compacted_history_length": 1}

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None
            self.restored_compaction = None

        def restore_message_history(self, restored):
            self.restored_history = restored

        def dump_compaction_state(self):
            return compaction

        def restore_compaction_state(self, data):
            self.restored_compaction = data

        def dump_message_history(self):
            return history

        async def cleanup(self):
            return None

    class FakePlanningAgent(FakeCoderAgent):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.instances.append(self)

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        restore_calls = []
        render_calls = []

        def spy_restore(restored):
            restore_calls.append(restored)

        def spy_render():
            render_calls.append(True)

        monkeypatch.setattr(app, "_restore_conversation_history", spy_restore)
        monkeypatch.setattr(app, "_render_conversation", spy_render)

        await app._set_interaction_mode("plan")

        assert restore_calls == []
        assert render_calls == []
        assert app.interaction_mode == "plan"
        assert "plan" in str(app.query_one("#session_meta", Static).render())
        assert "Plan" in str(app.query_one("#status_dashboard", Static).render())

        assert FakePlanningAgent.instances
        planning_agent = FakePlanningAgent.instances[-1]
        assert planning_agent.restored_history == history
        assert planning_agent.restored_compaction == compaction


@pytest.mark.asyncio
async def test_textual_app_mode_switch_preserves_transcript_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry

    history = [{"role": "user", "content": [{"type": "text", "text": "persisted"}]}]

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return history

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakeAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        startup = app.conversation_entries[0]
        user = ConversationEntry(kind="user", content="hello")
        assistant = ConversationEntry(kind="assistant", content="hi", complete=True)
        tool = ConversationEntry(
            kind="tool_result",
            content="done",
            complete=True,
            tool_name="read_file",
            tool_call_id="tool-1",
            full_content="done",
        )
        app.conversation_entries = [startup, user, assistant, tool]
        non_startup_entries = app.conversation_entries[1:]

        await app._set_interaction_mode("plan")

        assert app.conversation_entries[0] is startup
        assert app.conversation_entries[1:] == non_startup_entries
        assert app.conversation_entries[1] is user
        assert app.conversation_entries[2] is assistant
        assert app.conversation_entries[3] is tool


@pytest.mark.asyncio
async def test_textual_app_turn_status_formats_error_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 0.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        app._begin_turn_progress()
        now = 83.0
        app._finish_turn_progress("Stopped due to an error: boom", TurnState.ERROR)

        assert "Errored after 1m 23s" in str(app.query_one("#turn_status", Static).render())


@pytest.mark.asyncio
async def test_textual_app_shift_tab_toggles_between_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.constants import BUILD_INTERACTION_MODE, PLAN_INTERACTION_MODE
    from kolega_code.cli.tui.state import PendingQuestion

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.cleaned = False

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            self.cleaned = True

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        toggle_binding = next(binding for binding in app.BINDINGS if binding.action == "toggle_interaction_mode")
        assert toggle_binding.key == "shift+tab"
        assert toggle_binding.key_display == "Shift+Tab"
        assert toggle_binding.priority is True

        assert isinstance(app.agent, FakeCoderAgent)
        assert app.interaction_mode == BUILD_INTERACTION_MODE

        await pilot.press("shift+tab")

        assert app.interaction_mode == PLAN_INTERACTION_MODE
        assert isinstance(app.agent, FakePlanningAgent)
        startup = app.conversation_entries[0].content
        assert "Interaction: plan" in startup

        app._latest_plan = "# Plan\n\nDo it."
        app._plan_decision_active = False
        app._set_plan_actions_visible(True)
        question_future = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(
            question="Choose?",
            options=["A", "B"],
            future=question_future,
        )
        app._set_question_actions_visible(True)

        await pilot.press("shift+tab")

        assert app.interaction_mode == BUILD_INTERACTION_MODE
        assert isinstance(app.agent, FakeCoderAgent)
        assert app._latest_plan == "# Plan\n\nDo it."
        assert app._plan_decision_active is False
        assert app._pending_question is None
        assert question_future.cancelled()
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# Plan\n\nDo it."
        assert app.query_one("#plan_actions").display is False
        assert app.query_one("#question_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nDo it."
        assert loaded.interaction_mode == BUILD_INTERACTION_MODE


@pytest.mark.asyncio
async def test_textual_app_ctrl_p_toggles_permission_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.permissions import PermissionMode

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.permission_mode = kwargs["permission_mode"]

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        def set_permission_mode(self, permission_mode):
            self.permission_mode = permission_mode

        def set_permission_callback(self, permission_callback):
            self.permission_callback = permission_callback

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        toggle_binding = next(binding for binding in app.BINDINGS if binding.action == "toggle_permission_mode")
        assert toggle_binding.key == "ctrl+p"
        assert app.permission_mode == PermissionMode.ASK
        assert app.agent.kwargs["permission_mode"] == PermissionMode.ASK

        await pilot.press("ctrl+p")

        assert app.permission_mode == PermissionMode.AUTO
        assert app.agent.permission_mode == PermissionMode.AUTO
        assert store.load(session.session_id).permission_mode == "auto"
        assert SettingsStore(store.root).load().permission_mode == "auto"
        assert "Permissions: auto" in app.conversation_entries[0].content
        assert "Auto" in str(app.query_one("#status_dashboard", Static).render())

        await app._command_permissions("ask")

        assert app.permission_mode == PermissionMode.ASK
        assert app.agent.permission_mode == PermissionMode.ASK
        assert store.load(session.session_id).permission_mode == "ask"
        assert SettingsStore(store.root).load().permission_mode == "ask"


@pytest.mark.asyncio
async def test_textual_app_ctrl_o_toggles_sidebar_and_keeps_active_tab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        toggle_binding = next(binding for binding in app.BINDINGS if binding.action == "toggle_sidebar")
        assert toggle_binding.key == "ctrl+o"
        assert toggle_binding.key_display == "Ctrl+O"
        assert toggle_binding.priority is True

        side_panel = app.query_one("#side_panel")
        tabs = app.query_one("#events", TabbedContent)
        tabs.active = "terminal_pane"
        await pilot.pause()

        assert app.sidebar_visible is True
        assert side_panel.display is True

        await pilot.press("ctrl+o")

        assert app.sidebar_visible is False
        assert side_panel.display is False
        assert tabs.active == "terminal_pane"

        await pilot.press("ctrl+o")

        assert app.sidebar_visible is True
        assert side_panel.display is True
        assert tabs.active == "terminal_pane"


@pytest.mark.asyncio
async def test_textual_app_permission_approval_actions_show_rule_labels_without_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

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
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request,
            future=future,
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
    from kolega_code.cli.tui.widgets import ActionList, PromptPanel
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test(size=(100, 40)) as pilot:
        long_command = 'python -c "' + "print('approval layout') ; " * 80 + '"'
        request = permission_request_for_tool("exec_command", {"command": long_command})
        assert request is not None
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request,
            future=future,
            rule_options=allow_rule_options(request),
        )

        app._set_approval_actions_visible(True)
        await pilot.pause()

        approval_prompt = app.query_one("#approval_prompt", PromptPanel)
        approval_actions = app.query_one("#approval_actions", ActionList)
        assert approval_prompt.display is True
        assert approval_actions.display is True
        assert approval_actions.option_count == 5
        assert app.focused is approval_actions
        assert approval_actions.region.y + approval_actions.region.height <= (
            approval_prompt.region.y + approval_prompt.region.height
        )
        assert approval_actions.region.y + approval_actions.region.height <= app.size.height

        await pilot.press("1")
        decision = await asyncio.wait_for(future, timeout=1)

        assert decision.allowed is True
        assert app._pending_approval is None
        assert approval_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_long_question_keeps_actions_visible_and_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, PromptPanel

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test(size=(100, 40)) as pilot:
        long_question = "Which migration path should we use? " + "Consider all edge cases and rollout steps. " * 80
        options = ["Keep current path", "Use bounded prompt header", "Defer the decision"]
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(
            question=long_question,
            options=options,
            future=future,
            descriptions=["Least change", "Fixes the layout", "Needs follow-up"],
        )

        app._show_question_options(long_question, options, app._pending_question.descriptions)
        await pilot.pause()

        question_prompt = app.query_one("#question_prompt", PromptPanel)
        question_actions = app.query_one("#question_actions", ActionList)
        assert question_prompt.display is True
        assert question_actions.display is True
        assert question_actions.option_count == 3
        assert app.focused is question_actions
        assert question_actions.region.y + question_actions.region.height <= (
            question_prompt.region.y + question_prompt.region.height
        )
        assert question_actions.region.y + question_actions.region.height <= app.size.height

        await pilot.press("2")
        answer = await asyncio.wait_for(future, timeout=1)

        assert answer == "Use bounded prompt header"
        assert app._pending_question is None
        assert question_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_restores_saved_plan_and_interaction_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    saved_plan = "# Saved plan\n\nUse the restored plan."
    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = saved_history
    session.latest_plan_markdown = saved_plan
    session.plan_pending = True
    session.interaction_mode = "plan"
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)
        assert app._latest_plan == saved_plan
        assert app._plan_pending is True
        assert app._plan_decision_active is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == saved_plan
        plan_actions = app.query_one("#plan_actions", ActionList)
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == ["implement_plan"]
        assert app.query_one("#composer", ChatComposer).disabled is False


@pytest.mark.asyncio
async def test_textual_app_restores_saved_plan_in_build_mode_without_plan_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    saved_plan = "# Saved plan\n\nKeep this visible."
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.latest_plan_markdown = saved_plan
    session.plan_pending = True
    session.interaction_mode = "build"
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app._latest_plan == saved_plan
        assert app.query_one("#planning_plan_markdown", Markdown).source == saved_plan
        # Even with a pending plan, the action stays hidden outside plan mode.
        assert app.query_one("#plan_actions").display is False


@pytest.mark.asyncio
async def test_textual_app_invalid_saved_interaction_mode_falls_back_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.constants import BUILD_INTERACTION_MODE

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.interaction_mode = "invalid"
    store.save(session)
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.interaction_mode == BUILD_INTERACTION_MODE
        assert app.session.interaction_mode == BUILD_INTERACTION_MODE
        assert isinstance(app.agent, FakeCoderAgent)


@pytest.mark.asyncio
async def test_textual_app_passes_shared_task_list_tools_to_build_agent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        task_list_extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list")
        build_tools = task_list_extension.tools
        assert {"get_task_list", "update_task_list"} == set(build_tools)
        # The task list is single-owner; it must not be inherited by sub-agents.
        assert task_list_extension.propagate_to_sub_agents is False
        assert all("ask_user_choice" not in extension.tools for extension in app.agent.kwargs["tool_extensions"])
        build_task_list_prompt = app.agent.kwargs["prompt_extensions"][0].markdown
        assert "After each meaningful task is completed" in build_task_list_prompt
        assert "Do not wait until every TODO is complete" in build_task_list_prompt
        update_task_list_doc = build_tools["update_task_list"].__doc__ or ""
        assert "progress is visible incrementally" in update_task_list_doc
        assert "do not wait" in update_task_list_doc.lower()

        assert await build_tools["get_task_list"]() == "No task list has been set."
        assert await build_tools["update_task_list"]("- [ ] inspect\n- [x] plan") == "Task list updated."
        assert app.session.task_list_markdown == "- [ ] inspect\n- [x] plan"
        assert app.query_one("#planning_task_list_markdown", Markdown).source == "- [ ] inspect\n- [x] plan"
        assert store.load(session.session_id).task_list_markdown == "- [ ] inspect\n- [x] plan"

        await pilot.press("shift+tab")

        assert isinstance(app.agent, FakePlanningAgent)
        plan_extension_names = {getattr(ext, "name", None) for ext in app.agent.kwargs["tool_extensions"]}
        # Plan mode no longer gets the shared task list (build-mode only)...
        assert "cli-shared-task-list" not in plan_extension_names
        # ...but still gets the planning-question tool.
        assert "cli-planning-questions" in plan_extension_names
        question_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools
        assert {"ask_user_choice"} == set(question_tools)
        prompt_markdown = "\n".join(extension.markdown for extension in app.agent.kwargs["prompt_extensions"])
        assert "multiple-choice" in prompt_markdown
        # The task list captured in build mode persists and is untouched by plan mode.
        assert app.session.task_list_markdown == "- [ ] inspect\n- [x] plan"


@pytest.mark.asyncio
async def test_textual_app_passes_skill_extensions_to_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        skill_prompt = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-agent-skills")
        skill_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-agent-skills").tools

        assert "demo-skill" in skill_prompt.markdown
        assert {"list_skills", "activate_skill", "read_skill_resource"} == set(skill_tools)
        assert "demo-skill" in await skill_tools["list_skills"]()

        await pilot.press("shift+tab")

        planning_skill_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-agent-skills")
        assert "activate_skill" in planning_skill_tools.tools


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_accepts_option_list_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import OptionList

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools[
            "ask_user_choice"
        ]

        app._turn_active = True
        answer_task = asyncio.create_task(
            ask_user_choice(
                questions=question_payload(
                    "Which approach should we use?",
                    [("Keep state local", "Store in memory"), ("Persist it", "Write to disk")],
                    header="Approach",
                )
            )
        )
        await pilot.pause()

        assert app._pending_question is not None
        assert app.query_one("#composer", ChatComposer).disabled is False
        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.display is True
        assert app.focused is question_actions
        assert question_actions.highlighted == 0
        assert question_actions.get_option("question_option_0").prompt == "1. Keep state local — Store in memory"
        # While pending, the prompt lives only in the combined panel — no chat bubble.
        assert all(entry.kind != "question" for entry in app.conversation_entries)

        selected = question_actions.get_option("question_option_1")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 1))

        assert json.loads(await answer_task) == {"Approach": "Persist it"}
        assert app._pending_question is None
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER
        # After answering, the question is recorded followed by the chosen answer.
        assert app.conversation_entries[-2].kind == "question"
        assert app.conversation_entries[-2].content == "Which approach should we use?"
        assert app.conversation_entries[-1].kind == "user"
        assert app.conversation_entries[-1].content == "Persist it"
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_planning_question_supports_arrow_and_digit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools[
            "ask_user_choice"
        ]

        options = ["Alpha", "Beta", "Gamma", "Delta"]
        answer_task = asyncio.create_task(
            ask_user_choice(questions=question_payload("Pick one of four?", options, header="Pick"))
        )
        await pilot.pause()

        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.option_count == 4
        assert app.focused is question_actions

        await pilot.press("down", "down", "enter")
        assert json.loads(await answer_task) == {"Pick": "Gamma"}
        assert question_actions.display is False

        answer_task = asyncio.create_task(
            ask_user_choice(questions=question_payload("Pick again?", options, header="Pick"))
        )
        await pilot.pause()

        assert app.focused is app.query_one("#question_actions", ActionList)
        await pilot.press("4")
        assert json.loads(await answer_task) == {"Pick": "Delta"}


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_accepts_custom_text_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools[
            "ask_user_choice"
        ]

        answer_task = asyncio.create_task(
            ask_user_choice(questions=question_payload("Which scope?", ["Small fix", "Full workflow"], header="Scope"))
        )
        await pilot.pause()

        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.get_option("question_option_0").prompt == "1. Small fix — details"

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("Start with the small fix, but keep the API extensible.")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert json.loads(await answer_task) == {"Scope": "Start with the small fix, but keep the API extensible."}
        assert composer.text == ""
        assert app._pending_question is None
        assert question_actions.display is False
        assert question_actions.option_count == 0
        assert app.conversation_entries[-1].content == "Start with the small fix, but keep the API extensible."


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_asks_multiple_questions_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList
    from textual.widgets import OptionList

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools[
            "ask_user_choice"
        ]

        questions = question_payload("First?", ["A1", "B1"], header="First") + question_payload(
            "Second?", ["A2", "B2"], header="Second"
        )
        answer_task = asyncio.create_task(ask_user_choice(questions=questions))
        await pilot.pause()

        # First question is presented; answer it, then the second appears.
        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.option_count == 2
        selected = question_actions.get_option("question_option_0")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 0))
        await pilot.pause()

        assert app._pending_question is not None
        question_actions = app.query_one("#question_actions", ActionList)
        selected = question_actions.get_option("question_option_1")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 1))

        assert json.loads(await answer_task) == {"First": "A1", "Second": "B2"}
        assert app._pending_question is None


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_rejects_malformed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.tools import ToolError

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools[
            "ask_user_choice"
        ]

        # Empty / non-list questions.
        with pytest.raises(ToolError):
            await ask_user_choice(questions=[])
        with pytest.raises(ToolError):
            await ask_user_choice(questions="Which approach?")

        # Fewer than two valid options.
        with pytest.raises(ToolError):
            await ask_user_choice(questions=question_payload("Q?", ["only one"]))

        # Options that are bare strings rather than {label, description} objects.
        with pytest.raises(ToolError):
            await ask_user_choice(
                questions=[{"question": "Q?", "header": "H", "multiSelect": False, "options": ["A", "B"]}]
            )

        # Missing question text.
        with pytest.raises(ToolError):
            await ask_user_choice(
                questions=[
                    {
                        "question": "  ",
                        "header": "H",
                        "multiSelect": False,
                        "options": [
                            {"label": "A", "description": "d"},
                            {"label": "B", "description": "d"},
                        ],
                    }
                ]
            )

        assert app._pending_question is None


@pytest.mark.asyncio
async def test_textual_app_blocks_mode_toggle_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._turn_active = True

        await app.action_toggle_interaction_mode()

        assert app.interaction_mode == "build"
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER
        hint = app.query_one("#composer_hint", Static)
        assert hint.display is True
        assert "Stop the current turn before switching modes." in str(hint.render())


@pytest.mark.asyncio
async def test_textual_app_shows_plan_decision_when_planning_agent_writes_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    class FakePlanningAgent(FakeCoderAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.completed_plan = "# Plan\n\n" + "\n".join(
                f"- Step {index}: keep the planning sidebar readable." for index in range(1, 26)
            )

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "I have a plan.", "complete": True, "uuid": "response-1"}

        def consume_completed_plan(self):
            plan = self.completed_plan
            self.completed_plan = None
            return plan

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        await app._process_message("plan this")

        initial_plan = app.agent.completed_plan or app._latest_plan
        assert app._plan_decision_active is True
        assert app._plan_pending is True
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert (
            app.query_one("#composer", ChatComposer).placeholder
            == "Plan ready. Choose Implement plan or Discuss further."
        )
        plan_actions = app.query_one("#plan_actions", ActionList)
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == [
            "implement_plan",
            "implement_plan_clear",
            "discuss_plan",
        ]
        assert app.focused is plan_actions
        assert app.query_one("#planning_plan_markdown", Markdown).source == initial_plan
        assert "Step 25" in app.query_one("#planning_plan_markdown", Markdown).source
        assert app.conversation_entries[-1].kind == "plan"
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.plan_reofferable is True
        assert loaded.interaction_mode == "plan"

        await app._discuss_pending_plan()

        assert app._plan_decision_active is False
        assert app._plan_pending is False
        assert app._plan_reofferable is True
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == initial_plan
        assert plan_actions.display is False
        assert plan_actions.option_count == 0
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is True

        await app._process_message("keep discussing")

        assert app._plan_decision_active is True
        assert app._plan_pending is True
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == [
            "implement_plan",
            "implement_plan_clear",
            "discuss_plan",
        ]
        assert app.conversation_entries[-1].kind == "plan"
        assert app.conversation_entries[-1].content == initial_plan
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.plan_pending is True
        assert loaded.plan_reofferable is True

        await app._discuss_pending_plan()

        app.agent.completed_plan = "# Revised plan\n\nBuild planning mode carefully."
        await app._capture_completed_plan()

        assert app._plan_decision_active is True
        assert app._plan_pending is True
        assert app._latest_plan == "# Revised plan\n\nBuild planning mode carefully."
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == [
            "implement_plan",
            "implement_plan_clear",
            "discuss_plan",
        ]
        assert (
            app.query_one("#planning_plan_markdown", Markdown).source
            == "# Revised plan\n\nBuild planning mode carefully."
        )
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Revised plan\n\nBuild planning mode carefully."
        assert loaded.plan_reofferable is True


@pytest.mark.asyncio
async def test_textual_app_implement_plan_switches_to_build_and_sends_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "implemented", "complete": True, "uuid": "response-1"}

    class FakePlanningAgent(FakeCoderAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.messages
        assert "# Plan\n\nBuild it." in app.agent.messages[-1]
        assert app._plan_decision_active is False
        # The plan is kept as a read-only sidebar reference, but it is no longer
        # pending a decision so the action must not be re-offered.
        assert app._plan_pending is False
        assert app._plan_reofferable is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# Plan\n\nBuild it."
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nBuild it."
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is False
        assert loaded.interaction_mode == "build"


@pytest.mark.asyncio
async def test_textual_app_implemented_plan_not_reoffered_on_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.constants import PLAN_INTERACTION_MODE
    from kolega_code.cli.tui.widgets import ActionList

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "implemented", "complete": True, "uuid": "response-1"}

    class FakePlanningAgent(FakeCoderAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        # Enter plan mode with a freshly captured plan awaiting a decision.
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        # Implement it: switches to build and runs the plan.
        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()
        assert app.interaction_mode == "build"

        # Re-enter plan mode. The already-implemented plan must NOT be re-offered,
        # but it stays visible in the sidebar as a read-only reference.
        await app._set_interaction_mode(PLAN_INTERACTION_MODE)

        assert app._plan_pending is False
        assert app._plan_reofferable is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        plan_actions = app.query_one("#plan_actions", ActionList)
        assert plan_actions.display is False
        assert plan_actions.option_count == 0
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# Plan\n\nBuild it."

        # A restart (reloading from the persisted session) must also not re-offer it.
        loaded = store.load(session.session_id)
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is False


@pytest.mark.asyncio
async def test_textual_app_clear_context_and_implement_plan_starts_build_agent_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []
            self.history = []
            self.last_compression_index = None

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "implemented", "complete": True, "uuid": "response-1"}

    class FakePlanningAgent(FakeCoderAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        # Seed the planning agent with prior conversation that the normal implement flow
        # would carry forward into the build agent.
        app.agent.history = ["planning message 1", "planning message 2"]
        prior_entry_count = len(app.conversation_entries)
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._implement_pending_plan(clear_context=True)
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        # The build agent starts fresh: the planning conversation was wiped before the
        # mode switch, so it never reached the new agent.
        assert app.agent.history == []
        assert app.session.history == []
        # The plan is still delivered to the build agent via the implement prompt.
        assert app.agent.messages
        assert "# Plan\n\nBuild it." in app.agent.messages[-1]
        # The plan itself is preserved (sidebar keeps showing it).
        assert app._plan_decision_active is False
        assert app._plan_pending is False
        assert app._plan_reofferable is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# Plan\n\nBuild it."
        assert app.query_one("#plan_actions").display is False
        # LLM-context-only clear: the visible transcript is preserved, plus the new
        # "Implement the approved plan." entry.
        assert len(app.conversation_entries) > prior_entry_count
        assert any(
            entry.kind == "user" and entry.content == "Implement the approved plan."
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_discuss_plan_preserves_old_plan_until_new_plan_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "implemented", "complete": True, "uuid": "response-1"}

    class FakePlanningAgent(FakeCoderAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it after discussing."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._discuss_pending_plan()

        assert app._latest_plan == "# Plan\n\nBuild it after discussing."
        assert app._plan_pending is False
        assert app._plan_reofferable is True
        assert app._plan_decision_active is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# Plan\n\nBuild it after discussing."
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nBuild it after discussing."
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is True

        await app._implement_pending_plan()
        assert app.agent_worker is None
        assert app.interaction_mode == "plan"

        app._latest_plan = "# New plan\n\nBuild this instead."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert "# New plan\n\nBuild this instead." in app.agent.messages[-1]
        assert "# Plan\n\nBuild it after discussing." not in app.agent.messages[-1]
        assert app._latest_plan == "# New plan\n\nBuild this instead."
        assert app._plan_reofferable is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# New plan\n\nBuild this instead."
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# New plan\n\nBuild this instead."
        assert loaded.plan_reofferable is False
