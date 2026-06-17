from pathlib import Path
import asyncio
import json

import pytest

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


def extension_by_name(extensions, name: str):
    return next(
        extension
        for extension in extensions
        if getattr(extension, "name", None) == name or getattr(extension, "id", None) == name
    )


def question_payload(question, options, *, header="Choice", multi_select=False):
    """Build a structured `questions` list for a single question.

    options: a list of labels, or (label, description) tuples.
    """
    built = []
    for option in options:
        label, description = option if isinstance(option, tuple) else (option, "details")
        built.append({"label": label, "description": description})
    return [{"question": question, "header": header, "multiSelect": multi_select, "options": built}]


def build_test_config(project: Path):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )


@pytest.mark.asyncio
async def test_textual_app_mounts_with_fake_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Collapsible, Header, Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history_restored = False

        def restore_message_history(self, history):
            self.history_restored = bool(history)

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))

    app = KolegaCodeApp(
        project_path=project,
        config=config,
        mode="code",
        store=store,
        session=session,
    )

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.mode == AgentMode.CLI.value
        assert app.interaction_mode == "build"
        assert app.session.mode == AgentMode.CLI.value
        assert app.agent.kwargs["agent_mode"] == AgentMode.CLI
        assert list(app.query(Header)) == []
        assert app.query_one("#conversation") is not None
        assert app.query_one("#composer") is not None
        assert app.query_one("#planning_pane") is not None
        assert app.query_one("#planning_form", VerticalScroll) is not None
        assert app.query_one("#planning_plan", Collapsible).collapsed is False
        assert app.query_one("#planning_task_list", Collapsible).collapsed is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "No plan captured yet."
        assert app.query_one("#planning_task_list_markdown", Markdown).source == "No task list has been set."
        assert app.conversation_entries[0].kind == "startup"
        startup = app.conversation_entries[0].content
        assert "____          _" in startup
        assert f"Project: {project}" in startup
        assert f"Session: {session.session_id[:8]}" in startup
        assert "Mode: cli" in startup
        assert "Interaction: build" in startup
        expected_model = f"{config.long_context_config.provider.value}/{config.long_context_config.model}"
        assert f"Model: {expected_model}" in startup


@pytest.mark.asyncio
async def test_textual_app_status_tab_is_default_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static, TabbedContent

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.query_one("#events", TabbedContent).active == "status_pane"
        dashboard_widget = app.query_one("#status_dashboard", Static)
        dashboard = str(dashboard_widget.render())

        assert "Status" in dashboard
        assert f"{config.long_context_config.provider.value}/{config.long_context_config.model}" in dashboard
        assert "Build" in dashboard
        assert "Idle" in dashboard
        assert "Waiting for first context count" in dashboard
        assert dashboard_widget.styles.border == app.query_one("#logs").styles.border
        assert list(app.query("#status")) == []


@pytest.mark.asyncio
async def test_textual_app_context_usage_updates_status_without_raw_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakeAgent)

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
async def test_textual_app_turn_status_formats_error_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp, TurnState

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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


def test_turn_state_styles_do_not_depend_on_content_text() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import TURN_STATE_STYLES, TurnState

    assert TURN_STATE_STYLES[TurnState.ERROR] == "red"
    assert TURN_STATE_STYLES[TurnState.STOPPED] == "yellow"
    assert TURN_STATE_STYLES[TurnState.STOPPING] == "yellow"
    assert TURN_STATE_STYLES[TurnState.IDLE] == "green"
    assert TURN_STATE_STYLES.get(TurnState.GENERATING) is None  # falls back to accent


@pytest.mark.asyncio
async def test_progress_entry_tone_drives_styling_not_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp, TurnState

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        # Prose mentioning "error" with a warning tone must not render as an error
        warning_entry = ConversationEntry(
            kind="progress", content="Stopped before the error handler ran", complete=True, tone="warning"
        )
        rendered = app._format_conversation_entry(warning_entry)
        assert "[yellow]" in str(rendered)
        assert "[red]" not in str(rendered)

        error_entry = ConversationEntry(kind="progress", content="All good otherwise", complete=True, tone="error")
        rendered = app._format_conversation_entry(error_entry)
        assert "[red]" in str(rendered)

        # Explicit state drives the dashboard, not content keywords
        app._turn_active = True
        app._begin_turn_progress()
        app._finish_turn_progress("Wrapped up without issue", TurnState.STOPPED)
        assert app._status_state.turn_state is TurnState.STOPPED


@pytest.mark.asyncio
async def test_textual_app_keeps_command_c_for_screen_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        cancel_binding = next(binding for binding in app.BINDINGS if binding.action == "cancel_generation")
        assert cancel_binding.key == "ctrl+c"
        assert all("super+c" not in binding.key for binding in app.BINDINGS)


@pytest.mark.asyncio
async def test_textual_app_shift_tab_toggles_between_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import BUILD_INTERACTION_MODE, PLAN_INTERACTION_MODE, KolegaCodeApp, PendingQuestion

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.cleaned = False

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            self.cleaned = True

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

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
async def test_textual_app_ctrl_p_toggles_permission_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.permissions import PermissionMode

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.permission_mode = kwargs["permission_mode"]

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        def set_permission_mode(self, permission_mode):
            self.permission_mode = permission_mode

        def set_permission_callback(self, permission_callback):
            self.permission_callback = permission_callback

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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
        assert "Permissions: auto" in app.conversation_entries[0].content
        assert "Auto" in str(app.query_one("#status_dashboard", Static).render())


@pytest.mark.asyncio
async def test_textual_app_restores_saved_plan_and_interaction_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    saved_plan = "# Saved plan\n\nUse the restored plan."
    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = saved_history
    session.latest_plan_markdown = saved_plan
    session.interaction_mode = "plan"
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)
        assert app._latest_plan == saved_plan
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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    saved_plan = "# Saved plan\n\nKeep this visible."
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.latest_plan_markdown = saved_plan
    session.interaction_mode = "build"
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app._latest_plan == saved_plan
        assert app.query_one("#planning_plan_markdown", Markdown).source == saved_plan
        assert app.query_one("#plan_actions").display is False


@pytest.mark.asyncio
async def test_textual_app_invalid_saved_interaction_mode_falls_back_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import BUILD_INTERACTION_MODE, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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
async def test_textual_app_passes_shared_task_list_tools_to_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert isinstance(app.agent, FakeCoderAgent)
        build_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        assert {"get_task_list", "update_task_list"} == set(build_tools)
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
        plan_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        question_tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-planning-questions").tools
        prompt_markdown = "\n".join(extension.markdown for extension in app.agent.kwargs["prompt_extensions"])
        assert await plan_tools["get_task_list"]() == "- [ ] inspect\n- [x] plan"
        assert await plan_tools["update_task_list"]("- [x] inspect\n- [x] plan") == "Task list updated."
        assert {"ask_user_choice"} == set(question_tools)
        assert "multiple-choice" in prompt_markdown
        assert app.session.task_list_markdown == "- [x] inspect\n- [x] plan"
        assert app.query_one("#planning_task_list_markdown", Markdown).source == "- [x] inspect\n- [x] plan"


@pytest.mark.asyncio
async def test_textual_app_passes_skill_extensions_to_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

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
async def test_textual_app_skill_slash_commands_list_and_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/skills")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.conversation_entries[-1].kind == "system"
        assert "`/demo-skill`" in app.conversation_entries[-1].content

        composer.load_text("/demo-skill")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.conversation_entries[-1].kind == "skill"
        assert '<skill_content name="demo-skill">' in app.agent.history[-1].get_text_content()
        assert '<skill_content name="demo-skill">' in store.load(session.session_id).history[-1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_textual_app_skill_slash_command_with_prompt_starts_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.messages = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/demo-skill Build the feature")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["Build the feature"]
        assert any(entry.kind == "skill" for entry in app.conversation_entries)
        assert any(entry.kind == "user" and entry.content == "Build the feature" for entry in app.conversation_entries)


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_accepts_option_list_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import OptionList

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ActionList, ChatComposer, KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

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
        assert app.conversation_entries[-1].kind == "question"

        selected = question_actions.get_option("question_option_1")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 1))

        assert json.loads(await answer_task) == {"Approach": "Persist it"}
        assert app._pending_question is None
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER
        assert app.conversation_entries[-1].kind == "user"
        assert app.conversation_entries[-1].content == "Persist it"
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_planning_question_supports_arrow_and_digit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        answer_task = asyncio.create_task(
            ask_user_choice(
                questions=question_payload("Which scope?", ["Small fix", "Full workflow"], header="Scope")
            )
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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, KolegaCodeApp
    from textual.widgets import OptionList

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.tools import ToolError

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli import app as app_module
    from textual.widgets import Markdown

    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    class FakePlanningAgent(FakeCoderAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.completed_plan = "# Plan\n\n" + "\n".join(
                f"- Step {index}: keep the planning sidebar readable."
                for index in range(1, 26)
            )

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "I have a plan.", "complete": True, "uuid": "response-1"}

        def consume_completed_plan(self):
            plan = self.completed_plan
            self.completed_plan = None
            return plan

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

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
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert app.query_one("#composer", ChatComposer).placeholder == "Plan ready. Choose Implement plan or Discuss further."
        plan_actions = app.query_one("#plan_actions", ActionList)
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == ["implement_plan", "discuss_plan"]
        assert app.focused is plan_actions
        assert app.query_one("#planning_plan_markdown", Markdown).source == initial_plan
        assert "Step 25" in app.query_one("#planning_plan_markdown", Markdown).source
        assert app.conversation_entries[-1].kind == "plan"
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.interaction_mode == "plan"

        app._discuss_pending_plan()

        assert app._plan_decision_active is False
        assert app._latest_plan is None
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "No plan captured yet."
        assert plan_actions.display is False
        assert plan_actions.option_count == 0
        assert store.load(session.session_id).latest_plan_markdown == ""

        app.agent.completed_plan = "# Revised plan\n\nBuild planning mode carefully."
        app._capture_completed_plan()

        assert app._plan_decision_active is True
        assert app._latest_plan == "# Revised plan\n\nBuild planning mode carefully."
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == ["implement_plan", "discuss_plan"]
        assert (
            app.query_one("#planning_plan_markdown", Markdown).source
            == "# Revised plan\n\nBuild planning mode carefully."
        )
        assert store.load(session.session_id).latest_plan_markdown == "# Revised plan\n\nBuild planning mode carefully."


@pytest.mark.asyncio
async def test_textual_app_implement_plan_switches_to_build_and_sends_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "implemented", "complete": True, "uuid": "response-1"}

    class FakePlanningAgent(FakeCoderAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_decision_active = True

        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.messages
        assert "# Plan\n\nBuild it." in app.agent.messages[-1]
        assert app._plan_decision_active is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# Plan\n\nBuild it."
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nBuild it."
        assert loaded.interaction_mode == "build"


@pytest.mark.asyncio
async def test_textual_app_discuss_plan_clears_old_plan_until_new_plan_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "implemented", "complete": True, "uuid": "response-1"}

    class FakePlanningAgent(FakeCoderAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it after discussing."
        app._plan_decision_active = True

        app._discuss_pending_plan()

        assert app._latest_plan is None
        assert app._plan_decision_active is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "No plan captured yet."
        assert app.query_one("#plan_actions").display is False
        assert store.load(session.session_id).latest_plan_markdown == ""

        await app._implement_pending_plan()
        assert app.agent_worker is None
        assert app.interaction_mode == "plan"

        app._latest_plan = "# New plan\n\nBuild this instead."
        app._plan_decision_active = True

        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert "# New plan\n\nBuild this instead." in app.agent.messages[-1]
        assert "# Plan\n\nBuild it after discussing." not in app.agent.messages[-1]
        assert app._latest_plan == "# New plan\n\nBuild this instead."
        assert app.query_one("#planning_plan_markdown", Markdown).source == "# New plan\n\nBuild this instead."
        assert app.query_one("#plan_actions").display is False
        assert store.load(session.session_id).latest_plan_markdown == "# New plan\n\nBuild this instead."


@pytest.mark.asyncio
async def test_textual_app_does_not_save_startup_entry_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return saved_history

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.conversation_entries[0].kind == "startup"
        app._save_session_history()

        assert session.history == saved_history
        assert all("Kolega Code" not in str(item) for item in session.history)


@pytest.mark.asyncio
async def test_textual_app_composer_shift_enter_inserts_line_break_and_enter_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("shift+enter")
        await pilot.press("t", "h", "e", "r", "e")
        assert composer.text == "hi\nthere"

        await pilot.press("enter")
        await pilot.pause()

        assert app.agent is not None
        assert app.agent.messages == ["hi\nthere"]
        assert composer.text == ""
        user_entries = [entry for entry in app.conversation_entries if entry.kind == "user"]
        assert user_entries[-1].content == "hi\nthere"


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_enter_still_inserts_line_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("ctrl+enter")
        await pilot.press("t", "h", "e", "r", "e")

        assert composer.text == "hi\nthere"


@pytest.mark.asyncio
async def test_textual_app_composer_preserves_multiline_paste(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    pasted = "line one\n    line two\nline three"
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await composer._on_paste(events.Paste(pasted))
        assert composer.text == pasted

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()

        assert app.agent is not None
        assert app.agent.messages == [pasted]
        assert composer.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/clear", "/reset"])
async def test_textual_app_reset_command_clears_current_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp, PendingQuestion, THREAD_RESET_MESSAGE

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            raise AssertionError("reset commands should not be sent to the agent")

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    saved_history = [
        Message(role="user", content=[TextBlock("old request")]).to_dict(),
        Message(role="assistant", content=[TextBlock("old response")]).to_dict(),
    ]
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = saved_history
    session.task_list_markdown = "- [ ] old task"
    session.latest_plan_markdown = "# Plan\n\nOld plan."
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.agent is not None
        assert len(app.agent.history) == 2
        assert any(entry.content == "old request" for entry in app.conversation_entries)
        app._latest_plan = "# Plan\n\nOld plan."
        app._plan_decision_active = False
        app._set_plan_actions_visible(True)
        question_future = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(
            question="Old question?",
            options=["A", "B"],
            future=question_future,
        )
        app._set_question_actions_visible(True)

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text(command)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.agent_worker is None
        assert len(app.agent.history) == 0
        assert app.session.history == []
        assert app.session.task_list_markdown == ""
        assert app._latest_plan is None
        assert app._plan_decision_active is False
        assert app._pending_question is None
        assert question_future.cancelled()
        assert app.query_one("#plan_actions").display is False
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "No plan captured yet."
        assert app.query_one("#planning_task_list_markdown", Markdown).source == "No task list has been set."
        assert store.load(session.session_id).history == []
        assert store.load(session.session_id).task_list_markdown == ""
        assert store.load(session.session_id).latest_plan_markdown == ""
        assert composer.text == ""
        assert [entry.kind for entry in app.conversation_entries] == ["startup", "progress"]
        assert app.conversation_entries[-1].content == THREAD_RESET_MESSAGE
        assert all(entry.content != command for entry in app.conversation_entries)


@pytest.mark.asyncio
async def test_textual_app_reset_command_waits_for_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = ["old history"]

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    saved_history = [Message(role="user", content=[TextBlock("old request")]).to_dict()]
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = saved_history
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/clear")
        app._turn_active = True

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.session.history == saved_history
        assert store.load(session.session_id).history == saved_history
        assert composer.text == "/clear"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        hint = app.query_one("#composer_hint", Static)
        assert hint.display is True
        assert "Stop the current turn before resetting the thread." in str(hint.render())


@pytest.mark.asyncio
async def test_textual_app_mounts_settings_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            raise AssertionError("agent should not be built without a valid API key")

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    settings_store = SettingsStore(tmp_path / "state")
    session = store.create(project, "code", {})

    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert app.agent is None
        assert app.query_one("#composer", ChatComposer).disabled is True
        startup = app.conversation_entries[0].content
        assert "Model: not configured" in startup
        assert "API key: not checked until a model is configured" in startup
        stored_settings = settings_store.load()
        assert stored_settings.active_provider is None
        assert stored_settings.active_model is None
        status = str(app.query_one("#settings_status").render())
        assert "Configuration incomplete" in status
        assert "No provider/model configured" in status


@pytest.mark.asyncio
async def test_textual_app_does_not_select_model_from_api_key_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            raise AssertionError("agent should not be built from an API key alone")

    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    session = store.create(project, "code", {})

    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert app.agent is None
        assert app.query_one("#composer", ChatComposer).disabled is True
        startup = app.conversation_entries[0].content
        assert "Model: not configured" in startup
        assert f"Model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" not in startup
        stored_settings = settings_store.load()
        assert stored_settings.active_provider is None
        assert stored_settings.active_model is None


@pytest.mark.asyncio
async def test_textual_app_mounts_with_stored_kimi_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})

    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.MOONSHOT
        assert app.agent.kwargs["config"].long_context_config.model == UI_DEFAULT_MODEL
        assert app.agent.kwargs["config"].long_context_config.thinking_effort == "auto"
        assert app.query_one("#composer", ChatComposer).disabled is False


@pytest.mark.asyncio
async def test_textual_app_mounts_with_stored_deepseek_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=ModelProvider.DEEPSEEK.value, active_model=DEEPSEEK_DEFAULT_MODEL)
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})

    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.DEEPSEEK
        assert app.agent.kwargs["config"].long_context_config.model == DEEPSEEK_DEFAULT_MODEL
        assert app.agent.kwargs["config"].long_context_config.thinking_effort == "high"
        assert app.query_one("#composer", ChatComposer).disabled is False


@pytest.mark.asyncio
async def test_textual_app_saves_settings_and_builds_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert app.agent is None
        app.query_one("#api_key_input", Input).value = "moonshot-key"
        await app._save_settings_from_ui()

        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.MOONSHOT
        assert settings_store.load().get_api_key(UI_DEFAULT_PROVIDER) == "moonshot-key"
        assert settings_store.load().active_thinking_effort == "auto"
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert [entry.kind for entry in app.conversation_entries].count("startup") == 1
        startup = app.conversation_entries[0].content
        assert f"Model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in startup
        assert "Thinking effort: auto" in startup
        assert "API key: present in local settings" in startup


@pytest.mark.asyncio
async def test_textual_app_saves_deepseek_settings_and_builds_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input, Select

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert app.agent is None
        app.query_one("#provider_select", Select).value = ModelProvider.DEEPSEEK.value
        model_select = app.query_one("#model_select", Select)
        model_select.set_options([("DeepSeek V4 Pro", DEEPSEEK_DEFAULT_MODEL)])
        model_select.value = DEEPSEEK_DEFAULT_MODEL
        app.query_one("#api_key_input", Input).value = "deepseek-key"
        await app._save_settings_from_ui()

        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.DEEPSEEK
        assert app.agent.kwargs["config"].long_context_config.model == DEEPSEEK_DEFAULT_MODEL
        assert app.agent.kwargs["config"].long_context_config.thinking_effort == "high"
        assert settings_store.load().get_api_key(ModelProvider.DEEPSEEK.value) == "deepseek-key"
        assert settings_store.load().active_thinking_effort == "high"
        assert app.query_one("#composer", ChatComposer).disabled is False


@pytest.mark.asyncio
async def test_textual_app_merges_streamed_response_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    chunks = [
        {"type": "response", "content": "hello ", "complete": False, "uuid": "response-1"},
        {"type": "response", "content": "world", "complete": False, "uuid": "response-1"},
        {"type": "response", "content": "", "complete": True, "uuid": "response-1"},
    ]

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        assistant_entries = [entry for entry in app.conversation_entries if entry.kind == "assistant"]
        assert len(assistant_entries) == 1
        assert assistant_entries[0].content == "hello world"
        assert assistant_entries[0].complete is True
        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_merges_streamed_thinking_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    chunks = [
        {"type": "thinking", "content": "checking ", "complete": False, "uuid": "thinking-1"},
        {"type": "thinking", "content": "context", "complete": False, "uuid": "thinking-1"},
        {"type": "thinking", "content": "", "complete": True, "uuid": "thinking-1"},
        {"type": "response", "content": "done", "complete": True, "uuid": "response-1"},
    ]

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        thinking_entries = [entry for entry in app.conversation_entries if entry.kind == "thinking"]
        assert len(thinking_entries) == 1
        assert thinking_entries[0].content == "checking context"
        assert thinking_entries[0].complete is True


@pytest.mark.asyncio
async def test_textual_app_formats_thinking_as_italic_chat_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        formatted = app._format_conversation_entry(
            ConversationEntry(kind="thinking", content="inspect [red]markup[/red]", complete=False)
        )

        assert "[dim italic]Thinking[/dim italic]" in formatted
        assert "\\[red]" in formatted
        assert "[italic dim]" in formatted
        assert "…" in formatted  # streaming indicator in the header


@pytest.mark.asyncio
async def test_textual_app_renders_one_widget_per_chat_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, ConversationEntryWidget, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first"),
            ConversationEntry(kind="assistant", content="second", complete=False),
            ConversationEntry(kind="user", content="third"),
        ]
        app._render_conversation()
        await pilot.pause()

        widgets = list(app.query(ConversationEntryWidget))
        assert len(widgets) == 3
        assert [widget.entry.content for widget in widgets] == ["first", "second", "third"]
        assert widgets[0].has_class("entry-user")
        assert widgets[1].has_class("entry-assistant")

        # Streaming into an entry updates its widget in place without remounting
        app.conversation_entries[1].content = "second updated"
        app._invalidate_conversation(app.conversation_entries[1])
        app._flush_conversation_render()
        await pilot.pause()

        same_widgets = list(app.query(ConversationEntryWidget))
        assert len(same_widgets) == 3
        assert same_widgets[1] is widgets[1]
        assert "second updated" in str(same_widgets[1]._formatted)


@pytest.mark.asyncio
async def test_conversation_entry_widget_extracts_plain_selected_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, ConversationEntryWidget, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="copy this")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        selected = widget.get_selection(Selection(None, None))

        assert selected is not None
        text, ending = selected
        assert ending == "\n"
        assert "Agent" in text
        assert "copy this" in text
        assert "\x1b" not in text
        assert "[bold]" not in text


@pytest.mark.asyncio
async def test_conversation_entry_supports_mouse_drag_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, ConversationEntryWidget, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="select this text")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()

        await pilot.mouse_down(widget, offset=(0, 1))
        await pilot._post_mouse_events([events.MouseMove], widget, offset=(19, 1), button=1)
        await pilot.mouse_up(widget, offset=(19, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert selected_text.strip() == "select this text"


@pytest.mark.asyncio
async def test_command_c_copies_selected_chat_text_to_macos_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, ConversationEntryWidget, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    pbcopy_calls: list[dict] = []

    def fake_run(args, *, input, text, check):
        pbcopy_calls.append({"args": args, "input": input, "text": text, "check": check})

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module.subprocess, "run", fake_run)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="copy this")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        app.screen.selections = {widget: Selection(None, None)}

        await pilot.press("super+c")

        assert "copy this" in app.clipboard
        assert "\x1b" not in app.clipboard
        assert len(pbcopy_calls) == 1
        assert pbcopy_calls[0]["args"] == ["pbcopy"]
        assert pbcopy_calls[0]["input"] == app.clipboard


@pytest.mark.asyncio
async def test_textual_app_formats_agent_and_tool_chat_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assistant = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="hello", complete=False)
        )
        tool_call = app._format_conversation_entry(
            ConversationEntry(
                kind="tool_call",
                content="inspect [red]markup[/red]\nthen continue",
                tool_name="read_file",
                complete=False,
            )
        )
        tool_result = app._format_conversation_entry(
            ConversationEntry(kind="tool_result", content="completed\nok", tool_name="read_file")
        )
        tool_error = app._format_conversation_entry(
            ConversationEntry(kind="tool_error", content="Permission denied", tool_name="write_file")
        )

        assert "[magenta]●[/magenta] [bold]Agent[/bold]" in assistant
        assert "Kolega" not in assistant
        assert "[cyan]⏺[/cyan] [bold]read_file[/bold]" in tool_call
        assert "· running" in tool_call
        assert "[dim]  │[/dim] inspect \\[red]markup\\[/red]" in tool_call
        assert "[dim]  │[/dim] then continue" in tool_call
        assert "[green]⏺[/green] [bold]read_file[/bold]" in tool_result
        assert "· done" in tool_result
        assert "[red]⏺[/red] [bold]write_file[/bold]" in tool_error
        assert "· failed" in tool_error


@pytest.mark.asyncio
async def test_textual_app_ignores_empty_final_response_without_existing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "", "complete": True, "uuid": "response-empty"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        assert [entry for entry in app.conversation_entries if entry.kind == "assistant"] == []
        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_shows_working_progress_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    started = asyncio.Event()
    release = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            started.set()
            await release.wait()
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 100.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        assert turn_status.display is False

        task = asyncio.create_task(app._process_message("hi"))
        await started.wait()

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is True
        assert "Working…" in str(turn_status.render())
        assert "0s" in str(turn_status.render())

        now = 103.0
        app._render_event(
            AgentEvent(event_type="status_update", sender="coder", content={"text": "Indexing workspace"})
        )
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        app._refresh_turn_status_strip()
        assert "Indexing workspace" in str(turn_status.render())
        assert "3s" in str(turn_status.render())

        now = 423.0
        release.set()
        await task

        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is False
        assert "Done in 5m 23s" in str(turn_status.render())


@pytest.mark.asyncio
async def test_textual_app_renders_tool_events_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp, TOOL_RESULT_PREVIEW_CHARS

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "short result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_error",
                    "text": "x" * (TOOL_RESULT_PREVIEW_CHARS + 10),
                    "tool_description": "read_file",
                    "tool_call_id": "tool-2",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert [entry.kind for entry in tool_entries] == ["tool_result", "tool_error"]
        assert tool_entries[0].content == "short result"
        assert tool_entries[0].tool_call_id == "tool-1"
        assert tool_entries[1].content.endswith("…")
        assert tool_entries[1].tool_call_id == "tool-2"
        assert len(tool_entries[1].content) == TOOL_RESULT_PREVIEW_CHARS + 1


@pytest.mark.asyncio
async def test_textual_app_appends_append_mode_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling think_hard",
                    "tool_description": "think_hard",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "partial analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "\ncontinued analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "partial analysis\ncontinued analysis"
        assert tool_entries[0].complete is False

        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "final analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": True,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "final analysis"
        assert tool_entries[0].complete is True
        assert app._tool_stream_buffers == {}


@pytest.mark.asyncio
async def test_textual_app_replaces_default_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "Fetching content...",
                    "tool_name": "web_fetch",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "Processing content...",
                    "tool_name": "web_fetch",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Processing content..."


@pytest.mark.asyncio
async def test_textual_app_caps_long_append_mode_tool_streaming_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp, TOOL_STREAM_PREVIEW_CHARS

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "a" * (TOOL_STREAM_PREVIEW_CHARS + 10),
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].content.startswith(f"[stream truncated to the last {TOOL_STREAM_PREVIEW_CHARS} characters]")
        assert tool_entries[0].content.endswith("a" * TOOL_STREAM_PREVIEW_CHARS)


@pytest.mark.asyncio
async def test_textual_app_renders_queued_tool_events_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    started = asyncio.Event()
    release = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.connection_manager = kwargs["connection_manager"]
            self.workspace_id = kwargs["workspace_id"]
            self.thread_id = kwargs["thread_id"]

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_call",
                        "text": "Calling read_file",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            started.set()
            await release.wait()
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_result",
                        "text": "README contents",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    async def wait_for_tool_entries(app: KolegaCodeApp, count: int) -> list:
        while True:
            entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
            if len(entries) >= count:
                return entries
            await asyncio.sleep(0.01)

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 10.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "hi"))
        worker = app.agent_worker
        assert worker is not None
        assert worker.group == "turns"

        await started.wait()
        event_worker = next(worker for worker in app.workers if worker.name == "kolega-events")
        assert event_worker.group == "events"
        assert not event_worker.is_cancelled

        tool_entries = await asyncio.wait_for(wait_for_tool_entries(app, 1), timeout=1)
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Calling read_file"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Running read_file…" in str(turn_status.render())

        now = 25.0
        release.set()
        await worker.wait()

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "README contents"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Done in 15s" in str(turn_status.render())


@pytest.mark.asyncio
async def test_textual_app_late_tool_result_updates_existing_tool_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._active_progress_entry = None
        app._turn_active = False

        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "late result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "late result"


@pytest.mark.asyncio
async def test_textual_app_cancellation_is_visible_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

    started = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            started.set()
            while True:
                await asyncio.sleep(1)
                yield {"type": "thinking", "content": "still working", "complete": False, "uuid": "thinking-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 10.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        task = asyncio.create_task(app._process_message("hi"))
        app.agent_worker = task
        await started.wait()

        now = 52.0
        app.action_cancel_generation()
        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Stopping…" in str(turn_status.render())
        assert "42s" in str(turn_status.render())

        await task

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped by user."
        assert progress_entries[0].complete is True
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Stopped after 42s" in str(turn_status.render())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,provider,model,expected_message",
    [
        pytest.param(
            LLMBillingError(
                "DeepSeek APIError: Insufficient Balance",
                provider=ModelProvider.DEEPSEEK.value,
            ),
            ModelProvider.DEEPSEEK,
            DEEPSEEK_DEFAULT_MODEL,
            "DeepSeek/deepseek-v4-pro could not run this request",
            id="billing",
        ),
        pytest.param(
            LLMContextWindowExceededError("context too large", provider=ModelProvider.ANTHROPIC.value),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "The conversation context became too large for the model",
            id="context-window",
        ),
        pytest.param(
            LLMInternalServerError(
                "provider overloaded",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "There is high traffic on our LLM provider",
            id="internal-server",
        ),
        pytest.param(
            LLMAuthenticationError(
                "invalid key",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "Anthropic/claude-haiku-4-5-20251001 could not authenticate",
            id="authentication",
        ),
        pytest.param(
            LLMError(
                "unexpected provider error",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "Anthropic/claude-haiku-4-5-20251001 returned an error",
            id="generic-llm",
        ),
    ],
)
async def test_textual_app_handles_llm_error_without_worker_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error,
    provider,
    model,
    expected_message,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp, TurnState

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            raise error
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    config.long_context_config.provider = provider
    config.long_context_config.model = model
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    monkeypatch.setattr(app, "_now", lambda: 10.0)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)

        await app._process_message("hi")

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert expected_message in progress_entries[0].content
        assert progress_entries[0].tone == "error"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is False
        assert app.agent_worker is None
        assert app._status_state.turn_state is TurnState.ERROR
        assert "Errored after" in str(turn_status.render())


@pytest.mark.asyncio
async def test_textual_app_reraises_non_llm_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp, TurnState

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            raise RuntimeError("tool host exploded")
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)

        with pytest.raises(RuntimeError, match="tool host exploded"):
            await app._process_message("hi")

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped due to an error: tool host exploded"
        assert progress_entries[0].tone == "error"
        assert composer.disabled is False
        assert app._status_state.turn_state is TurnState.ERROR


@pytest.mark.asyncio
async def test_textual_app_renders_resumed_history_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None

        def restore_message_history(self, history):
            self.restored_history = history

        def dump_message_history(self):
            return self.restored_history or []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = [
        Message(role="user", content=[TextBlock("Please read the README")]).to_dict(),
        Message(
            role="assistant",
            content=[
                TextBlock("I'll inspect it."),
                ToolCall(id="tool-1", name="read_file", input={"relative_path": "README.md"}),
            ],
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="tool-1", content="README contents", name="read_file", is_error=False)],
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="tool-2", content="Permission denied", name="write_file", is_error=True)],
        ).to_dict(),
        Message(role="assistant", content=[TextBlock("Done.")]).to_dict(),
    ]

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.agent.restored_history == session.history
        assert app.conversation_entries[0].kind == "startup"
        startup = app.conversation_entries[0].content
        expected_model = f"{config.long_context_config.provider.value}/{config.long_context_config.model}"
        assert f"Project: {project}" in startup
        assert f"Model: {expected_model}" in startup
        assert [(entry.kind, entry.content, entry.tool_name) for entry in app.conversation_entries[1:]] == [
            ("user", "Please read the README", None),
            ("assistant", "I'll inspect it.", None),
            ("tool_call", "Calling read_file", "read_file"),
            ("tool_result", "README contents", "read_file"),
            ("tool_error", "Permission denied", "write_file"),
            ("assistant", "Done.", None),
        ]


# ---------------------------------------------------------------------------
# Parallel sub-agent rendering
# ---------------------------------------------------------------------------


def _build_sub_agent_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


def _sub_agent_event(
    agent_id="agent-1",
    agent_name="general-agent",
    task="inspect sessions",
    parent_tool_call_id="tc-1",
    uuid=None,
    **content,
):
    kwargs = {"uuid": uuid} if uuid is not None else {}
    return AgentEvent(
        event_type="chat_message",
        sender=agent_name,
        content=content,
        sub_agent_info={
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": task,
            "parent_tool_call_id": parent_tool_call_id,
            "conversation_id": None,
            "depth": 1,
        },
        **kwargs,
    )


def _sub_agent_entries(app):
    return [entry for entry in app.conversation_entries if entry.kind == "sub_agent"]


@pytest.mark.asyncio
async def test_sub_agent_stream_chunks_group_into_single_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(uuid="u1", text="The session store wri"))
        app._render_event(_sub_agent_event(uuid="u1", text="tes JSON records"))

        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert not any(entry.kind == "message" for entry in app.conversation_entries)
        assert "general-agent" in entries[0].content
        assert "#1" in entries[0].content
        assert "The session store writes JSON records" in entries[0].content
        assert "Task: inspect sessions" in entries[0].content


@pytest.mark.asyncio
async def test_parallel_sub_agents_create_separate_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(agent_id="a1", task="task one", uuid="u1", text="alpha"))
        app._render_event(_sub_agent_event(agent_id="a2", task="task two", parent_tool_call_id="tc-2", uuid="u2", text="beta"))
        app._render_event(_sub_agent_event(agent_id="a1", task="task one", uuid="u1", text=" more"))

        entries = _sub_agent_entries(app)
        assert len(entries) == 2
        assert "#1" in entries[0].content and "alpha more" in entries[0].content
        assert "#2" in entries[1].content and "beta" in entries[1].content
        assert "alpha" not in entries[1].content


@pytest.mark.asyncio
async def test_sub_agent_tool_events_update_counters_not_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _sub_agent_event(message_type="tool_call", text="Calling search_codebase", tool_description="search_codebase")
        )
        app._render_event(
            _sub_agent_event(message_type="tool_result", text="found things", tool_description="search_codebase")
        )

        assert not any(entry.kind.startswith("tool") for entry in app.conversation_entries)
        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert "1 tool" in entries[0].content
        assert "last: search_codebase done" in entries[0].content
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.tool_calls == 1


@pytest.mark.asyncio
async def test_sub_agent_status_events_complete_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting general-agent task"))
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.status == "running"
        assert activity.entry.complete is False

        app._render_event(_sub_agent_event(status="STOPPED", message="Completed general-agent task"))
        assert activity.status == "completed"
        assert activity.entry.complete is True
        assert "completed in" in activity.entry.content


@pytest.mark.asyncio
async def test_sub_agent_error_status_marks_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting general-agent task"))
        app._render_event(_sub_agent_event(status="ERROR", message="Error in general-agent: boom"))

        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.status == "failed"
        assert "failed after" in activity.entry.content


@pytest.mark.asyncio
async def test_activity_strip_running_sub_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._turn_active = True
        app._render_event(_sub_agent_event(agent_id="a1", status="GENERATING", message="Starting"))
        assert app._status_state.activity == "Running sub-agent general-agent #1…"

        app._render_event(
            _sub_agent_event(agent_id="a2", parent_tool_call_id="tc-2", status="GENERATING", message="Starting")
        )
        assert app._status_state.activity == "Running 2 sub-agents…"
        assert app._status_state.turn_state == "Running sub-agents"

        app._render_event(_sub_agent_event(agent_id="a1", status="STOPPED", message="Completed"))
        app._render_event(_sub_agent_event(agent_id="a2", parent_tool_call_id="tc-2", status="STOPPED", message="Completed"))
        assert app._status_state.activity == "Working…"


@pytest.mark.asyncio
async def test_main_agent_tool_events_unaffected_by_sub_agent_routing(
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
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind == "tool_call"]
        assert len(tool_entries) == 1
        assert not _sub_agent_entries(app)


@pytest.mark.asyncio
async def test_sub_agent_event_without_agent_id_uses_fallback_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        event1 = _sub_agent_event(uuid="u1", text="part one ")
        event2 = _sub_agent_event(uuid="u1", text="part two")
        for event in (event1, event2):
            del event.sub_agent_info["agent_id"]
            app._render_event(event)

        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert "part one part two" in entries[0].content
        assert "tc-1" in app._sub_agent_activities


@pytest.mark.asyncio
async def test_sub_agent_tool_streaming_update_routes_to_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        event = _sub_agent_event(text="ignored")
        streaming = AgentEvent(
            event_type="tool_streaming_update",
            sender="general-agent",
            content={"text": "partial", "tool_call_id": "t1", "tool_name": "run_command_tracked", "is_complete": False},
            sub_agent_info=event.sub_agent_info,
        )
        app._render_event(streaming)

        assert not any(entry.kind.startswith("tool") for entry in app.conversation_entries)
        entries = _sub_agent_entries(app)
        assert len(entries) == 1
        assert "run_command_tracked streaming" in entries[0].content


@pytest.mark.asyncio
async def test_cancel_finalizes_running_sub_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting"))
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.status == "running"

        app._finalize_sub_agent_activities()

        assert activity.status == "stopped"
        assert activity.entry.complete is True
        assert "stopped after" in activity.entry.content


@pytest.mark.asyncio
async def test_thread_reset_clears_sub_agent_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(uuid="u1", text="some output"))
        assert app._sub_agent_activities

        app._reset_current_thread()

        assert app._sub_agent_activities == {}
        assert app._sub_agent_by_tool_call == {}
        assert not _sub_agent_entries(app)


@pytest.mark.asyncio
async def test_rapid_stream_chunks_coalesce_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        render_calls = 0
        original_render = app._render_conversation

        def counting_render() -> None:
            nonlocal render_calls
            render_calls += 1
            original_render()

        monkeypatch.setattr(app, "_render_conversation", counting_render)

        for index in range(50):
            app._apply_stream_chunk({"uuid": "chunk-1", "content": f"word{index} ", "complete": False}, kind="assistant")
        app._apply_stream_chunk({"uuid": "chunk-1", "content": "done", "complete": True}, kind="assistant")

        await pilot.pause(0.1)

        assert render_calls < 10
        entry = app._stream_entries["chunk-1"]
        assert entry.complete is True
        assert "word0" in entry.content
        assert "word49" in entry.content
        assert entry.content.endswith("done")


@pytest.mark.asyncio
async def test_conversation_scroll_position_survives_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ConversationEntry, JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(40):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        assert view.max_scroll_y > 0
        # Anchored: streaming keeps the view pinned to the bottom
        assert view.scroll_y == view.max_scroll_y

        # User scrolls up; new entries must not yank the view back down
        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        for index in range(5):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"late message {index}"))
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        assert view.scroll_y == 0
        assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is True

        # Jump-to-bottom restores the anchor and hides the bar
        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(app.query_one("#jump_to_bottom", JumpToBottomBar)))
        await pilot.pause()
        assert view.scroll_y == view.max_scroll_y
        assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is False


@pytest.mark.asyncio
async def test_assistant_entries_render_markdown_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from rich.console import Group
    from rich.markdown import Markdown as RichMarkdown

    from kolega_code.cli.app import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        streaming = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=False)
        )
        assert isinstance(streaming, str)
        assert "…" in streaming  # header carries the streaming indicator

        complete = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=True)
        )
        assert isinstance(complete, Group)
        renderables = list(complete.renderables)
        assert any(
            isinstance(getattr(item, "renderable", item), RichMarkdown) for item in renderables
        )

        plan = app._format_conversation_entry(
            ConversationEntry(kind="plan", content="- step one\n- step two", complete=True)
        )
        assert isinstance(plan, Group)


@pytest.mark.asyncio
async def test_confirmations_surface_as_toasts_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        notifications: list[tuple[str, str]] = []
        logged: list[tuple[str, str]] = []

        def fake_notify(message, *, severity="information", title=None, **kwargs):
            notifications.append((message, severity))

        original_log_status = app._log_status

        def spy_log_status(text, level="info"):
            logged.append((text, level))
            original_log_status(text, level)

        monkeypatch.setattr(app, "notify", fake_notify)
        monkeypatch.setattr(app, "_log_status", spy_log_status)

        await app._set_interaction_mode("plan")

        assert ("Switched to plan mode.", "information") in notifications
        assert ("Switched to plan mode.", "ok") in logged  # diagnostic record kept

        # Blockers surface as warning toasts
        app._turn_active = True
        await app.action_toggle_interaction_mode()
        assert ("Stop the current turn before switching modes.", "warning") in notifications


@pytest.mark.asyncio
async def test_turn_status_strip_shows_spinner_and_outcome_glyph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.app import TurnState

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    now = 0.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        app._begin_turn_progress()
        content = app._turn_status_content()
        assert any(frame in content for frame in theme.spinner_frames())
        assert "Working…" in content

        now = 12.0
        app._finish_turn_progress("Finished.", TurnState.IDLE)
        content = app._turn_status_content()
        assert theme.g(theme.Glyph.CHECK) in content
        assert "Done in 12s" in content


@pytest.mark.asyncio
async def test_tool_entries_render_as_collapsibles_with_full_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Collapsible

    from kolega_code.cli.app import TOOL_RESULT_PREVIEW_CHARS, ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        long_output = "x" * (TOOL_RESULT_PREVIEW_CHARS + 200)
        app._add_tool_message(
            "tool_call", {"tool_name": "read_file", "tool_call_id": "tc-1", "text": "Calling read_file"}
        )
        app._flush_conversation_render()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        assert collapsible.collapsed is True
        assert "running" in str(collapsible.title)

        app._add_tool_message(
            "tool_result", {"tool_name": "read_file", "tool_call_id": "tc-1", "text": long_output}
        )
        app._flush_conversation_render()
        await pilot.pause()

        # The same widget is updated in place: title flips to done, body holds full output
        same_widget = app.query(ToolEntryWidget).last()
        assert same_widget is widget
        assert "done" in str(widget.query_one(Collapsible).title)
        entry = widget.entry
        assert len(entry.content) == TOOL_RESULT_PREVIEW_CHARS + 1  # preview stays truncated
        assert entry.full_content == long_output  # expand-on-demand shows everything


@pytest.mark.asyncio
async def test_log_lines_carry_timestamp_and_level_glyph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    import re

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        line = app._format_log_line("boom", "error")
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2} \S+ boom", line.plain)

        written: list[object] = []
        monkeypatch.setattr(app._logs, "write", written.append)
        app._render_event(
            AgentEvent(event_type="log_message", sender="coder", content={"level": "error", "message": "it [broke]"})
        )
        assert len(written) == 1
        assert "[error]" not in written[0].plain  # no raw level prefix
        assert "it [broke]" in written[0].plain  # brackets survive without markup errors


@pytest.mark.asyncio
async def test_terminal_commands_render_as_styled_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent
    from kolega_code.cli import theme

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        formatted = app._format_terminal_command("ls -la")
        assert formatted.plain == f"{theme.g(theme.Glyph.USER)} ls -la"

        written: list[object] = []
        monkeypatch.setattr(app._terminal, "write", written.append)
        app._render_event(AgentEvent(event_type="terminal_command", sender="coder", content={"command": "echo one"}))
        app._render_event(AgentEvent(event_type="terminal_output", sender="coder", content={"output": "one"}))
        app._render_event(AgentEvent(event_type="terminal_command", sender="coder", content={"command": "echo two"}))

        plains = [item.plain if hasattr(item, "plain") else item for item in written]
        # Second command block is preceded by a blank separator line
        assert plains == [f"{theme.g(theme.Glyph.USER)} echo one", "one", "", f"{theme.g(theme.Glyph.USER)} echo two"]


@pytest.mark.asyncio
async def test_status_dashboard_context_note_uses_alert_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        def context_event(alert_level):
            return AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 1000,
                    "max_tokens": 2000,
                    "usage_percentage": 50.0,
                    "compression_threshold": 80.0,
                    "alert_level": alert_level,
                    "message": "Context is getting large.",
                },
            )

        app._render_event(context_event("info"))
        dashboard = app._format_status_dashboard()
        assert "[yellow]Context is getting large.[/yellow]" in dashboard

        app._render_event(context_event("critical"))
        dashboard = app._format_status_dashboard()
        assert "[red]Context is getting large.[/red]" in dashboard


@pytest.mark.asyncio
async def test_save_settings_toasts_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        notifications: list[tuple[str, str]] = []

        def fake_notify(message, *, severity="information", title=None, **kwargs):
            notifications.append((message, severity))

        monkeypatch.setattr(app, "notify", fake_notify)

        app.query_one("#api_key_input", Input).value = "moonshot-key"
        await app._save_settings_from_ui()

        assert ("Settings saved.", "information") in notifications
        status_text = str(app.query_one("#settings_status").render())
        assert "Active model:" in status_text


@pytest.mark.asyncio
async def test_planning_sidebar_marks_empty_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import PLAN_EMPTY_MESSAGE

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        plan_md = app.query_one("#planning_plan_markdown", Markdown)
        assert plan_md.source == PLAN_EMPTY_MESSAGE
        assert plan_md.has_class("empty-state")

        app._latest_plan = "# Plan\n\n- do the thing"
        app._refresh_planning_sidebar()

        assert plan_md.source == "# Plan\n\n- do the thing"
        assert not plan_md.has_class("empty-state")


@pytest.mark.asyncio
async def test_logs_tab_shows_activity_dot_until_visited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli import theme

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        tabs = app.query_one("#events", TabbedContent)
        assert tabs.active == "status_pane"

        app._write_log("background activity")
        dot = theme.g(theme.Glyph.STATUS)
        assert str(tabs.get_tab("logs_pane").label) == f"Logs {dot}"

        tabs.active = "logs_pane"
        await pilot.pause()
        assert str(tabs.get_tab("logs_pane").label) == "Logs"

        # Writing while the tab is active does not re-add the dot
        app._write_log("foreground activity")
        assert str(tabs.get_tab("logs_pane").label) == "Logs"


def _build_mention_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.messages = []
            self.attachments = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            self.attachments.append(attachments)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (project / "src" / "alpine.txt").write_text("mountains\n", encoding="utf-8")
    (project / "README.md").write_text("# Readme\n", encoding="utf-8")
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_opens_and_escape_dismisses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        await pilot.pause()

        composer.insert("@alp")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.option_count > 0

        await pilot.press("escape")
        assert not dropdown.is_open
        assert composer.text == "@alp"


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_not_opened_by_email_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("mail user@example")
        await pilot.pause()
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_down_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("@alp")
        await pilot.pause()
        assert dropdown.is_open

        expected = dropdown.entry_at(1).path
        await pilot.press("down")
        assert dropdown.highlighted == 1
        await pilot.press("tab")
        assert composer.text == f"@{expected} "
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_mention_enter_completes_instead_of_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("@README")
        await pilot.pause()
        assert dropdown.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "@README.md "
        assert not dropdown.is_open
        # No message was submitted, only the completion was applied.
        assert app.agent.messages == []


@pytest.mark.asyncio
async def test_textual_app_submitting_mention_attaches_file_and_keeps_short_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("summarize @src/alpha.py please")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["summarize @src/alpha.py please"]
        attachments = app.agent.attachments[0]
        assert attachments is not None and len(attachments) == 1
        assert attachments[0]["type"] == "file"
        assert attachments[0]["path"] == "src/alpha.py"
        assert attachments[0]["content"] == "print('alpha')\n"
        assert any(
            entry.kind == "user" and entry.content == "summarize @src/alpha.py please"
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_unresolved_mention_shows_hint_and_sends_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("look at @does/not/exist.py")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        # The hint is visible while the turn runs; end-of-turn cleanup restores the placeholder.
        hint = app.query_one("#composer_hint", Static)
        assert "does/not/exist.py" in str(hint.render())

        await pilot.pause()
        assert app.agent.messages == ["look at @does/not/exist.py"]
        assert app.agent.attachments == [None]


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_opens_filters_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown
    from kolega_code.cli.slash_commands import SlashCommandEntry

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        await pilot.pause()

        composer.insert("/")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.option_count > 1
        assert isinstance(dropdown.highlighted_entry(), SlashCommandEntry)

        composer.insert("pl")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.highlighted_entry().name == "plan"

        await pilot.press("tab")
        assert composer.text == "/plan "
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_lists_skills_with_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)
    skill_dir = app.project_path / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("/demo")
        await pilot.pause()

        assert dropdown.is_open
        entry = dropdown.highlighted_entry()
        assert entry.name == "demo-skill"
        assert entry.description == "Use this demo skill."

        await pilot.press("tab")
        assert composer.text == "/demo-skill "


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_enter_completes_instead_of_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("/versio")
        await pilot.pause()
        assert dropdown.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "/version "
        assert not dropdown.is_open
        assert app.agent.messages == []


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_does_not_open_mid_text_or_after_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()

        composer.insert("see src/")
        await pilot.pause()
        assert not dropdown.is_open

        composer.load_text("")
        composer.insert("first line")
        composer.action_insert_newline()
        composer.insert("/")
        await pilot.pause()
        assert not dropdown.is_open

        composer.load_text("")
        composer.insert("/skills extra")
        await pilot.pause()
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_chat_composer_active_slash_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.insert("/")
        assert composer.active_slash_query() == ("", 0, 1)

        composer.insert("mod")
        assert composer.active_slash_query() == ("mod", 0, 4)

        composer.load_text("")
        composer.insert("  /he")
        assert composer.active_slash_query() == ("he", 2, 5)

        composer.load_text("")
        composer.insert("hello /he")
        assert composer.active_slash_query() is None

        composer.load_text("")
        composer.insert("/model kimi")
        assert composer.active_slash_query() is None


@pytest.mark.asyncio
async def test_textual_app_plan_and_build_slash_commands_switch_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    class FakeCoderAgent(FakeAgent):
        pass

    class FakePlanningAgent(FakeAgent):
        pass

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        assert app.interaction_mode == "build"

        composer.load_text("/plan")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)
        assert composer.text == ""

        composer.load_text("/build")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_shows_and_switches_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(
        app_module,
        "ui_model_options",
        lambda provider: [
            ("Kimi K2.7 Code", UI_DEFAULT_MODEL),
            ("Kimi K2.6", "kimi-k2.6"),
            ("Kimi K3", "kimi-k3"),
        ],
    )
    monkeypatch.setattr(
        app_module,
        "ui_thinking_effort_options",
        lambda provider, model: [("High", "high")] if model == "kimi-k3" else [("Auto", "auto")],
    )
    monkeypatch.setattr(
        app_module,
        "default_ui_thinking_effort",
        lambda provider, model: "high" if model == "kimi-k3" else "auto",
    )

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        from textual.widgets import Input

        from kolega_code.cli.app import ActionList

        app.query_one("#api_key_input", Input).value = "moonshot-key"
        await app._save_settings_from_ui()
        assert isinstance(app.agent, FakeCoderAgent)

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        effort_entry = app.conversation_entries[-1]
        assert effort_entry.kind == "system"
        assert "Available thinking efforts:" in effort_entry.content
        assert "`auto`" in effort_entry.content
        assert "`none`" not in effort_entry.content
        effort_actions = app.query_one("#effort_actions", ActionList)
        assert effort_actions.display is True
        assert app.focused is effort_actions
        assert effort_actions.get_option("effort_option_0").prompt.startswith("1. Auto (auto)")

        composer.load_text("/effort auto")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_thinking_effort == "auto"
        assert effort_actions.display is False
        first_agent = app.agent
        assert isinstance(first_agent, FakeCoderAgent)

        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert UI_DEFAULT_MODEL in entry.content and "kimi-k2.6" in entry.content and "kimi-k3" in entry.content
        assert "Thinking effort: auto" in entry.content
        model_actions = app.query_one("#model_actions", ActionList)
        assert model_actions.display is True
        assert app.focused is model_actions
        assert model_actions.option_count == 3
        assert model_actions.get_option("model_option_0").prompt.startswith(f"1. Kimi K2.7 Code ({UI_DEFAULT_MODEL})")

        # kimi-k3 is a fake model the real config builder rejects, so stub it for the rebuild step.
        saved_config = app.config
        monkeypatch.setattr(app_module, "build_agent_config", lambda *args, **kwargs: saved_config)

        composer.load_text("/model kimi-k3")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        switched_settings = settings_store.load()
        assert switched_settings.active_model == "kimi-k3"
        assert switched_settings.active_thinking_effort == "high"
        assert model_actions.display is False
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent is not first_agent

        composer.load_text("/model does-not-exist")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_model == "kimi-k3"


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_selects_from_action_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.MOONSHOT.value,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(ModelProvider.MOONSHOT.value, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        model_actions = app.query_one("#model_actions", ActionList)
        assert model_actions.display is True
        assert app.focused is model_actions
        assert model_actions.option_count == 3

        await pilot.press("down", "enter")
        await pilot.pause()
        assert settings_store.load().active_model == MOONSHOT_K26_MODEL
        assert model_actions.display is False

        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.focused is model_actions

        await pilot.press("1")
        await pilot.pause()
        assert settings_store.load().active_model == UI_DEFAULT_MODEL
        assert model_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_accepts_typed_selection_and_rejects_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "unexpected"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.MOONSHOT.value,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(ModelProvider.MOONSHOT.value, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        model_actions = app.query_one("#model_actions", ActionList)
        assert model_actions.display is True

        composer.load_text("bogus-model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_model == UI_DEFAULT_MODEL
        assert model_actions.display is True
        assert not any(entry.kind == "user" and entry.content == "bogus-model" for entry in app.conversation_entries)
        assert app.agent.messages == []

        composer.load_text(MOONSHOT_K26_MODEL.upper())
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        switched_settings = settings_store.load()
        assert switched_settings.active_model == MOONSHOT_K26_MODEL
        assert switched_settings.active_thinking_effort == "auto"
        assert model_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_blocks_selector_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.MOONSHOT.value,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(ModelProvider.MOONSHOT.value, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app._pending_model_selection is None
        assert app.query_one("#model_actions", ActionList).display is False
        assert settings_store.load().active_model == UI_DEFAULT_MODEL
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_effort_slash_command_selects_from_action_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
        active_thinking_effort="high",
    )
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        effort_actions = app.query_one("#effort_actions", ActionList)
        assert effort_actions.display is True
        assert app.focused is effort_actions
        assert effort_actions.option_count == 3

        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert settings_store.load().active_thinking_effort == "max"
        assert effort_actions.display is False

        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.focused is effort_actions

        await pilot.press("1")
        await pilot.pause()
        assert settings_store.load().active_thinking_effort == "none"
        assert effort_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_effort_slash_command_accepts_typed_selection_and_rejects_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "unexpected"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
        active_thinking_effort="high",
    )
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        effort_actions = app.query_one("#effort_actions", ActionList)
        assert effort_actions.display is True

        composer.load_text("bogus")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_thinking_effort == "high"
        assert effort_actions.display is True
        assert not any(entry.kind == "user" and entry.content == "bogus" for entry in app.conversation_entries)
        assert app.agent.messages == []

        composer.load_text("MAX")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_thinking_effort == "max"
        assert effort_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_effort_slash_command_blocks_selector_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ActionList, ChatComposer, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
        active_thinking_effort="high",
    )
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app._pending_effort_selection is None
        assert app.query_one("#effort_actions", ActionList).display is False
        assert settings_store.load().active_thinking_effort == "high"
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_copy_and_version_slash_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    import kolega_code
    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, ConversationEntry
    from kolega_code.cli.updater import UpdateCheckResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    copied: list[str] = []
    monkeypatch.setattr(
        app_module,
        "check_for_update",
        lambda: UpdateCheckResult(current_version=kolega_code.__version__, latest_version=kolega_code.__version__),
    )

    async with app.run_test():
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        composer = app.query_one("#composer", ChatComposer)

        composer.load_text("/copy")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert copied == []

        app._add_conversation_entry(ConversationEntry(kind="assistant", content="the answer"))
        composer.load_text("/copy")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert copied == ["the answer"]

        composer.load_text("/version")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert kolega_code.__version__ in entry.content
        assert "up to date" in entry.content


@pytest.mark.asyncio
async def test_textual_app_update_slash_command_runs_self_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer
    from kolega_code.cli.updater import UpdateRunResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        app_module,
        "run_self_update",
        lambda *, capture_output=False: UpdateRunResult(returncode=0, stdout="installed\n"),
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/update")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert "Kolega Code update completed" in entry.content
        assert "installed" in entry.content


@pytest.mark.asyncio
async def test_textual_app_startup_update_check_notifies_when_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.updater import UpdateCheckResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    app.check_for_updates = True
    monkeypatch.setattr(
        app_module,
        "check_for_update",
        lambda: UpdateCheckResult(current_version="0.2.0", latest_version="0.3.0", update_available=True),
    )

    async with app.run_test():
        for _ in range(20):
            if any("Update available: 0.2.0 -> 0.3.0" in entry.content for entry in app.conversation_entries):
                break
            await asyncio.sleep(0.05)

        assert any("Update available: 0.2.0 -> 0.3.0" in entry.content for entry in app.conversation_entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/quit", "/exit"])
async def test_textual_app_quit_slash_command_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text(command)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

    assert app.return_value is None
    assert not app.is_running


@pytest.mark.asyncio
async def test_textual_app_unknown_slash_command_falls_through_to_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/help")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["/help"]
