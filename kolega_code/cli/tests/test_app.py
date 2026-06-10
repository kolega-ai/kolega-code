from pathlib import Path
import asyncio

import pytest

from kolega_code.agent.config import ModelProvider
from kolega_code.agent.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.agent.models.public import AgentEvent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import DEEPSEEK_DEFAULT_MODEL, UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore


def extension_by_name(extensions, name: str):
    return next(
        extension
        for extension in extensions
        if getattr(extension, "name", None) == name or getattr(extension, "id", None) == name
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        # Prose mentioning "error" with a warning tone must not render as an error
        warning_entry = ConversationEntry(
            kind="progress", content="Stopped before the error handler ran", complete=True, tone="warning"
        )
        rendered = app._format_conversation_entry(warning_entry)
        assert "bold yellow" in str(rendered)
        assert "bold red" not in str(rendered)

        error_entry = ConversationEntry(kind="progress", content="All good otherwise", complete=True, tone="error")
        rendered = app._format_conversation_entry(error_entry)
        assert "bold red" in str(rendered)

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

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan").display is False
        assert app.query_one("#discuss_plan").display is False
        assert app.query_one("#question_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nDo it."
        assert loaded.interaction_mode == BUILD_INTERACTION_MODE


@pytest.mark.asyncio
async def test_textual_app_restores_saved_plan_and_interaction_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan", Button).display is True
        assert app.query_one("#discuss_plan", Button).display is False
        assert app.query_one("#composer", ChatComposer).disabled is False


@pytest.mark.asyncio
async def test_textual_app_restores_saved_plan_in_build_mode_without_plan_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Markdown

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan", Button).display is False
        assert app.query_one("#discuss_plan", Button).display is False


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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
async def test_textual_app_planning_question_tool_accepts_option_button_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Markdown

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, ChatComposer, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
            ask_user_choice("Which approach should we use?", ["Keep state local", "Persist it"])
        )
        await pilot.pause()

        assert app._pending_question is not None
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert app.query_one("#question_option_0", Button).label.plain == "1. Keep state local"
        assert app.conversation_entries[-1].kind == "question"

        await app.on_button_pressed(Button.Pressed(app.query_one("#question_option_1", Button)))

        assert await answer_task == "Persist it"
        assert app._pending_question is None
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER
        assert app.conversation_entries[-1].kind == "user"
        assert app.conversation_entries[-1].content == "Persist it"
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_accepts_custom_text_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            app.agent.kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        answer_task = asyncio.create_task(
            ask_user_choice("Which scope?", ["Small fix", "Full workflow"])
        )
        await pilot.pause()

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("Start with the small fix, but keep the API extensible.")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert await answer_task == "Start with the small fix, but keep the API extensible."
        assert composer.text == ""
        assert app._pending_question is None
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#question_option_0", Button).label.plain == "1. Small fix"
        assert app.conversation_entries[-1].content == "Start with the small fix, but keep the API extensible."


@pytest.mark.asyncio
async def test_textual_app_blocks_mode_toggle_during_active_turn(
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._turn_active = True

        await app.action_toggle_interaction_mode()

        assert app.interaction_mode == "build"
        assert app.query_one("#composer", ChatComposer).placeholder == "Stop the current turn before switching modes."


@pytest.mark.asyncio
async def test_textual_app_shows_plan_decision_when_planning_agent_writes_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from textual.widgets import Button, Markdown

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan", Button).display is True
        assert app.query_one("#discuss_plan", Button).display is True
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
        assert app.query_one("#implement_plan", Button).display is False
        assert app.query_one("#discuss_plan", Button).display is False
        assert store.load(session.session_id).latest_plan_markdown == ""

        app.agent.completed_plan = "# Revised plan\n\nBuild planning mode carefully."
        app._capture_completed_plan()

        assert app._plan_decision_active is True
        assert app._latest_plan == "# Revised plan\n\nBuild planning mode carefully."
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert app.query_one("#implement_plan", Button).display is True
        assert app.query_one("#discuss_plan", Button).display is True
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

    from textual.widgets import Button, Markdown

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan", Button).display is False
        assert app.query_one("#discuss_plan", Button).display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nBuild it."
        assert loaded.interaction_mode == "build"


@pytest.mark.asyncio
async def test_textual_app_discuss_plan_clears_old_plan_until_new_plan_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Markdown

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan", Button).display is False
        assert app.query_one("#discuss_plan", Button).display is False
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
        assert app.query_one("#implement_plan", Button).display is False
        assert app.query_one("#discuss_plan", Button).display is False
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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

    from textual.widgets import Button, Markdown

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert app.query_one("#implement_plan", Button).display is False
        assert app.query_one("#discuss_plan", Button).display is False
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

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ChatComposer, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert composer.placeholder == "Stop the current turn before resetting the thread."


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
        assert f"Model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in startup
        assert "API key: missing" in startup
        status = str(app.query_one("#settings_status").render())
        assert "Configuration incomplete" in status
        assert "MOONSHOT_API_KEY" in status


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
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert [entry.kind for entry in app.conversation_entries].count("startup") == 1
        startup = app.conversation_entries[0].content
        assert f"Model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in startup
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
        assert settings_store.load().get_api_key(ModelProvider.DEEPSEEK.value) == "deepseek-key"
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        formatted = app._format_conversation_entry(
            ConversationEntry(kind="thinking", content="inspect [red]markup[/red]", complete=False)
        )

        assert formatted.startswith("[dim italic]Thinking[/dim italic]\n[italic]")
        assert "\\[red]" in formatted
        assert "[/italic]" in formatted
        assert formatted.endswith("\n[dim]...[/dim]")


def test_textual_app_separates_chat_entries_with_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp

    class FakeConversation:
        def __init__(self) -> None:
            self.cleared = False
            self.writes: list[object] = []

        def clear(self) -> None:
            self.cleared = True

        def write(self, renderable: object) -> None:
            self.writes.append(renderable)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    fake_conversation = FakeConversation()
    monkeypatch.setattr(KolegaCodeApp, "_conversation", property(lambda self: fake_conversation))
    app.conversation_entries = [
        ConversationEntry(kind="user", content="first"),
        ConversationEntry(kind="assistant", content="second"),
        ConversationEntry(kind="user", content="third"),
    ]

    app._render_conversation()

    assert fake_conversation.cleared is True
    assert fake_conversation.writes == [
        "[bold cyan]You[/bold cyan]\nfirst",
        "",
        "[bold magenta]Agent[/bold magenta]\nsecond",
        "",
        "[bold cyan]You[/bold cyan]\nthird",
    ]


@pytest.mark.asyncio
async def test_copyable_rich_log_extracts_plain_selected_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import CopyableRichLog, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        conversation = app.query_one("#conversation", CopyableRichLog)
        conversation.clear()
        conversation.write("[bold magenta]Agent[/bold magenta]\ncopy [red]this[/red]")

        selected = conversation.get_selection(Selection(None, None))

        assert selected == ("Agent\ncopy this", "\n")


@pytest.mark.asyncio
async def test_copyable_rich_log_supports_mouse_drag_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import CopyableRichLog, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        conversation = app.query_one("#conversation", CopyableRichLog)
        conversation.clear()
        conversation.write("[bold magenta]Agent[/bold magenta]\nselect this text")

        await pilot.mouse_down(conversation, offset=(1, 2))
        await pilot._post_mouse_events([events.MouseMove], conversation, offset=(18, 2), button=1)
        await pilot.mouse_up(conversation, offset=(18, 2))

        assert app.screen.get_selected_text() == "select this text"


@pytest.mark.asyncio
async def test_command_c_copies_selected_chat_text_to_macos_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import CopyableRichLog, KolegaCodeApp

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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        conversation = app.query_one("#conversation", CopyableRichLog)
        conversation.clear()
        conversation.write("[bold magenta]Agent[/bold magenta]\ncopy [red]this[/red]")
        app.screen.selections = {conversation: Selection(None, None)}

        await pilot.press("super+c")

        assert app.clipboard == "Agent\ncopy this"
        assert pbcopy_calls == [
            {"args": ["pbcopy"], "input": "Agent\ncopy this", "text": True, "check": True}
        ]


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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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

        assert assistant.startswith("[bold magenta]Agent[/bold magenta]")
        assert "Kolega" not in assistant
        assert "[black on yellow] TOOL [/black on yellow]" in tool_call
        assert "[dim]  │[/dim] inspect \\[red]markup\\[/red]" in tool_call
        assert "[dim]  │[/dim] then continue" in tool_call
        assert "[black on green] TOOL [/black on green]" in tool_result
        assert "[white on red] TOOL ERROR [/white on red]" in tool_error


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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert tool_entries[0].content == "completed\nshort result"
        assert tool_entries[0].tool_call_id == "tool-1"
        assert tool_entries[1].content.endswith("...")
        assert tool_entries[1].tool_call_id == "tool-2"
        assert len(tool_entries[1].content) == TOOL_RESULT_PREVIEW_CHARS + 3


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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert tool_entries[0].content == "completed\nfinal analysis"
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert tool_entries[0].content.startswith(f"[stream truncated to last {TOOL_STREAM_PREVIEW_CHARS} chars]")
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert tool_entries[0].content == "completed\nREADME contents"
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
        assert tool_entries[0].content == "completed\nlate result"


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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
            ("tool_result", "completed\nREADME contents", "read_file"),
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
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
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
