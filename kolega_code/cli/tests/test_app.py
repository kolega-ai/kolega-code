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


def renderable_text(renderable) -> str:
    from rich.console import Console

    console = Console(width=240, color_system=None, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable, soft_wrap=True, end="")
    return capture.get()


def first_text_styles(renderable) -> list[str]:
    renderables = list(getattr(renderable, "renderables", [renderable]))
    text = renderables[0]
    return [str(span.style) for span in getattr(text, "spans", [])]


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

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history_restored = False

        def restore_message_history(self, history):
            self.history_restored = bool(history)

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
async def test_settings_tab_grouped_into_model_and_appearance_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.history = []

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
        # Bordered, titled sections.
        assert app.query_one("#settings_model").border_title == "Model"
        assert app.query_one("#settings_agent_models").border_title == "Agent Models"
        assert app.query_one("#settings_appearance").border_title == "Appearance"
        # Every control still resolves by id (wiring is unchanged).
        for control_id in (
            "#provider_select",
            "#model_select",
            "#thinking_effort_select",
            "#api_key_input",
            "#save_settings",
            "#settings_status",
            "#theme_select",
        ):
            app.query_one(control_id)
        # Grouping: model controls in the Model card, theme in the Appearance card.
        assert app.query_one("#settings_model #provider_select")
        assert app.query_one("#settings_appearance #theme_select")
        assert not list(app.query("#settings_model #theme_select"))
        # Save is a form-level action, not nested inside the Model card.
        assert app.query_one("#settings_actions #save_settings")
        assert not list(app.query("#settings_model #save_settings"))


@pytest.mark.asyncio
async def test_web_search_settings_section_reveal_and_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input, Select

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

    async with app.run_test() as pilot:
        assert app.query_one("#settings_web_search").border_title == "Web Search"
        backend_select = app.query_one("#web_search_backend_select", Select)
        # Keyless DuckDuckGo is the default; key + base-url fields are hidden.
        assert str(backend_select.value) == "duckduckgo"
        assert app.query_one("#web_search_api_key_input").display is False
        assert app.query_one("#web_search_base_url_input").display is False

        # A cloud backend reveals the key field (no NoMatches on the initial Changed).
        backend_select.value = "tavily"
        await pilot.pause()
        assert app.query_one("#web_search_api_key_input").display is True
        assert app.query_one("#web_search_base_url_input").display is False

        # SearXNG reveals the base-url field instead.
        backend_select.value = "searxng"
        await pilot.pause()
        assert app.query_one("#web_search_base_url_input").display is True
        assert app.query_one("#web_search_api_key_input").display is False

        # Firecrawl is keyless-capable but still offers an OPTIONAL key field.
        backend_select.value = "firecrawl"
        await pilot.pause()
        assert app.query_one("#web_search_api_key_input").display is True
        assert app.query_one("#web_search_base_url_input").display is False
        assert "Optional" in app.query_one("#web_search_api_key_input", Input).placeholder

        # Configure Tavily with a key and save.
        backend_select.value = "tavily"
        await pilot.pause()
        app.query_one("#web_search_api_key_input", Input).value = "tvly-secret"
        await app._save_settings_from_ui()
        # Secret is never echoed back into the field after saving.
        assert app.query_one("#web_search_api_key_input", Input).value == ""

    stored = settings_store.load()
    assert stored.web_search_backend == "tavily"
    assert stored.get_api_key("tavily") == "tvly-secret"


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
async def test_textual_app_startup_entry_updates_incrementally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

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

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        startup = app.conversation_entries[0]
        original_content = startup.content
        invalidated = []
        rendered = []

        def spy_invalidate(entry):
            invalidated.append(entry)

        def spy_render():
            rendered.append(True)

        monkeypatch.setattr(app, "_invalidate_conversation", spy_invalidate)
        monkeypatch.setattr(app, "_render_conversation", spy_render)

        app.interaction_mode = "plan"
        app._ensure_startup_entry()

        assert app.conversation_entries[0] is startup
        assert startup.content != original_content
        assert "Interaction: plan" in startup.content
        assert invalidated == [startup]
        assert rendered == []

        app.conversation_entries = []
        app._ensure_startup_entry()

        assert app.conversation_entries[0].kind == "startup"
        assert rendered == [True]


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
async def test_textual_app_turn_status_formats_error_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_turn_state_styles_do_not_depend_on_content_text() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import TurnState, turn_state_color

    # Resolves role names against the active theme's live Color attributes.
    assert turn_state_color(TurnState.ERROR) == theme.Color.ERROR
    assert turn_state_color(TurnState.STOPPED) == theme.Color.WARNING
    assert turn_state_color(TurnState.STOPPING) == theme.Color.WARNING
    assert turn_state_color(TurnState.IDLE) == theme.Color.SUCCESS
    assert turn_state_color(TurnState.GENERATING) == theme.Color.ACCENT  # falls back to accent


@pytest.mark.asyncio
async def test_progress_entry_tone_drives_styling_not_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry, TurnState

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
        # Prose mentioning "error" with a warning tone must not render as an error
        warning_entry = ConversationEntry(
            kind="progress", content="Stopped before the error handler ran", complete=True, tone="warning"
        )
        rendered = app._format_conversation_entry(warning_entry)
        assert theme.Color.WARNING in first_text_styles(rendered)
        assert theme.Color.ERROR not in first_text_styles(rendered)

        error_entry = ConversationEntry(kind="progress", content="All good otherwise", complete=True, tone="error")
        rendered = app._format_conversation_entry(error_entry)
        assert theme.Color.ERROR in first_text_styles(rendered)

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
async def test_textual_app_ctrl_p_toggles_permission_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        assert "Permissions: auto" in app.conversation_entries[0].content
        assert "Auto" in str(app.query_one("#status_dashboard", Static).render())


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
        tabs.active = "logs_pane"
        await pilot.pause()

        assert app.sidebar_visible is True
        assert side_panel.display is True

        await pilot.press("ctrl+o")

        assert app.sidebar_visible is False
        assert side_panel.display is False
        assert tabs.active == "logs_pane"

        await pilot.press("ctrl+o")

        assert app.sidebar_visible is True
        assert side_panel.display is True
        assert tabs.active == "logs_pane"


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
async def test_textual_app_skill_slash_commands_list_and_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.messages = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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
                f"- Step {index}: keep the planning sidebar readable."
                for index in range(1, 26)
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
        assert app.query_one("#composer", ChatComposer).placeholder == "Plan ready. Choose Implement plan or Discuss further."
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


@pytest.mark.asyncio
async def test_textual_app_does_not_save_startup_entry_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]

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
            return saved_history

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
        assert app.conversation_entries[0].kind == "startup"
        await app._save_session_history_async()

        assert session.history == saved_history
        assert all("Kolega Code" not in str(item) for item in session.history)


@pytest.mark.asyncio
async def test_textual_app_history_save_runs_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]

    class FakeAgent:
        def dump_message_history(self):
            return saved_history

        def dump_compaction_state(self):
            return {}

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    original_save = store.save

    def slow_save(record):
        time.sleep(0.2)
        original_save(record)

    monkeypatch.setattr(store, "save", slow_save)
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    app.agent = FakeAgent()

    save_task = asyncio.create_task(app._save_session_history_async())
    marker_task = asyncio.create_task(asyncio.sleep(0.05))
    done, _ = await asyncio.wait({save_task, marker_task}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)

    assert marker_task in done
    assert not save_task.done()
    await save_task
    assert store.load(session.session_id).history == saved_history


@pytest.mark.asyncio
async def test_textual_app_history_save_persists_session_and_compaction(tmp_path: Path) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("persist me")]).to_dict()]
    saved_compaction = {"summary": "older turns", "compacted_through": 3, "compacted_history_length": 5}

    class FakeAgent:
        def dump_message_history(self):
            return saved_history

        def dump_compaction_state(self):
            return saved_compaction

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    app.agent = FakeAgent()

    await app._save_session_history_async()

    assert app.session.history == saved_history
    assert app.session.compaction == saved_compaction
    loaded = store.load(session.session_id)
    assert loaded.history == saved_history
    assert loaded.compaction == saved_compaction


@pytest.mark.asyncio
async def test_textual_app_overlapping_saves_preserve_later_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("history from first save")]).to_dict()]

    class FakeAgent:
        def dump_message_history(self):
            return saved_history

        def dump_compaction_state(self):
            return {}

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    app.agent = FakeAgent()

    original_save = store.save
    saved_snapshots: list[tuple[str, list[dict]]] = []

    def slow_first_save(record):
        saved_snapshots.append((record.task_list_markdown, list(record.history)))
        if len(saved_snapshots) == 1:
            time.sleep(0.2)
        original_save(record)

    monkeypatch.setattr(store, "save", slow_first_save)

    first_save = asyncio.create_task(app._save_session_history_async())
    await asyncio.sleep(0.05)
    app.session.task_list_markdown = "- [x] later state"
    second_save = asyncio.create_task(app._save_session_async())

    await asyncio.gather(first_save, second_save)

    assert len(saved_snapshots) == 2
    assert saved_snapshots[0] == ("", saved_history)
    assert saved_snapshots[1] == ("- [x] later state", saved_history)
    loaded = store.load(session.session_id)
    assert loaded.task_list_markdown == "- [x] later state"
    assert loaded.history == saved_history


@pytest.mark.asyncio
async def test_textual_app_composer_shift_enter_inserts_line_break_and_enter_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import THREAD_RESET_MESSAGE

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
            raise AssertionError("reset commands should not be sent to the agent")

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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
        app._plan_reofferable = True
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
        assert app._plan_reofferable is False
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
        assert store.load(session.session_id).plan_reofferable is False
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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = ["old history"]

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            raise AssertionError("agent should not be built without a valid API key")

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            raise AssertionError("agent should not be built from an API key alone")

    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

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

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry

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
        formatted = app._format_conversation_entry(
            ConversationEntry(kind="thinking", content="inspect [red]markup[/red]", complete=False)
        )
        rendered = renderable_text(formatted)

        assert "Thinking" in rendered
        assert "[red]markup[/red]" in rendered
        assert "…" in rendered  # streaming indicator in the header
        assert any("italic" in style and "dim" in style for style in first_text_styles(formatted))


@pytest.mark.asyncio
async def test_textual_app_renders_one_widget_per_chat_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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
        assert "second updated" in renderable_text(same_widgets[1]._formatted)


@pytest.mark.asyncio
async def test_conversation_render_skips_detached_view_during_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coalesced render timer can fire while the app is tearing down. The view is
    detached from the DOM (is_attached is False) but query_one still resolves it, so
    mounting into it used to raise MountError and crash the CLI on exit."""
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget, ConversationView

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
        app.conversation_entries = [ConversationEntry(kind="user", content="first")]
        app._render_conversation()
        await pilot.pause()
        assert len(app.query(ConversationEntryWidget)) == 1

        # Queue a new entry so a flush would reach view.mount(...), then simulate the
        # exit-time race by detaching the view (is_attached False) without removing it
        # from the DOM, so query_one still resolves it.
        app.conversation_entries.append(
            ConversationEntry(kind="assistant", content="late", complete=False)
        )
        app._render_pending = True
        ConversationView.is_attached = property(lambda self: False)
        try:
            app._flush_conversation_render()  # must not raise (pre-fix: MountError)
            app._render_conversation()  # must not raise
        finally:
            del ConversationView.is_attached

        # The detached render was skipped, so nothing new was mounted.
        assert len(app.query(ConversationEntryWidget)) == 1


@pytest.mark.asyncio
async def test_conversation_flush_uses_dirty_entry_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content=f"message {index}") for index in range(50)
        ]
        app._render_conversation()
        await pilot.pause()

        widgets = list(app.query(ConversationEntryWidget))
        assert len(widgets) == 50

        def fail_full_rebuild() -> None:
            raise AssertionError("dirty-entry refresh should not rebuild the transcript")

        monkeypatch.setattr(app, "_render_conversation", fail_full_rebuild)
        entry = app.conversation_entries[25]
        entry.content = "message 25 updated"
        app._dirty_entry_ids.add(entry.entry_id)
        app._render_pending = True

        app._flush_conversation_render()
        await pilot.pause()

        refreshed = list(app.query(ConversationEntryWidget))
        assert refreshed == widgets
        assert "message 25 updated" in renderable_text(refreshed[25]._formatted)


@pytest.mark.asyncio
async def test_repeated_progress_updates_refresh_status_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import TurnState

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        refresh_count = 0

        def refresh_status_dashboard() -> None:
            nonlocal refresh_count
            refresh_count += 1

        monkeypatch.setattr(app, "_refresh_status_dashboard", refresh_status_dashboard)
        app._turn_status_text = ""
        app._status_state.turn_state = TurnState.IDLE
        app._status_state.activity = "Ready"

        app._update_progress("Reading response", complete=False, state=TurnState.GENERATING)
        app._update_progress("Reading response", complete=False, state=TurnState.GENERATING)
        app._update_progress("Reading response", complete=False, state=TurnState.GENERATING)

        assert refresh_count == 1

        app._update_progress("Reading response", complete=False, state=TurnState.THINKING)

        assert refresh_count == 2


@pytest.mark.asyncio
async def test_tab_activity_label_changes_only_on_state_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli import theme
    from kolega_code.cli.tui.constants import TAB_BASE_LABELS
    from kolega_code.cli.theme import Glyph

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        tabs = app.query_one("#events", TabbedContent)
        logs_tab = tabs.get_tab("logs_pane")
        expected_marked = f"{TAB_BASE_LABELS['logs_pane']} {theme.g(Glyph.STATUS)}"

        app._mark_tab_activity("logs_pane")
        marked_label = logs_tab.label
        assert str(marked_label) == expected_marked

        app._mark_tab_activity("logs_pane")
        assert logs_tab.label is marked_label

        app._clear_tab_activity("logs_pane")
        cleared_label = logs_tab.label
        assert str(cleared_label) == TAB_BASE_LABELS["logs_pane"]

        app._clear_tab_activity("logs_pane")
        assert logs_tab.label is cleared_label


@pytest.mark.asyncio
async def test_conversation_entry_widget_skips_unchanged_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="stable")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        updates = 0

        def update(renderable) -> None:
            nonlocal updates
            updates += 1

        monkeypatch.setattr(widget, "update", update)
        widget.refresh_content()
        assert updates == 0

        widget.entry.content = "changed"
        widget.refresh_content()
        assert updates == 1


@pytest.mark.asyncio
async def test_conversation_entry_widget_extracts_plain_selected_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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
async def test_conversation_entry_selection_styles_rendered_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="select this user text"),
            ConversationEntry(kind="assistant", content="select this agent text"),
            ConversationEntry(kind="assistant", content="select this streaming text", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        widgets = list(app.query(ConversationEntryWidget))[-3:]
        for widget in widgets:
            app.screen.selections = {widget: Selection(Offset(0, 1), Offset(12, 1))}
            strip = widget.render_line(1)
            selection_bg = widget.selection_style.bgcolor

            assert any(
                segment.style is not None
                and segment.style.bgcolor == selection_bg
                and segment.style.meta.get("offset") is not None
                for segment in strip
            )


@pytest.mark.asyncio
async def test_conversation_entry_selection_preserves_text_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="select this user text"),
        ]
        app._render_conversation()
        await pilot.pause()

        # Exercise every theme: the fix must keep selected text readable regardless
        # of the active palette. Theme state is process-global, so restore it after.
        try:
            for theme_name in theme.available_themes():
                app._apply_theme(theme_name)
                await pilot.pause()
                widget = app.query(ConversationEntryWidget).last()

                # Foreground colors present on line 1 with no selection.
                app.screen.selections = {}
                plain_strip = widget.render_line(1)
                plain_colors = {
                    segment.style.color
                    for segment in plain_strip
                    if segment.text and segment.style is not None
                }
                assert plain_colors, f"expected colored content on line 1 for {theme_name}"

                # Select a span on line 1 and re-render.
                app.screen.selections = {widget: Selection(Offset(0, 1), Offset(20, 1))}
                selected_strip = widget.render_line(1)
                selection_style = widget.selection_style
                selection_bg = selection_style.bgcolor

                highlighted = [
                    segment
                    for segment in selected_strip
                    if segment.text
                    and segment.style is not None
                    and segment.style.bgcolor == selection_bg
                ]
                assert highlighted, f"expected highlighted segments for {theme_name}"

                # The selection must not blank out the text: every highlighted segment
                # keeps its original foreground (one seen unselected) and never the
                # transparent selection foreground.
                for segment in highlighted:
                    assert segment.style.color in plain_colors, (
                        f"selection wiped the text foreground for {theme_name}"
                    )
                    if selection_style.color is not None:
                        assert segment.style.color != selection_style.color, (
                            f"selection foreground overrode text color for {theme_name}"
                        )
        finally:
            theme.apply_theme(theme.DEFAULT_THEME_NAME)


@pytest.mark.asyncio
async def test_conversation_selection_can_start_in_blank_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first"),
            ConversationEntry(kind="assistant", content="second", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        first, second = list(app.query(ConversationEntryWidget))[-2:]
        separator_y = first.region.height - 1

        await pilot.mouse_down(first, offset=(0, separator_y))
        await pilot._post_mouse_events([events.MouseMove], second, offset=(20, 1), button=1)
        await pilot.mouse_up(second, offset=(20, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "second" in selected_text


@pytest.mark.asyncio
async def test_conversation_selection_can_start_after_line_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first"),
            ConversationEntry(kind="assistant", content="second", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        first, second = list(app.query(ConversationEntryWidget))[-2:]

        await pilot.mouse_down(first, offset=(30, 1))
        await pilot._post_mouse_events([events.MouseMove], second, offset=(20, 1), button=1)
        await pilot.mouse_up(second, offset=(20, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "second" in selected_text


@pytest.mark.asyncio
async def test_conversation_selection_spans_multiple_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first message"),
            ConversationEntry(kind="assistant", content="second message", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        first, second = list(app.query(ConversationEntryWidget))[-2:]

        await pilot.mouse_down(first, offset=(0, 1))
        await pilot._post_mouse_events([events.MouseMove], second, offset=(20, 1), button=1)
        await pilot.mouse_up(second, offset=(20, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "first message" in selected_text
        assert "second message" in selected_text
        assert selected_text.index("first message") < selected_text.index("second message")


@pytest.mark.asyncio
async def test_collapsed_tool_title_supports_drag_selection_and_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events
    from textual.widgets import Collapsible
    from textual.widgets._collapsible import CollapsibleTitle

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(
                kind="tool_result",
                content="preview text",
                full_content="full text",
                tool_name="read_file",
            )
        ]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        title = widget.query_one(CollapsibleTitle)

        await pilot.mouse_down(title, offset=(1, 0))
        await pilot._post_mouse_events([events.MouseMove], title, offset=(20, 0), button=1)
        await pilot.mouse_up(title, offset=(20, 0))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "read_file" in selected_text

        await pilot.click(title, offset=(1, 0))
        await pilot.pause()
        assert collapsible.collapsed is False

        title.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert collapsible.collapsed is True


@pytest.mark.asyncio
async def test_expanded_tool_body_line_start_selection_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events
    from textual.widgets import Collapsible, Static
    from textual.widgets._collapsible import CollapsibleTitle

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test(size=(100, 40)) as pilot:
        app.conversation_entries = [
            ConversationEntry(
                kind="tool_result",
                content="preview text",
                full_content="alpha line\nbeta line\ngamma line",
                tool_name="read_file",
            )
        ]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        title = widget.query_one(CollapsibleTitle)
        await pilot.click(title, offset=(1, 0))
        await pilot.pause()

        body = widget.query_one(".tool-body", Static)
        assert widget.query_one(Collapsible).collapsed is False
        assert body.region.x == widget.region.x + 3

        body_y = body.region.y - widget.region.y
        await pilot.mouse_down(widget, offset=(0, body_y))
        await pilot._post_mouse_events([events.MouseMove], widget, offset=(11, body_y + 1), button=1)
        await pilot.mouse_up(widget, offset=(11, body_y + 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert selected_text == "alpha line\nbeta line"

        await pilot.press("super+c")
        assert copied == ["alpha line\nbeta line"]


@pytest.mark.asyncio
async def test_command_c_copies_selected_chat_text_to_macos_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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

    pbcopy_calls: list[dict] = []

    def fake_run(args, *, input, text, check):
        pbcopy_calls.append({"args": args, "input": input, "text": text, "check": check})

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry

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
        assistant_text = renderable_text(assistant)
        tool_call_text = renderable_text(tool_call)
        tool_result_text = renderable_text(tool_result)
        tool_error_text = renderable_text(tool_error)

        assert "● Agent" in assistant_text
        assert "Kolega" not in assistant_text
        assert "⏺ read_file" in tool_call_text
        assert "· running" in tool_call_text
        assert "inspect [red]markup[/red]" in tool_call_text
        assert "then continue" in tool_call_text
        assert "⏺ read_file" in tool_result_text
        assert "· done" in tool_result_text
        assert "⏺ write_file" in tool_error_text
        assert "· failed" in tool_error_text


@pytest.mark.asyncio
async def test_textual_app_ignores_empty_final_response_without_existing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

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

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "", "complete": True, "uuid": "response-empty"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    started = asyncio.Event()
    release = asyncio.Event()

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

        async def process_message_stream(self, message):
            started.set()
            await release.wait()
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.theme import TOOL_RESULT_PREVIEW_CHARS

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.theme import TOOL_STREAM_PREVIEW_CHARS

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

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

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

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

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    started = asyncio.Event()

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

        async def process_message_stream(self, message):
            started.set()
            while True:
                await asyncio.sleep(1)
                yield {"type": "thinking", "content": "still working", "complete": False, "uuid": "thinking-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState
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

        async def process_message_stream(self, message):
            raise error
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState
    from kolega_code.cli.tui.widgets import ChatComposer

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

        async def process_message_stream(self, message):
            raise RuntimeError("tool host exploded")
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None

        def restore_message_history(self, history):
            self.restored_history = history

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.restored_history or []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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
                ToolCall(id="tool-1", name="read_file", input={"path": "README.md"}),
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
            ("tool_result", "README contents", "read_file"),
            ("tool_error", "Permission denied", "write_file"),
            ("assistant", "Done.", None),
        ]


@pytest.mark.asyncio
async def test_textual_app_restore_tool_history_matches_legacy_and_execution_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None

        def restore_message_history(self, history):
            self.restored_history = history

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.restored_history or []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    legacy_call = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Legacy first."},
            {
                "type": "tool_call",
                "id": "provider-legacy",
                "name": "read_file",
                "input": {"path": "legacy.md"},
            },
        ],
        "stop_reason": "tool_use",
        "usage_metadata": {},
    }
    legacy_result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "provider-legacy",
                "content": "legacy contents",
                "name": "read_file",
                "is_error": False,
                "cache_checkpoint": False,
            }
        ],
        "stop_reason": None,
        "usage_metadata": {},
    }
    execution_call = Message(
        role="assistant",
        content=[
            TextBlock("Modern first."),
            ToolCall(
                id="provider-modern",
                name="search_codebase",
                input={"pattern": "needle"},
                execution_id="tool_exec_modern",
            ),
        ],
    ).to_dict()
    execution_result = Message(
        role="user",
        content=[
            ToolResult(
                tool_use_id="provider-modern",
                content="modern contents",
                name="search_codebase",
                is_error=False,
                execution_id="tool_exec_modern",
            ),
            TextBlock("Thanks for the tool output."),
        ],
    ).to_dict()
    pending_call = Message(
        role="assistant",
        content=[ToolCall(id="provider-pending", name="list_directory", input={"path": "."})],
    ).to_dict()
    orphan_result = Message(
        role="user",
        content=[
            ToolResult(tool_use_id="provider-orphan", content="orphan failed", name="write_file", is_error=True)
        ],
    ).to_dict()
    session.history = [legacy_call, legacy_result, execution_call, execution_result, pending_call, orphan_result]

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        restored = app.conversation_entries[1:]
        assert [(entry.kind, entry.content, entry.tool_name) for entry in restored] == [
            ("assistant", "Legacy first.", None),
            ("tool_result", "legacy contents", "read_file"),
            ("assistant", "Modern first.", None),
            ("tool_result", "modern contents", "search_codebase"),
            ("user", "Thanks for the tool output.", None),
            ("tool_call", "Calling list_directory", "list_directory"),
            ("tool_error", "orphan failed", "write_file"),
        ]
        modern_entry = next(entry for entry in restored if entry.tool_name == "search_codebase")
        assert modern_entry.tool_call_id == "tool_exec_modern"
        pending_entry = next(entry for entry in restored if entry.tool_name == "list_directory")
        assert pending_entry.complete is False


@pytest.mark.asyncio
async def test_textual_app_model_rebuild_rerenders_completed_tool_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.compaction = {}

        def restore_message_history(self, history):
            self.history = history

        def dump_compaction_state(self):
            return self.compaction

        def restore_compaction_state(self, data):
            self.compaction = data or {}

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    history = [
        Message(
            role="assistant",
            content=[ToolCall(id="provider-1", name="list_directory", input={"path": "."})],
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="provider-1", content="files", name="list_directory", is_error=False)],
        ).to_dict(),
    ]

    async with app.run_test():
        app.agent.history = history
        await app._build_agent(config, rebuild=True)

        tool_entries = [entry for entry in app.conversation_entries if entry.tool_name == "list_directory"]
        assert [(entry.kind, entry.content) for entry in tool_entries] == [("tool_result", "files")]


# ---------------------------------------------------------------------------
# Parallel sub-agent rendering
# ---------------------------------------------------------------------------


def _build_sub_agent_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")

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


def _sub_agent_context_event(usage_percentage, *, input_tokens=5000, agent_id="agent-1", agent_name="general-agent"):
    return AgentEvent(
        event_type="llm_context_update",
        sender=agent_name,
        content={
            "input_tokens": input_tokens,
            "max_tokens": 200000,
            "usage_percentage": usage_percentage,
            "alert_level": "normal",
            "message": None,
            "compression_threshold": 80.0,
        },
        sub_agent_info={
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": "inspect sessions",
            "parent_tool_call_id": "tc-1",
            "conversation_id": None,
            "depth": 1,
        },
    )


@pytest.mark.asyncio
async def test_sub_agent_context_update_does_not_stomp_main_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Main agent reports its context usage -> the status dashboard reflects it.
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 123456,
                    "max_tokens": 200000,
                    "usage_percentage": 61.7,
                    "alert_level": "normal",
                    "message": None,
                    "compression_threshold": 80.0,
                },
            )
        )
        assert "61.7%" in app._format_status_dashboard()

        # A sub-agent reports its much smaller context usage. It must NOT overwrite the
        # main dashboard; it lands on the sub-agent's own card instead.
        app._render_event(_sub_agent_context_event(3.0, input_tokens=6000))

        dashboard = app._format_status_dashboard()
        assert "61.7%" in dashboard  # main agent's value preserved
        assert "3.0%" not in dashboard

        activities = list(app._sub_agent_activities.values())
        assert len(activities) == 1
        assert activities[0].context_percentage == 3.0
        assert activities[0].context_input_tokens == 6000
        # The cumulative-token field is a different metric and stays untouched.
        assert activities[0].tokens is None
        # The per-agent context shows on its own card.
        assert "ctx 3%" in activities[0].entry.content


@pytest.mark.asyncio
async def test_concurrent_sub_agent_context_updates_keep_main_dashboard_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 100000,
                    "max_tokens": 200000,
                    "usage_percentage": 50.0,
                    "alert_level": "normal",
                    "message": None,
                    "compression_threshold": 80.0,
                },
            )
        )

        # Several sub-agents interleave context updates; none may move the main bar.
        for pct, aid in [(2.0, "agent-1"), (4.0, "agent-2"), (1.5, "agent-3")]:
            app._render_event(_sub_agent_context_event(pct, agent_id=aid, agent_name=aid))

        assert "50.0%" in app._format_status_dashboard()
        assert len(app._sub_agent_activities) == 3


@pytest.mark.asyncio
async def test_sub_agent_status_update_routes_to_card_not_main_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Create the sub-agent card first; this is what legitimately updates the main
        # activity line (with the "running sub-agent" notice), before the spy is set.
        app._render_event(_sub_agent_event(text="working"))

        calls: list = []
        monkeypatch.setattr(app, "_update_activity_progress", lambda *a, **k: calls.append(a))

        app._render_event(
            AgentEvent(
                event_type="llm_status_update",
                sender="general-agent",
                content={"status": "overloaded", "message": "Provider overloaded, retrying"},
                sub_agent_info={
                    "agent_id": "agent-1",
                    "agent_name": "general-agent",
                    "task": "inspect sessions",
                    "parent_tool_call_id": "tc-1",
                    "conversation_id": None,
                    "depth": 1,
                },
            )
        )

        assert calls == []  # the main activity line is untouched
        activity = next(iter(app._sub_agent_activities.values()))
        assert "Provider overloaded, retrying" in activity.last_activity


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

        await app._reset_current_thread()

        assert app._sub_agent_activities == {}
        assert app._sub_agent_by_tool_call == {}
        assert not _sub_agent_entries(app)


@pytest.mark.asyncio
async def test_sub_agent_steps_capture_full_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # A tool call + its result pair into one step keyed by tool_call_id.
        app._render_event(
            _sub_agent_event(
                message_type="tool_call", text="Reading app.py", tool_description="read_file", tool_call_id="t1"
            )
        )
        app._render_event(
            _sub_agent_event(
                message_type="tool_result", text="file contents", tool_description="read_file", tool_call_id="t1"
            )
        )
        # Thinking + streamed response accumulate into steps by uuid.
        app._render_event(_sub_agent_event(message_type="thinking", uuid="th1", text="planning the edit"))
        app._render_event(_sub_agent_event(uuid="r1", text="Here is "))
        app._render_event(_sub_agent_event(uuid="r1", text="the answer"))

        activity = next(iter(app._sub_agent_activities.values()))
        # Seeded task step + 1 paired tool step + 1 thinking + 1 response = 4 steps.
        assert len(activity.steps) == 4
        assert activity.steps[0].kind == "sub_agent_task"
        assert activity.steps[0].content == "inspect sessions"
        tool_step = activity.steps[1]
        assert tool_step.kind == "tool_result"
        assert tool_step.tool_call_id == "t1"
        assert tool_step.full_content == "file contents"
        assert activity.steps[2].kind == "thinking"
        assert activity.steps[2].content == "planning the edit"
        assert activity.steps[3].kind == "assistant"
        assert activity.steps[3].content == "Here is the answer"
        # The card advertises the inspect affordance once steps exist.
        from kolega_code.cli import messages as cli_messages

        assert cli_messages.SUB_AGENT_INSPECT_HINT in activity.entry.content


@pytest.mark.asyncio
async def test_sub_agent_completion_event_records_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(status="GENERATING", message="Starting"))
        app._render_event(_sub_agent_event(status="STOPPED", message="Completed", total_tokens=3100))

        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.tokens == 3100
        assert "3.1k tok" in activity.entry.content


@pytest.mark.asyncio
async def test_open_sub_agent_inspector_renders_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.sub_agent_screen import SubAgentInspectorScreen

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(
            _sub_agent_event(
                message_type="tool_call", text="Reading", tool_description="read_file", tool_call_id="t1"
            )
        )
        app._render_event(_sub_agent_event(uuid="r1", text="some response"))

        app.action_open_sub_agent()
        await pilot.pause()

        assert isinstance(app._sub_agent_inspector, SubAgentInspectorScreen)
        screen = app._sub_agent_inspector
        # One roster row per agent, and a trajectory widget per captured step.
        assert len(screen._rows) == 1
        # Seeded task step + tool_call + assistant response.
        assert len(screen._step_widgets) == 3
        # The mounted widgets wrap the real step entries (not empty placeholders).
        kinds = {w.entry.kind for w in screen._step_widgets.values()}
        contents = {w.entry.content for w in screen._step_widgets.values()}
        assert kinds == {"sub_agent_task", "tool_call", "assistant"}
        assert "some response" in contents
        assert "inspect sessions" in contents


@pytest.mark.asyncio
async def test_sub_agent_inspector_switches_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(agent_id="a1", task="one", uuid="u1", text="alpha"))
        app._render_event(
            _sub_agent_event(agent_id="a2", task="two", parent_tool_call_id="tc-2", uuid="u2", text="beta")
        )

        app.action_open_sub_agent("a1")
        await pilot.pause()
        screen = app._sub_agent_inspector
        assert screen._selected_key == "a1"

        screen.action_next_agent()
        await pilot.pause()
        assert screen._selected_key == "a2"


@pytest.mark.asyncio
async def test_sub_agent_inspector_escape_closes_without_cancelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        cancelled = False

        def fake_cancel() -> None:
            nonlocal cancelled
            cancelled = True

        monkeypatch.setattr(app, "action_cancel_generation", fake_cancel)

        app._render_event(_sub_agent_event(uuid="u1", text="output"))
        app.action_open_sub_agent()
        await pilot.pause()
        assert app._sub_agent_inspector is not None

        await pilot.press("escape")
        await pilot.pause()

        assert app._sub_agent_inspector is None
        assert cancelled is False


@pytest.mark.asyncio
async def test_open_sub_agent_inspector_empty_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        notes: list[str] = []
        monkeypatch.setattr(app, "_notify_user", lambda message, **kw: notes.append(message))

        app.action_open_sub_agent()

        assert app._sub_agent_inspector is None
        from kolega_code.cli import messages as cli_messages

        assert cli_messages.SUB_AGENT_INSPECTOR_EMPTY in notes


@pytest.mark.asyncio
async def test_sub_agent_tool_error_step_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _sub_agent_event(message_type="tool_call", text="Running", tool_description="run_command", tool_call_id="t1")
        )
        app._render_event(
            _sub_agent_event(
                message_type="tool_error", text="boom: exit 1", tool_description="run_command", tool_call_id="t1"
            )
        )

        activity = next(iter(app._sub_agent_activities.values()))
        assert len(activity.steps) == 2  # seeded task step + the paired tool_error
        step = activity.steps[1]
        assert step.kind == "tool_error"
        assert step.complete is True
        assert "boom" in step.full_content
        assert "last: run_command failed" in activity.entry.content


@pytest.mark.asyncio
async def test_sub_agent_tool_steps_without_id_do_not_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Defensive: two separate executions of the same tool, neither carrying a tool_call_id,
    # must produce two distinct paired steps rather than overwriting one shared step.
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(_sub_agent_event(message_type="tool_call", text="grep a", tool_description="grep"))
        app._render_event(_sub_agent_event(message_type="tool_result", text="result a", tool_description="grep"))
        app._render_event(_sub_agent_event(message_type="tool_call", text="grep b", tool_description="grep"))
        app._render_event(_sub_agent_event(message_type="tool_result", text="result b", tool_description="grep"))

        activity = next(iter(app._sub_agent_activities.values()))
        assert len(activity.steps) == 3  # seeded task step + two distinct paired tool steps
        assert [s.kind for s in activity.steps] == ["sub_agent_task", "tool_result", "tool_result"]
        assert activity.steps[1].full_content == "result a"
        assert activity.steps[2].full_content == "result b"


@pytest.mark.asyncio
async def test_sub_agent_stream_without_uuid_merges_by_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Empty uuid on the response chunks: they must merge into one step, not fragment.
        app._render_event(_sub_agent_event(uuid="", text="part one "))
        app._render_event(_sub_agent_event(uuid="", text="part two"))
        # A thinking chunk with an empty uuid must stay a separate step (kind-qualified sentinel).
        app._render_event(_sub_agent_event(uuid="", message_type="thinking", text="a thought"))

        activity = next(iter(app._sub_agent_activities.values()))
        kinds = [s.kind for s in activity.steps]
        assert kinds.count("assistant") == 1
        assert kinds.count("thinking") == 1
        assistant = next(s for s in activity.steps if s.kind == "assistant")
        assert assistant.content == "part one part two"


@pytest.mark.asyncio
async def test_sub_agent_inspector_shows_empty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import Static

    from kolega_code.cli import messages as cli_messages

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        # A task-less lifecycle event: the agent exists but has captured no trajectory steps
        # (and has no task to seed), so the empty-state placeholder still applies.
        app._render_event(_sub_agent_event(task="", status="GENERATING", message="Starting"))

        app.action_open_sub_agent()
        await pilot.pause()
        screen = app._sub_agent_inspector
        assert screen is not None
        assert screen._empty_shown is True
        assert not screen._step_widgets
        view = screen.query_one("#inspector_trajectory")
        placeholder = view.query(Static).first()
        assert cli_messages.SUB_AGENT_INSPECTOR_NO_STEPS in str(placeholder.render())

        # Once a real step arrives, the placeholder is replaced by step widgets.
        app._render_event(_sub_agent_event(uuid="r1", text="now working"))
        screen._flush()
        await pilot.pause()
        assert screen._empty_shown is False
        assert len(screen._step_widgets) == 1


def _workflow_event(message_type, run_id="wf-1", **content):
    return AgentEvent(
        event_type="chat_message",
        sender="gigacode",
        content={
            "message_type": message_type,
            "workflow_run_id": run_id,
            "text": content.pop("text", ""),
            **content,
        },
    )


@pytest.mark.asyncio
async def test_workflow_card_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _workflow_event(
                "workflow_start",
                name="review-changes",
                description="Review then verify",
                phases=[{"title": "Review", "detail": "scan"}, {"title": "Verify"}],
            )
        )
        cards = [e for e in app.conversation_entries if e.kind == "workflow"]
        assert len(cards) == 1
        card = next(iter(app._workflow_activities.values()))
        assert card.name == "review-changes"
        assert [p.title for p in card.phases] == ["Review", "Verify"]
        assert all(p.state == "pending" for p in card.phases)

        # The rich renderable used by the widget builds and prints without markup errors.
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, width=80).print(app._format_workflow_renderable(card))
        rendered = buf.getvalue()
        assert "review-changes" in rendered
        assert "Review" in rendered and "Verify" in rendered

        # A phase event marks it active; a log lands in the footer.
        app._render_event(_workflow_event("workflow_phase", text="Review"))
        assert card.phase_by_title("Review").state == "active"
        app._render_event(_workflow_event("workflow_log", text="grepping"))
        assert card.latest_log == "grepping"

        # Moving to the next phase retires the prior one.
        app._render_event(_workflow_event("workflow_phase", text="Verify"))
        assert card.phase_by_title("Review").state == "done"
        assert card.phase_by_title("Verify").state == "active"

        # End completes the card and any remaining phases.
        app._render_event(_workflow_event("workflow_end", status="completed"))
        assert card.status == "completed"
        assert all(p.state == "done" for p in card.phases)
        assert card.entry.complete is True


@pytest.mark.asyncio
async def test_workflow_card_counts_sub_agents_by_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._render_event(
            _workflow_event("workflow_start", name="wf", description="d", phases=[{"title": "Verify"}])
        )
        card = next(iter(app._workflow_activities.values()))

        # A workflow sub-agent carrying run_id + phase rolls into the card even though no
        # workflow_phase event was emitted (the agent(phase=...) kwarg path).
        evt = _sub_agent_event(agent_id="wf-a1", task="do it", text="working")
        evt.sub_agent_info["workflow_run_id"] = "wf-1"
        evt.sub_agent_info["phase"] = "Verify"
        app._render_event(evt)

        assert card.agent_count == 1
        verify = card.phase_by_title("Verify")
        assert verify.state == "active"
        assert verify.agents_total == 1
        assert verify.agents_done == 0

        # Completion bumps the done count and rolls up tokens.
        done = _sub_agent_event(
            agent_id="wf-a1", task="do it", status="STOPPED", message="Completed", total_tokens=500
        )
        done.sub_agent_info["workflow_run_id"] = "wf-1"
        done.sub_agent_info["phase"] = "Verify"
        app._render_event(done)

        assert verify.agents_done == 1
        assert card.tokens == 500


@pytest.mark.asyncio
async def test_thread_reset_closes_open_inspector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(uuid="u1", text="output"))
        app.action_open_sub_agent()
        await pilot.pause()
        assert app._sub_agent_inspector is not None

        await app._reset_current_thread()
        await pilot.pause()

        assert app._sub_agent_inspector is None


@pytest.mark.asyncio
async def test_sub_agent_inspector_tick_follow_and_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr(app, "copy_to_clipboard", lambda text: copied.append(text))
        monkeypatch.setattr(app, "_notify_user", lambda *a, **k: None)

        # A finished agent (exercises the completed status glyph) + a running one.
        app._render_event(_sub_agent_event(agent_id="done", status="GENERATING", message="Starting"))
        app._render_event(
            _sub_agent_event(
                agent_id="done",
                message_type="tool_call",
                text="Reading",
                tool_description="read_file",
                tool_call_id="t1",
            )
        )
        app._render_event(_sub_agent_event(agent_id="done", status="STOPPED", message="Completed", total_tokens=1500))

        app.action_open_sub_agent("done")
        await pilot.pause()
        screen = app._sub_agent_inspector
        assert screen is not None

        # Spinner/elapsed tick refresh must not raise on running or finished agents.
        screen._on_tick()
        await pilot.pause()

        # Follow toggles.
        assert screen._follow is True
        screen.action_toggle_follow()
        assert screen._follow is False

        # Copy gathers the selected agent's trajectory text.
        screen.action_copy_trajectory()
        assert copied and "read_file" in copied[0]


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
async def test_conversation_body_renders_rich_markup_tokens_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    literal = "I investigated [/dim]\n[bold]not bold[/bold]\npath\\"

    async with app.run_test():
        for entry in [
            ConversationEntry(kind="user", content=literal),
            ConversationEntry(kind="assistant", content=literal, complete=False),
            ConversationEntry(kind="thinking", content=literal, complete=False),
            ConversationEntry(kind="progress", content=literal, complete=True),
            ConversationEntry(kind="question", content=literal),
            ConversationEntry(kind="skill", content=literal),
            ConversationEntry(kind="system", content=literal),
            ConversationEntry(kind="message", content=literal),
        ]:
            rendered = app._format_conversation_entry(entry)
            text = renderable_text(rendered)
            assert "[/dim]" in text
            assert "[bold]not bold[/bold]" in text
            assert "path\\" in text

        app._render_event(
            _sub_agent_event(
                agent_name="agent[/dim]",
                task="inspect [red]task[/red]",
                uuid="u1",
                text="tail [/dim] [bold]literal[/bold]\\",
            )
        )
        sub_agent_entry = _sub_agent_entries(app)[0]
        sub_agent_text = renderable_text(app._format_conversation_entry(sub_agent_entry))
        assert "agent[/dim]" in sub_agent_text
        assert "[red]task[/red]" in sub_agent_text
        assert "[bold]literal[/bold]\\" in sub_agent_text


@pytest.mark.asyncio
async def test_streaming_assistant_refresh_accepts_literal_markup_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        entry = ConversationEntry(kind="assistant", content="start [/dim]", complete=False)
        app.conversation_entries = [entry]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        entry.content += "\n[bold]literal[/bold]\\"
        app._invalidate_conversation(entry)
        app._flush_conversation_render()
        await pilot.pause()

        assert app.query(ConversationEntryWidget).last() is widget
        rendered = renderable_text(widget._formatted)
        assert "[/dim]" in rendered
        assert "[bold]literal[/bold]\\" in rendered


@pytest.mark.asyncio
async def test_conversation_scroll_position_survives_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

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

    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        streaming = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=False)
        )
        assert not isinstance(streaming, str)
        assert "…" in renderable_text(streaming)  # header carries the streaming indicator

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
async def test_confirmations_surface_as_logs_without_toasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        logged: list[tuple[str, str]] = []

        def fake_notify(message, *, severity="information", title=None, **kwargs):
            raise AssertionError("TUI notices should not show transient popups")

        original_log_status = app._log_status

        def spy_log_status(text, level="info"):
            logged.append((text, level))
            original_log_status(text, level)

        monkeypatch.setattr(app, "notify", fake_notify)
        monkeypatch.setattr(app, "_log_status", spy_log_status)

        await app._set_interaction_mode("plan")

        assert ("Switched to plan mode.", "ok") in logged  # diagnostic record kept

        # Blockers are logged as warnings without transient popups.
        app._turn_active = True
        await app.action_toggle_interaction_mode()
        assert ("Stop the current turn before switching modes.", "warn") in logged


@pytest.mark.asyncio
async def test_turn_status_strip_shows_spinner_and_outcome_glyph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import TurnState

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

    from kolega_code.cli.tui.widgets import ToolEntryWidget
    from kolega_code.cli.theme import TOOL_RESULT_PREVIEW_CHARS

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
    from kolega_code.cli import theme

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
        warn = theme.Color.WARNING
        assert f"[{warn}]Context is getting large.[/{warn}]" in dashboard

        app._render_event(context_event("critical"))
        dashboard = app._format_status_dashboard()
        err = theme.Color.ERROR
        assert f"[{err}]Context is getting large.[/{err}]" in dashboard


@pytest.mark.asyncio
async def test_save_settings_logs_on_success_without_toast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        logged: list[tuple[str, str]] = []

        def fake_notify(message, *, severity="information", title=None, **kwargs):
            raise AssertionError("TUI notices should not show transient popups")

        original_log_status = app._log_status

        def spy_log_status(text, level="info"):
            logged.append((text, level))
            original_log_status(text, level)

        monkeypatch.setattr(app, "notify", fake_notify)
        monkeypatch.setattr(app, "_log_status", spy_log_status)

        app.query_one("#api_key_input", Input).value = "moonshot-key"
        await app._save_settings_from_ui()

        assert ("Settings saved.", "ok") in logged
        status_text = str(app.query_one("#settings_status").render())
        assert "Active model:" in status_text


@pytest.mark.asyncio
async def test_planning_sidebar_marks_empty_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.messages import PLAN_EMPTY_MESSAGE

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

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            self.attachments.append(attachments)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer

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
async def test_textual_app_unresolved_mention_clears_hint_and_sends_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("look at @does/not/exist.py")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        # Unresolved mentions are sent as plain text; the compose-time hint must
        # not linger after the message has been submitted.
        hint = app.query_one("#composer_hint", Static)
        assert str(hint.render()) == ""

        await pilot.pause()
        assert app.agent.messages == ["look at @does/not/exist.py"]
        assert app.agent.attachments == [None]


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_opens_filters_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown
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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

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

    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

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

    class FakeCoderAgent(FakeAgent):
        pass

    class FakePlanningAgent(FakeAgent):
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
async def test_textual_app_sidebar_slash_command_toggles_sidebar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        side_panel = app.query_one("#side_panel")

        composer.load_text("/sidebar")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert composer.text == ""
        assert app.sidebar_visible is False
        assert side_panel.display is False

        composer.load_text("/sidebar")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert composer.text == ""
        assert app.sidebar_visible is True
        assert side_panel.display is True


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_starts_agents_md_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/init focus on test commands")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert composer.text == ""
        assert len(app.agent.messages) == 1
        prompt = app.agent.messages[0]
        assert "Create or update `AGENTS.md` for this repository." in prompt
        assert "`focus on test commands`" in prompt
        assert "$ARGUMENTS" not in prompt
        assert any(entry.kind == "user" and entry.content == "/init focus on test commands" for entry in app.conversation_entries)


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_switches_from_plan_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []
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
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

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
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/plan")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)

        composer.load_text("/init focus on docs")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.messages
        assert "`focus on docs`" in app.agent.messages[0]


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_blocks_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/init")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.agent.messages == []
        assert "Stop the current turn before running /init." in str(app.query_one("#composer_hint", Static).render())
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_shows_and_switches_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui import command_handlers as command_handlers_module
    from kolega_code.cli.tui import settings_panel as settings_panel_module
    from kolega_code.cli.tui.widgets import ChatComposer

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
    def fake_model_options(provider):
        return [
            ("Kimi K2.7 Code", UI_DEFAULT_MODEL),
            ("Kimi K2.6", "kimi-k2.6"),
            ("Kimi K3", "kimi-k3"),
        ]

    def fake_effort_options(provider, model):
        return [("High", "high")] if model == "kimi-k3" else [("Auto", "auto")]

    def fake_default_effort(provider, model):
        return "high" if model == "kimi-k3" else "auto"
    for module in (settings_panel_module, command_handlers_module):
        monkeypatch.setattr(module, "ui_model_options", fake_model_options)
        monkeypatch.setattr(module, "ui_thinking_effort_options", fake_effort_options)
        monkeypatch.setattr(module, "default_ui_thinking_effort", fake_default_effort)

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

        from kolega_code.cli.tui.widgets import ActionList

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
        monkeypatch.setattr(agent_runtime_module, "build_agent_config", lambda *args, **kwargs: saved_config)

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

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "unexpected"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "unexpected"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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
    from kolega_code.cli.tui import command_handlers as command_handlers_module
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.updater import UpdateCheckResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    copied: list[str] = []
    monkeypatch.setattr(
        command_handlers_module,
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

    from kolega_code.cli.tui import command_handlers as command_handlers_module
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.updater import UpdateRunResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        command_handlers_module,
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

    from kolega_code.cli.tui.widgets import ChatComposer

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

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/help")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["/help"]


@pytest.mark.asyncio
async def test_textual_app_prompt_list_recovers_focus_after_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shown prompt list must regain keyboard focus if focus drifts away.

    Regression for: after a prompt appears, a background click or a resize could
    leave the option list without focus and the user had no keyboard way back
    (arrow keys / Enter dead). The composer is disabled during an approval, so the
    list is always the only valid focus target here.
    """
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            pass

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
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request, future=future, rule_options=allow_rule_options(request)
        )
        app._set_approval_actions_visible(True)
        app._set_chat_enabled(False)

        approval_actions = app.query_one("#approval_actions", ActionList)
        assert app.focused is approval_actions

        # Focus drifts to the conversation transcript (the AUTO_FOCUS magnet that
        # would otherwise win on resize/resume). The focus hook pulls it back.
        app.screen.set_focus(app.query_one("#conversation"))
        await pilot.pause()
        assert app.focused is approval_actions

        # A background (NoWidget) click does set_focus(None); the blur hook restores.
        app.screen.set_focus(None)
        await pilot.pause()
        assert app.focused is approval_actions

        app._pending_approval = None
        app._set_approval_actions_visible(False)


@pytest.mark.asyncio
async def test_textual_app_question_recovers_focus_but_allows_free_form_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During a question the option list self-heals, but a deliberate move to the
    enabled composer (to type a free-form answer) must NOT be fought."""
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            pass

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
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        question_actions = app.query_one("#question_actions", ActionList)
        assert app.focused is question_actions

        # Drift to the transcript is pulled back to the option list.
        app.screen.set_focus(app.query_one("#conversation"))
        await pilot.pause()
        assert app.focused is question_actions

        # A deliberate move to the ENABLED composer is preserved (free-form answer).
        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled is False
        app.screen.set_focus(composer)
        await pilot.pause()
        assert app.focused is composer

        app._pending_question = None
        app._set_question_actions_visible(False)


@pytest.mark.asyncio
async def test_agent_models_section_saves_override_and_builds_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input, Select

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

    async with app.run_test() as pilot:
        app.query_one("#api_key_input", Input).value = "moonshot-key"
        # Give the investigation role its own model (same provider keeps one API key).
        app.query_one("#am_provider_investigation", Select).value = UI_DEFAULT_PROVIDER
        await pilot.pause()  # let the provider->model cascade settle
        app.query_one("#am_model_investigation", Select).value = MOONSHOT_K26_MODEL
        await pilot.pause()
        await app._save_settings_from_ui()

        saved = settings_store.load().get_agent_model("investigation")
        assert saved is not None
        assert saved["provider"] == UI_DEFAULT_PROVIDER
        assert saved["model"] == MOONSHOT_K26_MODEL

        config = app.agent.kwargs["config"]
        assert config.model_config_for_agent("investigation-agent").model == MOONSHOT_K26_MODEL
        # Roles left on "Default" still inherit the active model.
        assert config.model_config_for_agent("coder").model == UI_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_agent_models_section_populates_and_clears_to_inherit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Select

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.provider_registry import INHERIT_SENTINEL

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
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")
    settings.set_agent_model("investigation", UI_DEFAULT_PROVIDER, MOONSHOT_K26_MODEL)
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
        # The saved override is reflected in the row on mount.
        assert app.query_one("#am_provider_investigation", Select).value == UI_DEFAULT_PROVIDER
        assert app.query_one("#am_model_investigation", Select).value == MOONSHOT_K26_MODEL

        # Switching the row back to "Default" clears the override on save.
        app.query_one("#am_provider_investigation", Select).value = INHERIT_SENTINEL
        await pilot.pause()
        await app._save_settings_from_ui()

        assert settings_store.load().get_agent_model("investigation") is None
