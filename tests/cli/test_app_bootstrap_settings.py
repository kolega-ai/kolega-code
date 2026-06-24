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
async def test_textual_app_mounts_with_fake_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Collapsible, Header

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import PlanningMarkdown

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
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "No plan captured yet."
        assert app.query_one("#planning_task_list_markdown", PlanningMarkdown).source == "No task list has been set."
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
async def test_textual_app_status_tab_is_default_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        assert dashboard_widget.styles.border == app.query_one("#terminal").styles.border
        assert list(app.query("#logs")) == []
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
async def test_web_search_settings_section_reveal_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled is True
        assert composer.placeholder == "Connect a model in Settings before chatting."
        assert app.query_one("#events").active == "status_pane"
        startup = app.conversation_entries[0].content
        assert "Not connected." in startup
        assert "Choose a provider and add an API key or sign in from the Settings tab before chatting." in startup
        assert "Press Ctrl+O to open the sidebar, then select Settings." in startup
        assert "Model: not configured" in startup
        assert "API key: not checked until a model is configured" in startup
        dashboard = str(app.query_one("#status_dashboard").render())
        assert "not connected" in dashboard
        assert "Open Settings and connect a provider to start chatting." in dashboard
        status = str(app.query_one("#settings_status").render())
        assert "Configuration incomplete" in status
        assert "No provider/model configured" in status
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "fix this"))
        assert composer.placeholder == "Connect a model in Settings before chatting."
        assert "Open Settings and connect a provider to start chatting." in str(
            app.query_one("#composer_hint").render()
        )
        stored_settings = settings_store.load()
        assert stored_settings.active_provider is None
        assert stored_settings.active_model is None


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
        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled is True
        assert composer.placeholder == "Connect a model in Settings before chatting."
        startup = app.conversation_entries[0].content
        assert "Not connected." in startup
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
        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled is False
        assert composer.placeholder == "Ask Kolega Code..."
        assert "Not connected." not in app.conversation_entries[0].content


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
