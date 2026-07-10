# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

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


class _GuardCoderAgent(FakeCoderAgent):
    """Fake whose construction raises; asserts the app must not build an agent."""

    def __init__(self, **kwargs):
        raise AssertionError("agent should not be built")


class _PromptOverrideCoderAgent(FakeCoderAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_override_errors = [
            "Could not render prompt override .kolega/prompts/CODER.md: boom. Falling back to the default prompt."
        ]


@pytest.mark.asyncio
async def test_textual_app_mounts_with_fake_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import Header

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import PlanningMarkdown

    install_fake_agents(monkeypatch)

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
        assert app.query_one("#planning_plan", Vertical) is not None
        assert app.query_one("#status_form", VerticalScroll) is not None
        assert app.query_one("#status_summary_section", Vertical) is not None
        assert app.query_one("#status_task_list_section", Vertical) is not None
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "No plan captured yet."
        assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == "No task list has been set."
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
async def test_textual_app_startup_shows_prompt_overrides_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, coder_cls=_PromptOverrideCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    prompt_dir = project / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("custom", encoding="utf-8")
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        startup = app.conversation_entries[0].content
        assert "Prompt overrides: CODER.md" in startup
        assert "Prompt override errors:" in startup
        assert "Could not render prompt override .kolega/prompts/CODER.md" in startup
        assert "Falling back to the default prompt" in startup


@pytest.mark.asyncio
async def test_textual_app_status_tab_is_default_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.containers import Vertical
    from textual.widgets import Static, TabbedContent

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

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

        assert dashboard.lstrip().startswith("Model")
        assert f"{config.long_context_config.provider.value}/{config.long_context_config.model}" in dashboard
        assert "Build" in dashboard
        assert "Idle" in dashboard
        assert "Waiting for first context count" in dashboard
        assert str(dashboard_widget.styles.border) == "Edges()"
        assert str(app.query_one("#status_summary_section", Vertical).styles.border) != "Edges()"
        assert str(app.query_one("#status_task_list_section", Vertical).styles.border) != "Edges()"
        assert str(app.query_one("#terminal").styles.border) == "Edges()"
        assert list(app.query("#logs")) == []
        assert list(app.query("#status")) == []


@pytest.mark.asyncio
async def test_settings_tab_grouped_into_model_and_appearance_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        # Bordered, titled section.
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

    install_fake_agents(monkeypatch)

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
    from textual.widgets import TabbedContent

    install_fake_agents(monkeypatch, coder_cls=_GuardCoderAgent)

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
        assert app.query_one("#events", TabbedContent).active == "status_pane"
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

    install_fake_agents(monkeypatch, coder_cls=_GuardCoderAgent)
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")

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
async def test_textual_app_ignores_project_dotenv_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch, coder_cls=_GuardCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("OPENAI_API_KEY=project-openai-key\n", encoding="utf-8")
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
        startup = app.conversation_entries[0].content
        assert "Not connected." in startup
        assert "Model: not configured" in startup
        assert "openai" not in startup.lower()


@pytest.mark.asyncio
async def test_textual_app_mounts_with_stored_kimi_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

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
async def test_textual_app_shows_process_env_model_override_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "process-openai-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", ModelProvider.OPENAI.value)
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "gpt-5.5")

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
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.OPENAI
        startup = app.conversation_entries[0].content
        assert "Model: openai/gpt-5.5" in startup
        assert (
            "Environment/CLI override active: using openai/gpt-5.5 instead of saved "
            f"{UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}."
        ) in startup
        assert "API key: present via OPENAI_API_KEY" in startup
        status = str(app.query_one("#settings_status", Static).render())
        assert "Environment/CLI override active" in status


@pytest.mark.asyncio
async def test_textual_app_mounts_with_stored_deepseek_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

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

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("OPENAI_API_KEY=project-openai-key\n", encoding="utf-8")
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
        assert app.agent.kwargs["config"].openai_api_key is None
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

    install_fake_agents(monkeypatch)

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
async def test_mcp_server_save_auto_generates_id_without_visible_id_field(tmp_path: Path) -> None:
    pytest.importorskip("textual")

    from textual.css.query import NoMatches
    from textual.widgets import Input, Select

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.mcp.config import global_mcp_config_path

    class NoAgentApp(KolegaCodeApp):
        CSS_PATH = str(Path(__file__).parents[2] / "kolega_code/cli/tui/styles.tcss")

        async def _ensure_agent_from_settings(self, *args, **kwargs):
            return None

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    session = store.create(project, "code", {})
    app = NoAgentApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        with pytest.raises(NoMatches):
            app.query_one("#mcp_server_id_input", Input)

        app.query_one("#mcp_name_input", Input).value = "DeepWiki Docs"
        app.query_one("#mcp_transport_select", Select).value = "streamable_http"
        app.query_one("#mcp_url_input", Input).value = "https://mcp.deepwiki.com/mcp"

        await app._save_mcp_server_from_ui()

        assert app.query_one("#mcp_server_select", Select).value == "deepwiki-docs"
        payload = json.loads(global_mcp_config_path(state_dir).read_text(encoding="utf-8"))
        assert len(payload["servers"]) == 1
        server = payload["servers"][0]
        assert server["id"] == "deepwiki-docs"
        assert server["name"] == "DeepWiki Docs"
        assert server["transport"] == "streamable_http"
        assert server["url"] == "https://mcp.deepwiki.com/mcp"
        assert server["enabled"] is True


@pytest.mark.asyncio
async def test_agent_models_section_saves_override_and_builds_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input, Select

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

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

        assert app.agent is not None
        config = getattr(app.agent, "kwargs")["config"]
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

    install_fake_agents(monkeypatch)

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


@pytest.mark.asyncio
async def test_browser_model_settings_preserve_and_reject_nonvision_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Select, Static

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings.set_agent_model("browser", ModelProvider.DEEPSEEK.value, DEEPSEEK_DEFAULT_MODEL)
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
        assert app.query_one("#am_provider_browser", Select).value == ModelProvider.DEEPSEEK.value
        assert app.query_one("#am_model_browser", Select).value == DEEPSEEK_DEFAULT_MODEL
        assert "does not support vision" in str(app.query_one("#am_status_browser", Static).render())

        await app._save_settings_from_ui()

        saved = settings_store.load().get_agent_model("browser")
        assert saved is not None
        assert saved["provider"] == ModelProvider.DEEPSEEK.value
        assert saved["model"] == DEEPSEEK_DEFAULT_MODEL
        assert "does not support vision" in str(app.query_one("#settings_status", Static).render())


@pytest.mark.asyncio
async def test_browser_model_settings_allow_nonvision_inheritance_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Select, Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.provider_registry import INHERIT_SENTINEL

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
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
        assert app.query_one("#am_provider_browser", Select).value == INHERIT_SENTINEL
        hint = str(app.query_one("#am_status_browser", Static).render())
        assert "Browser agent is unavailable" in hint
        assert "does not support vision" in hint

        await app._save_settings_from_ui()

        assert settings_store.load().get_agent_model("browser") is None
