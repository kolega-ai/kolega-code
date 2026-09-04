from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import DEEPSEEK_DEFAULT_MODEL, UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import install_fake_agents, stage_provider_api_key, wait_for_onboarding_screen


def _configured_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str = UI_DEFAULT_PROVIDER,
    model: str = UI_DEFAULT_MODEL,
    effort: str | None = None,
    agent_models: dict | None = None,
    model_slots: dict | None = None,
    extra_key_providers: tuple[str, ...] = (),
    custom_endpoints: dict | None = None,
    env: dict | None = None,
):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=provider, active_model=model, active_thinking_effort=effort)
    settings.set_api_key(provider, "stored-key")
    for extra_provider in extra_key_providers:
        settings.set_api_key(extra_provider, "stored-key")
    if agent_models:
        settings.agent_models = agent_models
    if model_slots:
        settings.model_slots = model_slots
    if custom_endpoints:
        settings.custom_endpoints = custom_endpoints
    settings_store.save(settings)
    config = build_agent_config(project, env=env or {}, settings=settings, settings_store=settings_store)
    store = SessionStore(state_dir)
    session = store.create(project, "code", config_summary(config))
    return (
        KolegaCodeApp(
            project_path=project,
            config=config,
            mode="code",
            store=store,
            settings_store=settings_store,
            session=session,
        ),
        settings_store,
    )


def _non_featured_openrouter_model() -> str:
    """A catalogued OpenRouter id outside the featured picker set, with efforts."""
    from kolega_code.llm.specs import MODEL_SPECS, is_featured_model, thinking_effort_options

    return next(
        model
        for provider, model in MODEL_SPECS
        if provider == "openrouter"
        and not is_featured_model("openrouter", model)
        and thinking_effort_options("openrouter", model)
    )


def _non_vision_openrouter_model() -> str:
    """A catalogued OpenRouter id that cannot accept image input."""
    from kolega_code.llm.specs import MODEL_SPECS, supports_vision

    return next(
        model
        for provider, model in MODEL_SPECS
        if provider == "openrouter" and not supports_vision("openrouter", model)
    )


@pytest.mark.asyncio
async def test_settings_screen_is_categorized_and_stages_credentials_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Input, OptionList, Select

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await pilot.pause()
        assert screen.dirty is False
        assert screen.query_one("#settings_categories", OptionList).option_count == 9
        assert screen.query_one("#settings_page_model").display is True
        assert screen.query_one("#settings_page_tools").display is False
        apply_button = screen.query_one("#save_settings", Button)
        assert apply_button.region.y + apply_button.region.height <= 24

        screen._show_category("tools")
        assert screen.query_one("#settings_page_model").display is False
        assert screen.query_one("#settings_page_tools").display is True

        # Credentials live on their own page, opened on the active provider's row.
        screen._show_category("providers")
        await pilot.pause()
        assert screen.credential_provider == UI_DEFAULT_PROVIDER
        remove_button = screen.query_one("#provider_remove_api_key", Button)
        assert remove_button.disabled is False
        app._settings_remove_api_key()
        await pilot.pause()
        assert screen.dirty is True
        assert UI_DEFAULT_PROVIDER in screen.pending_api_key_removals

        await app._save_settings_from_ui()
        assert settings_store.load().get_api_key(UI_DEFAULT_PROVIDER) == "stored-key"
        assert screen.dirty is True
        assert "Configuration incomplete" in str(screen.query_one("#settings_status").render())

        await stage_provider_api_key(screen, pilot, UI_DEFAULT_PROVIDER, "replacement-key")
        assert UI_DEFAULT_PROVIDER not in screen.pending_api_key_removals
        await app._save_settings_from_ui()

        assert settings_store.load().get_api_key(UI_DEFAULT_PROVIDER) == "replacement-key"
        assert screen.query_one("#provider_api_key_input", Input).value == ""
        assert screen.dirty is False
        assert apply_button.disabled is True

        # Decoupled: the Models provider picker no longer touches the key field.
        await stage_provider_api_key(screen, pilot, UI_DEFAULT_PROVIDER, "still-here")
        screen.query_one("#provider_select", Select).value = "deepseek"
        await pilot.pause()
        assert screen.query_one("#provider_api_key_input", Input).value == "still-here"
        assert screen.pending_api_keys[UI_DEFAULT_PROVIDER] == "still-here"


async def _wait_for_select_values(pilot, screen, expected: dict[str, str], *, attempts: int = 50) -> None:
    """Poll until the given Select widgets hold the expected values (or give up).

    The provider→model→effort cascade settles over several message passes, so a
    single ``pilot.pause()`` can race on loaded runners.
    """
    from textual.widgets import Select

    for _ in range(attempts):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if all(str(screen.query_one(f"#{widget_id}", Select).value) == value for widget_id, value in expected.items()):
            return


@pytest.mark.asyncio
async def test_settings_screen_retains_active_model_and_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    """Regression: stale mount-time Select.Changed events must not clobber the draft."""
    pytest.importorskip("textual")
    from textual.widgets import Select

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, _ = _configured_app(
        tmp_path,
        monkeypatch,
        provider="ollama_cloud",
        model="gemma4:31b",
        effort="high",
    )

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await _wait_for_select_values(
            pilot,
            screen,
            {
                "provider_select": "ollama_cloud",
                "model_select": "gemma4:31b",
                "thinking_effort_select": "high",
            },
        )
        provider = screen.query_one("#provider_select", Select)
        model = screen.query_one("#model_select", Select)
        effort = screen.query_one("#thinking_effort_select", Select)
        assert str(provider.value) == "ollama_cloud"
        assert str(model.value) == "gemma4:31b"
        assert str(effort.value) == "high"
        assert screen.dirty is False

        # A genuine provider change still cascades: model falls back to the new
        # provider's first model, then a manual model pick is retained.
        provider.value = "deepseek"
        await _wait_for_select_values(pilot, screen, {"model_select": "deepseek-v4-pro"})
        assert str(model.value) == "deepseek-v4-pro"

        model.value = "deepseek-v4-flash"
        await _wait_for_select_values(pilot, screen, {"model_select": "deepseek-v4-flash"})
        assert str(model.value) == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_settings_layout_uses_uniform_controls_and_quiet_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.containers import Vertical
    from textual.widgets import Button, Select, Static, TabbedContent

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, _ = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(140, 40)) as pilot:
        app.query_one("#events", TabbedContent).active = "settings_pane"
        await pilot.pause()
        open_settings = app.query_one("#open_settings", Button)
        assert open_settings.has_class("quiet")
        assert open_settings.styles.background.a == 0
        summary_section = app.query_one("#settings_summary_section", Vertical)
        assert open_settings.region.width < summary_section.region.width

        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        # One uniform width for every form control, aligned to one edge.
        provider = screen.query_one("#provider_select", Select)
        model = screen.query_one("#model_select", Select)
        effort = screen.query_one("#thinking_effort_select", Select)
        assert provider.region.width == model.region.width == effort.region.width == 46
        assert provider.region.x == model.region.x == effort.region.x

        # The Providers page shares the same control width.
        screen._show_category("providers")
        await pilot.pause()
        api_key = screen.query_one("#provider_api_key_input")
        assert api_key.region.width == 46
        screen._show_category("model")
        await pilot.pause()

        # Quiet secondary vs one solid primary action in the footer band.
        close_button = screen.query_one("#close_settings", Button)
        assert close_button.has_class("quiet")
        assert close_button.styles.background.a == 0
        apply_button = screen.query_one("#save_settings", Button)
        assert apply_button.has_class("solid-primary")
        assert apply_button.styles.background.a == 1
        # The footer is a single band: status text sits beside the buttons,
        # and the esc hint lives in the header.
        status = screen.query_one("#settings_status", Static)
        assert status.region.y < apply_button.region.y + apply_button.region.height
        assert apply_button.region.y < status.region.y + status.region.height
        hint = screen.query_one("#settings_screen_hint", Static)
        assert hint.region.y < screen.query_one("#settings_screen_body").region.y

        # Agent-model rows align to the same right edge as the stacked controls.
        screen._show_category("agents")
        await pilot.pause()
        role_select = screen.query_one("#agent_role_select", Select)
        row_model = screen.query_one("#am_model_planning", Select)
        assert row_model.region.x + row_model.region.width == role_select.region.x + role_select.region.width

        screen._show_category("mcp")
        await pilot.pause()
        server = screen.query_one("#mcp_server_select", Select)
        enabled = screen.query_one("#mcp_enabled_select", Select)
        assert server.region.width == 46
        assert enabled.region.width == 46

        reload_button = screen.query_one("#mcp_refresh", Button)
        trust_button = screen.query_one("#mcp_trust_project", Button)
        assert reload_button.region.x + reload_button.region.width < trust_button.region.x
        delete_button = screen.query_one("#mcp_delete_server", Button)
        assert delete_button.has_class("danger")
        assert list(screen.query("#mcp_enable_server")) == []
        assert list(screen.query("#mcp_disable_server")) == []


@pytest.mark.asyncio
async def test_first_run_onboarding_finishes_without_writing_partial_drafts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

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
        startup_config_error="No provider/model configured.",
    )

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await wait_for_onboarding_screen(app, pilot)
        assert screen.step_index == 0
        next_button = screen.query_one("#onboarding_next")
        assert next_button.region.y + next_button.region.height <= 24
        assert "No provider/model configured" in str(screen.query_one("#onboarding_status").render())
        assert settings_store.load().active_provider is None

        await screen._continue()
        assert screen.step_index == 1
        screen.query_one("#onboarding_api_key", Input).value = "new-key"
        await screen._continue()
        assert screen.step_index == 2
        assert settings_store.load().active_provider is None

        await screen._continue()
        assert screen.step_index == 3
        await screen._finish()
        await pilot.pause()

        saved = settings_store.load()
        assert saved.active_provider == UI_DEFAULT_PROVIDER
        assert saved.active_model == UI_DEFAULT_MODEL
        assert saved.get_api_key(UI_DEFAULT_PROVIDER) == "new-key"
        assert app.config is not None
        assert app.agent is not None
        assert app._onboarding_screen is None


@pytest.mark.asyncio
async def test_first_run_onboarding_skip_is_session_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")

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
        screen = await wait_for_onboarding_screen(app, pilot)
        screen.action_skip()
        await pilot.pause()
        assert app._onboarding_skipped is True
        assert app._onboarding_screen is None
        assert app.config is None
        assert settings_store.load() == CliSettings()


@pytest.mark.asyncio
async def test_onboarding_actions_stay_on_screen_at_small_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    """The wizard frame is fixed: step content scrolls, the action row never clips."""
    pytest.importorskip("textual")
    from textual.widgets import Button

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
        startup_config_error="No provider/model configured.",
    )

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await wait_for_onboarding_screen(app, pilot)
        next_button = screen.query_one("#onboarding_next", Button)
        assert next_button.has_class("solid-primary")
        assert screen.query_one("#onboarding_skip", Button).has_class("quiet")
        for step in range(4):
            screen._show_step(step)
            await pilot.pause()
            region = next_button.region
            assert region.height > 0, f"step {step}: Continue button not laid out"
            assert region.y + region.height <= 24, f"step {step}: Continue button clipped"


# --- "Other…" custom model picker ----------------------------------------


@pytest.mark.asyncio
async def test_settings_other_model_restores_a_saved_non_featured_openrouter_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    """A saved catalogued-but-non-featured model opens as Other… + typed id."""
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    custom_model_id = _non_featured_openrouter_model()
    app, _ = _configured_app(tmp_path, monkeypatch, provider="openrouter", model=custom_model_id)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await _wait_for_select_values(pilot, screen, {"model_select": CUSTOM_MODEL_SENTINEL})

        custom_input = screen.query_one("#model_custom_input", Input)
        assert custom_input.display is True
        assert custom_input.value == custom_model_id
        assert screen.dirty is False
        # Effort options come from the typed id, never the sentinel.
        from kolega_code.cli.provider_registry import ui_thinking_effort_options

        effort = screen.query_one("#thinking_effort_select", Select)
        valid_efforts = {value for _label, value in ui_thinking_effort_options("openrouter", custom_model_id)}
        assert valid_efforts
        assert str(effort.value) in valid_efforts


@pytest.mark.asyncio
async def test_settings_other_model_picks_and_applies_a_typed_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    custom_model_id = _non_featured_openrouter_model()
    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        model_select = screen.query_one("#model_select", Select)
        model_select.value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"model_select": CUSTOM_MODEL_SENTINEL})

        custom_input = screen.query_one("#model_custom_input", Input)
        assert custom_input.display is True
        await pilot.pause()
        assert app.focused is custom_input

        custom_input.value = custom_model_id
        await pilot.pause()
        await app._save_settings_from_ui()

        assert settings_store.load().active_model == custom_model_id
        # The picker keeps showing Other… with the typed id.
        assert screen.query_one("#model_select", Select).value == CUSTOM_MODEL_SENTINEL
        assert screen.query_one("#model_custom_input", Input).value == custom_model_id


@pytest.mark.asyncio
async def test_settings_other_model_rejects_an_unknown_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select, Static

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#model_select", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"model_select": CUSTOM_MODEL_SENTINEL})
        screen.query_one("#model_custom_input", Input).value = "vendor/not-real"
        await pilot.pause()
        await app._save_settings_from_ui()

        assert settings_store.load().active_model != "vendor/not-real"
        status = str(screen.query_one("#settings_status", Static).render())
        assert "vendor/not-real" in status


@pytest.mark.asyncio
async def test_settings_other_model_requires_a_typed_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select, Static

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")
    original_model = settings_store.load().active_model

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#model_select", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"model_select": CUSTOM_MODEL_SENTINEL})
        await app._save_settings_from_ui()

        assert settings_store.load().active_model == original_model
        status = str(screen.query_one("#settings_status", Static).render())
        assert "Type a model id" in status


@pytest.mark.asyncio
async def test_settings_other_model_is_cleared_by_a_provider_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    custom_model_id = _non_featured_openrouter_model()
    app, _ = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#model_select", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"model_select": CUSTOM_MODEL_SENTINEL})
        custom_input = screen.query_one("#model_custom_input", Input)
        custom_input.value = custom_model_id
        await pilot.pause()

        screen.query_one("#provider_select", Select).value = "anthropic"
        await _wait_for_select_values(pilot, screen, {"model_select": "claude-fable-5"})
        assert custom_input.display is False
        assert custom_input.value == ""


@pytest.mark.asyncio
async def test_agent_row_other_model_restores_a_saved_non_featured_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    custom_model_id = _non_featured_openrouter_model()
    agent_models = {"planning": {"provider": "openrouter", "model": custom_model_id}}
    app, _ = _configured_app(tmp_path, monkeypatch, provider="openrouter", agent_models=agent_models)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await _wait_for_select_values(
            pilot,
            screen,
            {"am_provider_planning": "openrouter", "am_model_planning": CUSTOM_MODEL_SENTINEL},
        )

        row_input = screen.query_one("#am_custom_model_planning", Input)
        assert row_input.display is True
        assert row_input.value == custom_model_id
        assert screen.dirty is False


@pytest.mark.asyncio
async def test_agent_row_other_model_picks_and_applies_a_typed_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    custom_model_id = _non_featured_openrouter_model()
    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#am_provider_planning", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"am_provider_planning": "openrouter"})
        screen.query_one("#am_model_planning", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"am_model_planning": CUSTOM_MODEL_SENTINEL})

        row_input = screen.query_one("#am_custom_model_planning", Input)
        assert row_input.display is True
        row_input.value = custom_model_id
        await pilot.pause()
        await app._save_settings_from_ui()

        saved = settings_store.load().get_agent_model("planning")
        assert saved is not None
        assert saved["provider"] == "openrouter"
        assert saved["model"] == custom_model_id


@pytest.mark.asyncio
async def test_agent_row_other_model_rejects_an_unknown_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select, Static

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#am_provider_planning", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"am_provider_planning": "openrouter"})
        screen.query_one("#am_model_planning", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"am_model_planning": CUSTOM_MODEL_SENTINEL})
        screen.query_one("#am_custom_model_planning", Input).value = "vendor/not-real"
        await pilot.pause()
        await app._save_settings_from_ui()

        assert settings_store.load().get_agent_model("planning") is None
        status = str(screen.query_one("#settings_status", Static).render())
        assert "vendor/not-real" in status


@pytest.mark.asyncio
async def test_agent_row_other_model_requires_a_typed_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select, Static

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#am_provider_planning", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"am_provider_planning": "openrouter"})
        screen.query_one("#am_model_planning", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"am_model_planning": CUSTOM_MODEL_SENTINEL})
        await app._save_settings_from_ui()

        assert settings_store.load().get_agent_model("planning") is None
        status = str(screen.query_one("#settings_status", Static).render())
        assert "Type a model id" in status


@pytest.mark.asyncio
async def test_agent_row_other_model_is_cleared_when_row_returns_to_inherit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.provider_registry import INHERIT_SENTINEL
    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    custom_model_id = _non_featured_openrouter_model()
    agent_models = {"planning": {"provider": "openrouter", "model": custom_model_id}}
    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter", agent_models=agent_models)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await _wait_for_select_values(pilot, screen, {"am_model_planning": CUSTOM_MODEL_SENTINEL})
        assert screen.query_one("#am_custom_model_planning", Input).display is True

        screen.query_one("#am_provider_planning", Select).value = INHERIT_SENTINEL
        await pilot.pause()
        await app._save_settings_from_ui()

        assert settings_store.load().get_agent_model("planning") is None
        assert screen.query_one("#am_custom_model_planning", Input).display is False
        assert screen.query_one("#am_custom_model_planning", Input).value == ""


@pytest.mark.asyncio
async def test_agent_row_other_model_keeps_the_browser_vision_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    """A typed non-vision id in the Browser row blocks save, like the listed path."""
    pytest.importorskip("textual")
    from textual.widgets import Input, Select, Static

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    non_vision = _non_vision_openrouter_model()
    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#am_provider_browser", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"am_provider_browser": "openrouter"})
        screen.query_one("#am_model_browser", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"am_model_browser": CUSTOM_MODEL_SENTINEL})
        screen.query_one("#am_custom_model_browser", Input).value = non_vision
        await pilot.pause()
        await app._save_settings_from_ui()

        assert settings_store.load().get_agent_model("browser") is None
        status = str(screen.query_one("#settings_status", Static).render())
        assert "does not support vision" in status


@pytest.mark.asyncio
async def test_agent_row_other_model_accepts_a_vision_capable_browser_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.llm.specs import MODEL_SPECS, supports_vision

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    vision_id = next(
        model for provider, model in MODEL_SPECS if provider == "openrouter" and supports_vision("openrouter", model)
    )
    app, settings_store = _configured_app(tmp_path, monkeypatch, provider="openrouter")

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings("agents")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.query_one("#am_provider_browser", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"am_provider_browser": "openrouter"})
        screen.query_one("#am_model_browser", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"am_model_browser": CUSTOM_MODEL_SENTINEL})
        screen.query_one("#am_custom_model_browser", Input).value = vision_id
        await pilot.pause()
        await app._save_settings_from_ui()

        saved = settings_store.load().get_agent_model("browser")
        assert saved is not None
        assert saved["provider"] == "openrouter"
        assert saved["model"] == vision_id


@pytest.mark.asyncio
async def test_model_slot_rows_default_to_inherit_and_name_the_active_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select, Static

    from kolega_code.cli.provider_registry import INHERIT_SENTINEL

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        await _wait_for_select_values(pilot, screen, {"slot_provider_fast": INHERIT_SENTINEL})

        assert str(screen.query_one("#slot_provider_fast", Select).value) == INHERIT_SENTINEL
        assert screen.query_one("#slot_model_fast", Select).value is Select.NULL
        hint = str(screen.query_one("#slot_hint_fast", Static).render())
        assert f"{UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in hint
        assert "inherited from the active model" in hint

        # Inheriting is the absence of an override, not a stored value.
        await app._save_settings_from_ui()
        assert settings_store.load().model_slots == {}


@pytest.mark.asyncio
async def test_model_slot_row_pins_a_model_on_another_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select, Static

    app, settings_store = _configured_app(tmp_path, monkeypatch, extra_key_providers=("deepseek",))

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        screen.query_one("#slot_provider_fast", Select).value = "deepseek"
        await _wait_for_select_values(pilot, screen, {"slot_provider_fast": "deepseek"})
        screen.query_one("#slot_model_fast", Select).value = "deepseek-v4-flash"
        await _wait_for_select_values(pilot, screen, {"slot_model_fast": "deepseek-v4-flash"})

        hint = str(screen.query_one("#slot_hint_fast", Static).render())
        assert "deepseek/deepseek-v4-flash" in hint
        assert "inherited" not in hint

        await app._save_settings_from_ui()

        assert settings_store.load().get_model_slot("fast") == {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
        }
        # The fast slot now diverges from the main model it used to shadow.
        assert app.config is not None
        assert app.config.fast_config.model == "deepseek-v4-flash"
        assert app.config.long_context_config.model == UI_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_model_slot_row_returning_to_inherit_clears_the_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select, Static

    from kolega_code.cli.provider_registry import INHERIT_SENTINEL

    app, settings_store = _configured_app(
        tmp_path,
        monkeypatch,
        model_slots={"fast": {"provider": "deepseek", "model": "deepseek-v4-flash"}},
        extra_key_providers=("deepseek",),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        # A saved slot is restored into its row.
        await _wait_for_select_values(
            pilot, screen, {"slot_provider_fast": "deepseek", "slot_model_fast": "deepseek-v4-flash"}
        )

        screen.query_one("#slot_provider_fast", Select).value = INHERIT_SENTINEL
        await _wait_for_select_values(pilot, screen, {"slot_provider_fast": INHERIT_SENTINEL})
        await app._save_settings_from_ui()

        assert settings_store.load().get_model_slot("fast") is None
        assert app.config is not None
        assert app.config.fast_config.model == UI_DEFAULT_MODEL
        hint = str(screen.query_one("#slot_hint_fast", Static).render())
        assert "inherited from the active model" in hint


@pytest.mark.asyncio
async def test_model_slot_row_other_model_applies_a_typed_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL

    custom_model_id = _non_featured_openrouter_model()
    app, settings_store = _configured_app(tmp_path, monkeypatch, extra_key_providers=("openrouter",))

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        screen.query_one("#slot_provider_fast", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"slot_provider_fast": "openrouter"})
        screen.query_one("#slot_model_fast", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"slot_model_fast": CUSTOM_MODEL_SENTINEL})

        custom_input = screen.query_one("#slot_custom_model_fast", Input)
        assert custom_input.display is True
        custom_input.value = custom_model_id
        await pilot.pause()
        await app._save_settings_from_ui()

        # The sentinel never persists; the typed id does.
        assert settings_store.load().get_model_slot("fast") == {
            "provider": "openrouter",
            "model": custom_model_id,
        }


@pytest.mark.asyncio
async def test_model_slot_row_rejects_an_unknown_typed_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL

    app, settings_store = _configured_app(tmp_path, monkeypatch, extra_key_providers=("openrouter",))

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        screen.query_one("#slot_provider_fast", Select).value = "openrouter"
        await _wait_for_select_values(pilot, screen, {"slot_provider_fast": "openrouter"})
        screen.query_one("#slot_model_fast", Select).value = CUSTOM_MODEL_SENTINEL
        await _wait_for_select_values(pilot, screen, {"slot_model_fast": CUSTOM_MODEL_SENTINEL})
        screen.query_one("#slot_custom_model_fast", Input).value = "not-a-real-model"
        await pilot.pause()

        await app._save_settings_from_ui()

        assert settings_store.load().model_slots == {}
        assert "not-a-real-model" in str(screen.query_one("#settings_status").render())


@pytest.mark.asyncio
async def test_model_slot_row_without_an_api_key_blocks_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select

    # No stored google key: the slot's provider needs its own credential.
    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(80, 24)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        screen.query_one("#slot_provider_fast", Select).value = "google"
        await _wait_for_select_values(pilot, screen, {"slot_provider_fast": "google"})

        await app._save_settings_from_ui()

        assert settings_store.load().model_slots == {}
        assert "GOOGLE_API_KEY" in str(screen.query_one("#settings_status").render())


@pytest.mark.asyncio
async def test_providers_page_stages_keys_for_several_providers_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    """The case that was impossible before: keys for providers you are not using."""
    pytest.importorskip("textual")

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings("providers")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        await stage_provider_api_key(screen, pilot, "deepseek", "deepseek-key")
        await stage_provider_api_key(screen, pilot, "google", "google-key")
        assert screen.dirty is True

        await app._save_settings_from_ui()

        saved = settings_store.load()
        assert saved.get_api_key("deepseek") == "deepseek-key"
        assert saved.get_api_key("google") == "google-key"
        # Neither was misfiled against the active model's provider.
        assert saved.get_api_key(UI_DEFAULT_PROVIDER) == "stored-key"
        assert saved.active_provider == UI_DEFAULT_PROVIDER
        assert screen.pending_api_keys == {}


@pytest.mark.asyncio
async def test_staged_key_survives_moving_the_selection_away_and_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, OptionList

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, _ = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings("providers")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        provider_list = screen.query_one("#provider_list", OptionList)

        await stage_provider_api_key(screen, pilot, "deepseek", "deepseek-key")
        # Move away: the field shows the other provider's (empty) staging...
        provider_list.highlighted = provider_list.get_option_index("provider_row_google")
        await pilot.pause()
        assert screen.query_one("#provider_api_key_input", Input).value == ""
        # ...and coming back restores what was typed rather than losing it.
        provider_list.highlighted = provider_list.get_option_index("provider_row_deepseek")
        await pilot.pause()
        assert screen.query_one("#provider_api_key_input", Input).value == "deepseek-key"
        assert screen.pending_api_keys == {"deepseek": "deepseek-key"}


@pytest.mark.asyncio
async def test_provider_rows_report_credential_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import OptionList

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    monkeypatch.setenv("GOOGLE_API_KEY", "from-the-environment")
    app, _ = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings("providers")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        provider_list = screen.query_one("#provider_list", OptionList)

        def row(provider: str) -> str:
            index = provider_list.get_option_index(f"provider_row_{provider}")
            return str(provider_list.get_option_at_index(index).prompt)

        assert "present in local settings" in row(UI_DEFAULT_PROVIDER)
        assert "present via GOOGLE_API_KEY" in row("google")
        assert "missing" in row("deepseek")

        # Staging a key updates the roster immediately, before Apply.
        await stage_provider_api_key(screen, pilot, "deepseek", "deepseek-key")
        assert "present in local settings" in row("deepseek")


@pytest.mark.asyncio
async def test_test_connection_probes_the_highlighted_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    """The probe follows the Providers selection, not the active model."""
    pytest.importorskip("textual")
    from textual.widgets import OptionList, Static

    from kolega_code.cli.tui.settings_screen import SettingsScreen
    from kolega_code.config import ModelProvider

    app, _ = _configured_app(tmp_path, monkeypatch, extra_key_providers=("deepseek",))
    probes: list[dict] = []

    async def fake_probe(provider, model, **kwargs):
        from kolega_code.cli.model_connection import ModelConnectionResult

        probes.append({"provider": provider, "model": model, **kwargs})
        return ModelConnectionResult(True, f"Connected to {provider.value}/{model}.")

    monkeypatch.setattr("kolega_code.cli.tui.settings_panel.test_model_connection", fake_probe)

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings("providers")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        provider_list = screen.query_one("#provider_list", OptionList)
        provider_list.highlighted = provider_list.get_option_index("provider_row_deepseek")
        await pilot.pause()

        await app._test_settings_connection()

        assert probes[-1]["provider"] == ModelProvider.DEEPSEEK
        assert probes[-1]["api_key"] == "stored-key"
        # Active model is moonshot's, so the probe falls back to deepseek's own default.
        assert probes[-1]["model"] == DEEPSEEK_DEFAULT_MODEL
        assert "deepseek" in str(screen.query_one("#provider_credential_status", Static).render())


@pytest.mark.asyncio
async def test_chatgpt_provider_row_offers_sign_in_instead_of_a_key_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import OptionList

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, _ = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings("providers")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        provider_list = screen.query_one("#provider_list", OptionList)

        assert screen.query_one("#provider_api_key_input").display is True
        assert screen.query_one("#provider_chatgpt_login").display is False

        provider_list.highlighted = provider_list.get_option_index("provider_row_openai_chatgpt")
        await pilot.pause()

        assert screen.query_one("#provider_api_key_input").display is False
        assert screen.query_one("#provider_remove_api_key").display is False
        assert screen.query_one("#provider_chatgpt_login").display is True
        assert screen.query_one("#provider_chatgpt_logout").display is True


# --- Custom Endpoints page -------------------------------------------------


def _fill_endpoint_form(
    screen, *, endpoint_id: str, base_url: str = "http://localhost:1234/v1", style: str = "openai_chat"
) -> None:
    from textual.widgets import Input, Select

    screen.query_one("#ep_id_input", Input).value = endpoint_id
    screen.query_one("#ep_base_url_input", Input).value = base_url
    screen.query_one("#ep_style_select", Select).value = style


@pytest.mark.asyncio
async def test_settings_adds_endpoint_and_apply_persists_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.provider_registry import ui_provider_options, ui_thinking_effort_options
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        _fill_endpoint_form(screen, endpoint_id="lmstudio")
        screen.query_one("#ep_thinking_mode_select", Select).value = "anthropic_budget"
        screen.query_one("#ep_max_output_input", Input).value = "8192"
        screen.query_one(
            "#ep_thinking_budgets_input", Input
        ).value = "low=2048, medium=4096, high=7168, xhigh=7168, max=7168"
        screen.query_one("#ep_reasoning_select", Select).value = "auto"
        await pilot.pause()
        app._handle_endpoint_settings_button("ep_save")

        await app._save_settings_from_ui()

    saved = settings_store.load().custom_endpoints["lmstudio"]
    assert saved["api_style"] == "openai_chat"
    assert saved["base_url"] == "http://localhost:1234/v1"
    assert saved["max_output_tokens"] == 8192
    assert saved["reasoning_replay"] == "auto"
    assert saved["thinking"]["mode"] == "anthropic_budget"
    assert saved["thinking"]["budgets"]["low"] == 2048
    assert "custom:lmstudio" in {value for _, value in ui_provider_options()}
    assert [value for _label, value in ui_thinking_effort_options("custom:lmstudio", "any-model")] == list(
        saved["thinking"]["options"]
    )


@pytest.mark.asyncio
async def test_settings_endpoint_edit_and_delete_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(
        tmp_path,
        monkeypatch,
        custom_endpoints={"lmstudio": {"api_style": "openai_chat", "base_url": "http://localhost:1234/v1"}},
    )

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        # Editing an existing endpoint: id input is locked, base URL updates.
        assert screen.query_one("#ep_id_input", Input).disabled is True
        screen.query_one("#ep_base_url_input", Input).value = "http://localhost:4321/v1"
        app._handle_endpoint_settings_button("ep_save")
        await app._save_settings_from_ui()

        assert settings_store.load().custom_endpoints["lmstudio"]["base_url"] == "http://localhost:4321/v1"

        # Delete round trip: staging then apply.
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        app._handle_endpoint_settings_button("ep_delete")
        await pilot.pause()
        await app._save_settings_from_ui()

    assert settings_store.load().custom_endpoints == {}


@pytest.mark.asyncio
async def test_settings_endpoint_staging_without_apply_leaves_file_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        _fill_endpoint_form(screen, endpoint_id="lmstudio")
        app._handle_endpoint_settings_button("ep_save")
        await pilot.pause()
        # The draft holds the endpoint but nothing is persisted until Apply.
        assert screen.pending_custom_endpoints["lmstudio"]["base_url"] == "http://localhost:1234/v1"

    assert settings_store.load().custom_endpoints == {}


@pytest.mark.asyncio
async def test_settings_endpoint_validation_rejects_bad_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Static

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        _fill_endpoint_form(screen, endpoint_id="Bad Slug!", base_url="localhost:1234")
        app._handle_endpoint_settings_button("ep_save")
        await pilot.pause()
        assert "slug" in str(screen.query_one("#endpoint_status", Static).render())

    assert settings_store.load().custom_endpoints == {}


@pytest.mark.asyncio
async def test_settings_deleting_active_endpoint_blocks_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Static

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(
        tmp_path,
        monkeypatch,
        provider="custom:lmstudio",
        model="qwen2.5",
        custom_endpoints={"lmstudio": {"api_style": "openai_chat", "base_url": "http://localhost:1234/v1"}},
    )

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        app._handle_endpoint_settings_button("ep_delete")
        await pilot.pause()
        await app._save_settings_from_ui()

        status_text = str(screen.query_one("#settings_status", Static).render())
        assert "No provider/model configured" in status_text

    # The failed apply must not have persisted the deletion.
    assert "lmstudio" in settings_store.load().custom_endpoints


@pytest.mark.asyncio
async def test_settings_custom_provider_model_uses_other_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(
        tmp_path,
        monkeypatch,
        provider="custom:lmstudio",
        model="qwen2.5",
        custom_endpoints={"lmstudio": {"api_style": "openai_chat", "base_url": "http://localhost:1234/v1"}},
    )

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        # The active custom model is restored through the free-text "Other…" entry.
        assert str(screen.query_one("#model_select", Select).value) == CUSTOM_MODEL_SENTINEL
        assert screen.query_one("#model_custom_input", Input).value == "qwen2.5"

        screen.query_one("#model_custom_input", Input).value = "qwen3-coder"
        await pilot.pause()
        await app._save_settings_from_ui()

    assert settings_store.load().active_model == "qwen3-coder"


@pytest.mark.asyncio
async def test_settings_env_defined_endpoints_in_pickers_but_not_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select

    from kolega_code.cli.provider_registry import ui_provider_options
    from kolega_code.cli.tui.settings_panel import ENDPOINT_NEW_VALUE
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    env_json = '{"envlocal": {"api_style": "openai_chat", "base_url": "http://localhost:9999/v1"}}'
    app, _ = _configured_app(tmp_path, monkeypatch, env={"KOLEGA_CODE_CUSTOM_ENDPOINTS": env_json})

    assert "custom:envlocal" in {value for _, value in ui_provider_options()}

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        endpoint_select = screen.query_one("#endpoint_select", Select)
        option_values = {value for _label, value in endpoint_select._options}  # type: ignore[attr-defined]
        assert option_values == {ENDPOINT_NEW_VALUE}


@pytest.mark.asyncio
async def test_settings_endpoint_temperature_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        _fill_endpoint_form(screen, endpoint_id="lmstudio")
        screen.query_one("#ep_temperature_input", Input).value = "0.4"
        app._handle_endpoint_settings_button("ep_save")
        await app._save_settings_from_ui()

    assert settings_store.load().custom_endpoints["lmstudio"]["temperature"] == 0.4

    # Anthropic style rejects temperatures above 1.
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    app2, _ = _configured_app(second_dir, monkeypatch)
    async with app2.run_test(size=(140, 40)) as pilot:
        app2.action_open_settings()
        await pilot.pause()
        screen2 = app2.screen
        assert isinstance(screen2, SettingsScreen)
        _fill_endpoint_form(screen2, endpoint_id="ap", style="anthropic")
        screen2.query_one("#ep_temperature_input", Input).value = "1.5"
        app2._handle_endpoint_settings_button("ep_save")
        await pilot.pause()
        from textual.widgets import Static

        assert "up to 1" in str(screen2.query_one("#endpoint_status", Static).render())


@pytest.mark.asyncio
async def test_settings_provider_switch_seeds_custom_endpoint_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, _ = _configured_app(
        tmp_path,
        monkeypatch,
        provider="deepseek",
        model=DEEPSEEK_DEFAULT_MODEL,
        custom_endpoints={
            "lmstudio": {
                "api_style": "openai_chat",
                "base_url": "http://localhost:1234/v1",
                "default_model": "qwen/qwen3.6-35b-a3b",
            }
        },
    )

    async with app.run_test(size=(140, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        screen.query_one("#provider_select", Select).value = "custom:lmstudio"
        await _wait_for_select_values(
            pilot, screen, {"provider_select": "custom:lmstudio", "model_select": CUSTOM_MODEL_SENTINEL}
        )
        await pilot.pause()
        assert screen.query_one("#model_custom_input", Input).value == "qwen/qwen3.6-35b-a3b"


@pytest.mark.asyncio
async def test_gateway_settings_page_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(80, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen._show_category("gateway")
        await pilot.pause()

        screen.query_one("#gateway_token_input", Input).value = "123:fake-bot-token-for-tests-only"
        screen.query_one("#gateway_allowed_users_input", Input).value = "111, 222"
        screen.query_one("#gateway_pairing_select", Select).value = "true"
        screen.query_one("#gateway_permission_select", Select).value = "auto"
        screen.query_one("#gateway_adapter_select", Select).value = "telegram"

        await app._save_settings_from_ui()
        settings = settings_store.load()
        assert settings.telegram_bot_token == "123:fake-bot-token-for-tests-only"
        assert settings.gateway["allowed_users"] == ["111", "222"]
        assert settings.gateway["pairing_enabled"] is True
        assert settings.gateway["permission_mode"] == "auto"
        assert settings.gateway["adapter"] == "telegram"


@pytest.mark.asyncio
async def test_tools_page_stt_settings_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(80, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen._show_category("tools")
        await pilot.pause()

        screen.query_one("#stt_enabled_select", Select).value = "true"
        # The single remote provider is preselected; blanking the model resets
        # to the provider default on save.
        assert screen.query_one("#stt_provider_select", Select).value == "groq"
        screen.query_one("#stt_model_input", Input).value = "whisper-large-v3"

        await app._save_settings_from_ui()
        settings = settings_store.load()
        assert settings.stt_enabled is True
        assert settings.stt_provider == "groq"
        assert settings.stt_model == "whisper-large-v3"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_gateway_settings_page_removes_the_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)
    settings = settings_store.load()
    settings.telegram_bot_token = "123:fake-bot-token-for-tests-only"
    settings_store.save(settings)
    app.settings = settings_store.load()

    async with app.run_test(size=(80, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen._show_category("gateway")
        await pilot.pause()
        assert "Token saved." in str(screen.query_one("#gateway_token_status").render())

        screen.query_one("#gateway_token_remove", Button).press()
        await pilot.pause()
        await app._save_settings_from_ui()
        assert settings_store.load().telegram_bot_token is None


@pytest.mark.asyncio
async def test_gateway_settings_page_rejects_a_bad_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    from kolega_code.cli.tui.settings_screen import SettingsScreen

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(80, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen._show_category("gateway")
        await pilot.pause()
        screen.query_one("#gateway_token_input", Input).value = "not-a-botfather-token"
        await app._save_settings_from_ui()
        assert settings_store.load().telegram_bot_token is None


@pytest.mark.asyncio
async def test_mcp_settings_page_oauth_controls_and_saving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.tui.settings_screen import SettingsScreen
    from kolega_code.mcp.config import load_mcp_config

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen._show_category("mcp")
        await pilot.pause()

        # Initially, OAuth is Disabled and OAuth detail inputs are hidden
        assert screen.query_one("#mcp_oauth_select", Select).value == "false"
        assert screen.query_one("#mcp_oauth_client_id_input", Input).display is False
        assert screen.query_one("#mcp_oauth_client_secret_input", Input).display is False

        # Toggling OAuth to Enabled shows the OAuth detail inputs
        screen.query_one("#mcp_oauth_select", Select).value = "true"
        await pilot.pause()
        assert screen.query_one("#mcp_oauth_client_id_input", Input).display is True
        assert screen.query_one("#mcp_oauth_client_secret_input", Input).display is True
        assert screen.query_one("#mcp_oauth_redirect_uri_input", Input).display is True

        # Switching transport to stdio hides all OAuth controls
        screen.query_one("#mcp_transport_select", Select).value = "stdio"
        await pilot.pause()
        assert screen.query_one("#mcp_oauth_select", Select).display is False
        assert screen.query_one("#mcp_oauth_client_id_input", Input).display is False

        # Switching back to streamable_http restores OAuth controls
        screen.query_one("#mcp_transport_select", Select).value = "streamable_http"
        await pilot.pause()
        assert screen.query_one("#mcp_oauth_select", Select).display is True
        assert screen.query_one("#mcp_oauth_client_id_input", Input).display is True

        # Fill in server details with pre-registered OAuth
        screen.query_one("#mcp_name_input", Input).value = "HubSpot CRM"
        screen.query_one("#mcp_url_input", Input).value = "https://mcp.hubspot.com"
        screen.query_one("#mcp_oauth_client_id_input", Input).value = "test-cid-123"
        screen.query_one("#mcp_oauth_client_secret_input", Input).value = "test-secret-456"
        screen.query_one("#mcp_oauth_client_secret_env_input", Input).value = "HUBSPOT_CLIENT_SECRET"
        screen.query_one("#mcp_oauth_redirect_uri_input", Input).value = "http://127.0.0.1:33418/callback"
        screen.query_one("#mcp_oauth_scope_input", Input).value = "crm.contacts.read"
        screen.query_one("#mcp_oauth_auth_method_select", Select).value = "client_secret_post"

        await app._handle_mcp_settings_button("mcp_save_server")
        await pilot.pause()

        # Check that server was saved to global MCP config
        cfg = load_mcp_config(app.project_path, settings_store.root, project_trusted=False)
        saved = next(s for s in cfg.servers.values() if s.name == "HubSpot CRM")
        assert saved.oauth.enabled is True
        assert saved.oauth.client_id == "test-cid-123"
        assert saved.oauth.client_secret == "test-secret-456"
        assert saved.oauth.client_secret_env == "HUBSPOT_CLIENT_SECRET"
        assert saved.oauth.redirect_uri == "http://127.0.0.1:33418/callback"
        assert saved.oauth.scope == "crm.contacts.read"
        assert saved.oauth.token_endpoint_auth_method == "client_secret_post"

        # Verify password mask on client secret input
        assert screen.query_one("#mcp_oauth_client_secret_input", Input).password is True

        # Test Clear OAuth: clears tokens from token store without erasing configured credentials
        app._clear_mcp_tokens_from_ui()
        cfg_after = load_mcp_config(app.project_path, settings_store.root, project_trusted=False)
        server_after = cfg_after.servers[saved.id]
        assert server_after.oauth.client_id == "test-cid-123"
        assert server_after.oauth.client_secret == "test-secret-456"
