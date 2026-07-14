from __future__ import annotations

from pathlib import Path

import pytest

from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import install_fake_agents, wait_for_onboarding_screen


def _configured_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "stored-key")
    settings_store.save(settings)
    config = build_agent_config(project, env={}, settings=settings, settings_store=settings_store)
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
        assert screen.query_one("#settings_categories", OptionList).option_count == 5
        assert screen.query_one("#settings_page_model").display is True
        assert screen.query_one("#settings_page_tools").display is False
        apply_button = screen.query_one("#save_settings", Button)
        assert apply_button.region.y + apply_button.region.height <= 24

        screen._show_category("tools")
        assert screen.query_one("#settings_page_model").display is False
        assert screen.query_one("#settings_page_tools").display is True

        screen._show_category("model")
        remove_button = screen.query_one("#settings_remove_api_key", Button)
        assert remove_button.disabled is False
        app._settings_remove_api_key()
        await pilot.pause()
        assert screen.dirty is True
        assert UI_DEFAULT_PROVIDER in screen.pending_api_key_removals

        await app._save_settings_from_ui()
        assert settings_store.load().get_api_key(UI_DEFAULT_PROVIDER) == "stored-key"
        assert screen.dirty is True
        assert "Configuration incomplete" in str(screen.query_one("#settings_status").render())

        screen.query_one("#api_key_input", Input).value = "replacement-key"
        await pilot.pause()
        assert UI_DEFAULT_PROVIDER not in screen.pending_api_key_removals
        await app._save_settings_from_ui()

        assert settings_store.load().get_api_key(UI_DEFAULT_PROVIDER) == "replacement-key"
        assert screen.query_one("#api_key_input", Input).value == ""
        assert screen.dirty is False
        assert apply_button.disabled is True

        screen.query_one("#api_key_input", Input).value = "provider-specific-key"
        screen.query_one("#provider_select", Select).value = "deepseek"
        await pilot.pause()
        assert screen.query_one("#api_key_input", Input).value == ""


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
        api_key = screen.query_one("#api_key_input")
        assert provider.region.width == model.region.width == effort.region.width == 46
        assert api_key.region.width == 46
        assert provider.region.x == model.region.x == effort.region.x == api_key.region.x

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
