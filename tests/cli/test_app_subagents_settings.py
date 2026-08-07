"""Settings-tab Sub-agents toggle: seeding, forced-by-flag status, and persistence.

Mirrors the LSP master-switch coverage: the Tools tab shows a Sub-agents select
seeded from ``settings.subagents_enabled``, a status note appears when the
``--subagents`` launch flag pins the value for the session, and saving with
"Disabled" writes ``subagents_enabled=False`` to settings.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kolega_code.cli.config import CliConfigOverrides, build_agent_config, config_summary
from kolega_code.cli.provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import install_fake_agents, open_settings_screen
from .test_app_lsp_settings_status import _static_text


def _configured_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    subagents_enabled: bool | None = None,
    overrides: CliConfigOverrides | None = None,
):
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=UI_DEFAULT_PROVIDER,
        active_model=UI_DEFAULT_MODEL,
        subagents_enabled=subagents_enabled,
    )
    settings.set_api_key(UI_DEFAULT_PROVIDER, "stored-key")
    settings_store.save(settings)
    config = build_agent_config(project, env={}, settings=settings, settings_store=settings_store, overrides=overrides)
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
            overrides=overrides,
        ),
        settings_store,
    )


@pytest.mark.asyncio
async def test_subagents_select_seeded_from_saved_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Select

    app, _ = _configured_app(tmp_path, monkeypatch, subagents_enabled=False)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        assert str(screen.query_one("#subagents_enabled_select", Select).value) == "false"


@pytest.mark.asyncio
async def test_subagents_status_notes_forced_launch_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Static

    app, _ = _configured_app(tmp_path, monkeypatch, overrides=CliConfigOverrides(subagents_mode="off"))
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        text = _static_text(screen.query_one("#subagents_status", Static))
        assert "--subagents" in text
        assert "forced off" in text


@pytest.mark.asyncio
async def test_subagents_status_quiet_without_forcing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Static

    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        assert _static_text(screen.query_one("#subagents_status", Static)) == ""


@pytest.mark.asyncio
async def test_subagents_save_persists_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Select

    app, settings_store = _configured_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        screen.query_one("#subagents_enabled_select", Select).value = "false"
        await pilot.pause()

        await app._save_settings_from_ui()

        assert settings_store.load().subagents_enabled is False
