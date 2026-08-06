"""TUI coverage for the compression-threshold settings control.

The select restores from persisted settings (including saved values outside the
preset list), collects back into a settings candidate ("default" -> None,
otherwise the percent), and reports when a launch flag or env var forces the
threshold for the session.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ._app_test_utils import build_test_config, install_fake_agents, open_settings_screen
from .test_app_lsp_settings_status import _static_text
from kolega_code.cli.tui.settings_panel import compression_threshold_options


def _make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, overrides=None, settings=None):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import config_summary
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.cli.settings import SettingsStore

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    if settings is not None:
        SettingsStore(tmp_path / "state").save(settings)
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(
        project_path=project, config=config, mode="code", store=store, session=session, overrides=overrides
    )


async def _wait_for_init(screen, pilot, timeout: float = 6.0) -> None:
    """Wait out the settings screen's restore pass before asserting select values."""
    deadline = time.monotonic() + timeout
    while screen._initializing and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert not screen._initializing


@pytest.mark.asyncio
async def test_compression_select_defaults_to_default_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Select

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        select = screen.query_one("#compression_threshold_select", Select)
        assert str(select.value) == "default"


@pytest.mark.asyncio
async def test_compression_select_restores_saved_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Select

    from kolega_code.cli.settings import CliSettings

    app = _make_app(tmp_path, monkeypatch, settings=CliSettings(compression_threshold=90.0))
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        select = screen.query_one("#compression_threshold_select", Select)
        assert str(select.value) == "90"


@pytest.mark.asyncio
async def test_compression_select_restores_saved_off_preset_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import Select

    from kolega_code.cli.settings import CliSettings

    app = _make_app(tmp_path, monkeypatch, settings=CliSettings(compression_threshold=85.0))
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        select = screen.query_one("#compression_threshold_select", Select)
        assert str(select.value) == "85"
        # The off-preset saved value was added as an extra option so the select can hold it.
        assert "85" in {value for _, value in compression_threshold_options(85.0)}


@pytest.mark.asyncio
async def test_compression_collect_maps_select_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ "default" collects to None; any other option collects to its percent."""
    from textual.widgets import Select

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        select = screen.query_one("#compression_threshold_select", Select)

        select.value = "default"
        app._collect_compression_from_ui()
        assert app.settings.compression_threshold is None

        select.value = "90"
        app._collect_compression_from_ui()
        assert app.settings.compression_threshold == 90.0


@pytest.mark.asyncio
async def test_compression_status_reports_forced_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--compression-threshold pins the session; the settings note must say so."""
    from textual.widgets import Static

    from kolega_code.cli.config import CliConfigOverrides

    app = _make_app(tmp_path, monkeypatch, overrides=CliConfigOverrides(compression_threshold="70"))
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        text = _static_text(screen.query_one("#compression_status", Static))
        assert "forced to 70%" in text
        assert "--compression-threshold" in text


@pytest.mark.asyncio
async def test_compression_status_reports_forced_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Static

    monkeypatch.setenv("KOLEGA_CODE_COMPRESSION_THRESHOLD", "65")
    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        text = _static_text(screen.query_one("#compression_status", Static))
        assert "forced to 65%" in text
        assert "KOLEGA_CODE_COMPRESSION_THRESHOLD" in text


@pytest.mark.asyncio
async def test_compression_status_blank_without_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import Static

    app = _make_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        text = _static_text(screen.query_one("#compression_status", Static))
        assert "forced" not in text
