from pathlib import Path

import pytest

from kolega_code.cli import theme
from kolega_code.cli.app import TurnState, tool_state_presentation, turn_state_color
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from .test_app import build_test_config


@pytest.fixture(autouse=True)
def _reset_theme():
    """Theme state is process-global; restore the default after each test."""
    yield
    theme.apply_theme(theme.DEFAULT_THEME_NAME)


class FakeCoderAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.history = []

    def restore_message_history(self, history):
        return None

    def dump_message_history(self):
        return []

    async def cleanup(self):
        return None


# --- theme module ----------------------------------------------------------


def test_available_themes_order_with_default_first():
    assert theme.available_themes() == ("Kolega Dark", "Nord", "Dracula", "Gruvbox", "Solarized")
    assert theme.DEFAULT_THEME_NAME == "Kolega Dark"


def test_default_theme_uses_ansi_role_colors():
    theme.apply_theme("Kolega Dark")
    # ANSI names render through the terminal's own palette (native in both
    # Ghostty and Terminal.app); bright variants clear contrast on dark bg.
    assert theme.Color.ACCENT == "cyan"
    assert theme.Color.AGENT == "bright_magenta"
    assert theme.Color.TOOL == "bright_blue"


def test_apply_theme_updates_color_log_levels_and_lazy_dicts():
    theme.apply_theme("Dracula")
    assert theme.Color.AGENT == "#ff79c6"
    # LOG_LEVEL_COLORS is rebuilt in place, not snapshotted at import.
    assert theme.LOG_LEVEL_COLORS["error"] == theme.Color.ERROR
    assert theme.LOG_LEVEL_COLORS["ok"] == theme.Color.SUCCESS
    # app.py's state dicts resolve role NAMES against the live Color attrs.
    assert turn_state_color(TurnState.IDLE) == theme.Color.SUCCESS
    assert turn_state_color(TurnState.ERROR) == theme.Color.ERROR
    assert tool_state_presentation("tool_call") == ("running", theme.Color.ACCENT)
    assert tool_state_presentation("tool_error")[1] == theme.Color.ERROR


def test_unknown_or_missing_theme_falls_back_to_default():
    assert theme.apply_theme("does-not-exist").name == theme.DEFAULT_THEME_NAME
    assert theme.apply_theme(None).name == theme.DEFAULT_THEME_NAME


def test_build_textual_themes_returns_five_named_by_slug_with_row_highlight():
    pytest.importorskip("textual")
    themes = theme.build_textual_themes()
    assert [t.name for t in themes] == ["kolega-dark", "nord", "dracula", "gruvbox", "solarized"]
    # Each theme pins a neutral choice-list highlight + muted text.
    assert all("surface-lighten-2" in t.variables for t in themes)
    assert all("text-muted" in t.variables for t in themes)


def test_markdown_code_theme_resolves_to_valid_pygments_style():
    styles = pytest.importorskip("pygments.styles")
    for name in theme.available_themes():
        theme.apply_theme(name)
        # Must resolve without raising (markdown_code_theme validates + falls back).
        styles.get_style_by_name(theme.markdown_code_theme())


def test_gradient_hex_interpolates_and_skips_ansi_endpoints():
    assert theme.gradient_hex("#000000", "#ffffff", 3) == ["#000000", "#808080", "#ffffff"]
    assert theme.gradient_hex("#000000", "#ffffff", 1) == ["#000000"]
    # ANSI-named endpoints (Kolega Dark) cannot interpolate -> empty -> flat fallback.
    assert theme.gradient_hex("cyan", "magenta", 6) == []


def test_supports_truecolor_reads_console_color_system():
    class FakeConsole:
        color_system = "truecolor"

    class Fake256:
        color_system = "256"

    assert theme.supports_truecolor(FakeConsole()) is True
    assert theme.supports_truecolor(Fake256()) is False


# --- settings persistence --------------------------------------------------


def test_settings_round_trip_active_theme(tmp_path: Path):
    store = SettingsStore(tmp_path)
    store.save(CliSettings(active_theme="Nord"))
    assert store.load().active_theme == "Nord"


def test_settings_back_compat_without_active_theme():
    legacy = {"schema_version": 2, "active_provider": "anthropic"}
    assert CliSettings.from_dict(legacy).active_theme is None


# --- app wiring ------------------------------------------------------------


def _make_app(tmp_path: Path, monkeypatch, *, persisted_theme=None):
    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    settings_store = SettingsStore(tmp_path / "settings")
    if persisted_theme is not None:
        settings_store.save(CliSettings(active_theme=persisted_theme))
    app = KolegaCodeApp(
        project_path=project,
        config=config,
        mode="code",
        store=store,
        session=session,
        settings_store=settings_store,
    )
    return app, settings_store


@pytest.mark.asyncio
async def test_persisted_theme_applied_at_startup(tmp_path: Path, monkeypatch):
    pytest.importorskip("textual")
    app, _ = _make_app(tmp_path, monkeypatch, persisted_theme="Nord")
    async with app.run_test():
        assert app.theme == "nord"
        assert theme.Color.ACCENT == "#88c0d0"
        # All five themes are registered with Textual.
        for slug in ("kolega-dark", "nord", "dracula", "gruvbox", "solarized"):
            assert slug in app.available_themes


@pytest.mark.asyncio
async def test_theme_command_switches_persists_and_reskins(tmp_path: Path, monkeypatch):
    pytest.importorskip("textual")
    app, settings_store = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._command_theme("dracula")  # case-insensitive match
        assert app.settings.active_theme == "Dracula"
        assert app.theme == "dracula"
        assert theme.Color.AGENT == "#ff79c6"
        assert settings_store.load().active_theme == "Dracula"


@pytest.mark.asyncio
async def test_theme_command_no_args_lists_options(tmp_path: Path, monkeypatch):
    pytest.importorskip("textual")
    app, _ = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._command_theme("")
        assert app._pending_theme_selection is not None
        assert [name for name, _ in app._pending_theme_selection.options] == list(theme.available_themes())


@pytest.mark.asyncio
async def test_theme_command_rejects_unknown(tmp_path: Path, monkeypatch):
    pytest.importorskip("textual")
    app, _ = _make_app(tmp_path, monkeypatch)
    async with app.run_test():
        before = app.theme
        await app._command_theme("bogus")
        assert app.theme == before
        assert app.settings.active_theme is None
        assert app._pending_theme_selection is None
