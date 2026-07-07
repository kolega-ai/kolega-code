# ruff: noqa: F401,F811,E402
from pathlib import Path

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module

from kolega_code.cli import theme
from kolega_code.cli.tui.state import TurnState, tool_state_presentation, turn_state_color
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import FakeCoderAgent, build_test_config


@pytest.fixture(autouse=True)
def _reset_theme():
    """Theme state is process-global; restore the default after each test."""
    yield
    theme.apply_theme(theme.DEFAULT_THEME_NAME)


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


def _is_gray(hexv: str) -> bool:
    return hexv.startswith("#") and len(hexv) == 7 and hexv[1:3] == hexv[3:5] == hexv[5:7]


def test_non_truecolor_chrome_is_neutral_gray_but_truecolor_keeps_tint():
    pytest.importorskip("textual")
    tinted = {t.name: t.to_color_system().generate() for t in theme.build_textual_themes(truecolor=True)}
    gray = {t.name: t.to_color_system().generate() for t in theme.build_textual_themes(truecolor=False)}
    for slug in ("nord", "solarized", "dracula", "gruvbox", "kolega-dark"):
        g, t = gray[slug], tinted[slug]
        # 256-color: structural chrome is exact gray (R==G==B), so it lands on the gray ramp.
        for key in ("background", "surface", "panel", "foreground", "text-muted", "surface-lighten-2"):
            assert _is_gray(g[key]), f"{slug}.{key} not neutral in 256-color: {g[key]}"
        # truecolor keeps Nord/Solarized's signature tinted backgrounds.
        # Role/semantic colors stay colorful in BOTH (carry the theme + match buttons).
        assert g["primary"] == t["primary"]
        assert g["accent"] == t["accent"]
    # Nord's tinted background is genuinely tinted (not already gray) in truecolor.
    assert not _is_gray(tinted["nord"]["background"])


def test_tinted_chrome_downsamples_to_cube_but_gray_to_ramp():
    """Regression guard for the macOS Terminal.app teal/blue-background bug."""
    from rich.color import Color as RichColor
    from rich.color import ColorSystem

    def to256(hexv: str) -> int:
        number = RichColor.parse(hexv).downgrade(ColorSystem.EIGHT_BIT).number
        assert number is not None
        return number

    for tinted in ("#2e3440", "#002b36"):  # Nord bg, Solarized bg
        assert to256(tinted) < 232, "tinted dark unexpectedly mapped to the gray ramp"
        assert to256(theme._luminance_gray(tinted)) >= 232, "neutral gray must map to the gray ramp"


def test_splash_endpoints_are_primary_then_secondary():
    # The wordmark's top/flat color is the primary == the $primary button color.
    for name in theme.available_themes():
        spec = theme.THEMES[name]
        assert theme.splash_colors(name) == (spec.tt_primary, spec.tt_secondary)
        # Every theme now has hex endpoints, so a gradient is available in truecolor.
        assert theme.gradient_hex(*theme.splash_colors(name), 6)


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
    from kolega_code.cli.app import KolegaCodeApp

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
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
