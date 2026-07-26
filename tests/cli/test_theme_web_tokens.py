"""Web design-token export: theme.py stays the single source of truth.

These tests are the contract that keeps hand-copied hex codes out of web assets:
every browser-facing value is generated from ``kolega_code.cli.theme``.
"""

import dataclasses
import re

import pytest

from kolega_code.cli import theme
from kolega_code.web import theme_css as css

HEX_RE = re.compile(r"^#[0-9a-f]{6}$")

MENU_SLUGS = ["kolega-dark", "nord", "dracula", "gruvbox", "solarized"]

# Every --kc-* custom property the web UI and replay player rely on.
EXPECTED_PROPERTIES = [
    "--kc-accent",
    "--kc-success",
    "--kc-warning",
    "--kc-error",
    "--kc-muted",
    "--kc-user",
    "--kc-agent",
    "--kc-tool",
    "--kc-thinking",
    "--kc-background",
    "--kc-surface",
    "--kc-panel",
    "--kc-primary",
    "--kc-secondary",
    "--kc-foreground",
    "--kc-text-muted",
    "--kc-row-highlight",
    "--kc-chrome-accent",
    "--kc-chrome-success",
    "--kc-chrome-warning",
    "--kc-chrome-error",
    "--kc-font-mono",
    "--kc-font-sans",
    "--kc-context-bar-width",
    "--kc-transcript-indent",
]

# Roles that are intentionally dimmer than WCAG AA against their own background.
# These are pre-existing palette choices (upstream Nord/Dracula/Gruvbox/Solarized
# colors, plus the deliberately dim muted/thinking roles) that the web UI must
# reproduce exactly as the TUI shows them, so the themes are NOT retuned here.
# (slug, role) -> floor the color must still clear. Measured ratios in comments.
LOW_CONTRAST_ROLES: dict[tuple[str, str], float] = {
    ("nord", "error"): 3.0,  # 3.65
    ("nord", "agent"): 3.0,  # 4.41
    ("nord", "tool"): 3.0,  # 3.10
    ("nord", "muted"): 2.5,  # 2.73 -- the only sub-3.0 pair in any theme
    ("nord", "thinking"): 2.5,  # 2.73 (same color as nord.muted)
    ("dracula", "muted"): 3.0,  # 4.02
    ("dracula", "thinking"): 3.0,  # 4.02
    ("gruvbox", "error"): 3.0,  # 4.29
    ("gruvbox", "muted"): 3.0,  # 4.02
    ("gruvbox", "thinking"): 3.0,  # 4.02
    ("solarized", "error"): 3.0,  # 3.55
    ("solarized", "user"): 3.0,  # 4.08
    ("solarized", "agent"): 3.0,  # 3.30
    ("solarized", "tool"): 3.0,  # 3.43
}

ROLE_KEYS = ("accent", "success", "warning", "error", "muted", "user", "agent", "tool", "thinking")


def _color_fields() -> list[str]:
    """ThemeSpec fields that hold a color (everything but names/code theme)."""
    skip = {"name", "slug", "markdown_code_theme"}
    return [f.name for f in dataclasses.fields(theme.ThemeSpec) if f.name not in skip]


def _dummy_spec() -> theme.ThemeSpec:
    return dataclasses.replace(theme.THEMES["Nord"], name="Test Dummy", slug="test-dummy")


# --- token export ----------------------------------------------------------


@pytest.mark.parametrize("name", theme.available_themes())
def test_web_tokens_resolve_every_color_to_hex(name: str):
    tokens = theme.web_tokens(theme.THEMES[name])
    colors = tokens["colors"]
    assert isinstance(colors, dict)
    for key, value in colors.items():
        # No ANSI name may survive: a browser has no terminal palette.
        assert HEX_RE.match(value), f"{name}.{key} is not #rrggbb: {value!r}"


def test_web_tokens_cover_every_theme_spec_color_field_exactly_once():
    fields = _color_fields()
    assert set(theme.WEB_COLOR_KEYS) == set(fields), "WEB_COLOR_KEYS drifted from ThemeSpec"
    web_keys = list(theme.WEB_COLOR_KEYS.values())
    assert len(set(web_keys)) == len(web_keys), "two ThemeSpec fields map to one web key"
    colors = theme.web_tokens(theme.THEMES[theme.DEFAULT_THEME_NAME])["colors"]
    assert isinstance(colors, dict)
    # Count equality makes a newly added ThemeSpec color field fail until exported.
    assert len(colors) == len(fields) == 21
    assert set(colors) == set(web_keys)
    # Both layers survive the name collisions instead of one clobbering the other.
    spec = theme.THEMES["Nord"]
    nord = theme.web_tokens(spec)["colors"]
    assert isinstance(nord, dict)
    assert colors["accent"] == theme.resolve_color(theme.THEMES[theme.DEFAULT_THEME_NAME].accent)
    assert nord["accent"] == spec.accent
    assert nord["chrome_accent"] == spec.tt_accent
    assert nord["text_muted"] == spec.tt_text_muted


def test_web_tokens_carry_glyphs_spinner_layout_and_fonts():
    tokens = theme.web_tokens(theme.THEMES[theme.DEFAULT_THEME_NAME])
    assert tokens["code_theme"] == theme.THEMES[theme.DEFAULT_THEME_NAME].markdown_code_theme
    spinner = tokens["spinner"]
    assert isinstance(spinner, dict)
    assert spinner["frames"] == list(theme.SPINNER_FRAMES)
    assert spinner["interval_ms"] == int(theme.SPINNER_INTERVAL * 1000)
    assert tokens["layout"] == {
        "context_bar_width": theme.CONTEXT_BAR_WIDTH,
        "transcript_indent": theme.TRANSCRIPT_INDENT,
    }
    fonts = tokens["fonts"]
    assert isinstance(fonts, dict)
    assert "monospace" in fonts["mono"]
    assert "sans-serif" in fonts["sans"]


def test_glyph_introspection_exports_every_public_attribute():
    tokens = theme.web_tokens(theme.THEMES[theme.DEFAULT_THEME_NAME])
    glyphs = tokens["glyphs"]
    assert isinstance(glyphs, dict)
    expected = {name.lower(): value for name, value in vars(theme.Glyph).items() if not name.startswith("_")}
    assert glyphs == expected
    assert glyphs["user"] == theme.Glyph.USER
    assert glyphs["inset_elbow"] == theme.Glyph.INSET_ELBOW
    assert glyphs["check"] == theme.Glyph.CHECK


def test_resolve_color_passes_hex_maps_ansi_and_rejects_junk():
    assert theme.resolve_color("#AABBCC") == "#aabbcc"
    assert theme.resolve_color("bright_cyan") == theme.ANSI_HEX["bright_cyan"]
    assert theme.resolve_color("CYAN") == theme.ANSI_HEX["cyan"]
    assert set(theme.ANSI_HEX) == {
        f"{prefix}{name}"
        for prefix in ("", "bright_")
        for name in ("black", "red", "green", "yellow", "blue", "magenta", "cyan", "white")
    }
    with pytest.raises(ValueError):
        theme.resolve_color("rebeccapurple")


def test_all_web_tokens_in_menu_order_with_kolega_dark_default():
    tokens = theme.all_web_tokens()
    assert list(tokens) == MENU_SLUGS
    assert theme.default_theme_slug() == theme.THEMES[theme.DEFAULT_THEME_NAME].slug == "kolega-dark"
    assert all(tokens[slug]["slug"] == slug for slug in MENU_SLUGS)


# --- contrast --------------------------------------------------------------


def test_contrast_ratio_matches_wcag_reference_values():
    assert css.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0, abs=0.01)
    assert css.contrast_ratio("#000000", "#000000") == pytest.approx(1.0, abs=0.01)


def test_bright_ansi_variants_clear_aa_on_kolega_dark_background():
    background = theme.ANSI_HEX["black"]
    assert background == theme.THEMES[theme.DEFAULT_THEME_NAME].tt_background
    for name, value in theme.ANSI_HEX.items():
        if not name.startswith("bright_"):
            continue
        ratio = css.contrast_ratio(value, background)
        assert ratio >= 4.5, f"ANSI {name} only {ratio:.2f}:1 on {background}"


@pytest.mark.parametrize("slug", MENU_SLUGS)
def test_theme_contrast_foreground_and_roles(slug: str):
    colors = theme.all_web_tokens()[slug]["colors"]
    background = colors["background"]
    fg_ratio = css.contrast_ratio(colors["foreground"], background)
    assert fg_ratio >= 4.5, f"{slug} foreground only {fg_ratio:.2f}:1"
    for role in ROLE_KEYS:
        ratio = css.contrast_ratio(colors[role], background)
        floor = LOW_CONTRAST_ROLES.get((slug, role), 4.5)
        assert ratio >= floor, f"{slug}.{role} only {ratio:.2f}:1 (floor {floor})"


def test_low_contrast_allowlist_references_real_themes_and_roles():
    for slug, role in LOW_CONTRAST_ROLES:
        assert slug in MENU_SLUGS
        assert role in ROLE_KEYS


# --- CSS generation --------------------------------------------------------


def test_theme_css_defines_every_theme_and_property():
    sheet = css.theme_css()
    assert "kolega_code/cli/theme.py" in sheet
    assert ":root {" in sheet
    for slug in MENU_SLUGS:
        assert f'[data-theme="{slug}"] {{' in sheet
    for prop in EXPECTED_PROPERTIES:
        # Present in :root and in every per-theme block.
        assert sheet.count(f"{prop}:") == len(MENU_SLUGS) + 1, prop


def _root_block(sheet: str) -> str:
    start = sheet.index(":root {")
    return sheet[start : sheet.index("}", start)]


def test_root_block_holds_only_default_theme_colors():
    sheet = css.theme_css()
    root = _root_block(sheet)
    default_slug = theme.default_theme_slug()
    all_tokens = theme.all_web_tokens()
    default_hexes = set(all_tokens[default_slug]["colors"].values())
    for slug, tokens in all_tokens.items():
        if slug == default_slug:
            continue
        for key, value in tokens["colors"].items():
            if value in default_hexes:
                continue  # shared value, indistinguishable by text search
            assert value not in root, f"{slug}.{key} leaked into :root"
    for value in default_hexes:
        assert value in root


def test_dummy_theme_surfaces_in_tokens_and_css(monkeypatch: pytest.MonkeyPatch):
    dummy = _dummy_spec()
    monkeypatch.setitem(theme.THEMES, dummy.name, dummy)
    tokens = theme.all_web_tokens()
    assert list(tokens) == [*MENU_SLUGS, "test-dummy"]
    assert tokens["test-dummy"]["colors"]["background"] == dummy.tt_background
    sheet = css.theme_css()
    assert '[data-theme="test-dummy"] {' in sheet
    # Nothing else changed: the default block is still the default theme's.
    assert theme.default_theme_slug() == "kolega-dark"
    for slug in MENU_SLUGS:
        assert f'[data-theme="{slug}"] {{' in sheet


# --- pygments --------------------------------------------------------------


def test_pygments_css_emits_scoped_rules_and_tolerates_bad_style_names():
    pytest.importorskip("pygments")
    known = css.pygments_css("monokai", selector=".kc-code")
    assert known.strip()
    assert ".kc-code" in known
    # Unknown style names fall back instead of raising (mirrors theme._resolved_code_theme).
    bogus = css.pygments_css("no-such-pygments-style", selector=".kc-code")
    assert bogus.strip()
    assert ".kc-code" in bogus


def test_all_pygments_css_scopes_each_theme_and_stylesheet_combines_both():
    pytest.importorskip("pygments")
    code_css = css.all_pygments_css()
    for slug in MENU_SLUGS:
        assert f'[data-theme="{slug}"] .kc-code' in code_css
    assert ":root .kc-code" in code_css  # default theme also applies without data-theme
    sheet = css.stylesheet()
    assert "--kc-accent:" in sheet
    assert ".kc-code" in sheet
