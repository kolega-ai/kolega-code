"""Visual design tokens for the Kolega Code CLI.

This module is the single source of truth for colors, glyphs, spacing, and
truncation limits used by both the Textual TUI (app.py) and the plain CLI
commands (main.py). It must stay importable without rich or textual installed,
so those libraries are only imported lazily inside helpers.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


class Color:
    """Semantic color roles as Rich style strings.

    Values are (re)assigned by :func:`apply_theme`; the defaults below mirror the
    Kolega Dark theme so the class is usable even before a theme is applied.
    """

    ACCENT: str = "cyan"
    SUCCESS: str = "green"
    WARNING: str = "yellow"
    ERROR: str = "bright_red"
    MUTED: str = "bright_black"
    USER: str = "bright_cyan"
    AGENT: str = "bright_magenta"
    TOOL: str = "bright_blue"
    THINKING: str = "bright_black"


class Glyph:
    """Unicode glyphs used in the UI. Use g() to apply ASCII fallbacks."""

    USER = "❯"  # ❯
    AGENT = "●"  # ●
    STATUS = "●"  # ●
    TOOL = "⏺"  # ⏺
    SUB_AGENT = "◆"  # ◆
    PLAN = "◆"  # ◆
    QUESTION = "?"
    INSET_BAR = "│"  # │
    INSET_ELBOW = "└"  # └
    ELLIPSIS = "…"  # …
    BULLET_SEP = "·"  # ·
    BAR_FILLED = "█"  # █
    BAR_EMPTY = "░"  # ░
    CHECK = "✓"  # ✓
    CROSS = "✗"  # ✗
    DOWN = "↓"  # ↓
    PENDING = "○"  # ○ — phase not started
    RUNNING = "▶"  # ▶ — phase in progress


ASCII_FALLBACKS = {
    Glyph.USER: ">",
    Glyph.AGENT: "*",
    Glyph.TOOL: "*",
    Glyph.SUB_AGENT: "*",
    Glyph.INSET_BAR: "|",
    Glyph.INSET_ELBOW: "`-",
    Glyph.ELLIPSIS: "...",
    Glyph.BULLET_SEP: "-",
    Glyph.BAR_FILLED: "#",
    Glyph.BAR_EMPTY: "-",
    Glyph.CHECK: "ok",
    Glyph.CROSS: "x",
    Glyph.DOWN: "v",
    Glyph.PENDING: "o",
    Glyph.RUNNING: ">",
}

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
SPINNER_FRAMES_ASCII = "|/-\\"
SPINNER_INTERVAL = 0.25

# Truncation and layout limits
TOOL_RESULT_PREVIEW_CHARS = 500
TOOL_STREAM_PREVIEW_CHARS = 4_000
TOOL_FULL_CONTENT_CAP_CHARS = 50_000
SUB_AGENT_TAIL_CHARS = 200
SUB_AGENT_TASK_PREVIEW_CHARS = 120
CONTEXT_BAR_WIDTH = 18
TRANSCRIPT_INDENT = 2
MARKDOWN_CODE_THEME = "monokai"
RENDER_COALESCE_INTERVAL = 0.05
# Coalesce less aggressively as the live streaming entry grows: each flush still costs
# Textual an O(height) re-measure of an auto-height widget, so for very large reasoning
# streams we trade a little update latency for far fewer full re-measures. Sizes are
# characters of the live entry; see transcript._invalidate_conversation.
RENDER_COALESCE_INTERVAL_MEDIUM = 0.12
RENDER_COALESCE_INTERVAL_LARGE = 0.25
RENDER_COALESCE_MEDIUM_CHARS = 40_000
RENDER_COALESCE_LARGE_CHARS = 200_000

# Scrollback window bounds (see tui.widgets.ScrollbackWindow). The transcript and the
# sub-agent inspector mount only a trailing window of their entries so Textual's
# O(mounted-widgets) reflow stays cheap no matter how long the session runs.
# Scrolling near the top mounts older chunks; following the bottom trims the oldest.
TRANSCRIPT_WINDOW_MAX = 300
TRANSCRIPT_WINDOW_TRIM_CHUNK = 100
TRANSCRIPT_WINDOW_EXPAND_CHUNK = 100
INSPECTOR_WINDOW_MAX = 150
INSPECTOR_WINDOW_EXPAND_CHUNK = 100


@lru_cache(maxsize=None)
def supports_unicode(encoding: Optional[str] = None) -> bool:
    """Whether the output encoding can represent the glyphs above.

    Probes the original stdout (sys.__stdout__) because Textual redirects
    sys.stdout while the app is running.
    """
    if encoding is None:
        encoding = getattr(sys.__stdout__, "encoding", None)
    if encoding is None:
        import locale

        encoding = locale.getpreferredencoding(False) or "ascii"
    try:
        Glyph.TOOL.encode(encoding)
        SPINNER_FRAMES.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def g(glyph: str) -> str:
    """Return the glyph, or its ASCII fallback on limited encodings."""
    if supports_unicode():
        return glyph
    return ASCII_FALLBACKS.get(glyph, glyph)


def spinner_frames() -> str:
    return SPINNER_FRAMES if supports_unicode() else SPINNER_FRAMES_ASCII


# Populated (and re-populated on every theme switch) by apply_theme() so log
# colors track the active theme instead of snapshotting Color at import time.
LOG_LEVEL_COLORS: dict[str, str] = {}


def log_level_color(level: str) -> str:
    """Semantic color for a log level, defaulting to muted."""
    return LOG_LEVEL_COLORS.get(level.lower(), Color.MUTED)


def styled(text: str, style: str) -> str:
    """Wrap text in Rich markup for the given style."""
    return f"[{style}]{text}[/{style}]"


def role_header(
    glyph: str,
    label: str,
    color: str,
    *,
    label_style: str = "bold",
    state: Optional[str] = None,
    detail: Optional[str] = None,
) -> str:
    """Render the shared entry-header grammar.

    GRAMMAR: <colored glyph> <bold label> [ · state] [ · detail]
    The glyph carries the semantic color; state and detail are muted.
    """
    parts = [styled(g(glyph), color), styled(label, label_style)]
    sep = g(Glyph.BULLET_SEP)
    if state:
        parts.append(styled(f"{sep} {state}", "dim"))
    if detail:
        parts.append(styled(f"{sep} {detail}", "dim"))
    return " ".join(parts)


def context_bar(usage_percentage: float, width: int = CONTEXT_BAR_WIDTH) -> str:
    """Render a usage bar like █████░░░░ for the status dashboard."""
    filled = max(0, min(width, round((usage_percentage / 100) * width)))
    return g(Glyph.BAR_FILLED) * filled + g(Glyph.BAR_EMPTY) * (width - filled)


def build_rich_theme():
    """Build a rich Theme mapping semantic names to the palette (lazy import)."""
    from rich.theme import Theme

    return Theme(
        {
            "accent": Color.ACCENT,
            "success": Color.SUCCESS,
            "warning": Color.WARNING,
            "error": Color.ERROR,
            "muted": Color.MUTED,
            "user": Color.USER,
            "agent": Color.AGENT,
            "tool": Color.TOOL,
        }
    )


# ---------------------------------------------------------------------------
# Themes
#
# A theme bundles two coordinated color layers: the nine Rich role colors above
# (used for inline markup: glyphs, headers, the splash) and the Textual chrome
# params (used to build a textual.theme.Theme that drives the $surface/$text/...
# CSS variables). apply_theme() swaps both in lockstep. Chrome surface/panel are
# kept near-neutral so 'round $surface' borders never render saturated in
# 256-color terminals (e.g. macOS Terminal.app); see app.py CSS comment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThemeSpec:
    name: str  # display name + registry key, e.g. "Kolega Dark"
    slug: str  # textual.theme.Theme.name, e.g. "kolega-dark"
    # Nine Rich role colors (ANSI names for Kolega Dark, #rrggbb otherwise).
    accent: str
    success: str
    warning: str
    error: str
    muted: str
    user: str
    agent: str
    tool: str
    thinking: str
    # Textual chrome (concrete colors; surface/panel kept neutral).
    tt_background: str
    tt_surface: str
    tt_panel: str
    tt_primary: str
    tt_secondary: str
    tt_accent: str
    tt_foreground: str
    tt_text_muted: str
    tt_success: str
    tt_warning: str
    tt_error: str
    row_highlight: str  # neutral choice-list highlight; overrides $surface-lighten-2
    markdown_code_theme: str  # Pygments style for fenced code blocks
    # NOTE: the splash wordmark uses tt_primary -> tt_secondary (the button color),
    # so it always matches the primary buttons; there are no separate splash fields.


_KOLEGA_DARK = ThemeSpec(
    name="Kolega Dark",
    slug="kolega-dark",
    # ANSI names -> render through the terminal's own palette (native in both
    # Ghostty and Terminal.app); bright variants clear AA on the dark background.
    accent="cyan",
    success="green",
    warning="yellow",
    error="bright_red",
    muted="bright_black",
    user="bright_cyan",
    agent="bright_magenta",
    tool="bright_blue",
    thinking="bright_black",
    tt_background="#14171b",
    tt_surface="#1e2228",
    tt_panel="#272c33",
    tt_primary="#5fb3c4",
    tt_secondary="#c678dd",
    tt_accent="#5fb3c4",
    tt_foreground="#d6d6d6",
    tt_text_muted="#8a8f98",
    tt_success="#5fa05f",
    tt_warning="#c9a13b",
    tt_error="#cc5555",
    row_highlight="#3a4047",
    markdown_code_theme="monokai",
)

_NORD = ThemeSpec(
    name="Nord",
    slug="nord",
    accent="#88c0d0",
    success="#a3be8c",
    warning="#ebcb8b",
    error="#cd6f78",
    muted="#6c7689",
    user="#8fbcbb",
    agent="#b48ead",
    tool="#5e81ac",
    thinking="#6c7689",
    tt_background="#2e3440",
    tt_surface="#3b4252",
    tt_panel="#434c5e",
    tt_primary="#88c0d0",
    tt_secondary="#b48ead",
    tt_accent="#88c0d0",
    tt_foreground="#d8dee9",
    tt_text_muted="#6c7689",
    tt_success="#a3be8c",
    tt_warning="#ebcb8b",
    tt_error="#bf616a",
    row_highlight="#4c566a",
    markdown_code_theme="nord",
)

_DRACULA = ThemeSpec(
    name="Dracula",
    slug="dracula",
    accent="#8be9fd",
    success="#50fa7b",
    warning="#f1fa8c",
    error="#ff6e6e",
    muted="#7a86b8",
    user="#bd93f9",
    agent="#ff79c6",
    tool="#80d4e8",
    thinking="#7a86b8",
    tt_background="#282a36",
    tt_surface="#343746",
    tt_panel="#424557",
    tt_primary="#bd93f9",
    tt_secondary="#ff79c6",
    tt_accent="#8be9fd",
    tt_foreground="#f8f8f2",
    tt_text_muted="#7a86b8",
    tt_success="#50fa7b",
    tt_warning="#f1fa8c",
    tt_error="#ff5555",
    row_highlight="#44475a",
    markdown_code_theme="dracula",
)

_GRUVBOX = ThemeSpec(
    name="Gruvbox",
    slug="gruvbox",
    accent="#83a598",
    success="#b8bb26",
    warning="#fabd2f",
    error="#fb4934",
    muted="#928374",
    user="#a9b665",
    agent="#d3869b",
    tool="#83a598",
    thinking="#928374",
    tt_background="#282828",
    tt_surface="#32302f",
    tt_panel="#3c3836",
    tt_primary="#83a598",
    tt_secondary="#d3869b",
    tt_accent="#83a598",
    tt_foreground="#ebdbb2",
    tt_text_muted="#928374",
    tt_success="#b8bb26",
    tt_warning="#fabd2f",
    tt_error="#fb4934",
    row_highlight="#504945",
    markdown_code_theme="gruvbox-dark",
)

_SOLARIZED = ThemeSpec(
    name="Solarized",
    slug="solarized",
    accent="#2aa198",
    success="#859900",
    warning="#b58900",
    error="#e0403c",
    muted="#839496",
    user="#268bd2",
    agent="#d33682",
    tool="#6c71c4",
    thinking="#839496",
    tt_background="#002b36",
    tt_surface="#073642",
    tt_panel="#2b3a40",
    tt_primary="#2aa198",
    tt_secondary="#d33682",
    tt_accent="#2aa198",
    tt_foreground="#93a1a1",
    tt_text_muted="#839496",
    tt_success="#859900",
    tt_warning="#b58900",
    tt_error="#dc322f",
    row_highlight="#2b3a40",
    markdown_code_theme="solarized-dark",
)

# Insertion order == menu order. First entry is the default.
THEMES: dict[str, ThemeSpec] = {spec.name: spec for spec in (_KOLEGA_DARK, _NORD, _DRACULA, _GRUVBOX, _SOLARIZED)}
DEFAULT_THEME_NAME = _KOLEGA_DARK.name

_active_theme_name = DEFAULT_THEME_NAME


def available_themes() -> tuple[str, ...]:
    """Display names of all themes, in menu order."""
    return tuple(THEMES.keys())


def active_theme() -> ThemeSpec:
    return THEMES[_active_theme_name]


def textual_theme_name(name: Optional[str] = None) -> str:
    """Textual Theme slug for a display name (active theme when name is None)."""
    spec = THEMES.get(name) if name else None
    return (spec or active_theme()).slug


def splash_colors(name: Optional[str] = None) -> tuple[str, str]:
    """(top, bottom) splash endpoints: the theme's primary -> secondary.

    The top/flat color is the primary (== the $primary button color), so the
    wordmark always matches the primary buttons.
    """
    spec = THEMES.get(name) if name else None
    spec = spec or active_theme()
    return spec.tt_primary, spec.tt_secondary


def apply_theme(name: Optional[str]) -> ThemeSpec:
    """Make ``name`` the active theme, syncing Color and LOG_LEVEL_COLORS.

    Unknown or missing names fall back to the default theme. This only mutates
    plain strings, so it stays importable without rich/textual installed.
    """
    global _active_theme_name
    spec = THEMES.get(name or "") or THEMES[DEFAULT_THEME_NAME]
    _active_theme_name = spec.name
    Color.ACCENT = spec.accent
    Color.SUCCESS = spec.success
    Color.WARNING = spec.warning
    Color.ERROR = spec.error
    Color.MUTED = spec.muted
    Color.USER = spec.user
    Color.AGENT = spec.agent
    Color.TOOL = spec.tool
    Color.THINKING = spec.thinking
    LOG_LEVEL_COLORS.clear()
    LOG_LEVEL_COLORS.update(
        {
            "debug": spec.muted,
            "info": spec.muted,
            "ok": spec.success,
            "warn": spec.warning,
            "warning": spec.warning,
            "error": spec.error,
            "critical": spec.error,
        }
    )
    return spec


@lru_cache(maxsize=None)
def _resolved_code_theme(name: str) -> str:
    """Validate a Pygments style name, falling back to the default on miss."""
    try:
        from pygments.styles import get_style_by_name

        get_style_by_name(name)
        return name
    except Exception:
        return MARKDOWN_CODE_THEME


def markdown_code_theme() -> str:
    """Pygments code-block style for the active theme (validated)."""
    return _resolved_code_theme(active_theme().markdown_code_theme)


def supports_truecolor(console=None) -> bool:
    """Whether 24-bit color is available (gates the splash gradient)."""
    system = getattr(console, "color_system", None) if console is not None else None
    if system is not None:
        return system == "truecolor"
    return os.environ.get("COLORTERM", "").strip().lower() in {"truecolor", "24bit"}


def _parse_hex(value: str) -> Optional[tuple[int, int, int]]:
    if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
        return None
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError:
        return None


def gradient_hex(top: str, bottom: str, steps: int) -> list[str]:
    """``steps`` hex colors interpolated top->bottom, or [] if not both hex."""
    start = _parse_hex(top)
    end = _parse_hex(bottom)
    if start is None or end is None or steps <= 0:
        return []
    if steps == 1:
        return ["#{:02x}{:02x}{:02x}".format(*start)]
    out: list[str] = []
    for i in range(steps):
        t = i / (steps - 1)
        rgb = tuple(round(start[c] + (end[c] - start[c]) * t) for c in range(3))
        out.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return out


def _luminance_gray(value: str) -> str:
    """Exact-neutral gray (#vvvvvv) at the perceptual luminance of ``value``.

    Rich's 8-bit downgrade maps near-neutral *tinted* darks onto the saturated
    6x6x6 color cube (e.g. #2e3440 -> teal), but exact grays (R==G==B) land on
    the gray ramp. Converting chrome colors to their luminance gray keeps each
    theme's relative lightness while guaranteeing neutral chrome in 256-color
    terminals (macOS Terminal.app). No-op if ``value`` is not a hex color.
    """
    rgb = _parse_hex(value)
    if rgb is None:
        return value
    v = round(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    return f"#{v:02x}{v:02x}{v:02x}"


def build_textual_theme(spec: ThemeSpec, truecolor: bool = True):
    """Construct a textual.theme.Theme for ``spec`` (lazy textual import).

    When ``truecolor`` is False (256-color terminals), the structural chrome is
    neutralized to exact grays so it doesn't quantize to a saturated cube color;
    the colorful role/semantic colors (primary/secondary/accent/success/warning/
    error) are left intact since they downsample correctly and carry the theme.
    """
    from textual.theme import Theme as TextualTheme

    gray = (lambda c: c) if truecolor else _luminance_gray
    return TextualTheme(
        name=spec.slug,
        primary=spec.tt_primary,
        secondary=spec.tt_secondary,
        accent=spec.tt_accent,
        foreground=gray(spec.tt_foreground),
        background=gray(spec.tt_background),
        surface=gray(spec.tt_surface),
        panel=gray(spec.tt_panel),
        success=spec.tt_success,
        warning=spec.tt_warning,
        error=spec.tt_error,
        dark=True,
        variables={
            "text-muted": gray(spec.tt_text_muted),
            # Pin the choice-list highlight ($surface-lighten-2) to a near-neutral
            # gray per theme, overriding Textual's auto-derived value so it never
            # quantizes to a saturated cell in 256-color terminals (Solarized).
            "surface-lighten-2": gray(spec.row_highlight),
        },
    )


def build_textual_themes(truecolor: bool = True) -> list:
    """All five themes as textual.theme.Theme objects, in menu order.

    Pass ``truecolor=False`` for 256-color terminals to get neutral-gray chrome.
    """
    return [build_textual_theme(spec, truecolor=truecolor) for spec in THEMES.values()]


# Populate Color / LOG_LEVEL_COLORS at import so consumers see a ready palette.
apply_theme(DEFAULT_THEME_NAME)
