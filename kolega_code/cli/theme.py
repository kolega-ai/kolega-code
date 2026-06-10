"""Visual design tokens for the Kolega Code CLI.

This module is the single source of truth for colors, glyphs, spacing, and
truncation limits used by both the Textual TUI (app.py) and the plain CLI
commands (main.py). It must stay importable without rich or textual installed,
so those libraries are only imported lazily inside helpers.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Optional


class Color:
    """Semantic color roles as Rich style strings."""

    ACCENT = "cyan"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    MUTED = "bright_black"
    USER = "cyan"
    AGENT = "magenta"
    TOOL = "blue"
    THINKING = "bright_black"


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
INSET_WIDTH = 2
MARKDOWN_CODE_THEME = "monokai"
RENDER_COALESCE_INTERVAL = 0.05


@lru_cache(maxsize=None)
def supports_unicode(encoding: Optional[str] = None) -> bool:
    """Whether the output encoding can represent the glyphs above."""
    resolved: str = encoding or getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        Glyph.TOOL.encode(resolved)
        SPINNER_FRAMES.encode(resolved)
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
