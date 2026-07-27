"""Generate the web stylesheet from the CLI design tokens.

:mod:`kolega_code.cli.theme` is the single source of truth: this module only
transposes ``theme.web_tokens()`` into CSS custom properties and asks Pygments
for the same code-highlighting styles the TUI uses, so the web UI and the replay
player look like the same product as the terminal. Nothing here hard-codes a
color; the only literals are units and selector names.
"""

from __future__ import annotations

from kolega_code.cli import theme

_HEADER = (
    "/* GENERATED FILE -- do not edit by hand.\n"
    " * Source of truth: kolega_code/cli/theme.py (see theme.web_tokens()).\n"
    " * Regenerate with kolega_code.web.theme_css.stylesheet().\n"
    " */\n"
)

CODE_SELECTOR = ".kc-code"


def _css_var(key: str) -> str:
    """``row_highlight`` -> ``--kc-row-highlight``."""
    return f"--kc-{key.replace('_', '-')}"


def _declarations(tokens: dict) -> list[str]:
    """Custom-property declarations for one theme's token dict."""
    colors: dict[str, str] = tokens["colors"]
    fonts: dict[str, str] = tokens["fonts"]
    layout: dict[str, int] = tokens["layout"]
    spinner: dict = tokens["spinner"]
    lines = [f"  {_css_var(key)}: {value};" for key, value in colors.items()]
    lines.append(f"  {_css_var('font_mono')}: {fonts['mono']};")
    lines.append(f"  {_css_var('font_sans')}: {fonts['sans']};")
    # Character-cell units keep the browser on the TUI's monospace grid.
    lines.append(f"  {_css_var('context_bar_width')}: {layout['context_bar_width']}ch;")
    lines.append(f"  {_css_var('transcript_indent')}: {layout['transcript_indent']}ch;")
    lines.append(f"  {_css_var('spinner_interval')}: {spinner['interval_ms']}ms;")
    return lines


def _block(selector: str, tokens: dict) -> str:
    body = "\n".join(_declarations(tokens))
    return f"/* {tokens['name']} */\n{selector} {{\n{body}\n}}\n"


def theme_css(tokens: dict[str, dict] | None = None) -> str:
    """All themes as CSS custom properties.

    Emits a ``:root`` block carrying the default theme plus one
    ``[data-theme="<slug>"]`` block per theme, so a host switches themes by
    setting a single attribute.
    """
    all_tokens = theme.all_web_tokens() if tokens is None else tokens
    if not all_tokens:
        return _HEADER
    default_slug = theme.default_theme_slug()
    default_tokens = all_tokens.get(default_slug) or next(iter(all_tokens.values()))
    parts = [_HEADER, "\n", _block(":root", default_tokens)]
    for slug, theme_tokens in all_tokens.items():
        parts.append("\n")
        parts.append(_block(f'[data-theme="{slug}"]', theme_tokens))
    return "".join(parts)


def pygments_css(style_name: str, *, selector: str = CODE_SELECTOR) -> str:
    """Pygments style definitions scoped to ``selector``.

    Uses the very same Pygments style the TUI renders fenced code with. Unknown
    style names fall back to :data:`theme.MARKDOWN_CODE_THEME` (mirroring
    ``theme._resolved_code_theme``); a missing Pygments returns ``""`` rather
    than raising, since code color is a nicety and not load-bearing.
    """
    try:
        from pygments.formatters import HtmlFormatter
    except Exception:
        return ""
    selectors = [part.strip() for part in selector.split(",") if part.strip()]
    for candidate in (style_name, theme.MARKDOWN_CODE_THEME):
        try:
            formatter = HtmlFormatter(style=candidate)
            # get_style_defs accepts "a string or list of selectors" at runtime;
            # the bundled stub only declares str.
            return str(formatter.get_style_defs(selectors))  # pyright: ignore[reportArgumentType]
        except Exception:
            continue
    return ""


def all_pygments_css() -> str:
    """Code-highlighting rules for every theme, scoped to that theme.

    Each block is emitted under ``[data-theme="<slug>"] .kc-code`` so a theme's
    code colors only apply while that theme is active. The default theme also
    gets a ``:root .kc-code`` selector so code is styled before any host sets
    ``data-theme``; it is emitted first, so explicit themes win on order.
    """
    default_slug = theme.default_theme_slug()
    parts: list[str] = []
    for slug, tokens in theme.all_web_tokens().items():
        scoped = f'[data-theme="{slug}"] {CODE_SELECTOR}'
        if slug == default_slug:
            scoped = f":root {CODE_SELECTOR}, {scoped}"
        defs = pygments_css(str(tokens["code_theme"]), selector=scoped)
        if not defs:
            continue
        parts.append(f"/* {tokens['name']} code: {tokens['code_theme']} */\n{defs}\n")
    return "\n".join(parts)


def stylesheet() -> str:
    """The whole generated stylesheet: theme tokens plus code highlighting."""
    return theme_css() + "\n" + all_pygments_css()


def _channel(value: float) -> float:
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance of a hex or ANSI-named color."""
    resolved = theme.resolve_color(hex_color)
    r, g, b = (_channel(int(resolved[i : i + 2], 16) / 255) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """WCAG 2.x contrast ratio between two colors (1.0 .. 21.0)."""
    lum_a = _relative_luminance(hex_a)
    lum_b = _relative_luminance(hex_b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)
