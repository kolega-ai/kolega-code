"""Markdown-to-Telegram-HTML conversion and message chunking.

Telegram's HTML parse mode accepts only a small tag subset (``<b>``, ``<i>``,
``<u>``, ``<s>``, ``<code>``, ``<pre>``, ``<a>``, ``<tg-spoiler>``,
``<blockquote>``). Model output arrives as loose Markdown, so the adapter
renders it into that subset and falls back to plain text when Telegram
rejects the payload.

Deliberately conservative:

- ``**bold**`` and ``*italic*`` are converted, but ``__bold__``/``_italic_``
  (underscore delimiters) are not — underscores are everywhere in code and
  identifiers.
- Fenced code blocks become ``<pre><code>``; inline `` `code` `` becomes
  ``<code>``.
- ``[text](url)`` becomes ``<a href="...">``; everything else is escaped.
- Headings flatten to bold lines; lists stay plain text.

``chunk_text`` splits at line boundaries before conversion, so a chunk boundary
can fall inside a fenced block; the per-message plain-text fallback absorbs
that edge case rather than complicating the splitter.
"""

from __future__ import annotations

import html
import re

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _inline(text: str) -> str:
    """Convert the inline Markdown subset, escaping everything else."""
    out: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end != -1:
                out.append("<b>" + html.escape(text[i + 2 : end]) + "</b>")
                i = end + 2
                continue
            # Unbalanced bold: keep the star literal and let the next
            # iteration handle the rest (it may pair as italic or stay text).
            out.append("*")
            i += 1
            continue
        if text[i] == "`":
            end = text.find("`", i + 1)
            if end != -1:
                out.append("<code>" + html.escape(text[i + 1 : end]) + "</code>")
                i = end + 1
                continue
        if text[i] == "*":
            end = _find_single_star(text, i + 1)
            if end != -1:
                out.append("<i>" + html.escape(text[i + 1 : end]) + "</i>")
                i = end + 1
                continue
        if text[i] == "[":
            match = _LINK_RE.match(text, i)
            if match is not None:
                label, url = match.group(1), match.group(2)
                out.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>')
                i = match.end()
                continue
        out.append(html.escape(text[i]))
        i += 1
    return "".join(out)


def _find_single_star(text: str, start: int) -> int:
    """Find a ``*`` that is not part of a ``**`` pair, or -1.

    ``lone * and **stars`` must stay literal: the only closing candidates are
    the paired stars of ``**``.
    """
    i = start
    while i < len(text):
        if text[i] == "*":
            if text.startswith("**", i):
                i += 2
                continue
            return i
        i += 1
    return -1


def telegram_html(text: str) -> str:
    """Render loose Markdown into Telegram's HTML subset."""
    lines: list[str] = []
    fence_lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                code = html.escape("\n".join(fence_lines))
                lines.append(f"<pre><code>{code}</code></pre>")
                fence_lines = []
            else:
                in_fence = True
            continue
        if in_fence:
            fence_lines.append(line)
            continue
        if not stripped:
            lines.append("")
        elif stripped.startswith("#"):
            lines.append("<b>" + _inline(stripped.lstrip("#").strip()) + "</b>")
        else:
            lines.append(_inline(line))
    if in_fence:  # Unclosed fence: render what we have.
        lines.append(f"<pre><code>{html.escape(chr(10).join(fence_lines))}</code></pre>")
    return "\n".join(lines)


def chunk_text(text: str, limit: int = 4000) -> list[str]:
    """Split text into chunks no longer than ``limit`` characters.

    Splits on line boundaries and hard-splits overlong single lines. Empty
    input yields no chunks.
    """
    if limit <= 0:
        raise ValueError(f"chunk limit must be positive, got {limit}")
    if not text:
        return []
    chunks: list[str] = []
    buffer: list[str] = []
    joined_len = 0  # exact length of "\n".join(buffer)
    for line in text.split("\n"):
        if len(line) > limit:
            if buffer:
                chunks.append("\n".join(buffer))
                buffer = []
                joined_len = 0
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
        add_cost = len(line) + (1 if buffer else 0)
        if joined_len + add_cost > limit:
            chunks.append("\n".join(buffer))
            buffer = []
            joined_len = 0
        buffer.append(line)
        joined_len += len(line) + (1 if len(buffer) > 1 else 0)
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks
