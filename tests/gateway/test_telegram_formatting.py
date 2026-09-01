"""Golden tests for the Markdown→Telegram-HTML renderer and chunker."""

import pytest

from kolega_code.gateway.adapters.telegram.formatting import chunk_text, telegram_html


def test_plain_text_is_escaped() -> None:
    assert telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_bold_and_italic() -> None:
    assert telegram_html("**bold** and *italic*") == "<b>bold</b> and <i>italic</i>"


def test_inline_code_wins_over_markers_inside_it() -> None:
    assert telegram_html("`**not bold**`") == "<code>**not bold**</code>"


def test_fenced_code_block() -> None:
    assert telegram_html("```python\nx = 1\nprint(x < 2)\n```") == ("<pre><code>x = 1\nprint(x &lt; 2)</code></pre>")


def test_unclosed_fence_renders_what_exists() -> None:
    assert telegram_html("```\nunclosed") == "<pre><code>unclosed</code></pre>"


def test_links_escape_urls_and_labels() -> None:
    assert telegram_html("[docs](https://example.com/?a=1&b=2)") == (
        '<a href="https://example.com/?a=1&amp;b=2">docs</a>'
    )


def test_headings_flatten_to_bold() -> None:
    assert telegram_html("## Title") == "<b>Title</b>"


def test_underscores_are_not_emphasis() -> None:
    # Identifiers and paths are full of underscores; never italicize them.
    assert telegram_html("snake_case and __dunder__") == "snake_case and __dunder__"


def test_unbalanced_delimiters_stay_literal() -> None:
    assert telegram_html("lone * and **stars") == "lone * and **stars"


def test_lists_and_blank_lines_keep_structure() -> None:
    assert telegram_html("- one\n- two\n\nplain") == "- one\n- two\n\nplain"


def test_chunk_text_empty_input() -> None:
    assert chunk_text("") == []


def test_chunk_text_short_text_passes_through() -> None:
    assert chunk_text("short", limit=100) == ["short"]


def test_chunk_text_splits_on_lines_within_limit() -> None:
    text = "a\nb\nc"
    assert chunk_text(text, limit=3) == ["a\nb", "c"]


def test_chunk_text_hard_splits_overlong_lines() -> None:
    assert chunk_text("abcdef", limit=2) == ["ab", "cd", "ef"]


def test_chunk_text_keeps_longer_lines_intact_when_possible() -> None:
    assert chunk_text("short\nline", limit=6) == ["short", "line"]


def test_chunk_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", limit=0)
