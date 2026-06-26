"""Incremental streaming-inset rendering must match the canonical renderer and stay O(n).

The live (incomplete) assistant/thinking entries are rendered by appending only the newly
arrived characters to a cached Rich Text (_extend_inset), instead of re-splitting and
re-rendering the whole growing buffer on every ~50ms flush (the O(n^2) freeze on slow
machines). These tests pin two properties:

  1. correctness  - the incremental result is character-for-character / style-for-style
                    identical to _format_inset_text for any chunk splitting.
  2. scaling      - building a large stream incrementally stays well under a generous
                    wall-clock bound (it was ~6s of O(n^2) CPU before; ~50ms after).
"""

import time

import pytest

from kolega_code.cli.tui.transcript import TranscriptRenderingMixin, _InsetRenderState


@pytest.fixture
def mixin() -> TranscriptRenderingMixin:
    # _format_inset_text / _extend_inset use no instance state.
    return TranscriptRenderingMixin.__new__(TranscriptRenderingMixin)


def _per_char(text):
    """(plain, [style-per-char]) — tolerant of harmless span fragmentation."""
    styles: list[object] = [None] * len(text.plain)
    for span in text._spans:
        for i in range(span.start, min(span.end, len(styles))):
            styles[i] = str(span.style)
    return text.plain, styles


def _incremental(mixin, content: str, style, chunk_sizes):
    state = _InsetRenderState(style)
    pos = 0
    for size in chunk_sizes:
        mixin._extend_inset(state, content[pos : pos + size])
        pos += size
    mixin._extend_inset(state, content[pos:])  # remainder (also handles empty content)
    return state.text


CASES = [
    "",
    "a",
    "abc",
    "a\n",
    "a\nb",
    "a\nb\n",
    "a\n\nb",
    "\n",
    "\n\n",
    "\n\n\n\n",
    "\n\na",
    "a\n\n",
    "end\n\n\n",
    "line one\nline two\nthird",
    "trailing spaces  \n  leading",
    "a\n\n\nb\n",
    "no newlines just one long line " * 10,
    ("word " * 20 + "\n") * 4,
]


@pytest.mark.parametrize("content", CASES)
@pytest.mark.parametrize("style", [None, "italic dim"])
def test_incremental_matches_canonical_for_uniform_chunks(mixin, content, style):
    canonical = _per_char(mixin._format_inset_text(content, style=style))
    for chunk in (1, 2, 3, 7, 4096):
        sizes = [chunk] * (len(content) // chunk + 1)
        assert _per_char(_incremental(mixin, content, style, sizes)) == canonical


def test_incremental_matches_canonical_for_irregular_chunks(mixin):
    import random

    rng = random.Random(2024)
    content = "".join(rng.choice(["alpha beta", "gamma", "\n", "  ", "delta epsilon zeta", "\n\n"]) for _ in range(400))
    for style in (None, "italic dim"):
        canonical = _per_char(mixin._format_inset_text(content, style=style))
        for _ in range(20):
            sizes = []
            remaining = len(content)
            while remaining > 0:
                step = rng.randint(1, min(11, remaining))
                sizes.append(step)
                remaining -= step
            assert _per_char(_incremental(mixin, content, style, sizes)) == canonical


def test_incremental_final_state_equals_canonical_for_large_stream(mixin):
    content = "a moderately sized line of reasoning output\n" * 12_000  # ~520 KB, many lines
    canonical = _per_char(mixin._format_inset_text(content))
    incremental = _per_char(_incremental(mixin, content, None, [4000] * (len(content) // 4000 + 1)))
    assert incremental == canonical


def test_large_stream_builds_in_linear_time(mixin):
    """Regression guard: flush-as-it-grows must not reintroduce O(n^2).

    Pre-fix this same 1 MB stream cost multiple seconds of CPU; incrementally it is tens
    of milliseconds. The 2.0s bound is deliberately generous to avoid CI flakiness while
    still failing loudly if the whole buffer is re-rendered on every flush again.
    """
    content = "a moderately sized line of reasoning output\n" * 23_000  # ~1 MB
    step = 4000
    state = _InsetRenderState(None)
    start = time.perf_counter()
    pos = 0
    while pos < len(content):
        nxt = min(len(content), pos + step)
        mixin._extend_inset(state, content[pos:nxt])
        pos = nxt
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"incremental 1MB stream took {elapsed:.2f}s (O(n^2) regression?)"
    assert state.text.plain == mixin._format_inset_text(content).plain
