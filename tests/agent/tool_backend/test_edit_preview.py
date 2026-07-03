"""Unit tests for the diff/head preview builders (kolega_code.agent.tool_backend.edit_preview)."""

from kolega_code.agent.tool_backend import edit_preview as ep


def _tags(preview):
    return [row[0] for row in preview["lines"]]


class TestBuildDiffPreview:
    def test_basic_diff(self):
        old = "def f():\n    return 1\n"
        new = "def f():\n    return 2\n"
        p = ep.build_diff_preview(old, new, "m.py")
        assert p["kind"] == "diff"
        assert p["language"] == "python"
        assert p["adds"] == 1
        assert p["dels"] == 1
        assert "add" in _tags(p)
        assert "del" in _tags(p)

    def test_noop_returns_none(self):
        assert ep.build_diff_preview("a\nb\n", "a\nb\n", "x.py") is None

    def test_binary_returns_none(self):
        assert ep.build_diff_preview("a\x00b", "c", "x.bin") is None
        assert ep.build_diff_preview("a", "c\x00d", "x.bin") is None

    def test_pure_addition_counts(self):
        p = ep.build_diff_preview("a\n", "a\nb\nc\n", "x.txt")
        assert p["adds"] == 2
        assert p["dels"] == 0

    def test_truncation_reports_more(self):
        new = "\n".join(f"line{i}" for i in range(100)) + "\n"
        p = ep.build_diff_preview("", new, "big.txt")
        assert len(p["lines"]) <= ep.MAX_DIFF_LINES
        assert p["more"] > 0

    def test_oversize_falls_back_to_head(self):
        big = "x\n" * (ep.MAX_DIFF_INPUT_LINES + 10)
        p = ep.build_diff_preview("", big, "big.txt")
        assert p is not None
        assert p["kind"] == "head"

    def test_long_line_is_clipped(self):
        new = "x" * (ep.MAX_LINE_CHARS + 50) + "\n"
        p = ep.build_diff_preview("", new, "x.txt")
        add_rows = [text for tag, text in p["lines"] if tag == "add"]
        assert add_rows
        # '+' marker + clipped body + ellipsis, never the full 290-char line.
        assert len(add_rows[0]) <= ep.MAX_LINE_CHARS + 2

    def test_no_cap_with_max_lines_zero(self):
        """All diff lines are included and more is 0 when max_lines=0."""
        old = "\n".join(f"old{i}" for i in range(30)) + "\n"
        new = "\n".join(f"new{i}" for i in range(30)) + "\n"
        p = ep.build_diff_preview(old, new, "many.py", max_lines=0)
        assert p["kind"] == "diff"
        # With 30 old lines replaced by 30 new lines, we expect 60 + some meta lines.
        assert len(p["lines"]) > ep.MAX_DIFF_LINES
        assert p["more"] == 0

    def test_custom_cap(self):
        """max_lines=N caps at exactly N lines."""
        old = "\n".join(f"old{i}" for i in range(50)) + "\n"
        new = "\n".join(f"new{i}" for i in range(50)) + "\n"
        p = ep.build_diff_preview(old, new, "custom.py", max_lines=5)
        assert len(p["lines"]) == 5
        assert p["more"] > 0

    def test_oversize_falls_back_to_head_with_max_lines_zero(self):
        """When oversized, fallback head preview also receives max_lines=0 (uncapped)."""
        big = "x\n" * (ep.MAX_DIFF_INPUT_LINES + 10)
        p = ep.build_diff_preview("", big, "big.txt", max_lines=0)
        assert p is not None
        assert p["kind"] == "head"
        # With max_lines=0, head should be uncapped (all lines shown).
        assert p["more"] == 0


class TestBuildHeadPreview:
    def test_basic_head(self):
        p = ep.build_head_preview("import os\nimport sys\nprint('hi')\n", "s.py")
        assert p["kind"] == "head"
        assert p["language"] == "python"
        assert p["adds"] == 3
        assert all(tag == "context" for tag in _tags(p))

    def test_empty_returns_none(self):
        assert ep.build_head_preview("", "x.py") is None

    def test_binary_returns_none(self):
        assert ep.build_head_preview("a\x00b", "x.bin") is None

    def test_truncation_reports_more(self):
        content = "\n".join(f"l{i}" for i in range(50)) + "\n"
        p = ep.build_head_preview(content, "big.txt")
        assert len(p["lines"]) == ep.MAX_HEAD_LINES
        assert p["more"] == 50 - ep.MAX_HEAD_LINES

    def test_no_cap_with_max_lines_zero(self):
        """All head lines are included and more is 0 when max_lines=0."""
        content = "\n".join(f"line{i}" for i in range(30)) + "\n"
        p = ep.build_head_preview(content, "full.txt", max_lines=0)
        assert len(p["lines"]) == 30
        assert p["more"] == 0

    def test_custom_cap(self):
        """max_lines=N caps the head at exactly N lines."""
        content = "\n".join(f"line{i}" for i in range(30)) + "\n"
        p = ep.build_head_preview(content, "capped.txt", max_lines=7)
        assert len(p["lines"]) == 7
        assert p["more"] == 30 - 7


class TestLanguageForPath:
    def test_known_extensions(self):
        assert ep.language_for_path("a/b.py") == "python"
        assert ep.language_for_path("x.TS") == "typescript"

    def test_unknown_extension(self):
        assert ep.language_for_path("file.unknownext") == "text"
        assert ep.language_for_path("noext") == "text"
