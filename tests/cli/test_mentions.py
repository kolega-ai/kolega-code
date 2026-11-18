"""Tests for @ mention parsing and file attachment expansion."""

from collections.abc import Callable
from io import FileIO
from pathlib import Path
from typing import IO, Any

import pytest

from kolega_code.cli.mentions import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_LINES,
    MAX_DIR_ENTRIES,
    build_file_attachments,
    parse_mentions,
)

SNIFF_BYTES = 8192
CONTINUATION = "ask the agent to read specific sections for more"


@pytest.fixture
def track_attachment_reads(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], list[int]]:
    original_open = Path.open

    def track(target: Path) -> list[int]:
        counts: list[int] = []
        opened = False

        class TrackedFile(FileIO):
            def read(self, size: int = -1, /) -> bytes:
                assert 0 < size <= SNIFF_BYTES
                data = super().read(size)
                assert data is not None
                counts.append(len(data))
                return data

        def open_file(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> IO[Any]:
            nonlocal opened
            if path != target:
                return original_open(path, mode, *args, **kwargs)
            assert not opened
            opened = True
            assert mode == "rb" and kwargs.get("buffering") == 0
            return TrackedFile(path, mode)

        monkeypatch.setattr(Path, "open", open_file)
        return counts

    return track


def test_parse_mentions_requires_start_or_whitespace() -> None:
    assert parse_mentions("email user@example.com please") == []
    assert [m.path for m in parse_mentions("@a.py and @b.py")] == ["a.py", "b.py"]
    assert [m.path for m in parse_mentions("look at @src/main.py")] == ["src/main.py"]


def test_parse_mentions_strips_trailing_punctuation() -> None:
    assert [m.path for m in parse_mentions("see @src/main.py, and @README.md.")] == ["src/main.py", "README.md"]
    assert [m.path for m in parse_mentions("(@notes.txt)")] == []  # @ after ( is not a mention
    assert [m.path for m in parse_mentions("read @notes.txt)")] == ["notes.txt"]


def test_parse_mentions_supports_quoted_paths() -> None:
    assert [m.path for m in parse_mentions('open @"my docs/plan.md" now')] == ["my docs/plan.md"]


def test_build_file_attachments_reads_existing_file(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    attachments, unresolved = build_file_attachments("check @hello.py", tmp_path)
    assert unresolved == []
    assert len(attachments) == 1
    assert attachments[0]["type"] == "file"
    assert attachments[0]["path"] == "hello.py"
    assert attachments[0]["content"] == "print('hi')\n"
    assert attachments[0]["truncated"] is False
    assert attachments[0]["is_dir"] is False


def test_build_file_attachments_nonexistent_path_is_unresolved(tmp_path: Path) -> None:
    attachments, unresolved = build_file_attachments("check @missing.py", tmp_path)
    assert attachments == []
    assert unresolved == ["missing.py"]


def test_build_file_attachments_dedupes_repeat_mentions(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    attachments, _ = build_file_attachments("@a.txt and @a.txt again", tmp_path)
    assert len(attachments) == 1


@pytest.mark.parametrize(
    ("unit", "preview", "limit", "read_bytes"),
    [
        (b"x\n", "x\n" * MAX_ATTACHMENT_LINES, "2,000-line", SNIFF_BYTES),
        (b"x", "x" * MAX_ATTACHMENT_BYTES, "48 KiB", MAX_ATTACHMENT_BYTES + 1),
    ],
    ids=["dense-log", "single-giant-line"],
)
def test_build_file_attachments_bounds_raw_reads(
    tmp_path: Path,
    track_attachment_reads: Callable[[Path], list[int]],
    unit: bytes,
    preview: str,
    limit: str,
    read_bytes: int,
) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(unit * (4 * 1024 * 1024 // len(unit)))
    counts = track_attachment_reads(path)

    attachments, _ = build_file_attachments("@large.txt", tmp_path)

    assert sum(counts) == read_bytes
    assert attachments[0]["truncated"] is True
    assert attachments[0]["content"] == f"{preview}\n[truncated at the {limit} attachment limit - {CONTINUATION}]"


@pytest.mark.parametrize("tail", ["", "\n", "\nx"], ids=["unterminated", "terminated", "one-over"])
def test_build_file_attachments_at_line_limit(tmp_path: Path, tail: str) -> None:
    preview = ("line\n" * MAX_ATTACHMENT_LINES).removesuffix("\n")
    (tmp_path / "lines.txt").write_bytes((preview + tail).encode("utf-8"))

    attachments, _ = build_file_attachments("@lines.txt", tmp_path)

    truncated = tail == "\nx"
    expected = preview + tail.removesuffix("x")
    if truncated:
        expected += f"\n[truncated at the 2,000-line attachment limit - {CONTINUATION}]"
    assert attachments[0]["truncated"] is truncated
    assert attachments[0]["content"] == expected


@pytest.mark.parametrize(
    ("tail", "remaining_bytes", "expected_tail", "truncated"),
    [
        ("x", 1, "x", False),
        ("xx", 1, "x", True),
        ("💡", 4, "💡", False),
        ("💡", 3, "", True),
    ],
    ids=["exact", "one-over", "unicode-exact", "unicode-straddles-cap"],
)
def test_build_file_attachments_at_byte_limit(
    tmp_path: Path, tail: str, remaining_bytes: int, expected_tail: str, truncated: bool
) -> None:
    prefix = "x" * (MAX_ATTACHMENT_BYTES - remaining_bytes)
    (tmp_path / "bytes.txt").write_bytes((prefix + tail).encode("utf-8"))

    attachments, _ = build_file_attachments("@bytes.txt", tmp_path)

    notice = f"\n[truncated at the 48 KiB attachment limit - {CONTINUATION}]" if truncated else ""
    assert attachments[0]["truncated"] is truncated
    assert attachments[0]["content"] == prefix + expected_tail + notice


@pytest.mark.parametrize("extra", ["", "💡"], ids=["exact", "extra-code-point"])
def test_build_file_attachments_at_both_limits_with_crlf(
    tmp_path: Path, track_attachment_reads: Callable[[Path], list[int]], extra: str
) -> None:
    width, longer_lines = divmod(MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_LINES)
    prefix = b"x" * (width - 1)
    raw = (prefix + b"x\r\n") * longer_lines + (prefix + b"\r\n") * (MAX_ATTACHMENT_LINES - longer_lines)
    path = tmp_path / "crlf.txt"
    path.write_bytes(raw + extra.encode("utf-8"))
    counts = track_attachment_reads(path)

    attachments, _ = build_file_attachments("@crlf.txt", tmp_path)

    assert sum(counts) == path.stat().st_size <= MAX_ATTACHMENT_BYTES + MAX_ATTACHMENT_LINES + 5
    preview = raw.replace(b"\r\n", b"\n").decode("ascii")
    notice = f"\n[truncated at the 2,000-line attachment limit - {CONTINUATION}]" if extra else ""
    assert attachments[0]["truncated"] is bool(extra)
    assert attachments[0]["content"] == preview + notice


@pytest.mark.parametrize(
    ("tail", "expected"),
    [(b"\r\nnext\rlast\r", "\nnext\nlast\n"), ("💡".encode("utf-8") + b"\xff end\xe2\x82", "💡� end�")],
    ids=["split-crlf-and-bare-cr", "split-utf8-and-invalid-input"],
)
def test_build_file_attachments_decodes_across_sniff_boundary(tmp_path: Path, tail: bytes, expected: str) -> None:
    prefix = "x" * (SNIFF_BYTES - 1)
    (tmp_path / "split.txt").write_bytes(prefix.encode("ascii") + tail)

    attachments, _ = build_file_attachments("@split.txt", tmp_path)

    assert attachments[0]["truncated"] is False
    assert attachments[0]["content"] == prefix + expected


def test_build_file_attachments_counts_unicode_line_separators(tmp_path: Path) -> None:
    preview = "first\u2028second\x85" * (MAX_ATTACHMENT_LINES // 2)
    (tmp_path / "unicode.txt").write_bytes((preview + "extra").encode("utf-8"))

    attachments, _ = build_file_attachments("@unicode.txt", tmp_path)

    assert attachments[0]["truncated"] is True
    assert attachments[0]["content"] == f"{preview}\n[truncated at the 2,000-line attachment limit - {CONTINUATION}]"


@pytest.mark.parametrize("nul_offset", [SNIFF_BYTES - 1, SNIFF_BYTES], ids=["inside-sniff", "outside-sniff"])
def test_build_file_attachments_sniffs_nul_in_first_8_kib(tmp_path: Path, nul_offset: int) -> None:
    raw = b"x" * nul_offset + b"\x00"
    (tmp_path / "blob.dat").write_bytes(raw)

    attachments, _ = build_file_attachments("@blob.dat", tmp_path)

    expected = "[binary file - content not attached]" if nul_offset < SNIFF_BYTES else raw.decode("ascii")
    assert attachments[0]["truncated"] is False
    assert attachments[0]["content"] == expected


def test_build_file_attachments_empty_text_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").touch()

    attachments, _ = build_file_attachments("@empty.txt", tmp_path)

    assert attachments[0]["truncated"] is False
    assert attachments[0]["content"] == ""


def test_build_file_attachments_text_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "unreadable.txt").touch()

    def fail_open(*_args: object, **_kwargs: object) -> FileIO:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", fail_open)

    attachments, _ = build_file_attachments("@unreadable.txt", tmp_path)

    assert attachments[0]["truncated"] is False
    assert attachments[0]["content"] == "[could not read file: denied]"


def test_build_file_attachments_allows_gitignored_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("hidden", encoding="utf-8")
    attachments, unresolved = build_file_attachments("@secret.txt", tmp_path)
    assert unresolved == []
    assert attachments[0]["content"] == "hidden"


def test_build_file_attachments_directory_shallow_listing(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "mod.py").write_text("", encoding="utf-8")
    attachments, _ = build_file_attachments("@pkg/", tmp_path)
    assert attachments[0]["is_dir"] is True
    assert attachments[0]["path"] == "pkg"
    assert "sub/" in attachments[0]["content"]
    assert "mod.py" in attachments[0]["content"]


def test_build_file_attachments_directory_listing_is_capped(tmp_path: Path) -> None:
    crowd = tmp_path / "crowd"
    crowd.mkdir()
    for index in range(MAX_DIR_ENTRIES + 10):
        (crowd / f"f{index:04}.txt").write_text("", encoding="utf-8")
    attachments, _ = build_file_attachments("@crowd", tmp_path)
    assert (
        f"[truncated: showing first {MAX_DIR_ENTRIES} of {MAX_DIR_ENTRIES + 10} entries]" in attachments[0]["content"]
    )


def test_build_file_attachments_rejects_paths_outside_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    attachments, unresolved = build_file_attachments(f"@{outside}", project)
    assert attachments == []
    assert unresolved == [str(outside)]


def test_build_file_attachments_accepts_absolute_path_inside_root(tmp_path: Path) -> None:
    (tmp_path / "inside.txt").write_text("yes", encoding="utf-8")
    attachments, unresolved = build_file_attachments(f"@{tmp_path / 'inside.txt'}", tmp_path)
    assert unresolved == []
    assert attachments[0]["path"] == "inside.txt"


def test_build_file_attachments_image_yields_image_attachment(tmp_path: Path) -> None:
    import base64

    payload = b"\x89PNG\r\n\x1a\nfake"
    (tmp_path / "shot.png").write_bytes(payload)
    attachments, unresolved = build_file_attachments("look at @shot.png", tmp_path)
    assert unresolved == []
    assert len(attachments) == 1
    assert attachments[0]["type"] == "image"
    assert attachments[0]["media_type"] == "image/png"
    assert attachments[0]["path"] == "shot.png"
    assert base64.b64decode(attachments[0]["data"]) == payload


def test_build_file_attachments_non_image_still_file(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("print(1)", encoding="utf-8")
    attachments, _ = build_file_attachments("see @code.py", tmp_path)
    assert len(attachments) == 1
    assert attachments[0]["type"] == "file"
    assert attachments[0]["content"] == "print(1)"


def test_build_file_attachments_jpeg_mention(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
    attachments, _ = build_file_attachments("@photo.jpg", tmp_path)
    assert len(attachments) == 1
    assert attachments[0]["type"] == "image"
    assert attachments[0]["media_type"] == "image/jpeg"
