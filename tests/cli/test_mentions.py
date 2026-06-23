"""Tests for @ mention parsing and file attachment expansion."""

from pathlib import Path

from kolega_code.cli.mentions import (
    MAX_ATTACHMENT_LINES,
    MAX_DIR_ENTRIES,
    build_file_attachments,
    parse_mentions,
)


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


def test_build_file_attachments_truncates_long_files(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("line\n" * (MAX_ATTACHMENT_LINES + 50), encoding="utf-8")
    attachments, _ = build_file_attachments("@big.txt", tmp_path)
    assert attachments[0]["truncated"] is True
    assert "[truncated: showing first" in attachments[0]["content"]


def test_build_file_attachments_binary_file_gets_stub(tmp_path: Path) -> None:
    (tmp_path / "blob.dat").write_bytes(b"\x00\x01\x02binary")
    attachments, _ = build_file_attachments("@blob.dat", tmp_path)
    assert attachments[0]["content"] == "[binary file - content not attached]"


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
    assert f"[truncated: showing first {MAX_DIR_ENTRIES} of {MAX_DIR_ENTRIES + 10} entries]" in attachments[0]["content"]


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
