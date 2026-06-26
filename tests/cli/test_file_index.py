"""Tests for the workspace file index behind @ mention autocomplete."""

from pathlib import Path

from kolega_code.cli.file_index import WorkspaceFileIndex, fuzzy_score


def _paths(entries) -> list[str]:
    return [entry.path for entry in entries]


def test_index_lists_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    paths = _paths(index.entries())
    assert "src/" in paths
    assert "src/main.py" in paths
    assert "README.md" in paths


def test_index_prunes_excluded_directories(tmp_path: Path) -> None:
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("", encoding="utf-8")
    (tmp_path / "app.js").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    paths = _paths(index.entries())
    assert paths == ["app.js"]


def test_index_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\nsecrets/\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key.pem").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    paths = _paths(index.entries())
    assert "kept.txt" in paths
    assert "ignored.txt" not in paths
    assert all(not path.startswith("secrets") for path in paths)


def test_index_skips_non_image_binaries(tmp_path: Path) -> None:
    (tmp_path / "archive.zip").write_bytes(b"zip")
    (tmp_path / "vector.svg").write_text("<svg/>", encoding="utf-8")  # not an attachable raster image
    (tmp_path / "code.py").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    paths = _paths(index.entries())
    assert "code.py" in paths
    assert "archive.zip" not in paths
    assert "vector.svg" not in paths


def test_index_includes_attachable_images(tmp_path: Path) -> None:
    # Images are @-mentionable (build_file_attachments attaches them), so they belong in completions.
    (tmp_path / "logo.png").write_bytes(b"png")
    (tmp_path / "photo.JPG").write_bytes(b"jpg")  # case-insensitive
    index = WorkspaceFileIndex(tmp_path)
    paths = _paths(index.entries())
    assert "logo.png" in paths
    assert "photo.JPG" in paths


def test_index_caps_total_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(WorkspaceFileIndex, "MAX_FILES", 10)
    for index_num in range(25):
        (tmp_path / f"f{index_num:03}.txt").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    assert len(index.entries()) == 10


def test_index_refreshes_after_ttl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(WorkspaceFileIndex, "TTL_SECONDS", 0.0)
    index = WorkspaceFileIndex(tmp_path)
    assert index.entries() == []
    (tmp_path / "late.txt").write_text("", encoding="utf-8")
    assert _paths(index.entries()) == ["late.txt"]


def test_cached_search_does_not_walk_until_refreshed(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    # No refresh yet: cached_search must not trigger a walk, so it returns nothing.
    assert index.cached_search("main") == []
    index.refresh()
    assert _paths(index.cached_search("main")) == ["main.py"]


def test_is_stale_tracks_refresh_and_ttl(tmp_path: Path, monkeypatch) -> None:
    index = WorkspaceFileIndex(tmp_path)
    assert index.is_stale() is True  # never refreshed
    index.refresh()
    assert index.is_stale() is False  # just refreshed, within TTL
    monkeypatch.setattr(WorkspaceFileIndex, "TTL_SECONDS", 0.0)
    assert index.is_stale() is True  # past (zero) TTL


def test_search_ranks_substring_above_subsequence(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "mandatory_ai_notes.txt").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    results = _paths(index.search("main"))
    assert results[0] == "main.py"


def test_search_empty_query_prefers_shallow_paths(tmp_path: Path) -> None:
    (tmp_path / "deep" / "nest").mkdir(parents=True)
    (tmp_path / "deep" / "nest" / "leaf.txt").write_text("", encoding="utf-8")
    (tmp_path / "top.txt").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    results = _paths(index.search(""))
    assert results[0] == "top.txt"


def test_fuzzy_score_no_match_returns_none() -> None:
    assert fuzzy_score("zzz", "src/main.py") is None


def test_fuzzy_score_subsequence_matches() -> None:
    assert fuzzy_score("smain", "src/main.py") is not None
    assert fuzzy_score("apppy", "kolega_code/cli/app.py") is not None


def test_search_dense_basename_match_beats_scattered_path_match(tmp_path: Path) -> None:
    (tmp_path / "cli").mkdir()
    (tmp_path / "agent").mkdir()
    (tmp_path / "cli" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "agent" / "prompts.py").write_text("", encoding="utf-8")
    index = WorkspaceFileIndex(tmp_path)
    results = _paths(index.search("apppy"))
    assert results[0] == "cli/app.py"
