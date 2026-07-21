from __future__ import annotations

import pytest

from kolega_code.services.file_system import LocalFileSystem
from kolega_code.services.lsp.edits import WorkspaceEditApplier, WorkspaceEditError


def _applier(tmp_path):
    return WorkspaceEditApplier(tmp_path, LocalFileSystem(root_path=tmp_path))


def test_applies_workspace_changes_with_utf16_positions(tmp_path):
    path = tmp_path / "emoji.py"
    path.write_text("value = '😀'\nname = old\n", encoding="utf-8")
    edit = {
        "changes": {
            path.as_uri(): [
                {
                    "range": {
                        "start": {"line": 1, "character": 7},
                        "end": {"line": 1, "character": 10},
                    },
                    "newText": "new",
                }
            ]
        }
    }

    result = _applier(tmp_path).apply(edit)

    assert result.applied is True
    assert path.read_text(encoding="utf-8") == "value = '😀'\nname = new\n"


def test_same_position_inserts_keep_lsp_order(tmp_path):
    path = tmp_path / "main.py"
    path.write_text("ab\n", encoding="utf-8")
    edit = {
        "changes": {
            path.as_uri(): [
                {
                    "range": {
                        "start": {"line": 0, "character": 1},
                        "end": {"line": 0, "character": 1},
                    },
                    "newText": "X",
                },
                {
                    "range": {
                        "start": {"line": 0, "character": 1},
                        "end": {"line": 0, "character": 1},
                    },
                    "newText": "Y",
                },
            ]
        }
    }

    _applier(tmp_path).apply(edit)

    assert path.read_text(encoding="utf-8") == "aXYb\n"


def test_overlapping_text_edits_are_rejected_without_writing(tmp_path):
    path = tmp_path / "main.py"
    path.write_text("abcdef\n", encoding="utf-8")
    edit = {
        "changes": {
            path.as_uri(): [
                {
                    "range": {
                        "start": {"line": 0, "character": 1},
                        "end": {"line": 0, "character": 4},
                    },
                    "newText": "X",
                },
                {
                    "range": {
                        "start": {"line": 0, "character": 3},
                        "end": {"line": 0, "character": 5},
                    },
                    "newText": "Y",
                },
            ]
        }
    }

    with pytest.raises(WorkspaceEditError, match="overlapping"):
        _applier(tmp_path).apply(edit)

    assert path.read_text(encoding="utf-8") == "abcdef\n"


def test_preserves_crlf_line_endings(tmp_path):
    path = tmp_path / "crlf.py"
    path.write_bytes(b"name = old\r\n")
    edit = {
        "changes": {
            path.as_uri(): [
                {
                    "range": {
                        "start": {"line": 0, "character": 7},
                        "end": {"line": 0, "character": 10},
                    },
                    "newText": "new",
                }
            ]
        }
    }

    _applier(tmp_path).apply(edit)

    assert path.read_bytes() == b"name = new\r\n"


def test_applies_text_edit_outside_project(tmp_path):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    outside = outside_dir / "outside.py"
    outside.write_text("x\n", encoding="utf-8")
    edit = {
        "changes": {
            outside.as_uri(): [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                    "newText": "y",
                }
            ]
        }
    }

    result = _applier(project).apply(edit)

    assert result.applied is True
    assert result.touched_paths == (str(outside),)
    assert outside.read_text(encoding="utf-8") == "y\n"


def test_applies_external_create_rename_and_delete_resource_operations(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    source = outside / "rename-source.py"
    renamed = outside / "renamed" / "rename-destination.py"
    created = outside / "created.py"
    deleted = outside / "deleted.py"
    source.write_text("source\n", encoding="utf-8")
    deleted.write_text("delete me\n", encoding="utf-8")
    edit = {
        "documentChanges": [
            {"kind": "create", "uri": created.as_uri()},
            {"kind": "rename", "oldUri": source.as_uri(), "newUri": renamed.as_uri()},
            {"kind": "delete", "uri": deleted.as_uri()},
        ]
    }

    result = _applier(project).apply(edit)

    assert result.applied is True
    assert created.read_text(encoding="utf-8") == ""
    assert not source.exists()
    assert renamed.read_text(encoding="utf-8") == "source\n"
    assert not deleted.exists()
    assert set(result.touched_paths) == {str(created), str(source), str(renamed), str(deleted)}


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("https://example.com/outside.py", "Unsupported URI scheme"),
        ("file://remote.example/outside.py", "Unsupported file URI host"),
    ],
)
def test_rejects_non_file_and_non_local_file_uris(tmp_path, uri, message):
    project = tmp_path / "project"
    project.mkdir()
    edit = {
        "changes": {
            uri: [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "newText": "x",
                }
            ]
        }
    }

    with pytest.raises(WorkspaceEditError, match=message):
        _applier(project).apply(edit)


def test_document_changes_rename_file(tmp_path):
    source = tmp_path / "old.py"
    dest = tmp_path / "pkg" / "new.py"
    source.write_text("value = 1\n", encoding="utf-8")
    edit = {
        "documentChanges": [
            {
                "kind": "rename",
                "oldUri": source.as_uri(),
                "newUri": dest.as_uri(),
                "options": {"overwrite": False},
            }
        ]
    }

    result = _applier(tmp_path).apply(edit)

    assert result.applied is True
    assert not source.exists()
    assert dest.read_text(encoding="utf-8") == "value = 1\n"
