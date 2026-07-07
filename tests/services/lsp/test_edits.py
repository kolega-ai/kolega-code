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


def test_rejects_outside_project_uri(tmp_path):
    path = tmp_path / "main.py"
    path.write_text("x\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.py"
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

    with pytest.raises(WorkspaceEditError, match="outside the project"):
        _applier(tmp_path).apply(edit)

    assert outside.read_text(encoding="utf-8") == "x\n"


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
