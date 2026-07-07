"""WorkspaceEdit application for trusted LSP edit operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from kolega_code.services.file_system import FileSystem


class WorkspaceEditError(ValueError):
    """Raised when an LSP WorkspaceEdit cannot be safely applied."""


@dataclass(frozen=True)
class TextChange:
    path: str
    old_text: str
    new_text: str


@dataclass(frozen=True)
class WorkspaceEditResult:
    applied: bool
    summaries: tuple[str, ...]
    touched_paths: tuple[str, ...]
    text_changes: tuple[TextChange, ...]


@dataclass(frozen=True)
class _ResolvedTextEdit:
    start: int
    end: int
    new_text: str
    index: int


@dataclass(frozen=True)
class _TextEditOp:
    path: str
    edits: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _ResourceOp:
    kind: str
    old_path: str | None = None
    new_path: str | None = None
    path: str | None = None
    overwrite: bool = False
    ignore_if_exists: bool = False
    ignore_if_missing: bool = False
    recursive: bool = False


_TEXT_DOCUMENT_EDIT_KINDS = {"textDocument/edit", "edit"}
_CREATE_FILE_KINDS = {"create", "createFile"}
_RENAME_FILE_KINDS = {"rename", "renameFile"}
_DELETE_FILE_KINDS = {"delete", "deleteFile"}


class WorkspaceEditApplier:
    """Validate, preview, and apply LSP WorkspaceEdit payloads."""

    def __init__(self, project_path: Path, filesystem: FileSystem) -> None:
        self.project_path = project_path.resolve()
        self.filesystem = filesystem

    def preview(self, edit: dict[str, Any] | None) -> WorkspaceEditResult:
        return self._apply(edit, should_write=False)

    def apply(self, edit: dict[str, Any] | None) -> WorkspaceEditResult:
        return self._apply(edit, should_write=True)

    def _apply(self, edit: dict[str, Any] | None, *, should_write: bool) -> WorkspaceEditResult:
        if not edit:
            return WorkspaceEditResult(
                applied=should_write,
                summaries=("No edits returned.",),
                touched_paths=(),
                text_changes=(),
            )

        ops = self._parse_workspace_edit(edit)
        if not ops:
            return WorkspaceEditResult(
                applied=should_write,
                summaries=("No edits returned.",),
                touched_paths=(),
                text_changes=(),
            )

        virtual_files: dict[str, str | None] = {}
        planned_text_changes: dict[int, TextChange] = {}
        summaries: list[str] = []
        touched_paths: list[str] = []
        text_changes: list[TextChange] = []

        for index, op in enumerate(ops):
            if isinstance(op, _TextEditOp):
                old_text = self._virtual_or_disk_content(op.path, virtual_files)
                new_text = self.apply_text_edits(old_text, op.edits)
                new_text = self._preserve_line_endings(op.path, old_text, new_text)
                virtual_files[op.path] = new_text
                change = TextChange(path=op.path, old_text=old_text, new_text=new_text)
                planned_text_changes[index] = change
                summaries.append(f"updated {op.path} ({len(op.edits)} text edit(s))")
                touched_paths.append(op.path)
                text_changes.append(change)
                continue

            self._validate_resource_op(op, virtual_files)
            self._apply_virtual_resource_op(op, virtual_files)
            summaries.append(self._resource_summary(op))
            for path in (op.path, op.old_path, op.new_path):
                if path:
                    touched_paths.append(path)

        if should_write:
            for index, op in enumerate(ops):
                if isinstance(op, _TextEditOp):
                    change = planned_text_changes[index]
                    self._write_text(change.path, change.new_text)
                else:
                    self._write_resource_op(op)

        return WorkspaceEditResult(
            applied=should_write,
            summaries=tuple(summaries),
            touched_paths=tuple(dict.fromkeys(touched_paths)),
            text_changes=tuple(text_changes),
        )

    def _parse_workspace_edit(self, edit: dict[str, Any]) -> list[_TextEditOp | _ResourceOp]:
        if not isinstance(edit, dict):
            raise WorkspaceEditError("WorkspaceEdit must be an object.")

        if edit.get("changes") is not None:
            changes = edit.get("changes")
            if not isinstance(changes, dict):
                raise WorkspaceEditError("WorkspaceEdit.changes must be an object.")
            return [
                _TextEditOp(
                    path=self._uri_to_relative_path(uri),
                    edits=tuple(self._validate_text_edit_list(text_edits)),
                )
                for uri, text_edits in changes.items()
                if text_edits
            ]

        document_changes = edit.get("documentChanges")
        if document_changes is None:
            return []
        if not isinstance(document_changes, list):
            raise WorkspaceEditError("WorkspaceEdit.documentChanges must be an array.")

        ops: list[_TextEditOp | _ResourceOp] = []
        for entry in document_changes:
            if not isinstance(entry, dict):
                raise WorkspaceEditError("WorkspaceEdit.documentChanges entries must be objects.")
            kind = entry.get("kind")
            if kind in _CREATE_FILE_KINDS:
                ops.append(self._parse_create_file(entry))
                continue
            if kind in _RENAME_FILE_KINDS:
                ops.append(self._parse_rename_file(entry))
                continue
            if kind in _DELETE_FILE_KINDS:
                ops.append(self._parse_delete_file(entry))
                continue
            if kind in _TEXT_DOCUMENT_EDIT_KINDS or entry.get("textDocument") is not None:
                text_document = entry.get("textDocument")
                if not isinstance(text_document, dict):
                    raise WorkspaceEditError("TextDocumentEdit.textDocument must be an object.")
                uri = text_document.get("uri")
                if not isinstance(uri, str):
                    raise WorkspaceEditError("TextDocumentEdit.textDocument.uri must be a string.")
                ops.append(
                    _TextEditOp(
                        path=self._uri_to_relative_path(uri),
                        edits=tuple(self._validate_text_edit_list(entry.get("edits", []))),
                    )
                )
                continue
            raise WorkspaceEditError(f"Unsupported WorkspaceEdit documentChanges entry: {kind!r}.")
        return ops

    def _parse_create_file(self, entry: dict[str, Any]) -> _ResourceOp:
        uri = entry.get("uri")
        if not isinstance(uri, str):
            raise WorkspaceEditError("CreateFile.uri must be a string.")
        options = entry.get("options") or {}
        if not isinstance(options, dict):
            raise WorkspaceEditError("CreateFile.options must be an object.")
        return _ResourceOp(
            kind="create",
            path=self._uri_to_relative_path(uri),
            overwrite=bool(options.get("overwrite")),
            ignore_if_exists=bool(options.get("ignoreIfExists")),
        )

    def _parse_rename_file(self, entry: dict[str, Any]) -> _ResourceOp:
        old_uri = entry.get("oldUri")
        new_uri = entry.get("newUri")
        if not isinstance(old_uri, str) or not isinstance(new_uri, str):
            raise WorkspaceEditError("RenameFile.oldUri and RenameFile.newUri must be strings.")
        options = entry.get("options") or {}
        if not isinstance(options, dict):
            raise WorkspaceEditError("RenameFile.options must be an object.")
        return _ResourceOp(
            kind="rename",
            old_path=self._uri_to_relative_path(old_uri),
            new_path=self._uri_to_relative_path(new_uri),
            overwrite=bool(options.get("overwrite")),
            ignore_if_exists=bool(options.get("ignoreIfExists")),
        )

    def _parse_delete_file(self, entry: dict[str, Any]) -> _ResourceOp:
        uri = entry.get("uri")
        if not isinstance(uri, str):
            raise WorkspaceEditError("DeleteFile.uri must be a string.")
        options = entry.get("options") or {}
        if not isinstance(options, dict):
            raise WorkspaceEditError("DeleteFile.options must be an object.")
        return _ResourceOp(
            kind="delete",
            path=self._uri_to_relative_path(uri),
            ignore_if_missing=bool(options.get("ignoreIfNotExists")),
            recursive=bool(options.get("recursive")),
        )

    def _validate_text_edit_list(self, edits: Any) -> list[dict[str, Any]]:
        if not isinstance(edits, list):
            raise WorkspaceEditError("Text edits must be an array.")
        for edit in edits:
            if not isinstance(edit, dict):
                raise WorkspaceEditError("Text edit entries must be objects.")
            if not isinstance(edit.get("range"), dict):
                raise WorkspaceEditError("Text edit range must be an object.")
            if not isinstance(edit.get("newText"), str):
                raise WorkspaceEditError("Text edit newText must be a string.")
        return edits

    def apply_text_edits(self, content: str, edits: tuple[dict[str, Any], ...]) -> str:
        resolved: list[_ResolvedTextEdit] = []
        seen_non_empty: set[tuple[int, int, str]] = set()
        for index, edit in enumerate(edits):
            edit_range = edit["range"]
            start = self._offset_for_position(content, edit_range.get("start"))
            end = self._offset_for_position(content, edit_range.get("end"))
            if end < start:
                raise WorkspaceEditError("Text edit range end is before start.")
            new_text = edit["newText"]
            key = (start, end, new_text)
            if start != end and key in seen_non_empty:
                continue
            if start != end:
                seen_non_empty.add(key)
            resolved.append(
                _ResolvedTextEdit(
                    start=start,
                    end=end,
                    new_text=new_text,
                    index=index,
                )
            )

        ascending = sorted(resolved, key=lambda item: (item.start, item.end, item.index))
        for previous, current in zip(ascending, ascending[1:]):
            if previous.end > current.start:
                raise WorkspaceEditError("WorkspaceEdit contains overlapping text edits.")

        updated = content
        for edit in sorted(resolved, key=lambda item: (item.start, item.index), reverse=True):
            updated = updated[: edit.start] + edit.new_text + updated[edit.end :]
        return updated

    def _offset_for_position(self, content: str, position: Any) -> int:
        if not isinstance(position, dict):
            raise WorkspaceEditError("Text edit position must be an object.")
        line = position.get("line")
        character = position.get("character")
        if not isinstance(line, int) or not isinstance(character, int):
            raise WorkspaceEditError("Text edit line and character must be integers.")
        if line < 0 or character < 0:
            raise WorkspaceEditError("Text edit line and character must be non-negative.")

        lines = content.splitlines(keepends=True)
        if not lines:
            if line == 0 and character == 0:
                return 0
            raise WorkspaceEditError("Text edit position is outside the document.")

        if line == len(lines) and character == 0:
            return len(content)
        if line >= len(lines):
            raise WorkspaceEditError("Text edit line is outside the document.")

        line_start = sum(len(existing_line) for existing_line in lines[:line])
        line_text = lines[line]
        line_body = line_text
        if line_body.endswith("\r\n"):
            line_body = line_body[:-2]
        elif line_body.endswith("\n") or line_body.endswith("\r"):
            line_body = line_body[:-1]
        return line_start + self._python_offset_for_utf16(line_body, character)

    def _python_offset_for_utf16(self, text: str, utf16_character: int) -> int:
        units_seen = 0
        for index, char in enumerate(text):
            if units_seen == utf16_character:
                return index
            units_seen += 2 if ord(char) > 0xFFFF else 1
            if units_seen > utf16_character:
                raise WorkspaceEditError("Text edit character splits a UTF-16 surrogate pair.")
        if units_seen == utf16_character:
            return len(text)
        raise WorkspaceEditError("Text edit character is outside the line.")

    def _uri_to_relative_path(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise WorkspaceEditError(f"Unsupported URI scheme for edit: {parsed.scheme!r}.")
        if parsed.netloc not in ("", "localhost"):
            raise WorkspaceEditError(f"Unsupported file URI host: {parsed.netloc!r}.")
        absolute_path = Path(unquote(parsed.path)).resolve()
        try:
            relative = absolute_path.relative_to(self.project_path)
        except ValueError as exc:
            raise WorkspaceEditError(f"WorkspaceEdit path is outside the project: {uri}") from exc
        relative_text = relative.as_posix()
        if not relative_text or relative_text == ".":
            raise WorkspaceEditError("WorkspaceEdit path must not be the project root.")
        return relative_text

    def _virtual_or_disk_content(self, path: str, virtual_files: dict[str, str | None]) -> str:
        if path in virtual_files:
            content = virtual_files[path]
            if content is None:
                raise WorkspaceEditError(f"WorkspaceEdit targets a deleted file: {path}")
            return content
        if not self.filesystem.exists(path):
            raise WorkspaceEditError(f"WorkspaceEdit targets a missing file: {path}")
        if self.filesystem.is_dir(path):
            raise WorkspaceEditError(f"WorkspaceEdit targets a directory as text: {path}")
        return self.filesystem.read_text(path)

    def _validate_resource_op(self, op: _ResourceOp, virtual_files: dict[str, str | None]) -> None:
        if op.kind == "create":
            assert op.path is not None
            exists = self._virtual_exists(op.path, virtual_files)
            if exists and not (op.overwrite or op.ignore_if_exists):
                raise WorkspaceEditError(f"CreateFile target already exists: {op.path}")
            return

        if op.kind == "rename":
            assert op.old_path is not None and op.new_path is not None
            old_exists = self._virtual_exists(op.old_path, virtual_files)
            new_exists = self._virtual_exists(op.new_path, virtual_files)
            if not old_exists:
                raise WorkspaceEditError(f"RenameFile source does not exist: {op.old_path}")
            if new_exists and not (op.overwrite or op.ignore_if_exists):
                raise WorkspaceEditError(f"RenameFile destination already exists: {op.new_path}")
            return

        if op.kind == "delete":
            assert op.path is not None
            exists = self._virtual_exists(op.path, virtual_files)
            if not exists and not op.ignore_if_missing:
                raise WorkspaceEditError(f"DeleteFile target does not exist: {op.path}")
            return

        raise WorkspaceEditError(f"Unsupported resource operation: {op.kind}")

    def _apply_virtual_resource_op(self, op: _ResourceOp, virtual_files: dict[str, str | None]) -> None:
        if op.kind == "create":
            assert op.path is not None
            if op.ignore_if_exists and self._virtual_exists(op.path, virtual_files):
                return
            virtual_files[op.path] = ""
            return

        if op.kind == "rename":
            assert op.old_path is not None and op.new_path is not None
            if op.ignore_if_exists and self._virtual_exists(op.new_path, virtual_files):
                return
            if op.old_path in virtual_files:
                virtual_files[op.new_path] = virtual_files.pop(op.old_path)
            elif self.filesystem.exists(op.old_path) and not self.filesystem.is_dir(op.old_path):
                virtual_files[op.new_path] = self.filesystem.read_text(op.old_path)
            virtual_files[op.old_path] = None
            return

        if op.kind == "delete":
            assert op.path is not None
            if op.ignore_if_missing and not self._virtual_exists(op.path, virtual_files):
                return
            virtual_files[op.path] = None
            return

    def _write_resource_op(self, op: _ResourceOp) -> None:
        if op.kind == "create":
            assert op.path is not None
            if op.ignore_if_exists and self.filesystem.exists(op.path):
                return
            if op.overwrite or not self.filesystem.exists(op.path):
                self._write_text(op.path, "")
            return

        if op.kind == "rename":
            assert op.old_path is not None and op.new_path is not None
            if op.ignore_if_exists and self.filesystem.exists(op.new_path):
                return
            if op.overwrite and self.filesystem.exists(op.new_path):
                self._remove_path(op.new_path, recursive=True)
            self._ensure_parent_directory(op.new_path)
            self.filesystem.rename(op.old_path, op.new_path)
            return

        if op.kind == "delete":
            assert op.path is not None
            if op.ignore_if_missing and not self.filesystem.exists(op.path):
                return
            self._remove_path(op.path, recursive=op.recursive)
            return

    def _virtual_exists(self, path: str, virtual_files: dict[str, str | None]) -> bool:
        if path in virtual_files:
            return virtual_files[path] is not None
        return self.filesystem.exists(path)

    def _write_text(self, path: str, content: str) -> None:
        self._ensure_parent_directory(path)
        self.filesystem.write_text(path, content)

    def _ensure_parent_directory(self, path: str) -> None:
        parent = self.filesystem.get_parent(path)
        if parent and parent != "." and not self.filesystem.exists(parent):
            self.filesystem.mkdir(parent, parents=True, exist_ok=True)

    def _remove_path(self, path: str, *, recursive: bool) -> None:
        if self.filesystem.is_dir(path):
            if recursive:
                self.filesystem.rmtree(path)
            else:
                self.filesystem.rmdir(path)
        else:
            self.filesystem.remove(path)

    def _resource_summary(self, op: _ResourceOp) -> str:
        if op.kind == "create":
            assert op.path is not None
            return f"created {op.path}"
        if op.kind == "rename":
            assert op.old_path is not None and op.new_path is not None
            return f"renamed {op.old_path} -> {op.new_path}"
        if op.kind == "delete":
            assert op.path is not None
            return f"deleted {op.path}"
        return op.kind

    def _preserve_line_endings(self, path: str, old_text: str, new_text: str) -> str:
        line_ending = self._detect_dominant_line_ending(path, old_text)
        if line_ending == "\n":
            return new_text
        return new_text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", line_ending)

    def _detect_dominant_line_ending(self, path: str, old_text: str) -> str:
        try:
            raw = self.filesystem.read_bytes(path)
        except Exception:
            raw_text = old_text
            crlf_count = raw_text.count("\r\n")
            without_crlf = raw_text.replace("\r\n", "")
            lf_count = without_crlf.count("\n")
            cr_count = without_crlf.count("\r")
        else:
            crlf_count = raw.count(b"\r\n")
            without_crlf = raw.replace(b"\r\n", b"")
            lf_count = without_crlf.count(b"\n")
            cr_count = without_crlf.count(b"\r")

        if crlf_count > 0 and crlf_count >= lf_count and crlf_count >= cr_count:
            return "\r\n"
        if cr_count > 0 and cr_count > lf_count:
            return "\r"
        return "\n"
