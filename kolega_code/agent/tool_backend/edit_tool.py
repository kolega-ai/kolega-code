from __future__ import annotations

from hashlib import sha256
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .base_tool import BaseTool
from .codex_patch import PatchOperation, apply_update_chunks, parse_codex_patch
from .edit_preview import build_diff_preview, build_head_preview
from .hashline_v2 import apply_hashline_edits, parse_edits

if TYPE_CHECKING:
    from kolega_code.services.lsp import LspManager
    from kolega_code.services.snapshots import SnapshotService


_BLOCK_PATTERN = re.compile(r"<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE", re.DOTALL)
_SMART_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)


@dataclass(frozen=True)
class SearchReplaceBlock:
    block_number: int
    search_text: str
    replace_text: str


@dataclass(frozen=True)
class ResolvedReplacement:
    block_number: int
    start: int
    end: int
    replace_text: str
    pass_name: str


class EditTool(BaseTool):
    def __init__(
        self,
        *args,
        lsp_manager: Optional["LspManager"] = None,
        snapshot_service: Optional["SnapshotService"] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._lsp_manager = lsp_manager
        self._snapshot_service = snapshot_service
        self._read_versions: dict[str, str] = {}

    def observe_read(self, path: str) -> None:
        """Record the current contents after a successful model-facing read."""

        normalized = self._normalize_claude_path(path)
        if self.filesystem.exists(normalized) and self.filesystem.is_file(normalized):
            self._read_versions[normalized] = self._content_digest(self.filesystem.read_text(normalized))

    async def claude_edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Apply a Claude-style exact string replacement."""

        path = self._normalize_claude_path(file_path)
        if old_string == new_string:
            raise ValueError("No changes to make: old_string and new_string are exactly the same.")

        exists = self.filesystem.exists(path)
        if exists and not self.filesystem.is_file(path):
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        if not exists and old_string:
            raise FileNotFoundError(f"File does not exist: {path}")

        original_content: Optional[str] = None
        if exists:
            original_content = self.filesystem.read_text(path)
            self._require_claude_read(path, original_content)
            if old_string == "":
                raise ValueError(
                    "old_string cannot be empty when editing an existing file. "
                    "Use write for an intentional full-file replacement."
                )

            line_ending = self._detect_dominant_line_ending(path, original_content)
            bom = "\ufeff" if original_content.startswith("\ufeff") else ""
            logical_original = self._normalize_line_endings(original_content[len(bom) :], "\n")
            logical_old = self._normalize_line_endings(old_string, "\n")
            logical_new = self._normalize_line_endings(new_string, "\n")
            occurrences = logical_original.count(logical_old)
            if occurrences == 0:
                raise ValueError("String to replace not found in file.")
            if occurrences > 1 and not replace_all:
                raise ValueError(
                    f"Found {occurrences} matches for old_string. "
                    "Provide more surrounding text to make it unique or set replace_all=true."
                )
            logical_updated = logical_original.replace(logical_old, logical_new, -1 if replace_all else 1)
            updated_content = bom + self._normalize_line_endings(logical_updated, line_ending)
        else:
            updated_content = new_string

        blocked_msg = self._enforce_vibe_edit_policy(path)
        if blocked_msg:
            return blocked_msg

        def _write() -> None:
            parent = self.filesystem.get_parent(path)
            if parent and parent != "." and not self.filesystem.exists(parent):
                self.filesystem.create_directory(parent)
            self.filesystem.write_text(path, updated_content)

        self._mutate_with_optional_snapshot(
            tool_name="edit",
            reason=f"claude edit {path}",
            paths=self._snapshot_paths_for_write(path),
            mutate=_write,
        )
        preview = (
            build_diff_preview(original_content, updated_content, path)
            if original_content is not None
            else build_head_preview(updated_content, path)
        )
        await self.send_edit_preview(
            preview,
            tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
            tool_name="edit",
        )
        self._read_versions[path] = self._content_digest(self.filesystem.read_text(path))
        result = f"Edited {path}" if exists else f"Created {path}"
        diagnostics = await self._maybe_append_lsp_diagnostics(path)
        return result + diagnostics

    async def claude_write(self, file_path: str, content: str) -> str:
        """Create or overwrite a file using the Claude-style write contract."""

        path = self._normalize_claude_path(file_path)
        exists = self.filesystem.exists(path)
        if exists and not self.filesystem.is_file(path):
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")

        original_content: Optional[str] = None
        if exists:
            original_content = self.filesystem.read_text(path)
            self._require_claude_read(path, original_content)
            source_has_bom = original_content.startswith("\ufeff")
            content_has_bom = content.startswith("\ufeff")
            content_body = content[1:] if content_has_bom else content
            content = ("\ufeff" if source_has_bom or content_has_bom else "") + self._normalize_line_endings(
                content_body, self._detect_dominant_line_ending(path, original_content)
            )

        blocked_msg = self._enforce_vibe_edit_policy(path)
        if blocked_msg:
            return blocked_msg

        def _write() -> None:
            parent = self.filesystem.get_parent(path)
            if parent and parent != "." and not self.filesystem.exists(parent):
                self.filesystem.create_directory(parent)
            self.filesystem.write_text(path, content)

        self._mutate_with_optional_snapshot(
            tool_name="write",
            reason=f"claude write {path}",
            paths=self._snapshot_paths_for_write(path),
            mutate=_write,
        )
        preview = (
            build_diff_preview(original_content, content, path)
            if original_content is not None
            else build_head_preview(content, path)
        )
        await self.send_edit_preview(
            preview,
            tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
            tool_name="write",
        )
        self._read_versions[path] = self._content_digest(self.filesystem.read_text(path))
        result = f"Wrote {path}"
        diagnostics = await self._maybe_append_lsp_diagnostics(path)
        return result + diagnostics

    async def edit(self, path: str, block: str) -> str:
        """
        Edit a file using a single search and replace block.

        Args:
            path: Path to the file to edit. Relative to the project root is preferred; an absolute path is also accepted.
            block: A single search and replace block

        Returns:
            A short summary of the update
        """
        parsed_blocks = self._parse_blocks(block)
        if len(parsed_blocks) > 1:
            error_msg = "Multiple search and replace blocks provided. Use multi_edit for multiple blocks."
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise ValueError(error_msg)
        result = await self._edit_blocks(path, parsed_blocks, tool_name="edit")
        lsp_diags = await self._maybe_append_lsp_diagnostics(path)
        if lsp_diags:
            result += lsp_diags
        return result

    async def multi_edit(self, path: str, blocks: str) -> str:
        """
        Edit a file using one or more search and replace blocks.

        All blocks are matched against the original file contents before any replacement is applied.
        Replacements are applied in reverse file order to avoid offset shifts.

        Args:
            path: Path to the file to edit. Relative to the project root is preferred; an absolute path is also accepted.
            blocks: One or more search and replace blocks

        Returns:
            A short summary of the update
        """
        parsed_blocks = self._parse_blocks(blocks)
        result = await self._edit_blocks(path, parsed_blocks, tool_name="multi_edit")
        lsp_diags = await self._maybe_append_lsp_diagnostics(path)
        if lsp_diags:
            result += lsp_diags
        return result

    async def write(self, path: str, content: str) -> str:
        """
        Write content to a file, creating the file if needed or replacing it if it exists.

        Args:
            path: Path to write. Relative to the project root is preferred; an absolute path is also accepted.
            content: Content to write to the file

        Returns:
            A short summary of the write
        """
        try:
            blocked_msg = self._enforce_vibe_edit_policy(path)
            if blocked_msg:
                return blocked_msg

            exists = self.filesystem.exists(path)
            old_content = None
            if exists:
                try:
                    old_content = self.filesystem.read_text(path)
                except Exception:
                    old_content = None

            parent_dir = self.filesystem.get_parent(path)

            # Preserve the file's dominant line ending on overwrite; new files
            # keep the line endings the caller provided (typically LF).
            if exists and old_content is not None:
                line_ending = self._detect_dominant_line_ending(path, old_content)
                content = self._normalize_line_endings(content, line_ending)

            def _write() -> None:
                if parent_dir and parent_dir != "." and not self.filesystem.exists(parent_dir):
                    self.filesystem.create_directory(parent_dir)
                self.filesystem.write_text(path, content)

            self._mutate_with_optional_snapshot(
                tool_name="write",
                reason=f"write {path}",
                paths=self._snapshot_paths_for_write(path),
                mutate=_write,
            )

            preview = (
                build_diff_preview(old_content, content, path)
                if old_content is not None
                else build_head_preview(content, path)
            )
            await self.send_edit_preview(
                preview,
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="write",
            )
            result = f"Wrote {path}"
            lsp_diags = await self._maybe_append_lsp_diagnostics(path)
            if lsp_diags:
                result += lsp_diags
            return result
        except PermissionError:
            error_msg = f"Permission denied when writing to file: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to write to file {path}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise

    async def hashline_edit(
        self,
        path: str,
        edits: list[dict[str, object]],
        delete: bool = False,
        rename: Optional[str] = None,
    ) -> str:
        """Apply one original Hashline v2 edit transaction to a file."""

        if not isinstance(delete, bool):
            raise ValueError("delete must be a boolean.")
        if rename is not None and not isinstance(rename, str):
            raise ValueError("rename must be a string path.")
        source = self._normalize_claude_path(path)
        destination = self._normalize_claude_path(rename) if rename else None
        if destination == source:
            destination = None
        if delete and destination is not None:
            raise ValueError("delete and rename cannot be combined.")
        if delete and edits:
            raise ValueError("delete requires an empty edits array.")

        for affected in dict.fromkeys(item for item in (source, destination) if item is not None):
            blocked = self._enforce_vibe_edit_policy(affected)
            if blocked:
                return blocked

        exists = self.filesystem.exists(source)
        if exists and not self.filesystem.is_file(source):
            raise IsADirectoryError(f"Path is a directory, not a file: {source}")

        parsed = parse_edits(edits)
        if delete:
            original = self.filesystem.read_text(source) if exists else None

            def _delete() -> None:
                if self.filesystem.exists(source):
                    self.filesystem.remove(source, missing_ok=True)

            self._mutate_with_optional_snapshot(
                tool_name="edit",
                reason=f"hashline delete {source}",
                paths=[source],
                mutate=_delete,
            )
            if original is not None:
                await self.send_edit_preview(
                    build_diff_preview(original, "", source),
                    tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                    tool_name="edit",
                )
            return f"Deleted {source}"

        if not exists and destination is not None:
            raise ValueError("Cannot rename a file that does not exist.")

        original: Optional[str] = self.filesystem.read_text(source) if exists else None
        if original is None:
            if any(
                edit.op not in {"append", "prepend"}
                or (edit.op == "append" and edit.after is not None)
                or (edit.op == "prepend" and edit.before is not None)
                for edit in parsed
            ):
                raise FileNotFoundError(
                    f"File not found: {source}. A missing file can only be created with unanchored append/prepend."
                )
            logical_original = ""
        else:
            logical_original = original

        bom = "\ufeff" if logical_original.startswith("\ufeff") else ""
        logical_original = logical_original[len(bom) :]
        line_ending = self._detect_dominant_line_ending(source, original) if original is not None else "\n"
        normalized_original = self._normalize_line_endings(logical_original, "\n")

        if original is None and not parsed:
            normalized_updated = ""
        elif parsed:
            normalized_updated = apply_hashline_edits(normalized_original, parsed)
        elif destination is not None:
            normalized_updated = normalized_original
        else:
            raise ValueError("No changes made. The edits array is empty.")

        updated = bom + self._normalize_line_endings(normalized_updated, line_ending)
        target = destination or source
        if self.filesystem.exists(target) and not self.filesystem.is_file(target):
            raise IsADirectoryError(f"Destination is a directory, not a file: {target}")
        self._validate_patch_parent(target)

        snapshot_paths = [source]
        if destination is not None:
            snapshot_paths.append(destination)
        snapshot_paths.extend(self._snapshot_paths_for_write(target))
        snapshot_paths = list(dict.fromkeys(snapshot_paths))

        def _write() -> None:
            parent = self.filesystem.get_parent(target)
            if parent and parent != "." and not self.filesystem.exists(parent):
                self.filesystem.create_directory(parent)
            self.filesystem.write_text(target, updated)
            if destination is not None and self.filesystem.exists(source):
                self.filesystem.remove(source, missing_ok=True)

        self._mutate_with_optional_snapshot(
            tool_name="edit",
            reason=(f"hashline move {source} -> {destination}" if destination else f"hashline edit {source}"),
            paths=snapshot_paths,
            mutate=_write,
        )

        if destination is not None:
            await self.send_edit_preview(
                build_diff_preview(original or "", "", source),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="edit",
            )
            await self.send_edit_preview(
                build_head_preview(updated, destination),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="edit",
            )
            result = f"Updated and moved {source} to {destination}"
        elif original is None:
            await self.send_edit_preview(
                build_head_preview(updated, source),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="edit",
            )
            result = f"Created {source}"
        else:
            await self.send_edit_preview(
                build_diff_preview(original, updated, source),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="edit",
            )
            result = f"Updated {source}"

        diagnostics = await self._maybe_append_lsp_diagnostics(target)
        return result + diagnostics

    async def hashline_write(self, path: str, content: str) -> str:
        """Create or overwrite a project file for the Hashline v2 surface."""

        normalized = self._normalize_claude_path(path)
        return await self.write(normalized, content)

    async def apply_patch(self, patch: str) -> str:
        """Apply a complete Codex patch atomically after in-memory validation."""
        operations = parse_codex_patch(patch)
        normalized_operations = [self._normalize_patch_operation(operation) for operation in operations]

        affected = list(
            dict.fromkeys(
                path
                for operation in normalized_operations
                for path in (operation.path, operation.move_to)
                if path is not None
            )
        )
        for path in affected:
            blocked = self._enforce_vibe_edit_policy(path)
            if blocked:
                return blocked

        initial: dict[str, Optional[str]] = {}
        working: dict[str, Optional[str]] = {}

        def load(path: str, *, must_exist: bool = False) -> Optional[str]:
            if path in working:
                value = working[path]
            elif self.filesystem.exists(path):
                if not self.filesystem.is_file(path):
                    raise IsADirectoryError(f"Patch path is not a file: {path}")
                value = self.filesystem.read_text(path)
                initial[path] = value
                working[path] = value
            else:
                value = None
                initial[path] = None
                working[path] = None
            if must_exist and value is None:
                raise FileNotFoundError(f"File not found: {path}")
            return value

        for operation in normalized_operations:
            if operation.kind == "add":
                previous = load(operation.path)
                content = "\n".join(operation.add_lines) + "\n"
                if previous is not None:
                    content = self._normalize_line_endings(
                        content, self._detect_dominant_line_ending(operation.path, previous)
                    )
                working[operation.path] = content
                continue

            if operation.kind == "delete":
                load(operation.path, must_exist=True)
                working[operation.path] = None
                continue

            original = load(operation.path, must_exist=True)
            assert original is not None
            updated = apply_update_chunks(original, operation.chunks, operation.path)
            if operation.move_to:
                load(operation.move_to)
                working[operation.path] = None
                working[operation.move_to] = updated
            else:
                working[operation.path] = updated

        for path, content in working.items():
            if content is not None:
                self._validate_patch_parent(path)

        changed = [path for path in working if initial.get(path) != working[path]]
        snapshot_paths = list(changed)
        for path in changed:
            if working[path] is not None:
                snapshot_paths.extend(self._snapshot_paths_for_write(path))
        snapshot_paths = list(dict.fromkeys(snapshot_paths))

        def mutate() -> None:
            for path in sorted(
                (item for item in changed if working[item] is None),
                key=lambda item: len(Path(item).parts),
                reverse=True,
            ):
                if self.filesystem.exists(path):
                    self.filesystem.remove(path, missing_ok=True)
            for path in sorted(
                (item for item in changed if working[item] is not None), key=lambda item: len(Path(item).parts)
            ):
                parent = self.filesystem.get_parent(path)
                if parent and parent != "." and not self.filesystem.exists(parent):
                    self.filesystem.create_directory(parent)
                self.filesystem.write_text(path, working[path] or "")

        self._mutate_with_optional_snapshot(
            tool_name="apply_patch",
            reason=f"apply_patch ({len(normalized_operations)} operations)",
            paths=snapshot_paths,
            mutate=mutate,
        )

        for path in changed:
            old = initial.get(path)
            new = working[path]
            if new is None:
                preview = build_diff_preview(old or "", "", path)
            elif old is None:
                preview = build_head_preview(new, path)
            else:
                preview = build_diff_preview(old, new, path)
            await self.send_edit_preview(
                preview,
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="apply_patch",
            )

        summaries: list[str] = []
        for operation in normalized_operations:
            if operation.kind == "add":
                summaries.append(f"A {operation.path}")
            elif operation.kind == "delete":
                summaries.append(f"D {operation.path}")
            elif operation.move_to:
                summaries.append(f"M {operation.path} -> {operation.move_to}")
            else:
                summaries.append(f"M {operation.path}")

        result = "Success. Updated the following files:\n" + "\n".join(summaries)
        diagnostics: list[str] = []
        for path in changed:
            if working[path] is not None:
                item = await self._maybe_append_lsp_diagnostics(path)
                if item:
                    diagnostics.append(item.strip())
        if diagnostics:
            result += "\n\n" + "\n\n".join(diagnostics)
        return result

    def _normalize_patch_operation(self, operation: PatchOperation) -> PatchOperation:
        return PatchOperation(
            kind=operation.kind,
            path=self._normalize_patch_path(operation.path),
            move_to=self._normalize_patch_path(operation.move_to) if operation.move_to else None,
            add_lines=operation.add_lines,
            chunks=operation.chunks,
        )

    def _normalize_patch_path(self, path: str) -> str:
        if not path or not path.strip():
            raise ValueError("Patch path is required.")
        return path

    def _normalize_claude_path(self, path: str) -> str:
        """Validate a model-provided path without changing its filesystem semantics."""

        if not path or not path.strip():
            raise ValueError("file_path is required")
        return path

    @staticmethod
    def _content_digest(content: str) -> str:
        return sha256(content.encode("utf-8")).hexdigest()

    def _require_claude_read(self, path: str, content: str) -> None:
        observed = self._read_versions.get(path)
        if observed is None:
            raise ValueError(f"File has not been read yet: {path}. Read it first before editing it.")
        if observed != self._content_digest(content):
            raise ValueError(f"File has changed since it was read: {path}. Read it again before editing it.")

    def _validate_patch_parent(self, path: str) -> None:
        parent = self.filesystem.get_parent(path)
        while parent and parent != ".":
            if self.filesystem.exists(parent) and not self.filesystem.is_dir(parent):
                raise NotADirectoryError(f"Patch parent is not a directory: {parent}")
            next_parent = self.filesystem.get_parent(parent)
            if next_parent == parent:
                break
            parent = next_parent

    async def _edit_blocks(self, path: str, blocks: list[SearchReplaceBlock], *, tool_name: str) -> str:
        if not self.filesystem.exists(path):
            error_msg = f"File not found: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise FileNotFoundError(error_msg)

        try:
            original_content = self.filesystem.read_text(path)
            resolved = [self._resolve_replacement(original_content, block) for block in blocks]
            self._validate_non_overlapping(resolved)
            updated_content = self._apply_replacements(original_content, resolved)

            # Preserve the file's dominant line ending. ``read_text`` may normalize
            # CRLF -> LF (Python universal newlines), so detect the true on-disk
            # ending and rewrite the result in that ending to keep unchanged
            # regions byte-identical and avoid mixed line endings.
            line_ending = self._detect_dominant_line_ending(path, original_content)
            normalized_original = self._normalize_line_endings(original_content, line_ending)
            normalized_updated = self._normalize_line_endings(updated_content, line_ending)

            if normalized_updated == normalized_original:
                await self.log_warning(
                    f"No changes made to {path}. All replacements were identical to original text.",
                    sender=self.caller.agent_name,
                )
                return f"# {path} (No changes made)\n\n```\n{normalized_original}\n```"

            blocked_msg = self._enforce_vibe_edit_policy(path)
            if blocked_msg:
                return blocked_msg
            self._mutate_with_optional_snapshot(
                tool_name=tool_name,
                reason=f"{tool_name} {path}",
                paths=[path],
                mutate=lambda: self.filesystem.write_text(path, normalized_updated),
            )

            await self.send_edit_preview(
                build_diff_preview(original_content, normalized_updated, path),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name=tool_name,
            )
            if tool_name == "multi_edit":
                return f"Edited {path} with {len(blocks)} replacements"
            return f"Edited {path}"
        except ValueError:
            raise
        except PermissionError:
            error_msg = f"Permission denied when writing to file: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to edit {path}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise

    def _parse_blocks(self, blocks: str) -> list[SearchReplaceBlock]:
        matches = list(_BLOCK_PATTERN.finditer(blocks))
        if not matches:
            raise ValueError(
                "No valid search and replace blocks found. Blocks must follow the format:\n"
                "<<<<<<< SEARCH\n[original code]\n=======\n[new code]\n>>>>>>> REPLACE"
            )

        parsed_blocks = []
        for block_number, match in enumerate(matches, 1):
            search_text = match.group(1)
            if not search_text.strip():
                raise ValueError(f"Empty search block in block #{block_number}. Search text cannot be empty.")
            parsed_blocks.append(
                SearchReplaceBlock(
                    block_number=block_number,
                    search_text=search_text,
                    replace_text=match.group(2),
                )
            )
        return parsed_blocks

    def _snapshot_paths_for_write(self, path: str) -> list[str]:
        paths = [path]
        parent = self.filesystem.get_parent(path)
        while parent and parent != "." and not self.filesystem.exists(parent):
            paths.append(parent)
            next_parent = self.filesystem.get_parent(parent)
            if next_parent == parent:
                break
            parent = next_parent
        return paths

    def _resolve_replacement(self, content: str, block: SearchReplaceBlock) -> ResolvedReplacement:
        passes: list[tuple[str, Callable[[str, str], list[tuple[int, int]]]]] = [
            ("exact", self._exact_spans),
            ("line_strip", self._line_strip_spans),
            ("line_endings", self._line_ending_spans),
            ("unicode_punctuation", self._unicode_punctuation_spans),
        ]

        for pass_name, matcher in passes:
            spans = matcher(content, block.search_text)
            if len(spans) > 1:
                raise ValueError(f"Search block #{block.block_number} matched {len(spans)} occurrences in the file.")
            if len(spans) == 1:
                start, end = spans[0]
                return ResolvedReplacement(
                    block_number=block.block_number,
                    start=start,
                    end=end,
                    replace_text=block.replace_text,
                    pass_name=pass_name,
                )

        preview = block.search_text[:100] + ("..." if len(block.search_text) > 100 else "")
        raise ValueError(
            f"Search block #{block.block_number} does not match any content in the file.\nSearch text: '{preview}'"
        )

    def _exact_spans(self, content: str, search_text: str) -> list[tuple[int, int]]:
        return self._find_spans(content, search_text)

    def _line_strip_spans(self, content: str, search_text: str) -> list[tuple[int, int]]:
        content_lines = self._line_records(content)
        search_lines = self._line_records(search_text)
        if not content_lines or not search_lines or len(search_lines) > len(content_lines):
            return []

        search_stripped = [line.strip() for line, _offset in search_lines]
        search_ends_with_newline = search_text.endswith(("\n", "\r"))
        spans = []

        for index in range(0, len(content_lines) - len(search_lines) + 1):
            window = content_lines[index : index + len(search_lines)]
            if [line.strip() for line, _offset in window] != search_stripped:
                continue

            start = window[0][1]
            last_line, last_offset = window[-1]
            if search_ends_with_newline:
                end = last_offset + len(last_line)
            else:
                end = last_offset + self._line_body_length(last_line)
            spans.append((start, end))

        return spans

    def _line_ending_spans(self, content: str, search_text: str) -> list[tuple[int, int]]:
        normalized_content, index_map = self._normalize_line_endings_with_index_map(content)
        normalized_search, _search_index_map = self._normalize_line_endings_with_index_map(search_text)
        normalized_spans = self._find_spans(normalized_content, normalized_search)
        return [self._map_normalized_span(span, index_map, len(content)) for span in normalized_spans]

    def _unicode_punctuation_spans(self, content: str, search_text: str) -> list[tuple[int, int]]:
        return self._find_spans(
            self._normalize_unicode_punctuation(content), self._normalize_unicode_punctuation(search_text)
        )

    def _validate_non_overlapping(self, replacements: list[ResolvedReplacement]) -> None:
        ordered = sorted(replacements, key=lambda replacement: (replacement.start, replacement.end))
        previous = None
        for replacement in ordered:
            if previous is not None and replacement.start < previous.end:
                raise ValueError(
                    f"Search block #{replacement.block_number} overlaps with search block #{previous.block_number}."
                )
            previous = replacement

    def _apply_replacements(self, content: str, replacements: list[ResolvedReplacement]) -> str:
        updated = content
        for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
            updated = updated[: replacement.start] + replacement.replace_text + updated[replacement.end :]
        return updated

    def _find_spans(self, content: str, search_text: str) -> list[tuple[int, int]]:
        if search_text == "":
            return []

        spans = []
        start = 0
        while True:
            index = content.find(search_text, start)
            if index == -1:
                break
            spans.append((index, index + len(search_text)))
            start = index + len(search_text)
        return spans

    def _line_records(self, text: str) -> list[tuple[str, int]]:
        records = []
        offset = 0
        for line in text.splitlines(keepends=True):
            records.append((line, offset))
            offset += len(line)
        return records

    def _line_body_length(self, line: str) -> int:
        if line.endswith("\r\n"):
            return len(line) - 2
        if line.endswith(("\n", "\r")):
            return len(line) - 1
        return len(line)

    def _normalize_line_endings_with_index_map(self, text: str) -> tuple[str, list[int]]:
        normalized = []
        index_map = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\r":
                normalized.append("\n")
                index_map.append(index)
                if index + 1 < len(text) and text[index + 1] == "\n":
                    index += 2
                else:
                    index += 1
                continue
            normalized.append(char)
            index_map.append(index)
            index += 1
        return "".join(normalized), index_map

    def _map_normalized_span(
        self, span: tuple[int, int], index_map: list[int], original_length: int
    ) -> tuple[int, int]:
        start, end = span
        original_start = index_map[start] if start < len(index_map) else original_length
        original_end = index_map[end] if end < len(index_map) else original_length
        return original_start, original_end

    def _normalize_unicode_punctuation(self, text: str) -> str:
        return text.translate(_SMART_PUNCTUATION_TRANSLATION)

    def _detect_dominant_line_ending(self, path: str, content: str) -> str:
        """Return the dominant line ending (``\\r\\n``, ``\\r``, or ``\\n``) on disk.

        ``read_text`` performs universal-newline translation (``\\r\\n`` -> ``\\n``)
        on local filesystems, so when the read content has no carriage returns we
        inspect the raw bytes to recover the true ending. Sandbox/MCP filesystems
        that return raw text with carriage returns are detected directly from
        ``content``.
        """
        if "\r" in content:
            return self._dominant_line_ending_from_text(content)
        try:
            raw = self.filesystem.read_bytes(path)
            raw_text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
            return self._dominant_line_ending_from_text(raw_text)
        except Exception:
            return "\n"

    @staticmethod
    def _dominant_line_ending_from_text(text: str) -> str:
        n_crlf = text.count("\r\n")
        n_lf = text.count("\n") - n_crlf
        n_cr = text.count("\r") - n_crlf
        if n_crlf > 0 and n_crlf >= n_lf and n_crlf >= n_cr:
            return "\r\n"
        if n_cr > 0 and n_cr > n_lf:
            return "\r"
        return "\n"

    @staticmethod
    def _normalize_line_endings(text: str, target: str) -> str:
        """Normalize all line endings in ``text`` to ``target``."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if target == "\n":
            return text
        return text.replace("\n", target)

    async def _maybe_append_lsp_diagnostics(self, path: str) -> str:
        """Query LSP diagnostics for *path* and format them for appending to the tool result.

        Returns an empty string if LSP is unavailable, disabled, or produces no diagnostics.
        """
        if self._lsp_manager is None or not self._lsp_manager.enabled:
            return ""

        if not self._lsp_manager._config.auto_diagnostics_on_edit:
            return ""

        if not self._lsp_manager._initialized:
            await self._lsp_manager.initialize()

        server_name = self._lsp_manager.server_for_path(path)
        if server_name is None:
            return ""

        try:
            diagnostics = await self._lsp_manager.get_fresh_diagnostics(path)
        except Exception:
            return ""

        if not diagnostics:
            return ""

        from kolega_code.services.lsp import format_diagnostics

        # Blank-line separator so the diagnostics block reads as a distinct section
        # after the result line (e.g. "Edited foo.py\n\nLSP diagnostics (...)").
        return "\n\n" + format_diagnostics(diagnostics, path, source=server_name)
