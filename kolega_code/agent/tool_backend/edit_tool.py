from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .base_tool import BaseTool
from .edit_preview import build_diff_preview, build_head_preview

if TYPE_CHECKING:
    from kolega_code.services.lsp import LspManager


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
    def __init__(self, *args, lsp_manager: Optional["LspManager"] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lsp_manager = lsp_manager

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
            if parent_dir and parent_dir != "." and not self.filesystem.exists(parent_dir):
                self.filesystem.create_directory(parent_dir)

            # Preserve the file's dominant line ending on overwrite; new files
            # keep the line endings the caller provided (typically LF).
            if exists and old_content is not None:
                line_ending = self._detect_dominant_line_ending(path, old_content)
                content = self._normalize_line_endings(content, line_ending)

            self.filesystem.write_text(path, content)

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
            self.filesystem.write_text(path, normalized_updated)

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
