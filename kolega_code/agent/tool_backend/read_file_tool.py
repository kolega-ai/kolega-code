from typing import Callable, Optional

from .base_tool import BaseTool


class ReadFileTool(BaseTool):
    MAX_LINES_FOR_ENTIRE_FILE = 2000
    MAX_CHARS_FOR_FILE_OUTPUT = 100_000

    def _format_file_content(
        self,
        path: str,
        content: str,
        *,
        line_range: str = "",
        line_truncation_notice: str = "",
        start_line: int = 1,
        continues_after: bool = False,
        line_formatter: Optional[Callable[[str, int], str]] = None,
    ) -> str:
        original_char_count = len(content)
        char_truncated = original_char_count > self.MAX_CHARS_FOR_FILE_OUTPUT
        if char_truncated:
            candidate = content[: self.MAX_CHARS_FOR_FILE_OUTPUT]
            if line_formatter is None:
                content = candidate
            else:
                # Hashline anchors may only describe complete source lines.
                # Discard the partial tail instead of publishing an unusable ID.
                cutoff = max(candidate.rfind("\n"), candidate.rfind("\r"))
                content = candidate[:cutoff] if cutoff >= 0 else ""

        if line_formatter is not None:
            if continues_after or char_truncated:
                if content.endswith("\r\n"):
                    content = content[:-2]
                elif content.endswith(("\n", "\r")):
                    content = content[:-1]
            content = line_formatter(content, start_line) if content else ""

        truncated = bool(line_truncation_notice) or char_truncated
        suffix_parts = []
        if line_range:
            suffix_parts.append(line_range)
        if truncated:
            suffix_parts.append("(TRUNCATED)")
        suffix = f" {' '.join(suffix_parts)}" if suffix_parts else ""

        notices = []
        if line_truncation_notice:
            notices.append(line_truncation_notice)
        if char_truncated:
            notices.append(
                f"**File truncated by size: Showing first {self.MAX_CHARS_FOR_FILE_OUTPUT:,} "
                f"of {original_char_count:,} characters**"
            )
        if notices:
            notices.append("To read specific sections, use `read_file_section` with start/end line numbers.")

        notice_text = "\n\n".join(notices)
        if notice_text:
            notice_text += "\n\n"

        return f"# {path}{suffix}\n\n{notice_text}```\n{content}\n```"

    async def read_entire_file(
        self,
        path: str,
        *,
        line_formatter: Optional[Callable[[str, int], str]] = None,
    ) -> str:
        """
        Read the contents of a file in the project.

        Note: Files exceeding 2000 lines will be truncated with a warning message.
        Use read_file_section to read specific portions of large files.

        Args:
            path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.

        Returns:
            The contents of the file as a string formatted as markdown.
            If the file exceeds 2000 lines, returns a truncated version with a warning.

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not self.filesystem.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        file_content = self.filesystem.read_text(path)
        lines = file_content.splitlines(keepends=True)
        total_lines = len(lines)

        if total_lines > self.MAX_LINES_FOR_ENTIRE_FILE:
            # Truncate the content to the maximum allowed lines
            truncated_lines = lines[: self.MAX_LINES_FOR_ENTIRE_FILE]
            truncated_content = "".join(truncated_lines)

            return self._format_file_content(
                path,
                truncated_content,
                line_truncation_notice=(
                    f"**⚠️ File truncated: Showing first {self.MAX_LINES_FOR_ENTIRE_FILE} of {total_lines} lines**"
                ),
                continues_after=True,
                line_formatter=line_formatter,
            )

        return self._format_file_content(path, file_content, line_formatter=line_formatter)

    async def read_file_section(
        self,
        path: str,
        start_line: int,
        end_line: int,
        *,
        line_formatter: Optional[Callable[[str, int], str]] = None,
    ) -> str:
        """
        Read a specific section of a file in the project from start_line to end_line (inclusive).

        Args:
            path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.
            start_line: The line number to start reading from (1-indexed)
            end_line: The line number to stop reading at (1-indexed, inclusive)

        Returns:
            The specified section of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If start_line or end_line are invalid
        """
        if not self.filesystem.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if start_line < 1:
            raise ValueError(f"Start line must be at least 1, got {start_line}")

        if end_line < start_line:
            raise ValueError(f"End line ({end_line}) must be greater than or equal to start line ({start_line})")

        file_content = self.filesystem.read_text(path)
        lines = file_content.splitlines(keepends=True)

        if start_line > len(lines):
            raise ValueError(f"Start line {start_line} exceeds file length {len(lines)}")

        # Adjust for 0-indexed list
        section_content = "".join(lines[start_line - 1 : end_line])
        line_range = f"(lines {start_line}-{min(end_line, len(lines))})"
        return self._format_file_content(
            path,
            section_content,
            line_range=line_range,
            start_line=start_line,
            continues_after=end_line < len(lines),
            line_formatter=line_formatter,
        )
