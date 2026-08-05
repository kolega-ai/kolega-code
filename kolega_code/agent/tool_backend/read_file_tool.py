# kolega_code/agent/tool_backend/read_file_tool.py

from typing import Callable, Optional

from .base_tool import BaseTool


class ReadFileTool(BaseTool):
    MAX_LINES = 2000
    MAX_BYTES = 50 * 1024

    def _format_file_content(
        self,
        path: str,
        content: str,
        *,
        first_line: int,
        last_line: int,
        total_lines: int,
        truncated: bool,
        cap_name: str,
        giant_line: Optional[int] = None,
        giant_line_size: int = 0,
    ) -> str:
        suffix_parts = []
        if last_line >= first_line and (first_line != 1 or last_line != total_lines):
            suffix_parts.append(f"(lines {first_line}-{last_line})")
        if truncated:
            suffix_parts.append("(TRUNCATED)")
        suffix = f" {' '.join(suffix_parts)}" if suffix_parts else ""

        notices = []
        if giant_line is not None:
            line_number = giant_line + 1
            size = giant_line_size
            if last_line < first_line:
                notices.append(
                    f"Line {line_number} is {size:,} bytes, exceeding the 50KB output budget, so it cannot be "
                    "displayed. Use rg or exec_command for targeted extraction; a partial excerpt would not "
                    "contain the answer."
                )
            else:
                notices.append(
                    f"Showing lines {first_line}-{last_line} of {total_lines}: line {line_number} is {size:,} "
                    f"bytes, exceeding the 50KB output budget, so it is omitted. Use rg or exec_command for "
                    "targeted extraction; the excerpt may not contain the answer."
                )
        elif truncated:
            notices.append(
                f"[Showing lines {first_line}-{last_line} of {total_lines} ({cap_name} limit). "
                f"Use offset={last_line + 1} to continue.]"
            )
        notice_text = "\n\n".join(notices)
        if notice_text:
            notice_text += "\n\n"

        return f"# {path}{suffix}\n\n{notice_text}```\n{content}\n```"

    async def read(
        self,
        file_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        *,
        line_formatter: Optional[Callable[[str, int], str]] = None,
    ) -> str:
        """
        Read the contents of a file, optionally starting at a 1-indexed offset
        line and reading at most limit lines.

        Args:
            file_path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.
            offset: The 1-indexed first line to read (default 1).
            limit: The maximum number of lines to read; omitted reads from the top.

        Returns:
            The contents of the file as a string formatted as markdown.

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If offset or limit are invalid
        """
        if not self.filesystem.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if offset < 1:
            raise ValueError(f"Offset must be at least 1, got {offset}")

        if limit is not None and limit < 1:
            raise ValueError(f"Limit must be at least 1, got {limit}")

        file_content = self.filesystem.read_text(file_path)
        lines = file_content.splitlines(keepends=True)
        total_lines = len(lines)

        if offset > total_lines:
            raise ValueError(f"Offset {offset} exceeds file length {total_lines}")

        start = offset - 1
        max_lines = min(limit if limit is not None else self.MAX_LINES, self.MAX_LINES)
        requested_end = min(start + max_lines, total_lines)

        used_bytes = 0
        shown_end = start
        giant_line: Optional[int] = None
        giant_line_size = 0
        while shown_end < requested_end:
            line = lines[shown_end]
            size = len(line.encode("utf-8"))
            if size > self.MAX_BYTES:
                giant_line = shown_end
                giant_line_size = size
                break
            if used_bytes + size > self.MAX_BYTES:
                break
            used_bytes += size
            shown_end += 1

        first_line = start + 1
        last_line = shown_end
        # The line cap only binds when the 2000-line ceiling (not an explicit
        # limit the model requested) cut the range short.
        line_cap_bound = (
            max_lines == self.MAX_LINES
            and shown_end == requested_end
            and requested_end == start + max_lines
            and requested_end < total_lines
        )
        byte_cap_bound = shown_end < requested_end
        truncated = line_cap_bound or byte_cap_bound

        section_content = "".join(lines[start:shown_end])
        if line_formatter is not None:
            # Hashline anchors may only describe complete source lines. Drop the
            # trailing terminator when the shown range ends before the file does,
            # so no phantom empty line gets an anchor.
            if last_line < total_lines and section_content.endswith(("\n", "\r")):
                if section_content.endswith("\r\n"):
                    section_content = section_content[:-2]
                else:
                    section_content = section_content[:-1]
            section_content = line_formatter(section_content, first_line) if section_content else ""

        cap_name = "50KB" if byte_cap_bound else "2000-line"
        return self._format_file_content(
            path=file_path,
            content=section_content,
            first_line=first_line,
            last_line=last_line,
            total_lines=total_lines,
            truncated=truncated,
            cap_name=cap_name,
            giant_line=giant_line,
            giant_line_size=giant_line_size,
        )
