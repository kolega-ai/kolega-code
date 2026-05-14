from .base_tool import BaseTool


class ReadFileTool(BaseTool):
    MAX_LINES_FOR_ENTIRE_FILE = 2000

    async def read_entire_file(self, relative_path: str) -> str:
        """
        Read the contents of a file in the project.

        Note: Files exceeding 2000 lines will be truncated with a warning message.
        Use read_file_section to read specific portions of large files.

        Args:
            relative_path: Path to the file, relative to the project root

        Returns:
            The contents of the file as a string formatted as markdown.
            If the file exceeds 2000 lines, returns a truncated version with a warning.

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not self.filesystem.exists(relative_path):
            raise FileNotFoundError(f"File not found: {relative_path}")

        file_content = self.filesystem.read_text(relative_path)
        lines = file_content.splitlines(keepends=True)
        total_lines = len(lines)

        if total_lines > self.MAX_LINES_FOR_ENTIRE_FILE:
            # Truncate the content to the maximum allowed lines
            truncated_lines = lines[: self.MAX_LINES_FOR_ENTIRE_FILE]
            truncated_content = "".join(truncated_lines)

            return (
                f"# {relative_path} (TRUNCATED)\n\n"
                f"**⚠️ File truncated: Showing first {self.MAX_LINES_FOR_ENTIRE_FILE} of {total_lines} lines**\n\n"
                f"To read specific sections, use `read_file_section` with start/end line numbers.\n\n"
                f"```\n{truncated_content}\n```"
            )

        return f"# {relative_path}\n\n```\n{file_content}\n```"

    async def read_file_section(self, relative_path: str, start_line: int, end_line: int) -> str:
        """
        Read a specific section of a file in the project from start_line to end_line (inclusive).

        Args:
            relative_path: Path to the file, relative to the project root
            start_line: The line number to start reading from (1-indexed)
            end_line: The line number to stop reading at (1-indexed, inclusive)

        Returns:
            The specified section of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If start_line or end_line are invalid
        """
        if not self.filesystem.exists(relative_path):
            raise FileNotFoundError(f"File not found: {relative_path}")

        if start_line < 1:
            raise ValueError(f"Start line must be at least 1, got {start_line}")

        if end_line < start_line:
            raise ValueError(f"End line ({end_line}) must be greater than or equal to start line ({start_line})")

        file_content = self.filesystem.read_text(relative_path)
        lines = file_content.splitlines(keepends=True)

        if start_line > len(lines):
            raise ValueError(f"Start line {start_line} exceeds file length {len(lines)}")

        # Adjust for 0-indexed list
        section_content = "".join(lines[start_line - 1 : end_line])
        line_range = f"(lines {start_line}-{min(end_line, len(lines))})"
        return f"# {relative_path} {line_range}\n\n```\n{section_content}\n```"
