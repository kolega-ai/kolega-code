from .base_tool import BaseTool
from .edit_preview import build_diff_preview


class ReplaceLinesTool(BaseTool):
    async def replace_lines(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        """
        Replace a range of lines in a file with new content.

        Args:
            path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.
            start_line: The starting line number (1-indexed)
            end_line: The ending line number (1-indexed, inclusive)
            new_content: The new content to replace the specified lines with

        Returns:
            The updated contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the line range is invalid
            PermissionError: If the file cannot be written to
        """
        if not self.filesystem.exists(path):
            error_msg = f"File not found: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise FileNotFoundError(error_msg)

        if start_line < 1:
            error_msg = f"Invalid start_line: {start_line}. Line numbers must be 1-indexed."
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise ValueError(error_msg)

        if end_line < start_line:
            error_msg = (
                f"Invalid line range: end_line ({end_line}) must be greater than or equal to start_line ({start_line})."
            )
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise ValueError(error_msg)

        try:
            # Read the original file content
            file_content = self.filesystem.read_text(path)
            lines = file_content.splitlines(keepends=True)

            # Check if the line range is valid
            if start_line > len(lines):
                error_msg = f"Invalid start_line: {start_line}. File only has {len(lines)} lines."
                await self.log_error(error_msg, sender=self.caller.agent_name)
                raise ValueError(error_msg)

            # Convert to 0-indexed for internal processing
            start_idx = start_line - 1
            end_idx = min(end_line, len(lines))

            # Handle newlines
            new_content_lines = new_content.splitlines()
            if not new_content_lines:
                # Empty content case
                formatted_new_content = "\n" if end_idx < len(lines) else ""
            else:
                # Non-empty content case
                formatted_new_content = "\n".join(new_content_lines)
                if end_idx < len(lines) or (lines and lines[-1].endswith("\n")):
                    formatted_new_content += "\n"

            # Replace the specified lines
            updated_content = "".join(lines[:start_idx]) + formatted_new_content + "".join(lines[end_idx:])

            # Write the updated content back to the file (with vibe policy enforcement)
            blocked_msg = self._enforce_vibe_edit_policy(path)
            if blocked_msg:
                return blocked_msg
            self.filesystem.write_text(path, updated_content)

            success_msg = f"Successfully replaced lines {start_line}-{end_line} in file: {path}"
            await self.log_info(success_msg, sender=self.caller.agent_name)

            # Surface the diff inline (UI-only; no model tokens).
            await self.send_edit_preview(
                build_diff_preview(file_content, updated_content, path),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="replace_lines",
            )
            return f"Replaced lines {start_line}-{end_line} in {path}"

        except PermissionError:
            error_msg = f"Permission denied when writing to file: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to replace lines in file {path}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
