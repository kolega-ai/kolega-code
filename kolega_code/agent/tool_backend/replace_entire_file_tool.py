from .base_tool import BaseTool
from .edit_preview import build_diff_preview, build_head_preview


class ReplaceEntireFileTool(BaseTool):
    async def replace_entire_file(self, path: str, content: str) -> str:
        """
        Replace the entire contents of a file in the project.

        Args:
            path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.
            content: New content to write to the file

        Returns:
            The updated contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be written to
        """
        if not self.filesystem.exists(path):
            error_msg = f"File not found: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise FileNotFoundError(error_msg)

        try:
            # Capture the old content before overwriting so we can show a diff.
            try:
                old_content = self.filesystem.read_text(path)
            except Exception:
                old_content = None
            blocked_msg = self._enforce_vibe_edit_policy(path)
            if blocked_msg:
                return blocked_msg
            self.filesystem.write_text(path, content)
            success_msg = f"Successfully replaced file: {path}"
            await self.log_info(success_msg, sender=self.caller.agent_name)
            preview = (
                build_diff_preview(old_content, content, path)
                if old_content is not None
                else build_head_preview(content, path)
            )
            await self.send_edit_preview(
                preview,
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="replace_entire_file",
            )
            return f"# {path} has been replaced."
        except PermissionError as e:
            error_msg = f"Permission denied when writing to file: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to write to file {path}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
