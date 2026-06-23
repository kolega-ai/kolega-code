from .base_tool import BaseTool
from .edit_preview import build_head_preview


class CreateFileTool(BaseTool):
    async def create_file(self, path: str, content: str) -> str:
        """
        Create a new file with the specified content.

        Args:
            path: Path to create the file at. Relative to the project root is preferred; an absolute path is also accepted.
            content: Content to write to the file

        Returns:
            A success message with the file content formatted as markdown

        Raises:
            FileExistsError: If the file already exists
            ValueError: If the parent directory doesn't exist
            PermissionError: If the file cannot be created
            Exception: If there is a general error creating the file
        """
        try:
            # Enforce vibe policy for blacklisted basenames
            blocked_msg = self._enforce_vibe_edit_policy(path)
            if blocked_msg:
                return blocked_msg

            # Check if file already exists
            if self.filesystem.exists(path):
                error_msg = f"File already exists: {path}"
                await self.log_error(error_msg, sender=self.caller.agent_name)
                return error_msg

            # Create parent directory if it doesn't exist
            parent_dir = self.filesystem.get_parent(path)
            if parent_dir and parent_dir != "." and not self.filesystem.exists(parent_dir):
                self.filesystem.create_directory(parent_dir)

            # Create the file
            self.filesystem.write_text(path, content)

            # Surface a syntax-highlighted head inline (UI-only; no model tokens).
            await self.send_edit_preview(
                build_head_preview(content, path),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="create_file",
            )
            return f"Created {path}"

        except PermissionError:
            error_msg = f"Permission denied: Cannot create file {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return error_msg
        except Exception:
            error_msg = "Error creating file"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return error_msg
