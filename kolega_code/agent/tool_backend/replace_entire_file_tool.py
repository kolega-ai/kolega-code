from .base_tool import BaseTool


class ReplaceEntireFileTool(BaseTool):
    async def replace_entire_file(self, relative_path: str, content: str) -> str:
        """
        Replace the entire contents of a file in the project.

        Args:
            relative_path: Path to the file, relative to the project root
            content: New content to write to the file

        Returns:
            The updated contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be written to
        """
        if not self.filesystem.exists(relative_path):
            error_msg = f"File not found: {relative_path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise FileNotFoundError(error_msg)

        try:
            blocked_msg = self._enforce_vibe_edit_policy(relative_path)
            if blocked_msg:
                return blocked_msg
            self.filesystem.write_text(relative_path, content)
            success_msg = f"Successfully replaced file: {relative_path}"
            await self.log_info(success_msg, sender=self.caller.agent_name)
            return f"# {relative_path} has been replaced."
        except PermissionError as e:
            error_msg = f"Permission denied when writing to file: {relative_path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to write to file {relative_path}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
