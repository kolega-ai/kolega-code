from .base_tool import BaseTool


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

            # Return success message with content
            return f"File created successfully\n\n```\n{content}\n```"

        except PermissionError:
            error_msg = f"Permission denied: Cannot create file {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return error_msg
        except Exception as e:
            error_msg = "Error creating file"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return error_msg
