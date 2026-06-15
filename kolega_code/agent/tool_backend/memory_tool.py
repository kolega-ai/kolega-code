from .base_tool import BaseTool

MEMORY_FILE = "AGENT_MEMORY.md"


class MemoryTool(BaseTool):
    async def read_memory(self) -> str:
        """
        Read the contents of the AGENT_MEMORY.md file which serves as the agent's memory.

        Returns:
            The contents of the AGENT_MEMORY.md file as a string

        Raises:
            FileNotFoundError: If the AGENT_MEMORY.md file doesn't exist
        """
        memory_file = MEMORY_FILE

        if not self.filesystem.exists(memory_file):
            error_msg = f"Memory file {memory_file} not found in project root"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise FileNotFoundError(error_msg)

        try:
            memory_content = self.filesystem.read_text(memory_file)

            await self.log_info(f"Successfully read memory file {memory_file}", sender=self.caller.agent_name)
            return memory_content
        except PermissionError:
            error_msg = f"Permission denied when reading memory file {memory_file}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to read memory file {memory_file}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise

    async def write_memory(self, memory_content: str) -> str:
        """
        Write a new memory to the AGENT_MEMORY.md file which serves as the agent's memory.

        The memory is added as a markdown bullet point to the file.

        Args:
            memory_content: The memory content to add to the file

        Returns:
            A confirmation message indicating success

        Raises:
            PermissionError: If the file cannot be written to
            Exception: If any other error occurs during writing
        """
        memory_file = MEMORY_FILE

        try:
            # Create the file if it doesn't exist
            if not self.filesystem.exists(memory_file):
                self.filesystem.write_text(memory_file, f"# Agent Memory\n\n- {memory_content}\n")
                success_msg = f"Created memory file {memory_file} and added new memory"
            else:
                # Read existing content and append the new memory
                existing_content = self.filesystem.read_text(memory_file)

                # Add the new memory as a bullet point
                updated_content = f"{existing_content.rstrip()}\n- {memory_content}\n"

                # Write the updated content back to the file
                self.filesystem.write_text(memory_file, updated_content)
                success_msg = f"Successfully added new memory to {memory_file}"

            await self.log_info(success_msg, sender=self.caller.agent_name)
            return success_msg
        except PermissionError:
            error_msg = f"Permission denied when writing to memory file {memory_file}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to write to memory file {memory_file}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
