import re

from .base_tool import BaseTool
from .edit_preview import build_diff_preview


class SearchAndReplaceTool(BaseTool):
    async def search_and_replace(self, path: str, blocks: str) -> str:
        """
        Edit a file using search and replace blocks.

        The blocks should be formatted as follows:
        ```
        <<<<<<< SEARCH
        [original code to find]
        =======
        [new code to replace with]
        >>>>>>> REPLACE
        ```

        Multiple search and replace blocks can be provided in sequence.
        The tool will process each block in order, updating the file incrementally.
        THE INDENTATION IN THE SEARCH BLOCK MUST BE IDENTICAL TO THE EXISTING FILE.

        Args:
            path: Path to the file to edit. Relative to the project root is preferred; an absolute path is also accepted.
            blocks: One or more search and replace blocks formatted as shown above

        Returns:
            The updated contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the search block doesn't match any content in the file
            ValueError: If the blocks are malformed or incorrectly formatted
            PermissionError: If the file cannot be written to
        """
        if not self.filesystem.exists(path):
            error_msg = f"File not found: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise FileNotFoundError(error_msg)

        try:
            # Read the original file content
            file_content = self.filesystem.read_text(path)

            original_content = file_content
            updated_content = file_content

            # Track all the replacements made for reporting
            replacements_made = []

            # Parse the search and replace blocks
            block_pattern = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)

            matches = list(block_pattern.finditer(blocks))

            if not matches:
                error_msg = (
                    "No valid search and replace blocks found. Blocks must follow the format:\n"
                    "<<<<<<< SEARCH\n[original code]\n=======\n[new code]\n>>>>>>> REPLACE"
                )
                await self.log_error(error_msg, sender=self.caller.agent_name)
                raise ValueError(error_msg)

            # Check if there are multiple search and replace blocks
            if len(matches) > 1:
                error_msg = "Multiple search and replace blocks provided. This tool only supports one block at a time."
                await self.log_error(error_msg, sender=self.caller.agent_name)
                raise ValueError(error_msg)

            # Process each search and replace block
            for i, match in enumerate(matches, 1):
                search_text = match.group(1)
                replace_text = match.group(2)

                # Validate the search and replace blocks
                if not search_text.strip():
                    error_msg = f"Empty search block in block #{i}. Search text cannot be empty."
                    await self.log_error(error_msg, sender=self.caller.agent_name)
                    raise ValueError(error_msg)

                # Check if the search text exists in the current content
                if search_text not in updated_content:
                    # Get some context around where we expect to find it for better error reporting
                    error_msg = f"Search block #{i} does not match any content in the file.\nSearch text: '{search_text[:100]}{'...' if len(search_text) > 100 else ''}'"
                    await self.log_error(error_msg, sender=self.caller.agent_name)
                    raise ValueError(error_msg)

                # Count occurrences to check for multiple matches
                occurrences = updated_content.count(search_text)
                if occurrences > 1:
                    error_msg = f"Search block #{i} matched {occurrences} occurrences in the file."
                    await self.log_error(error_msg, sender=self.caller.agent_name)
                    raise ValueError(error_msg)

                # Track replacement details
                replacements_made.append(
                    {
                        "block_number": i,
                        "occurrences_replaced": occurrences,
                        "search_text_preview": search_text[:50] + ("..." if len(search_text) > 50 else ""),
                        "replace_text_preview": replace_text[:50] + ("..." if len(replace_text) > 50 else ""),
                    }
                )

                # Apply the replacement
                updated_content = updated_content.replace(search_text, replace_text)

            # Check if anything was changed
            if updated_content == original_content:
                await self.log_warning(
                    f"No changes made to {path}. All replacements were identical to original text.",
                    sender=self.caller.agent_name,
                )
                return f"# {path} (No changes made)\n\n```\n{original_content}\n```"

            # Write the updated content back to the file (with vibe policy enforcement)
            blocked_msg = self._enforce_vibe_edit_policy(path)
            if blocked_msg:
                return blocked_msg
            self.filesystem.write_text(path, updated_content)

            # Surface the diff inline (UI-only; the change is already in the tool args).
            await self.send_edit_preview(
                build_diff_preview(original_content, updated_content, path),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                tool_name="search_and_replace",
            )
            return f"Edited {path}"

        except ValueError:
            # Re-raise ValueError exceptions which are used for validation errors
            raise
        except PermissionError:
            error_msg = f"Permission denied when writing to file: {path}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
        except Exception as e:
            error_msg = f"Failed to apply search and replace to {path}: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            raise
