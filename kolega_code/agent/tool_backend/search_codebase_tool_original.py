import re

from .base_tool import BaseTool


class SearchCodebaseToolOriginal(BaseTool):
    """Original implementation preserved for comparison testing"""

    async def search_codebase(self, pattern: str, file_pattern: str = "*", case_sensitive: bool = False) -> str:
        """
        Search the codebase for files containing a specific pattern (grep functionality).

        Args:
            pattern: The pattern to search for in files
            file_pattern: Optional glob pattern to filter which files to search (default: all files)
            case_sensitive: Whether the search should be case-sensitive (default: False)

        Returns:
            Markdown formatted list of files and matches, limited to 128 results

        Raises:
            Exception: If any error occurs during the search operation
        """
        try:
            await self.log_info(f"Searching codebase for pattern: '{pattern}'", sender=self.caller.agent_name)

            # Compile the regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                error_msg = f"Invalid regular expression: {str(e)}"
                await self.log_error(error_msg, sender=self.caller.agent_name)
                return f"Error: {error_msg}"

            # Get list of files matching the file pattern
            if file_pattern == "*":
                # Recursively find all files
                files = self.filesystem.glob("**/*")
            else:
                # Use the provided file pattern
                files = self.filesystem.glob(f"**/{file_pattern}")

            # Filter to only include files (not directories)
            files = [f for f in files if self.filesystem.is_file(f)]

            # Search through each file for the pattern
            results = []
            total_matches = 0
            max_results = 128
            reached_limit = False

            for file_path in files:
                # Skip binary files and common exclusions
                if self._is_binary_file(self.filesystem.get_path(file_path)) or self._should_exclude_file(
                    self.filesystem.get_path(file_path)
                ):
                    continue

                try:
                    content = self.filesystem.read_text(file_path)

                    matches = regex.findall(content)

                    if matches:
                        # Get the relative path for display
                        relative_path = file_path

                        # Get matching lines with context
                        matching_lines = []
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            if regex.search(line):
                                line_num = i + 1  # 1-indexed line numbers
                                line_content = line.strip()
                                # Truncate long lines to 200 characters
                                if len(line_content) > 200:
                                    line_content = line_content[:200] + "..."
                                matching_lines.append(f"  Line {line_num}: {line_content}")

                        # Limit the number of matching lines to display
                        if len(matching_lines) > 5:
                            matching_lines = matching_lines[:5]
                            matching_lines.append(f"  ... and {len(matches) - 5} more matches")

                        # Add to results
                        results.append(f"- **{relative_path}** ({len(matches)} matches)\n" + "\n".join(matching_lines))

                        total_matches += 1
                        if total_matches >= max_results:
                            reached_limit = True
                            break
                except Exception as e:
                    # Skip files that can't be read
                    continue

                if reached_limit:
                    break

            # Format the results
            if results:
                result_text = f"# Search Results for '{pattern}'\n\n"

                # Add note about results limitation if we reached the limit
                if reached_limit:
                    result_text += f"⚠️ **Note:** Showing only the first {max_results} results. There are more matches in the codebase.\n\n"

                result_text += "\n\n".join(results)
                return result_text
            else:
                return f"No matches found for pattern '{pattern}'"

        except Exception as e:
            error_msg = f"Error searching codebase: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return f"Error: {error_msg}"
