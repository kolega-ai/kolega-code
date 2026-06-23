import re
from pathlib import Path
from typing import List, Tuple

from .base_tool import BaseTool


class SearchCodebaseTool(BaseTool):
    # Define binary extensions to skip without checking content
    BINARY_EXTENSIONS = {
        ".pyc",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".jar",
        ".war",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".rar",
        ".7z",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".o",
        ".obj",
        ".class",
        ".binary",
        ".wasm",
        ".node",
    }

    # Define directories to exclude
    EXCLUDE_DIRS = {
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        "bin",
        "obj",
        ".next",
        ".nuxt",
        "coverage",
    }

    async def search_codebase(self, pattern: str, file_pattern: str = "*", case_sensitive: bool = False, literal: bool = True) -> str:
        """
        Search the codebase for files containing a specific pattern (grep functionality).

        Uses grep command in sandbox environments for maximum efficiency (single command).
        Uses optimized Python implementation for local filesystems.

        Args:
            pattern: The pattern to search for in files
            file_pattern: Optional glob pattern to filter which files to search (default: all files)
            case_sensitive: Whether the search should be case-sensitive (default: False)
            literal: Whether to treat the pattern as literal text (True) or as a regular expression (False) (default: True)

        Returns:
            Markdown formatted list of files and matches, limited to 128 results

        Raises:
            Exception: If any error occurs during the search operation
        """
        try:
            await self.log_info(f"Searching codebase for pattern: '{pattern}'", sender=self.caller.agent_name)

            # If literal search, escape special regex characters
            search_pattern = re.escape(pattern) if literal else pattern

            # Compile the regex pattern to validate it
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(search_pattern, flags)
            except re.error as e:
                error_msg = f"Invalid regular expression: {str(e)}"
                await self.log_error(error_msg, sender=self.caller.agent_name)
                return f"Error: {error_msg}"

            # Use grep for sandbox environments (single command execution)
            if hasattr(self.filesystem, "sandbox"):
                return await self._search_with_grep_sandbox(search_pattern, file_pattern, case_sensitive, pattern, literal)
            else:
                # Use optimized Python implementation for local filesystem
                return await self._search_with_python(search_pattern, file_pattern, case_sensitive, regex, pattern)

        except Exception as e:
            error_msg = f"Error searching codebase: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return f"Error: {error_msg}"

    async def _search_with_grep_sandbox(self, pattern: str, file_pattern: str, case_sensitive: bool, original_pattern: str, literal: bool) -> str:
        """Use single grep command for sandbox environments - most efficient approach"""

        # Build grep command
        grep_flags = [
            "-r",  # Recursive
            "-n",  # Show line numbers
            "--binary-files=without-match",  # Skip binary files
        ]

        # Case sensitivity
        if not case_sensitive:
            grep_flags.append("-i")

        # Use fixed string matching for literal searches, extended regex otherwise
        if literal:
            grep_flags.append("-F")  # Fixed string matching
        else:
            grep_flags.append("-E")  # Extended regex

        # File pattern
        if file_pattern != "*":
            grep_flags.append(f'--include={file_pattern}')

        # Exclude directories
        for exclude_dir in self.EXCLUDE_DIRS:
            grep_flags.append(f"--exclude-dir={exclude_dir}")

        # Exclude binary extensions
        for ext in self.BINARY_EXTENSIONS:
            grep_flags.append(f'--exclude=*{ext}')

        # Build command with awk processing for formatting
        # Use the original pattern for grep when literal=True (no escaping needed with -F flag)
        grep_pattern = original_pattern if literal else pattern
        grep_cmd = f"grep {' '.join(grep_flags)} '{grep_pattern}' . 2>/dev/null"

        # AWK script to format output exactly like the original
        awk_script = r"""| awk '
        BEGIN { 
            file_count = 0; 
            current_file = "";
            match_count = 0;
            lines = "";
            lines_shown = 0;
            max_lines_per_file = 5;
            max_files = 128;
            max_line_length = 200;
        }
        {
            # Parse grep output: filename:line_number:content
            colon1 = index($0, ":");
            colon2 = index(substr($0, colon1 + 1), ":") + colon1;
            
            file = substr($0, 1, colon1 - 1);
            line_num = substr($0, colon1 + 1, colon2 - colon1 - 1);
            line_content = substr($0, colon2 + 1);
            
            # Remove leading/trailing whitespace from content
            gsub(/^[ \t]+|[ \t]+$/, "", line_content);
            
            # Truncate long lines to 200 characters
            if (length(line_content) > max_line_length) {
                line_content = substr(line_content, 1, max_line_length) "...";
            }
            
            if (file != current_file) {
                # Print previous file results if any
                if (current_file != "") {
                    print "- **" current_file "** (" match_count " matches)";
                    print substr(lines, 1, length(lines)-1);  # Remove trailing newline
                    if (lines_shown < match_count && lines_shown >= max_lines_per_file) {
                        print "  ... and " (match_count - max_lines_per_file) " more matches";
                    }
                    print "";
                }
                
                # Check if we reached the file limit
                file_count++;
                if (file_count > max_files) {
                    reached_limit = 1;
                    exit;
                }
                
                # Start new file
                current_file = file;
                match_count = 0;
                lines = "";
                lines_shown = 0;
            }
            
            match_count++;
            if (lines_shown < max_lines_per_file) {
                lines = lines "  Line " line_num ": " line_content "\n";
                lines_shown++;
            }
        }
        END {
            # Print last file
            if (current_file != "" && file_count <= max_files) {
                print "- **" current_file "** (" match_count " matches)";
                print substr(lines, 1, length(lines)-1);  # Remove trailing newline
                if (lines_shown < match_count && lines_shown >= max_lines_per_file) {
                    print "  ... and " (match_count - max_lines_per_file) " more matches";
                }
            }
            
            # Add warning if limit reached
            if (file_count > max_files || reached_limit) {
                print "";
                print "⚠️ **Note:** Showing only the first " max_files " results. There are more matches in the codebase.";
            }
        }'
        """

        full_cmd = f"cd {self.filesystem.root_path} && {grep_cmd} {awk_script}"

        # Execute the single command - sandbox is always async
        result = await self.filesystem.sandbox.commands.run(full_cmd)

        if result.exit_code != 0 or not result.stdout.strip():
            return f"No matches found for pattern '{original_pattern}'"

        # Format the final output
        output = f"# Search Results for '{original_pattern}'\n\n"
        output += result.stdout.strip()

        return output

    async def _search_with_python(self, pattern: str, file_pattern: str, case_sensitive: bool, regex, original_pattern: str) -> str:
        """Optimized Python implementation for local filesystems"""

        # Get files with their info
        files_with_info = await self._get_files_batch_local(file_pattern)

        # Search through files
        results = []
        total_matches = 0
        max_results = 128
        max_file_size = 10 * 1024 * 1024  # 10MB

        for file_path, file_size in files_with_info:
            # Skip files that are too large
            if file_size > max_file_size:
                continue

            # Skip binary files by extension
            if self._is_likely_binary_by_extension(file_path):
                continue

            try:
                # Read file content once
                content = self.filesystem.read_text(file_path)

                # Quick binary check on content (first 1024 bytes)
                if "\x00" in content[:1024]:
                    continue

                # Search for matches
                matches = list(regex.finditer(content))

                if matches:
                    # Get matching lines with context
                    matching_lines = self._extract_matching_lines(content, matches, max_lines=5)

                    # Add to results
                    results.append(f"- **{file_path}** ({len(matches)} matches)\n" + "\n".join(matching_lines))

                    total_matches += 1
                    if total_matches >= max_results:
                        break

            except Exception:
                # Skip files that can't be read
                continue

        # Format results
        if results:
            result_text = f"# Search Results for '{original_pattern}'\n\n"
            if total_matches >= max_results:
                result_text += f"⚠️ **Note:** Showing only the first {max_results} results. There are more matches in the codebase.\n\n"
            result_text += "\n\n".join(results)
            return result_text
        else:
            return f"No matches found for pattern '{original_pattern}'"

    async def _get_files_batch_local(self, file_pattern: str) -> List[Tuple[str, int]]:
        """
        Get files for local filesystem with minimal stat calls.
        """
        # Get files using glob
        if file_pattern == "*":
            files = self.filesystem.glob("**/*")
        else:
            files = self.filesystem.glob(f"**/{file_pattern}")

        files_with_info = []
        for file_path in files:
            # Quick path-based exclusions
            if self._should_exclude_by_path(file_path):
                continue

            # Check if it's a file and get size in one stat call
            try:
                if self.filesystem.is_file(file_path):
                    path_obj = self.filesystem.get_path(file_path)
                    # Use the existing _should_exclude_file logic from base class
                    if self._should_exclude_file(path_obj):
                        continue

                    # Get size from stat
                    try:
                        stat_info = path_obj.stat()
                        size = stat_info.st_size
                    except:
                        size = 0

                    files_with_info.append((file_path, size))
            except Exception:
                continue

        return files_with_info

    def _should_exclude_by_path(self, file_path: str) -> bool:
        """
        Check if file should be excluded based on its path alone.
        """
        path_parts = Path(file_path).parts
        return any(part in self.EXCLUDE_DIRS for part in path_parts)

    def _is_likely_binary_by_extension(self, file_path: str) -> bool:
        """
        Quick binary check based on file extension.
        """
        return Path(file_path).suffix.lower() in self.BINARY_EXTENSIONS

    def _extract_matching_lines(self, content: str, matches: List, max_lines: int = 5) -> List[str]:
        """
        Extract lines containing matches with line numbers.
        """
        lines = content.splitlines()
        line_matches = {}

        # Find which lines have matches
        for match in matches:
            line_num = content[: match.start()].count("\n") + 1
            if line_num not in line_matches:
                line_content = lines[line_num - 1].strip()
                # Truncate long lines to 200 characters
                if len(line_content) > 200:
                    line_content = line_content[:200] + "..."
                line_matches[line_num] = line_content

        # Format output - matching original format exactly
        matching_lines = []
        for line_num in sorted(line_matches.keys())[:max_lines]:
            matching_lines.append(f"  Line {line_num}: {line_matches[line_num]}")

        # Important: show total matches minus shown lines, not line count
        if len(line_matches) > max_lines:
            matching_lines.append(f"  ... and {len(matches) - max_lines} more matches")

        return matching_lines
