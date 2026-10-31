"""Utility functions for sandbox operations."""

import logging
import shlex
from typing import List, Optional
from .base import SandboxManager

logger = logging.getLogger(__name__)

WORKSPACE_PATH = "/home/user/workspace"


def parse_git_status_output(status_output: str) -> List[str]:
    """Parse git status --porcelain output and return list of files to add.

    This is a shared parsing function used by both direct E2B sandbox operations
    and SandboxManager-based operations to ensure consistent git status parsing.

    Args:
        status_output: Raw output from 'git -c core.quotePath=false status --porcelain'

    Returns:
        List of file paths, excluding deleted files and handling renames correctly
    """
    logger.debug(f"Parsing git status output: {status_output!r}")

    if not status_output or not status_output.strip():
        return []

    modified_files = []
    for line in status_output.split("\n"):
        if not line or len(line) < 3:
            continue

        # Git status format: XY filename
        # X = staging area status, Y = working tree status
        # Status codes (per git-status --porcelain):
        #   ' ' = unmodified
        #   M = modified
        #   A = added
        #   D = deleted
        #   R = renamed
        #   C = copied
        #   U = updated but unmerged
        #   ? = untracked
        #   ! = ignored
        status = line[:2]  # First 2 characters are status codes
        file_path = line[3:].strip()  # Everything after "XY "

        # Skip STAGED deletions (D at position 0) - these are already removed from git index
        # These will fail with "pathspec did not match any files" error
        # Working tree deletions (" D" at position 1, like " D", "MD", "AD") can still be staged with git add, so we include them
        if status[0] == "D":
            logger.debug(f"Skipping staged deletion (not in git index): {file_path}")
            continue

        # Handle renamed files: "R  old -> new"
        # Git quotes filenames containing " -> ": R  "old -> a.txt" -> "new -> b.txt"
        if status.startswith("R"):
            logger.debug(f"Processing renamed file - status: {status!r}, file_path: {file_path!r}")
            if " -> " in file_path:
                # Find the separator " -> " that's NOT inside quotes
                in_quotes = False
                for i in range(len(file_path) - 4):  # -4 for " -> " (4 chars)
                    if file_path[i] == '"':
                        in_quotes = not in_quotes
                    elif not in_quotes and file_path[i : i + 4] == " -> ":
                        # Found the real separator between old and new filenames
                        file_path = file_path[i + 4 :].strip()
                        break

        # Remove quotes if present (git quotes filenames with special chars or " -> ")
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
            # Unescape git's quote escaping (\" becomes ")
            file_path = file_path.replace('\\"', '"')

        modified_files.append(file_path)

    return modified_files


async def get_modified_files_from_sandbox(sandbox_manager: SandboxManager, sandbox_id: str) -> List[str]:
    """
    Get list of modified files from sandbox using git status.

    Args:
        sandbox_manager: The sandbox manager instance
        sandbox_id: ID of the sandbox

    Returns:
        List of modified file paths
    """
    try:
        terminal_manager = sandbox_manager.get_terminal_manager(sandbox_id)

        # Run git status with core.quotePath=false to get raw filenames without escaping
        # This prevents issues with special characters (e.g., em-dash — being escaped as \342\200\224)
        result = await terminal_manager.run_command(
            "git -c core.quotePath=false status --porcelain", cwd=WORKSPACE_PATH
        )

        # Use shared parsing function
        return parse_git_status_output(result)

    except Exception as e:
        logger.error(f"Error getting modified files from sandbox {sandbox_id}: {e}")
        return []


async def get_git_diff_from_sandbox(
    sandbox_manager: SandboxManager, sandbox_id: str, files: Optional[List[str]] = None
) -> str:
    """
    Get git diff from sandbox.

    Args:
        sandbox_manager: The sandbox manager instance
        sandbox_id: ID of the sandbox
        files: Optional list of specific files to diff

    Returns:
        Git diff output as string
    """
    try:
        terminal_manager = sandbox_manager.get_terminal_manager(sandbox_id)

        # Include untracked files in the diff without staging their contents.
        try:
            await terminal_manager.run_command("git add -N .", cwd=WORKSPACE_PATH)
        except Exception as add_error:
            logger.warning(f"Failed to mark untracked files for diff in sandbox {sandbox_id}: {add_error}")

        if files:
            # Diff specific files
            files_arg = " ".join(shlex.quote(f) for f in files)
            cmd = f"git diff HEAD -- {files_arg}"
        else:
            # Diff all changes
            cmd = "git diff HEAD"

        # Run git diff
        result = await terminal_manager.run_command(cmd, cwd=WORKSPACE_PATH)

        return result or ""

    except Exception as e:
        logger.error(f"Error getting git diff from sandbox {sandbox_id}: {e}")
        return ""
