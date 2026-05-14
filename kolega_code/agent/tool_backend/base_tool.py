import os
from pathlib import Path
from typing import Union, Optional, Set

from ..common import LogMixin
from ..config import AgentConfig
from ..connection_manager import AgentConnectionManager
from ..services.file_system import FileSystem, LocalFileSystem
from ..services.base import TerminalManager, BrowserManager
from ..prompt_provider import AgentMode


class BaseTool(LogMixin):

    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        config: AgentConfig,
        caller,
        filesystem: Optional[FileSystem] = None,
        terminal_manager: Optional[TerminalManager] = None,
        browser_manager: Optional[BrowserManager] = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.project_path = Path(project_path) if isinstance(project_path, str) else project_path
        self.connection_manager = connection_manager
        self.config = config
        self.caller = caller

        # Create filesystem instance if not provided
        if filesystem is None:
            self.filesystem = LocalFileSystem(root_path=self.project_path)
        else:
            self.filesystem = filesystem

        # Store optional managers (individual tools will use them if needed)
        self.terminal_manager = terminal_manager
        self.browser_manager = browser_manager

    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Determine if a file is binary.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file is binary, False otherwise
        """
        # Check file extension for common binary formats
        binary_extensions = {
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
        }

        if file_path.suffix.lower() in binary_extensions:
            return True

        # Check file contents (sample the first 1024 bytes)
        try:
            with file_path.open("rb") as f:
                sample = f.read(1024)
                if b"\x00" in sample:  # If null byte is present, likely binary
                    return True
        except Exception:
            # If there's an error reading the file, consider it binary to be safe
            return True

        return False

    def _should_exclude_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be excluded from search.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be excluded, False otherwise
        """
        # Common directories to exclude
        exclude_directories = {
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
        }

        # Check if any parent directory is in the exclude list
        for parent in file_path.parents:
            if parent.name in exclude_directories:
                return True

        # Exclude very large files (> 10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return True
        except Exception:
            # If we can't get the file size, exclude it to be safe
            return True

        # Check if file is excluded by .gitignore
        if self._is_gitignored(file_path):
            return True

        return False

    def _is_gitignored(self, file_path: Path) -> bool:
        """
        Check if a file is excluded by .gitignore patterns.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file is ignored according to .gitignore, False otherwise
        """
        # Use cached gitignore patterns if available
        if not hasattr(self, "_gitignore_spec"):
            self._load_gitignore_patterns()

        if not hasattr(self, "_gitignore_spec") or self._gitignore_spec is None:
            return False

        # Get the path relative to the project root
        relative_path = str(file_path.relative_to(self.project_path))

        # Check if the path matches any gitignore pattern
        return self._gitignore_spec.match_file(relative_path)

    def _load_gitignore_patterns(self) -> None:
        """
        Load .gitignore patterns from the project root.
        Creates a pathspec matcher that can be used to check if files match gitignore patterns.
        """
        try:
            import pathspec

            if not self.filesystem.exists(".gitignore"):
                self._gitignore_spec = None
                return

            gitignore_content = self.filesystem.read_text(".gitignore", encoding="utf-8")

            # Parse gitignore patterns
            self._gitignore_spec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, gitignore_content.splitlines()
            )
        except Exception as e:
            # If there's an error loading gitignore, log it and continue without gitignore filtering
            print(f"Error loading .gitignore: {str(e)}")
            self._gitignore_spec = None

    # --- Vibe-mode edit policy helpers ---
    def _is_vibe_mode(self) -> bool:
        """Return True if caller agent is in VIBE mode (same check pattern as agents)."""
        return getattr(self.caller, "agent_mode", None) == AgentMode.VIBE.value

    def _get_vibe_blacklist_basenames(self) -> Set[str]:
        """Basename blacklist for files that should not be edited in vibe mode."""
        protected_files = getattr(self.caller, "protected_files", None) if self.caller else None
        if protected_files:
            return set(protected_files)
        return {"package.json", "tsconfig.json"}

    def _enforce_vibe_edit_policy(self, relative_path: str) -> Optional[str]:
        """Return message if blocked; None if allowed."""
        if not self._is_vibe_mode():
            return None

        if os.path.basename(relative_path) in self._get_vibe_blacklist_basenames():
            return f"You are not allowed to edit this file: {relative_path}"
        return None
