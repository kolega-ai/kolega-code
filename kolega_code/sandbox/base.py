"""Base interfaces for sandbox management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..services.base import TerminalManager, BrowserManager
from ..services.file_system import FileSystem


@runtime_checkable
class SandboxHandle(Protocol):
    """
    The duck-typed interface the generic sandbox services expect from a
    provider's sandbox object (e.g. E2B's AsyncSandbox).

    SandboxFileSystem, SandboxTerminalManager, and friends only touch these
    members; a provider object satisfying this protocol can back them all.
    """

    @property
    def commands(self) -> Any:
        """Command runner exposing run(...), and send_stdin(pid, data) where supported."""
        ...

    @property
    def files(self) -> Any:
        """File API exposing read/write/list/exists operations."""
        ...

    def get_host(self, port: int) -> str:
        """Return the externally reachable host for a port inside the sandbox."""
        ...


@dataclass
class SandboxConfig:
    """Configuration for sandbox creation."""

    git_url: str
    branch: str = "main"
    commit_hash: Optional[str] = None
    manifest_path: str = ".kolega-manifest.yaml"
    resources: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    auth_method: str = "group_token"  # group_token, pat, or ssh
    network_access_mode: str = "deny_all"  # deny_all | allow_all | custom
    network_allowed_hosts: List[str] = field(default_factory=list)
    # Pool mode flags
    skip_git: bool = False
    skip_s3_mount: bool = False
    skip_project_setup: bool = False
    skip_integration_env_sync: bool = False  # Skip user-specific environment variable sync


@dataclass
class ProjectManifest:
    """Project configuration manifest."""

    name: str
    runtime: str  # e.g., "node:18", "python:3.11"
    install_commands: Optional[List[str]] = None
    post_setup_commands: Optional[List[str]] = None
    dev_server_command: Optional[str] = None
    dev_server_commands: Optional[List[str]] = None
    test_commands: Optional[List[str]] = None
    build_command: Optional[str] = None
    backend_build_command: Optional[str] = None
    frontend_build_command: Optional[str] = None
    environment_setup: Optional[List[str]] = None


class SandboxManager(ABC):
    """Abstract base class for sandbox managers."""

    @abstractmethod
    async def create_sandbox(
        self,
        workspace_id: str,
        thread_id: str,
        config: Optional[SandboxConfig] = None,
        workspace: Optional[Any] = None,
        connection_manager: Optional[Any] = None,
    ) -> str:
        """
        Create a new sandbox.

        Args:
            workspace_id: ID of the workspace
            thread_id: Thread ID for terminal output streaming
            config: Sandbox configuration
            workspace: Optional workspace object for additional context
            connection_manager: Optional connection manager for dispatching events

        Returns:
            Sandbox ID
        """
        pass

    @abstractmethod
    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """
        Destroy a sandbox and clean up resources.

        Args:
            sandbox_id: ID of the sandbox to destroy

        Raises:
            ValueError: If sandbox doesn't exist
        """

    @abstractmethod
    async def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """
        Get current status of a sandbox.

        Args:
            sandbox_id: ID of the sandbox

        Returns:
            Dictionary with status information

        Raises:
            ValueError: If sandbox doesn't exist
        """

    @abstractmethod
    async def commit_changes(self, sandbox_id: str, message: str, files: Optional[List[str]] = None) -> str:
        """
        Commit changes in sandbox and return commit hash.

        Args:
            sandbox_id: ID of the sandbox
            message: Commit message
            files: Optional list of files to commit (None = all changes)

        Returns:
            Git commit hash

        Raises:
            ValueError: If sandbox doesn't exist
            Exception: If commit fails
        """

    @abstractmethod
    async def push_changes(self, sandbox_id: str) -> bool:
        """
        Push committed changes to remote repository.

        Args:
            sandbox_id: ID of the sandbox

        Returns:
            True if push succeeded, False otherwise

        Raises:
            ValueError: If sandbox doesn't exist
        """

    @abstractmethod
    def get_filesystem(self, sandbox_id: str) -> FileSystem:
        """
        Get filesystem interface for sandbox.

        Args:
            sandbox_id: ID of the sandbox

        Returns:
            FileSystem implementation for the sandbox

        Raises:
            ValueError: If sandbox doesn't exist
        """

    @abstractmethod
    def get_terminal_manager(self, sandbox_id: str) -> TerminalManager:
        """
        Get terminal manager for sandbox.

        Args:
            sandbox_id: ID of the sandbox

        Returns:
            TerminalManager implementation for the sandbox

        Raises:
            ValueError: If sandbox doesn't exist
        """

    @abstractmethod
    def get_browser_manager(self, sandbox_id: str) -> BrowserManager:
        """
        Get browser manager for sandbox.

        Args:
            sandbox_id: ID of the sandbox

        Returns:
            BrowserManager implementation for the sandbox

        Raises:
            ValueError: If sandbox doesn't exist
        """

    @abstractmethod
    async def get_host(self, sandbox_id: str, port: int) -> str:
        """
        Get the hostname for accessing services on the given port.

        Args:
            sandbox_id: ID of the sandbox
            port: The port number to access

        Returns:
            The hostname (e.g., 'localhost' or 'xxxx.e2b.dev')
        """
        pass

    @abstractmethod
    async def pause_sandbox(self, sandbox_id: str) -> str:
        """
        Pause sandbox and return persistent sandbox ID for resuming later.

        Args:
            sandbox_id: ID of the sandbox to pause

        Returns:
            Persistent sandbox ID that can be used to resume

        Raises:
            ValueError: If sandbox doesn't exist
        """
        pass

    @abstractmethod
    async def resume_sandbox(self, persistent_sandbox_id: str, workspace_id: str, thread_id: str) -> str:
        """
        Resume a paused sandbox from persistent ID.

        Args:
            persistent_sandbox_id: The persistent ID returned from pause_sandbox
            workspace_id: ID of the workspace
            thread_id: Thread ID for terminal output streaming

        Returns:
            New sandbox ID for the resumed sandbox

        Raises:
            Exception: If sandbox cannot be resumed
        """
        pass

    @abstractmethod
    def has_sandbox(self, sandbox_id: str) -> bool:
        """
        Check if a sandbox is currently active in memory.

        Args:
            sandbox_id: ID of the sandbox to check

        Returns:
            True if sandbox is active, False otherwise
        """
        pass

    @abstractmethod
    async def adopt_sandbox(self, sandbox_id: str, workspace_id: str, thread_id: str) -> str:
        """
        Adopt an existing sandbox into this manager instance.

        Args:
            sandbox_id: The sandbox ID to adopt
            workspace_id: ID of the workspace
            thread_id: Thread ID for terminal output streaming

        Returns:
            The sandbox ID (same as input)

        Raises:
            Exception: If sandbox cannot be connected to
        """
        pass

    @abstractmethod
    async def sync_sandbox_env_vars(
        self, sandbox_id: str, workspace_id: str, sandbox: Optional[Any] = None, skip_integration_env_sync: bool = False
    ) -> None:
        """
        Sync current environment variables to an existing sandbox.

        Args:
            sandbox_id: ID of the sandbox
            workspace_id: ID of the workspace (used to fetch env vars from database)
            sandbox: Optional sandbox instance (avoids reconnection if provided)

        Raises:
            Exception: If sandbox cannot be connected to or env vars cannot be synced
        """
        pass
