"""Local development implementation of sandbox manager."""

from typing import Dict, Any, Optional, List

from .base import SandboxConfig, SandboxManager
from ..services.file_system import FileSystem, LocalFileSystem
from ..services.base import TerminalManager, BrowserManager
from ..services.terminal import LocalTerminalManager
from ..services.browser import PlaywrightBrowserManager


class LocalSandboxManager(SandboxManager):
    """Local development implementation that doesn't use actual sandboxes."""

    def __init__(self, connection_manager=None):
        """Initialize local sandbox manager."""
        self.connection_manager = connection_manager
        # Use local services
        self.filesystem = None
        self.terminal_manager = None
        self.browser_manager = None

    async def create_sandbox(
        self,
        workspace_id: str,
        thread_id: str,
        config: Optional[SandboxConfig] = None,
        workspace: Optional[Any] = None,
        connection_manager: Optional[Any] = None,
    ) -> str:
        """Create a 'local' sandbox (just returns a fake ID)."""
        # In local mode, we don't actually create sandboxes
        # Use a simple consistent ID
        sandbox_id = "local"

        # Initialize local services if not already done
        if self.filesystem is None and workspace:
            self.filesystem = LocalFileSystem(root_path=workspace.directory_path)
        if self.terminal_manager is None:
            self.terminal_manager = LocalTerminalManager(workspace_id, thread_id, self.connection_manager)
        if self.browser_manager is None:
            self.browser_manager = PlaywrightBrowserManager()

        return sandbox_id

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Destroy a 'local' sandbox (no-op in local mode)."""
        # In local mode, we don't need to destroy anything
        pass

    async def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Get sandbox status (always 'alive' in local mode)."""
        return {
            "sandbox_id": sandbox_id,
            "is_alive": True,
            "is_local": True,
        }

    async def commit_changes(self, sandbox_id: str, message: str, files: Optional[List[str]] = None) -> str:
        """Commit changes (no-op in local mode, returns empty string)."""
        # In local mode, we don't manage git
        return ""

    async def push_changes(self, sandbox_id: str) -> bool:
        """Push changes (no-op in local mode, returns False)."""
        # In local mode, we don't manage git
        return False

    def get_filesystem(self, sandbox_id: str) -> FileSystem:
        """Get filesystem for local development."""
        if self.filesystem is None:
            raise ValueError("Filesystem not initialized - call create_sandbox first")
        return self.filesystem

    def get_terminal_manager(self, sandbox_id: str) -> TerminalManager:
        """Get terminal manager for local development."""
        if self.terminal_manager is None:
            raise ValueError("Terminal manager not initialized - call create_sandbox first")
        return self.terminal_manager

    def get_browser_manager(self, sandbox_id: str) -> BrowserManager:
        """Get browser manager for local development."""
        if self.browser_manager is None:
            raise ValueError("Browser manager not initialized - call create_sandbox first")
        return self.browser_manager

    async def get_host(self, sandbox_id: str, port: int) -> str:
        """Always returns localhost for local development."""
        return f"localhost:{port}"

    async def pause_sandbox(self, sandbox_id: str) -> str:
        """Pause sandbox (no-op in local mode, returns sandbox_id)."""
        # In local mode, we don't pause sandboxes
        # Just return the sandbox_id as the "persistent" ID
        return sandbox_id

    async def resume_sandbox(self, persistent_sandbox_id: str, workspace_id: str, thread_id: str) -> str:
        """Resume sandbox (no-op in local mode, returns same ID)."""
        # In local mode, we don't actually resume sandboxes
        # Just return the same ID
        return persistent_sandbox_id

    def has_sandbox(self, sandbox_id: str) -> bool:
        """Check if a sandbox is currently active (always true in local mode if initialized)."""
        # In local mode, if we have services initialized, the "sandbox" is active
        return self.filesystem is not None

    async def adopt_sandbox(self, sandbox_id: str, workspace_id: str, thread_id: str) -> str:
        """Adopt sandbox (no-op in local mode, returns same ID)."""
        # In local mode, we don't actually adopt sandboxes
        # Just return the same ID
        return sandbox_id

    async def sync_sandbox_env_vars(self, sandbox_id: str, workspace_id: str, sandbox: Optional[Any] = None, skip_integration_env_sync: bool = False) -> None:
        """Sync environment variables (no-op in local mode)."""
        # In local mode, environment variables are managed by the local system
        # No need to sync to a sandbox profile
        pass
