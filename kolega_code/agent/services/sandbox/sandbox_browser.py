"""Browser manager implementation for sandbox environments."""

from typing import Any

from ..browser import PlaywrightBrowserManager


class SandboxBrowserManager(PlaywrightBrowserManager):
    """
    Browser manager for sandbox environments using Browserless.

    This class extends PlaywrightBrowserManager with Browserless support enabled,
    providing remote browser capabilities for sandbox environments.
    """

    def __init__(self, sandbox: Any = None):
        """
        Initialize sandbox browser manager with Browserless support.

        Args:
            sandbox: The sandbox instance (optional, kept for compatibility)
        """
        # Initialize parent class with Browserless backend
        super().__init__(browser_backend="browserless")
        self.sandbox = sandbox
