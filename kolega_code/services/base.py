from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TerminalManager(ABC):
    """
    Abstract base class for terminal managers.
    Implementations should handle creation, management, and cleanup of terminal instances.
    """

    @abstractmethod
    async def launch_terminal(self, terminal_id: Optional[str] = None, **terminal_kwargs) -> str:
        """
        Create a new terminal instance with the given ID or generate a random ID.

        Args:
            terminal_id: Optional identifier for the terminal (random UUID if not provided)
            **terminal_kwargs: Arguments to pass to terminal constructor

        Returns:
            The ID of the created terminal
        """

    @abstractmethod
    async def send_command(
        self, term_id: str, command: str, purpose: Optional[str] = None, timeout: Optional[int] = None
    ) -> bool:
        """
        Send a command to a specific terminal.

        Args:
            term_id: ID of the terminal to send command to
            command: The command to execute
            purpose: Optional description of command purpose
            timeout: Optional timeout in seconds (0 or None for no timeout)

        Returns:
            True if command was sent successfully

        Raises:
            KeyError: If terminal_id doesn't exist
        """

    @abstractmethod
    async def send_input(
        self, term_id: str, text: str, submit: bool = True, command_id: Optional[str] = None
    ) -> bool:
        """
        Send input to a command that is already running in a terminal.

        Args:
            term_id: ID of the terminal to send input to
            text: Input text to send
            submit: Whether to append a newline before sending
            command_id: Optional active command ID when the backend requires disambiguation

        Returns:
            True if input was sent successfully

        Raises:
            KeyError: If terminal_id doesn't exist
            ValueError: If no running command can receive input
        """

    @abstractmethod
    async def get_output(self, terminal_id: str, **kwargs) -> str:
        """
        Get output from a specific terminal.

        Args:
            terminal_id: ID of the terminal to get output from
            **kwargs: Arguments to pass to get output method

        Returns:
            Output from the specified terminal

        Raises:
            KeyError: If terminal_id doesn't exist
        """

    @abstractmethod
    async def close_terminal(self, terminal_id: str) -> None:
        """
        Close a specific terminal.

        Args:
            terminal_id: ID of the terminal to close

        Raises:
            KeyError: If terminal_id doesn't exist
        """

    @abstractmethod
    async def close_all(self) -> None:
        """Close all terminal instances."""

    @abstractmethod
    async def list_terminals(self) -> Dict[str, Any]:
        """
        Get information about all terminals.

        Returns:
            Dictionary mapping terminal IDs to terminal information
        """

    @abstractmethod
    async def run_command(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """
        Run a command directly (convenience method for utilities).

        Args:
            command: Command to execute
            cwd: Optional working directory
            timeout: Optional timeout in seconds

        Returns:
            Command output as string
        """


class BrowserManager(ABC):
    """
    Abstract base class for browser management.
    Defines the interface for browser operations.
    """

    @abstractmethod
    async def launch_browser(self, url: str) -> str:
        """
        Launch a new browser instance and navigate to the specified URL.

        Args:
            url: The URL to navigate to

        Returns:
            Browser ID string

        Raises:
            Exception: If browser launch fails
        """

    @abstractmethod
    async def list_browsers(self) -> dict:
        """
        Get information about all active browser instances.

        Returns:
            Dictionary mapping browser IDs to browser information
        """

    @abstractmethod
    async def get_browser_console_logs(
        self,
        browser_id: str,
        max_logs: int = 50,
        log_types: Optional[List[str]] = None,
        minutes_back: Optional[int] = None,
        max_chars: Optional[int] = 8000,
    ) -> dict:
        """
        Get console logs from a specific browser with configurable filtering.

        Args:
            browser_id: ID of the browser to get logs from
            max_logs: Maximum number of logs to return (most recent)
            log_types: List of log types to include (e.g., ['error', 'warning', 'assert'])
            minutes_back: Only return logs from the last N minutes
            max_chars: Maximum total character count for all log messages combined

        Returns:
            Dictionary containing filtered console logs and metadata

        Raises:
            KeyError: If browser_id doesn't exist
        """

    @abstractmethod
    async def get_browser_interactive_elements(self, browser_id: str) -> list:
        """
        Get interactive elements from a specific browser.

        Args:
            browser_id: ID of the browser to get elements from

        Returns:
            Dictionary containing current URL, title, and interactive elements

        Raises:
            KeyError: If browser_id doesn't exist
        """

    @abstractmethod
    async def take_browser_screenshot(self, browser_id: str) -> dict:
        """
        Take a screenshot of a specific browser.

        Args:
            browser_id: ID of the browser to take screenshot of

        Returns:
            Dictionary containing current URL, title, and base64-encoded screenshot

        Raises:
            KeyError: If browser_id doesn't exist
        """

    @abstractmethod
    async def interact_with_browser(self, browser_id: str, action: str, selector: str, text: str, scroll_px) -> dict:
        """
        Interact with a specific browser.

        Args:
            browser_id: ID of the browser to interact with
            action: Type of interaction (click, type, scroll, navigate)
            selector: CSS selector for the element to interact with
            text: Text to type or URL to navigate to
            scroll_px: Number of pixels to scroll

        Returns:
            Dictionary containing status and interaction details

        Raises:
            KeyError: If browser_id doesn't exist
            ValueError: If action is unknown
        """

    @abstractmethod
    async def set_select_value(self, browser_id: str, selector: str, value: str) -> dict:
        """
        Set the value of a select box in a specific browser.

        Args:
            browser_id: ID of the browser to interact with
            selector: CSS selector for the select element
            value: The value to set for the select option

        Returns:
            Dictionary containing status, current URL, and selected value

        Raises:
            KeyError: If browser_id doesn't exist
            ValueError: If the element is not a select box or value doesn't exist
        """

    @abstractmethod
    async def close_browser(self, browser_id: str) -> None:
        """
        Close a specific browser.

        Args:
            browser_id: ID of the browser to close

        Raises:
            KeyError: If browser_id doesn't exist
        """
