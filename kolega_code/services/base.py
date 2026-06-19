from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExecResult:
    """Result of an exec_command / write_stdin / kill_session call.

    A "session" is one running process. ``status`` is "exited" when the process
    has finished (``exit_code`` is set) or "running" when it is still alive
    (``session_id`` is set so it can be driven with write_stdin / kill_session).
    ``output`` is the head-tail-truncated, token-capped delta produced since the
    previous read for this session.
    """

    status: str
    session_id: Optional[str] = None
    exit_code: Optional[int] = None
    output: str = ""
    truncated: bool = False
    original_token_count: int = 0
    duration_ms: int = 0


class TerminalManager(ABC):
    """Abstract base class for terminal managers (codex-style unified exec).

    Each command runs as its own process (a session). Implementations stream
    output into a bounded buffer, report real exit codes, support writing stdin
    to a running session, and signal/kill sessions. Local backends allocate a
    PTY; sandbox backends run non-TTY background commands.
    """

    @abstractmethod
    async def exec_command(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
        login: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> "ExecResult":
        """Run ``command`` as a fresh process and collect output for a window.

        Waits up to ``yield_time_ms`` for the process to exit. If it exits, the
        result has status "exited" and an exit code. If it is still running, the
        result has status "running" and a ``session_id`` that can be driven with
        ``write_stdin`` and stopped with ``kill_session``.
        """

    @abstractmethod
    async def write_stdin(
        self,
        session_id: str,
        chars: str = "",
        *,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
    ) -> "ExecResult":
        """Write ``chars`` to a running session's stdin and read recent output.

        ``chars`` is sent raw (include a trailing newline to submit a line, or
        send ``"\\x03"`` for Ctrl-C). An empty ``chars`` polls for new output
        without writing. Raises ``KeyError`` if the session is unknown.
        """

    @abstractmethod
    async def kill_session(self, session_id: str, signal: str = "TERM") -> "ExecResult":
        """Terminate a running session and its process group.

        ``signal`` is "TERM" (graceful, then SIGKILL after a grace period) or
        "INT" (Ctrl-C / SIGINT). Raises ``KeyError`` if the session is unknown.
        """

    @abstractmethod
    async def list_sessions(self) -> Dict[str, Any]:
        """Return information about currently running sessions, keyed by id."""

    @abstractmethod
    async def close_all(self) -> None:
        """Terminate all sessions and release resources."""

    @abstractmethod
    async def run_command(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """Run a command to completion and return its output as a string.

        Convenience method for internal utilities (builds, sandbox setup); not
        exposed to the model.
        """

    async def cleanup_all(self) -> None:
        """Backwards-compatible alias for ``close_all`` used during teardown."""
        await self.close_all()


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
