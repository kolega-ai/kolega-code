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
    """Abstract single-session browser interface used by the browser agent."""

    @abstractmethod
    async def navigate(self, url: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def snapshot(self, target: Optional[str] = None, depth: Optional[int] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def find(self, *, text: Optional[str] = None, regex: Optional[str] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def click(
        self,
        target: str,
        *,
        double_click: bool = False,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def type_text(
        self, target: str, text: str, *, submit: bool = False, slowly: bool = False
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def fill_form(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]: ...

    @abstractmethod
    async def select_option(self, target: str, values: List[str]) -> Dict[str, Any]: ...

    @abstractmethod
    async def hover(self, target: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def drag(self, start_target: str, end_target: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def drop(
        self,
        target: str,
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        data: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def press_key(self, key: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def navigate_back(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def wait_for(
        self, *, time: Optional[float] = None, text: Optional[str] = None, text_gone: Optional[str] = None
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def resize(self, width: int, height: int) -> Dict[str, Any]: ...

    @abstractmethod
    async def tabs(self, action: str, *, index: Optional[int] = None, url: Optional[str] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def handle_dialog(self, accept: bool, prompt_text: Optional[str] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def file_upload(self, files: List[Dict[str, Any]]) -> Dict[str, Any]: ...

    @abstractmethod
    async def console_messages(self, level: str = "info", *, all_messages: bool = False) -> Dict[str, Any]: ...

    @abstractmethod
    async def network_requests(
        self, *, include_static: bool = False, filter_pattern: Optional[str] = None
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def network_request(self, index: int, part: Optional[str] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def screenshot(
        self,
        *,
        target: Optional[str] = None,
        image_type: str = "png",
        full_page: bool = False,
        scale: str = "css",
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def evaluate(self, function: str, target: Optional[str] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> Optional[str]: ...

    async def cleanup_all_browsers(self) -> None:
        """Close the current browser session during agent teardown."""
        await self.close()
