from typing import Any

from kolega_code.services.base import ExecResult, TerminalManager


class _StubSandbox:
    def get_host(self, port: int) -> str:
        return f"stub-sandbox:{port}"


class SandboxCarryingTerminalManager(TerminalManager):
    """ABC-based terminal-manager stand-in carrying a sandbox host provider."""

    def __init__(self) -> None:
        self.sandbox = _StubSandbox()

    async def exec_command(
        self,
        command: str,
        *,
        workdir: str | None = None,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
        login: bool = False,
        env: dict[str, str] | None = None,
        background: bool = False,
    ) -> ExecResult:
        _ = (command, workdir, yield_time_ms, max_output_tokens, login, env, background)
        raise AssertionError("exec_command is not used by tool inventory tests")

    async def write_stdin(
        self,
        session_id: str,
        chars: str = "",
        *,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
    ) -> ExecResult:
        _ = (session_id, chars, yield_time_ms, max_output_tokens)
        raise AssertionError("write_stdin is not used by tool inventory tests")

    async def kill_session(self, session_id: str, signal: str = "TERM") -> ExecResult:
        _ = (session_id, signal)
        raise AssertionError("kill_session is not used by tool inventory tests")

    async def list_sessions(self) -> dict[str, Any]:
        raise AssertionError("list_sessions is not used by tool inventory tests")

    async def close_all(self) -> None:
        raise AssertionError("close_all is not used by tool inventory tests")

    async def run_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        _ = (command, cwd, timeout)
        raise AssertionError("run_command is not used by tool inventory tests")
