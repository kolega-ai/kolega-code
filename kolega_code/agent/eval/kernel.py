"""Host-side eval kernel lifecycle: subprocess management and the cell executor.

``BaseKernel`` owns one kernel subprocess and speaks the NDJSON protocol from
``protocol.py``: a background reader task routes frames to the currently
executing cell, timeouts interrupt via SIGINT (escalating to SIGKILL, which
costs the kernel its state), and shutdown drains gracefully.

``EvalKernelManager`` owns one kernel per language for an agent session
(workspace_id, thread_id). Managers are shared with sub-agents through a
thread-keyed registry so kernel state persists across cells, tool calls, and
sub-agent dispatches within the session. While a cell runs, the manager
registers the executing agent on the loopback tool bridge so in-kernel
``tool.<name>(args)`` calls route through that agent's standard tool path.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Coroutine, Dict, List, Optional, Tuple

from kolega_code.llm.models import ImageBlock, TextBlock, ToolCall
from kolega_code.services.terminal import build_child_env
from kolega_code.tools import ToolError

from .bridge import BridgeRegistration, ToolBridge
from .env import EvalEnvironmentManager, KernelEnvInfo, default_state_dir
from .protocol import (
    FRAME_DISPLAY,
    FRAME_DONE,
    FRAME_ERROR,
    FRAME_RESULT,
    FRAME_STATUS,
    FRAME_STDERR,
    FRAME_STDOUT,
    FRAME_STREAM_LIMIT,
    STATUS_OK,
    ProtocolError,
    encode_frame,
    parse_frame,
)

STARTUP_TIMEOUT_S = 30.0
INTERRUPT_GRACE_S = 5.0
SHUTDOWN_GRACE_S = 1.0
CELL_PROTOCOL_VERSION = 1


@dataclass
class KernelErrorInfo:
    ename: str
    evalue: str
    traceback: List[str]


@dataclass
class EvalCellResult:
    """Accumulated outcome of one executed cell."""

    status: str = "ok"
    stdout: str = ""
    stderr: str = ""
    result_bundle: Dict[str, Any] = field(default_factory=dict)
    displays: List[Dict[str, Any]] = field(default_factory=list)
    statuses: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[KernelErrorInfo] = None
    interrupted: bool = False
    restarted: bool = False
    notes: List[str] = field(default_factory=list)


class KernelUnavailableError(ToolError):
    """The requested kernel language cannot run on this machine."""


class _Execution:
    """Accumulator for the frames of one in-flight cell."""

    def __init__(self, msg_id: int, run_id: str) -> None:
        self.msg_id = msg_id
        self.run_id = run_id
        self.done = asyncio.Event()
        self.result = EvalCellResult()

    def handle_frame(self, frame: Dict[str, Any]) -> None:
        frame_type = frame.get("type")
        result = self.result
        if frame_type == FRAME_STDOUT:
            result.stdout += str(frame.get("data") or "")
        elif frame_type == FRAME_STDERR:
            result.stderr += str(frame.get("data") or "")
        elif frame_type == FRAME_DISPLAY:
            bundle = frame.get("bundle")
            if isinstance(bundle, dict):
                result.displays.append(bundle)
        elif frame_type == FRAME_RESULT:
            bundle = frame.get("bundle")
            if isinstance(bundle, dict):
                result.result_bundle = bundle
        elif frame_type == FRAME_STATUS:
            event = frame.get("event")
            if isinstance(event, dict):
                result.statuses.append(event)
        elif frame_type == FRAME_ERROR:
            traceback = frame.get("traceback")
            result.error = KernelErrorInfo(
                ename=str(frame.get("ename") or "Error"),
                evalue=str(frame.get("evalue") or ""),
                traceback=[str(line) for line in traceback] if isinstance(traceback, list) else [],
            )
        elif frame_type == FRAME_DONE:
            result.status = str(frame.get("status") or "error")
            self.done.set()
        # FRAME_STARTED needs no handling.


class BaseKernel:
    """One kernel subprocess speaking the NDJSON cell protocol."""

    def __init__(self, *, language: str, cwd: str, env: Dict[str, str]) -> None:
        self.language = language
        self.cwd = cwd
        self.env = env
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stderr_tail: List[str] = []
        self._current: Optional[_Execution] = None
        self._msg_id = 0
        self._alive = False
        # Guards against overlapping execute() calls; the manager serializes per
        # language, and a non-blocking check there turns re-entry into an error.
        self._exec_lock = asyncio.Lock()

    # -- subclass hooks -----------------------------------------------------

    def argv(self) -> List[str]:
        raise NotImplementedError

    def init_cells(self) -> List[str]:
        """Silent setup cells run once after spawn (prelude etc.)."""
        return []

    # -- lifecycle ----------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._alive and self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        if self.alive:
            return
        await self.shutdown()
        self._stderr_tail.clear()
        self._proc = await asyncio.create_subprocess_exec(
            *self.argv(),
            cwd=self.cwd,
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=FRAME_STREAM_LIMIT,
        )
        self._alive = True
        self._reader_task = asyncio.create_task(self._reader_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())
        try:
            for cell in self.init_cells():
                await asyncio.wait_for(self._run_silent(cell), timeout=STARTUP_TIMEOUT_S)
        except Exception:
            await self.shutdown()
            raise

    async def shutdown(self, timeout: float = SHUTDOWN_GRACE_S) -> None:
        self._alive = False
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(encode_frame({"type": "exit"}))
                    await proc.stdin.wait_closed()
            except (BrokenPipeError, ConnectionError, RuntimeError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self._kill_proc(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        self._reader_task = None
        self._stderr_task = None
        current = self._current
        if current is not None and not current.done.is_set():
            current.result.status = "error"
            current.result.notes.append("kernel was shut down mid-cell")
            current.done.set()
        self._current = None

    def _kill_proc(self, proc: asyncio.subprocess.Process) -> None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    # -- protocol I/O -------------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    frame = parse_frame(line)
                except ProtocolError:
                    # C-level writes bypass the kernel's stream capture and land
                    # raw on fd 1; surface them as stdout rather than crashing.
                    current = self._current
                    if current is not None:
                        current.result.stdout += line.decode("utf-8", "replace")
                    continue
                self._route_frame(frame)
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            self._alive = False
            current = self._current
            if current is not None and not current.done.is_set():
                current.result.status = "error"
                current.result.notes.append("kernel exited unexpectedly; its state was lost")
                if self._stderr_tail:
                    current.result.stderr += "".join(self._stderr_tail[-20:])
                current.done.set()

    async def _stderr_loop(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                chunk = await self._proc.stderr.read(4096)
                if not chunk:
                    break
                self._stderr_tail.append(chunk.decode("utf-8", "replace"))
                del self._stderr_tail[:-50]
        except (ConnectionError, ValueError):
            pass

    def _route_frame(self, frame: Dict[str, Any]) -> None:
        current = self._current
        if current is None or frame.get("id") != current.msg_id:
            return
        current.handle_frame(frame)

    async def _write_frame(self, frame: Dict[str, Any]) -> None:
        if not self.alive or self._proc is None or self._proc.stdin is None:
            raise KernelUnavailableError(f"{self.language} kernel is not running")
        try:
            self._proc.stdin.write(encode_frame(frame))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionError, RuntimeError) as exc:
            self._alive = False
            raise KernelUnavailableError(f"{self.language} kernel pipe broke: {exc}") from exc

    # -- execution ----------------------------------------------------------

    async def _run_silent(self, code: str) -> None:
        execution = await self._begin_execution(code, run_id="", silent=True)
        await execution.done.wait()
        if execution.result.status != STATUS_OK:
            detail = ""
            if execution.result.error is not None:
                detail = f"{execution.result.error.ename}: {execution.result.error.evalue}"
            raise KernelUnavailableError(f"{self.language} kernel init failed: {detail or 'unknown error'}")

    async def _begin_execution(self, code: str, *, run_id: str, silent: bool) -> _Execution:
        self._msg_id += 1
        execution = _Execution(self._msg_id, run_id)
        self._current = execution
        await self._write_frame(
            {
                "type": "exec",
                "id": execution.msg_id,
                "run": run_id,
                "code": code,
                "cwd": self.cwd,
                "silent": silent,
            }
        )
        return execution

    async def execute(self, code: str, *, run_id: str, timeout: Optional[float]) -> EvalCellResult:
        """Run one cell, interrupting on timeout/cancellation."""
        if self._exec_lock.locked():
            raise ToolError(f"{self.language} kernel is busy with another cell")
        async with self._exec_lock:
            if not self.alive:
                raise KernelUnavailableError(f"{self.language} kernel is not running")
            execution = await self._begin_execution(code, run_id=run_id, silent=False)
            try:
                if timeout is not None and timeout > 0:
                    await asyncio.wait_for(execution.done.wait(), timeout=timeout)
                else:
                    await execution.done.wait()
            except asyncio.TimeoutError:
                execution.result.interrupted = True
                note = await self._interrupt(execution)
                execution.result.status = "error"
                execution.result.notes.append(f"cell timed out after {timeout}s; {note}")
                return self._finish(execution)
            except asyncio.CancelledError:
                execution.result.interrupted = True
                await self._interrupt(execution)
                raise
            finally:
                if self._current is execution:
                    self._current = None
            return self._finish(execution)

    async def _interrupt(self, execution: _Execution) -> str:
        """SIGINT the cell; escalate to kill. Returns a human note."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return "kernel was already dead"
        if sys.platform == "win32":
            # No reliable per-process Ctrl-C for detached children here.
            self._kill_proc(proc)
        else:
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                return "kernel was already dead"
        try:
            await asyncio.wait_for(execution.done.wait(), timeout=INTERRUPT_GRACE_S)
            return "interrupted (KeyboardInterrupt)"
        except asyncio.TimeoutError:
            self._kill_proc(proc)
            try:
                await asyncio.wait_for(execution.done.wait(), timeout=SHUTDOWN_GRACE_S)
            except asyncio.TimeoutError:
                pass
            return "cell did not interrupt; the kernel was killed and its state was lost"

    def _finish(self, execution: _Execution) -> EvalCellResult:
        return execution.result


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


def _cfg_str(config: object, name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a string AgentConfig field, tolerating Mock configs in tests."""
    value = getattr(config, name, default)
    return value if isinstance(value, str) and value.strip() else default


def _cfg_str_list(config: object, name: str) -> List[str]:
    value = getattr(config, name, None)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def serialize_tool_result(result: Any) -> Dict[str, Any]:
    """Flatten an agent ToolResult into the JSON value returned to kernels."""
    content = getattr(result, "content", "")
    texts: List[str] = []
    images: List[Dict[str, str]] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ImageBlock):
                images.append({"mime_type": block.media_type, "data": block.data})
            elif hasattr(block, "to_markdown"):
                texts.append(str(block.to_markdown()))
            else:
                texts.append(str(block))
    else:
        texts.append(str(content))
    value: Dict[str, Any] = {"text": "\n".join(texts)}
    if images:
        value["images"] = images
    if getattr(result, "is_error", False):
        raise ToolError(value["text"] or "tool call failed")
    return value


class EvalKernelManager:
    """Owns the session's kernels (one per language) and bridge registrations."""

    _registry: ClassVar[Dict[Tuple[str, str], "EvalKernelManager"]] = {}

    @classmethod
    def for_thread(
        cls,
        *,
        workspace_id: str,
        thread_id: str,
        project_path: Path,
        config: object,
    ) -> "EvalKernelManager":
        """Shared manager for an agent session; sub-agents join the same one."""
        key = (workspace_id, thread_id)
        manager = cls._registry.get(key)
        if manager is None or manager._shutdown:
            manager = cls(workspace_id=workspace_id, thread_id=thread_id, project_path=project_path, config=config)
            cls._registry[key] = manager
        return manager

    def __init__(self, *, workspace_id: str, thread_id: str, project_path: Path, config: object) -> None:
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.project_path = Path(project_path)
        self.config = config
        self.session_id = uuid.uuid4().hex
        self._kernels: Dict[str, BaseKernel] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._env_manager: Optional[EvalEnvironmentManager] = None
        self._env_info: Optional[KernelEnvInfo] = None
        self._shutdown = False

    # -- kernels ------------------------------------------------------------

    def _env_manager_for_thread(self) -> EvalEnvironmentManager:
        if self._env_manager is None:
            env_path_raw = _cfg_str(self.config, "eval_env_path")
            self._env_manager = EvalEnvironmentManager(
                env_path=Path(env_path_raw).expanduser() if env_path_raw else None,
                python_version=_cfg_str(self.config, "eval_python_version", "3.12") or "3.12",
                extra_packages=_cfg_str_list(self.config, "eval_kernel_packages"),
                override_python=_cfg_str(self.config, "eval_python_path"),
            )
        return self._env_manager

    async def _ensure_env(self) -> KernelEnvInfo:
        if self._env_info is None or self._env_info.degraded:
            info = await self._env_manager_for_thread().ensure()
            if info.provisioned_now:
                # The one-time setup note belongs only to the kernel start that
                # actually provisioned; later starts must reuse the env quietly.
                self._env_info = replace(info, provisioned_now=False, note=None)
                return info
            self._env_info = info
        return self._env_info

    async def _start_python_kernel(self) -> Tuple[BaseKernel, List[str]]:
        from .py_kernel import PythonKernel

        env_info = await self._ensure_env()
        bridge = await ToolBridge.ensure()
        kernel_env = build_child_env()
        kernel_env.update(
            {
                "KOLEGA_TOOL_BRIDGE_URL": bridge.url,
                "KOLEGA_TOOL_BRIDGE_TOKEN": bridge.token,
                "KOLEGA_TOOL_BRIDGE_SESSION": self.session_id,
                "KOLEGA_EVAL_ENV_PATH": str(env_info.env_path or ""),
                "KOLEGA_EVAL_BUNDLE": ",".join(env_info.bundle),
                "KOLEGA_EVAL_PIP_INSTALL_CMD": json.dumps(env_info.pip_install_cmd),
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        kernel = PythonKernel(
            cwd=str(self.project_path),
            env=kernel_env,
            python_path=env_info.python,
            state_dir=default_state_dir(),
        )
        await kernel.start()
        notes = [env_info.note] if env_info.note else []
        return kernel, notes

    async def _start_js_kernel(self) -> Tuple[BaseKernel, List[str]]:
        from .js_kernel import JsKernel, probe_js_runtime

        runtime = probe_js_runtime(self.config)
        if runtime is None:
            raise KernelUnavailableError(
                "JavaScript kernel unavailable: neither 'bun' nor 'node' (>= 18) was found on PATH. "
                "Install Bun (https://bun.sh) or Node.js, or use language='py'."
            )
        bridge = await ToolBridge.ensure()
        js_home = default_state_dir() / "eval-js"
        js_home.mkdir(parents=True, exist_ok=True)
        package_json = js_home / "package.json"
        if not package_json.exists():
            package_json.write_text("{}\n", encoding="utf-8")
        kernel_env = build_child_env()
        npm_cmd = runtime.npm_install_cmd(js_home)
        kernel_env.update(
            {
                "KOLEGA_TOOL_BRIDGE_URL": bridge.url,
                "KOLEGA_TOOL_BRIDGE_TOKEN": bridge.token,
                "KOLEGA_TOOL_BRIDGE_SESSION": self.session_id,
                "KOLEGA_EVAL_JS_HOME": str(js_home),
                "KOLEGA_EVAL_NPM_INSTALL_CMD": json.dumps(npm_cmd),
                "NODE_PATH": str(js_home / "node_modules"),
            }
        )
        kernel = JsKernel(
            cwd=str(self.project_path),
            env=kernel_env,
            runtime=runtime,
            state_dir=default_state_dir(),
        )
        await kernel.start()
        return kernel, [f"JavaScript kernel running on {runtime.name} ({runtime.path})."]

    async def _get_or_start_kernel(self, language: str) -> Tuple[BaseKernel, List[str], bool]:
        kernel = self._kernels.get(language)
        if kernel is not None and kernel.alive:
            return kernel, [], False
        restarted = kernel is not None
        if kernel is not None:
            await kernel.shutdown()
            self._kernels.pop(language, None)
        if language == "py":
            kernel, notes = await self._start_python_kernel()
        elif language == "js":
            kernel, notes = await self._start_js_kernel()
        else:
            raise ToolError(f"unsupported eval language {language!r}; use 'py' or 'js'")
        self._kernels[language] = kernel
        if restarted:
            notes = list(notes) + ["The previous kernel died; restarted with fresh state."]
        return kernel, notes, restarted

    async def _stop_kernel(self, language: str) -> None:
        kernel = self._kernels.pop(language, None)
        if kernel is not None:
            await kernel.shutdown()

    # -- bridge -------------------------------------------------------------

    def _make_tool_caller(self, agent: object) -> Callable[[str, Any], Coroutine[Any, Any, Any]]:
        async def call(name: str, args: Any) -> Any:
            if not isinstance(args, dict):
                raise ToolError(f"tool {name!r} via the eval bridge needs an object of arguments")
            if name == "eval":
                language = str(args.get("language") or "")
                if language in self._locks and self._locks[language].locked():
                    raise ToolError(
                        f"the {language} kernel is busy running this cell; call eval with another "
                        "language or run it after this cell finishes"
                    )
            tool_call = ToolCall(id=f"eval-{uuid.uuid4().hex}", name=name, input=args)
            result = await agent.execute_single_tool(tool_call)  # type: ignore[attr-defined]
            return serialize_tool_result(result)

        return call

    def _make_tool_lister(self, agent: object) -> Callable[[], List[Dict[str, str]]]:
        def list_tools() -> List[Dict[str, str]]:
            collection = getattr(agent, "tool_collection", None)
            if collection is None:
                return []
            tools: List[Dict[str, str]] = []
            for definition in collection.get_tool_list():
                summary = (definition.description or "").strip().splitlines()
                tools.append({"name": definition.name, "summary": summary[0] if summary else ""})
            return tools

        return list_tools

    # -- public API ---------------------------------------------------------

    async def execute(
        self,
        *,
        language: str,
        code: str,
        agent: object,
        timeout: Optional[float],
        reset: bool = False,
    ) -> EvalCellResult:
        """Run one cell on the session's kernel for ``language``."""
        if self._shutdown:
            raise ToolError("the eval kernel manager for this session has been shut down")
        lock = self._locks.setdefault(language, asyncio.Lock())
        if lock.locked():
            raise ToolError(
                f"the {language} kernel is busy with another cell. Wait for it to finish, "
                "or pass reset=true to restart it."
            )
        async with lock:
            if reset:
                await self._stop_kernel(language)
            kernel, notes, restarted = await self._get_or_start_kernel(language)
            run_id = uuid.uuid4().hex
            bridge = await ToolBridge.ensure()
            registration = BridgeRegistration(
                tool_caller=self._make_tool_caller(agent),
                tool_lister=self._make_tool_lister(agent),
            )
            unregister = bridge.register(self.session_id, run_id, registration)
            try:
                try:
                    result = await kernel.execute(code, run_id=run_id, timeout=timeout)
                except KernelUnavailableError:
                    # Kernel died between liveness check and write; retry once on
                    # a fresh kernel so a crash does not fail the user's cell.
                    await self._stop_kernel(language)
                    kernel, notes, restarted = await self._get_or_start_kernel(language)
                    notes = list(notes) + ["The kernel died at cell start; retried on a fresh kernel (state lost)."]
                    result = await kernel.execute(code, run_id=run_id, timeout=timeout)
            finally:
                unregister()
            result.notes = notes + result.notes
            result.restarted = restarted
            return result

    async def shutdown(self) -> None:
        """Shut down every kernel and unregister the manager (owner only)."""
        if self._shutdown:
            return
        self._shutdown = True
        for language in list(self._kernels):
            await self._stop_kernel(language)
        key = (self.workspace_id, self.thread_id)
        if EvalKernelManager._registry.get(key) is self:
            del EvalKernelManager._registry[key]
