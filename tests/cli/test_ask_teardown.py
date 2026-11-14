from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar, Token
import json
from pathlib import Path
import sys
from typing import Any

from anyio import CancelScope
import atif
import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli import main as main_module
from kolega_code.cli.peer_messaging import HeadlessPeerInbox
from kolega_code.cli.session_store import SessionStore, default_state_dir
from kolega_code.cli.session_usage import SessionUsageSink
from kolega_code.extensions import ExtensionSelection, KolegaExtensionBundle
from kolega_code.hooks import HookEvent, HookOutcome
from kolega_code.llm.exceptions import LLMBillingError
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import Message
from kolega_code.session.inbox import PeerSocketServer

pytestmark = pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")

_CLAIM = """
from pathlib import Path
import sys
from kolega_code.cli.session_store import SessionLockedError, SessionStore
store = SessionStore(Path(sys.argv[1]))
try:
    store.claim_for_resume(sys.argv[2], Path(sys.argv[3]))
except SessionLockedError:
    sys.exit(3)
else:
    store.release_session_locks()
"""


async def _claim_in_child(root: Path, session_id: str, project: Path) -> int:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CLAIM,
        str(root),
        session_id,
        str(project),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), 20)
        assert child.returncode in (0, 3), (stdout, stderr)
        return child.returncode
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "outcome", "cancel_usage"),
    [
        (phase, outcome, False)
        for outcome in ("completed", "failed", "cancelled")
        for phase in ("hook", "agent", "extension", "inbox")
    ]
    + [
        pytest.param("stream", "failed", True, id="cancel-drainer-before-producer-failure"),
        pytest.param("extension", "completed", True, id="cancel-drainer-before-late-cleanup-request"),
    ],
)
async def test_ask_teardown_retains_task_affinity_resources_terminal_export_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    outcome: str,
    cancel_usage: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-offline-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    store = SessionStore(default_state_dir())
    monkeypatch.setattr(main_module, "_store_from_args", lambda args: store)
    entered, release, streaming = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tasks: list[asyncio.Task[Any] | None] = []
    processes: list[asyncio.subprocess.Process] = []
    servers: list[asyncio.AbstractServer] = []
    marker: ContextVar[str] = ContextVar("ask-extension")
    token: Token[str] | None = None
    scope: CancelScope | None = None
    ledger: UsageLedger | None = None
    request_id: str | None = None
    sinks: list[SessionUsageSink] = []
    drainer_cancelled = asyncio.Event()
    session_id = "0123456789abcdef0123456789abcdef"
    output = tmp_path / "trajectory.json"
    unlock_snapshots: list[tuple[bool, list[str]]] = []
    original_unlock = store.release_session_locks

    def unlock() -> None:
        events = [json.loads(line) for line in store.events_path_for(session_id).read_text().splitlines()]
        unlock_snapshots.append((output.exists(), [event["type"] for event in events]))
        original_unlock()

    class ObservedUsageSink(SessionUsageSink):
        async def _drain(self) -> None:
            sinks.append(self)
            original_get = self._queue.get

            async def get() -> Any:
                try:
                    return await original_get()
                except asyncio.CancelledError:
                    drainer_cancelled.set()
                    raise

            monkeypatch.setattr(self._queue, "get", get)
            await super()._drain()

    if cancel_usage:
        monkeypatch.setattr(main_module, "SessionUsageSink", ObservedUsageSink)
        monkeypatch.setattr(store, "release_session_locks", unlock)

    async def checkpoint(name: str) -> None:
        if phase == name:
            entered.set()
            await release.wait()

    class ScriptedAgent(CoderAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
            nonlocal request_id
            assert marker.get() == "bound"
            recorder = self.session_recorder
            assert recorder is not None
            recorder.start_turn(Message(role="user", content=message))
            streaming.set()
            try:
                if cancel_usage and phase == "stream":
                    assert ledger is not None
                    request_id = ledger.begin("anthropic", "offline-model")
                    await checkpoint("stream")
                if outcome == "cancelled":
                    await asyncio.Event().wait()
                if outcome == "failed":
                    raise LLMBillingError("scripted rejection", provider="anthropic")
                assistant = Message(role="assistant", content="offline response", stop_reason="end_turn")
                recorder.record_assistant(assistant)
                self.history.extend([Message(role="user", content=message), assistant])
                yield {"type": "response", "content": "offline response", "complete": True}
            except LLMBillingError as exc:
                if request_id is not None:
                    assert ledger is not None
                    ledger.record_failure(request_id, str(exc))
                raise
            finally:
                recorder.finish_turn(outcome)

        async def fire_hook(self, name: HookEvent, payload: dict[str, Any], *, target: str = "") -> HookOutcome:
            if name == HookEvent.SESSION_END:
                await checkpoint("hook")
            return await super().fire_hook(name, payload, target=target)

        async def cleanup(self) -> None:
            await checkpoint("agent")
            await super().cleanup()

    async def bind(agent: CoderAgent) -> None:
        nonlocal ledger, scope, token
        ledger = agent.usage_ledger
        tasks.append(asyncio.current_task())
        scope = CancelScope()
        scope.__enter__()
        token = marker.set("bound")
        processes.append(await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(120)"))

    async def cleanup() -> None:
        nonlocal request_id
        await checkpoint("extension")
        tasks.append(asyncio.current_task())
        assert tasks[0] is tasks[-1]
        assert marker.get() == "bound"
        if cancel_usage and phase == "extension":
            assert ledger is not None
            request_id = ledger.begin("anthropic", "offline-model")
            ledger.record_failure(request_id, "scripted cleanup rejection")
        processes[0].terminate()
        await processes[0].wait()
        assert token is not None and scope is not None
        marker.reset(token)
        scope.__exit__(None, None, None)

    selection = ExtensionSelection(
        "offline:task_affine",
        lambda host, path: KolegaExtensionBundle(bind_agent=bind, cleanup=cleanup),
        None,
    )
    monkeypatch.setattr(main_module, "CoderAgent", ScriptedAgent)
    monkeypatch.setattr(main_module, "_resolve_extension_selection_from_args", lambda args: selection)
    original_start = main_module.start_headless_peer_inbox
    original_stop = HeadlessPeerInbox.stop

    async def start_inbox(**kwargs: Any) -> HeadlessPeerInbox | None:
        kwargs["enabled"] = True
        return await original_start(**kwargs)

    async def start_transport(server: PeerSocketServer) -> Path:
        # TCP keeps this real transport fixture independent of AF_UNIX path-length limits.
        transport = await asyncio.start_server(server._handle_connection, "127.0.0.1", 0)
        server._server = transport
        servers.append(transport)
        return server.path

    async def stop_inbox(inbox: HeadlessPeerInbox) -> None:
        await checkpoint("inbox")
        await original_stop(inbox)

    monkeypatch.setattr(main_module, "start_headless_peer_inbox", start_inbox)
    monkeypatch.setattr(PeerSocketServer, "start", start_transport)
    monkeypatch.setattr(HeadlessPeerInbox, "stop", stop_inbox)
    args = main_module.parse_args(
        ["ask", "offline turn", "--project", str(project), "--session", session_id, "--atif-output", str(output)]
    )
    command = asyncio.create_task(main_module._run_ask(args))
    try:
        if outcome == "cancelled":
            await asyncio.wait_for(streaming.wait(), 20)
            command.cancel()
        await asyncio.wait_for(entered.wait(), 20)
        assert await _claim_in_child(store.root, session_id, project) == 3
        assert len(servers) == 1 and servers[0].is_serving()
        if phase != "inbox":
            assert processes[0].returncode is None
        if phase != "stream":
            command.cancel()
            await asyncio.sleep(0)
            command.cancel()
            await asyncio.sleep(0)
        if cancel_usage:
            assert len(sinks) == 1
            sink = sinks[0]
            assert await sink.flush(timeout=20)
            assert sink._drainer is not None
            sink._drainer.cancel()
            await asyncio.wait_for(drainer_cancelled.wait(), 20)
            assert await sink.flush(timeout=20)
            assert processes[0].returncode is None
            assert servers[0].is_serving()
            assert len(tasks) == 1
        assert not command.done(), "caller cancellation abandoned an unfinished teardown phase"
        assert await _claim_in_child(store.root, session_id, project) == 3
        assert not output.exists()
        raw = store.events_path_for(session_id).read_text()
        assert not any(json.loads(line)["type"].startswith("run.") for line in raw.splitlines())
        release.set()
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(command, 20)
        else:
            assert await asyncio.wait_for(command, 20) == (1 if outcome == "failed" else 0)
        assert processes[0].returncode is not None
        assert not servers[0].is_serving()
        assert len(tasks) == 2 and tasks[0] is tasks[1]
        assert await _claim_in_child(store.root, session_id, project) == 0
        events = [json.loads(line) for line in store.events_path_for(session_id).read_text().splitlines()]
        terminals = [event for event in events if event["type"].startswith("run.")]
        assert [event["type"] for event in terminals] == [f"run.{outcome}"]
        kinds = [event["type"] for event in events]
        assert kinds.index("llm.run_finished") < kinds.index(f"run.{outcome}")
        assert kinds[-1] == f"run.{outcome}"
        if cancel_usage:
            accounting = [event for event in events if event["type"].startswith("llm.")]
            assert [event["type"] for event in accounting] == [
                "llm.run_started",
                "llm.request_started",
                "llm.request_settled",
                "llm.request_failed",
                "llm.run_finished",
            ]
            started, settled, failed = accounting[1:4]
            assert started["payload"]["request_id"] == settled["payload"]["request_id"] == request_id
            assert failed["payload"]["request_id"] == request_id
            assert settled["payload"]["state"] == "failed"
            counts = accounting[-1]["payload"]["counts"]
            assert counts["requests"] == counts["failed"] == 1
            assert counts["open"] == counts["responses"] == 0
            assert unlock_snapshots == [(True, kinds)]
        trajectory = atif.Trajectory.model_validate_json(output.read_text())
        assert trajectory.extra is not None
        assert trajectory.extra["kolega"]["status"]["outcome"] == outcome
    finally:
        release.set()
        if not command.done():
            command.cancel()
        await asyncio.wait_for(asyncio.gather(command, return_exceptions=True), 20)
        for child in processes:
            if child.returncode is None:
                child.kill()
                await child.wait()
        for server in servers:
            server.close()
            await server.wait_closed()
        original_unlock()
