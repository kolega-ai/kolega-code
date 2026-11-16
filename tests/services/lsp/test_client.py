"""Unit tests for LSP JSON-RPC client (message framing, request/response, notifications)."""

import asyncio
import json
import signal
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from kolega_code.services.lsp import client as client_module
from kolega_code.services.lsp.client import LspClient, LspClientError, parse_publish_diagnostics


# ---------------------------------------------------------------------------
# message framing (Content-Length header)
# ---------------------------------------------------------------------------


def test_parse_content_length_from_buffer():
    """Content-Length is parsed from a well-formed header."""
    # We test indirectly via the private method since the class is designed for
    # subprocess I/O, but we can verify the parser directly.
    client = LspClient(["echo"])
    # The header-parsing code is inside _process_buffer, but the private
    # _parse_content_length is directly available to test.
    assert client._parse_content_length("Content-Length: 42\r\n") == 42  # noqa: SLF001
    assert client._parse_content_length("Content-Length: 0\r\n") == 0  # noqa: SLF001
    assert client._parse_content_length("Other: val\r\nContent-Length: 128\r\nX: y\r\n") == 128  # noqa: SLF001


def test_parse_content_length_missing():
    """Missing Content-Length returns None."""
    client = LspClient(["echo"])
    assert client._parse_content_length("Other: val\r\n") is None  # noqa: SLF001
    assert client._parse_content_length("") is None  # noqa: SLF001


def test_parse_content_length_malformed():
    """Malformed Content-Length returns None."""
    client = LspClient(["echo"])
    assert client._parse_content_length("Content-Length: abc\r\n") is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# diagnostic parsing
# ---------------------------------------------------------------------------


def test_parse_publish_diagnostics():
    """PublishDiagnosticsParams is parsed from notification payload."""
    params = {
        "uri": "file:///test.py",
        "diagnostics": [
            {
                "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
                "severity": 1,
                "code": "undefined-variable",
                "message": "'foo' is not defined",
                "source": "pyright",
            },
            {
                "range": {"start": {"line": 4, "character": 0}, "end": {"line": 4, "character": 10}},
                "severity": 2,
                "message": "Unused import",
            },
        ],
    }
    result = parse_publish_diagnostics(params)
    assert result.uri == "file:///test.py"
    assert len(result.diagnostics) == 2
    assert result.diagnostics[0].message == "'foo' is not defined"
    assert result.diagnostics[0].severity == 1
    assert result.diagnostics[0].source == "pyright"
    assert result.diagnostics[0].code == "undefined-variable"
    assert result.diagnostics[1].severity == 2


def test_parse_publish_diagnostics_empty():
    """Empty diagnostics list is handled."""
    params = {"uri": "file:///empty.py", "diagnostics": []}
    result = parse_publish_diagnostics(params)
    assert result.uri == "file:///empty.py"
    assert result.diagnostics == []


# ---------------------------------------------------------------------------
# client lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_start_stop():
    """Client can be started and stopped cleanly."""
    # Use `sleep` as a dummy "server" that stays alive but does nothing.
    client = LspClient(["sleep", "10"])
    await client.start()
    assert client.running
    await client.stop()
    assert not client.running


@pytest.mark.asyncio
async def test_request_while_not_running_raises():
    """Request before start raises LspClientError."""
    client = LspClient(["echo"])
    with pytest.raises(LspClientError):
        await client.request("test/method")


@pytest.mark.asyncio
async def test_notification_handlers():
    """Registered notification handlers are called."""
    client = LspClient(["sleep", "10"])
    events = []

    def handler(params):
        events.append(params)

    client.on_notification("textDocument/publishDiagnostics", handler)
    await client.start()

    # Simulate a notification dispatch (direct call to _dispatch)
    await client._dispatch(
        {  # noqa: SLF001
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": "file:///test.py", "diagnostics": []},
        }
    )
    assert len(events) == 1
    assert events[0]["uri"] == "file:///test.py"

    await client.stop()


# ---------------------------------------------------------------------------
# server→client request handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_request_handler_called_and_response_sent():
    """A server→client request (id + method) triggers the handler and sends a response."""
    client = LspClient(["sleep", "10"])
    await client.start()

    sent_messages: list[dict] = []

    async def fake_send(payload):
        sent_messages.append(payload)

    client._send = fake_send  # type: ignore[assignment]  # noqa: SLF001

    handler_calls: list[dict] = []

    def config_handler(params):
        handler_calls.append(params)
        return [{"someSetting": True}]

    client.on_request("workspace/configuration", config_handler)

    # Simulate a server→client request
    await client._dispatch(
        {  # noqa: SLF001
            "jsonrpc": "2.0",
            "id": 42,
            "method": "workspace/configuration",
            "params": {"items": [{"section": "python"}]},
        }
    )

    assert len(handler_calls) == 1
    assert handler_calls[0]["items"][0]["section"] == "python"
    # A response was sent
    assert len(sent_messages) == 1
    assert sent_messages[0]["id"] == 42
    assert sent_messages[0]["result"] == [{"someSetting": True}]

    await client.stop()


@pytest.mark.asyncio
async def test_unhandled_server_request_sends_error_response():
    """An unhandled server→client request gets a -32601 error response (no hang)."""
    client = LspClient(["sleep", "10"])
    await client.start()

    sent_messages: list[dict] = []

    async def fake_send(payload):
        sent_messages.append(payload)

    client._send = fake_send  # type: ignore[assignment]  # noqa: SLF001

    await client._dispatch(
        {  # noqa: SLF001
            "jsonrpc": "2.0",
            "id": 99,
            "method": "some/unknown/method",
            "params": {},
        }
    )

    assert len(sent_messages) == 1
    assert sent_messages[0]["id"] == 99
    assert sent_messages[0]["error"]["code"] == -32601

    await client.stop()


@pytest.mark.asyncio
async def test_server_request_handler_exception_sends_error():
    """If a request handler raises, an error response is sent."""
    client = LspClient(["sleep", "10"])
    await client.start()

    sent_messages: list[dict] = []

    async def fake_send(payload):
        sent_messages.append(payload)

    client._send = fake_send  # type: ignore[assignment]  # noqa: SLF001

    def bad_handler(params):
        raise RuntimeError("boom")

    client.on_request("window/workDoneProgress/create", bad_handler)

    await client._dispatch(
        {  # noqa: SLF001
            "jsonrpc": "2.0",
            "id": 7,
            "method": "window/workDoneProgress/create",
            "params": {"token": "abc"},
        }
    )

    assert len(sent_messages) == 1
    assert sent_messages[0]["error"]["code"] == -32603

    await client.stop()


@pytest.mark.asyncio
async def test_late_response_to_timed_out_request_no_crash():
    """A response arriving after timeout is silently dropped (no crash)."""
    client = LspClient(["sleep", "10"])
    await client.start()

    # Dispatch a response for a request id that was never registered (simulating
    # a response arriving after the future was popped due to timeout)
    await client._dispatch(
        {  # noqa: SLF001
            "jsonrpc": "2.0",
            "id": 9999,
            "result": {"late": True},
        }
    )
    # No crash — the method simply returns

    await client.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [[], None, False, 0, "", {"ok": True}])
async def test_response_preserves_exact_jsonrpc_result(value):
    """JSON-RPC responses preserve valid falsey result values."""
    client = LspClient(["sleep", "10"])
    fut = client._pending[1] = asyncio.get_event_loop().create_future()  # noqa: SLF001

    await client._dispatch({"jsonrpc": "2.0", "id": 1, "result": value})  # noqa: SLF001

    assert fut.done()
    assert fut.result() == value


def test_subprocess_env_allowlist_withholds_secrets_and_strips_loader_vars(monkeypatch):
    """F2: only an allowlisted base env is forwarded; secrets and loader vars withheld.

    - Allowlisted vars (PATH, HOME) are inherited and can be overridden by the
      server's declared env.
    - Non-allowlisted inherited vars (e.g. provider API keys) are NOT forwarded.
    - Server-declared env vars are applied (so servers can opt into extra vars).
    - Loader-injection vars (PYTHONPATH, LD_PRELOAD, NODE_OPTIONS) are stripped
      even when declared by the server's env.
    """
    monkeypatch.setenv("PATH", "/inherited")
    monkeypatch.setenv("HOME", "/users/test")
    monkeypatch.setenv("PYTHONPATH", "/bad")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("KEEP_ME", "yes")

    client = LspClient(
        ["echo"],
        env={
            "PATH": "/custom",
            "SERVER_ONLY": "1",
            "PYTHONPATH": "/inject",
            "LD_PRELOAD": "/evil.so",
            "NODE_OPTIONS": "--require /evil",
        },
    )
    env = client._subprocess_env()  # noqa: SLF001

    # Allowlisted base vars are inherited...
    assert env["HOME"] == "/users/test"
    # ...and server env overrides them.
    assert env["PATH"] == "/custom"

    # Server-declared (non-dangerous) vars are applied.
    assert env["SERVER_ONLY"] == "1"

    # Secrets / non-allowlisted inherited vars are withheld.
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "KEEP_ME" not in env

    # Loader-injection vars are stripped from both the inherited and server env.
    assert "PYTHONPATH" not in env
    assert "LD_PRELOAD" not in env
    assert "NODE_OPTIONS" not in env


# ---------------------------------------------------------------------------
# state tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_tracks_pid_and_status():
    """Client stores server_pid and status after start."""
    client = LspClient(["sleep", "10"])
    assert client.status == "stopped"
    assert client.server_pid is None

    await client.start()
    assert client.status == "starting"
    assert client.server_pid is not None
    assert client.server_pid > 0

    await client.stop()
    assert client.status == "stopped"


@pytest.mark.asyncio
async def test_client_request_accepts_custom_timeout():
    """The request method accepts a custom timeout."""
    client = LspClient(["sleep", "10"])
    await client.start()

    # A request to a dummy server will time out quickly with a small timeout
    with pytest.raises(LspClientError, match="timed out after 0.1s"):
        await client.request("test/method", timeout=0.1)

    await client.stop()


class _DrainGate:
    def __init__(self) -> None:
        self.entered: asyncio.Event = asyncio.Event()
        self.release: asyncio.Event = asyncio.Event()
        self.cancelled: asyncio.Event = asyncio.Event()

    async def __call__(self) -> None:
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _ObservedLock(asyncio.Lock):
    def __init__(self) -> None:
        super().__init__()
        self.entered: asyncio.Event = asyncio.Event()

    async def acquire(self) -> bool:
        self.entered.set()
        return await super().acquire()


@pytest_asyncio.fixture
async def live_client() -> AsyncIterator[LspClient]:
    existing_tasks = asyncio.all_tasks() - {asyncio.current_task()}
    client = LspClient([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"])
    await asyncio.wait_for(client.start(), timeout=10)
    proc, reader = client._proc, client._reader_task
    assert proc is not None and reader is not None
    try:
        yield client
    finally:
        await asyncio.wait_for(client.stop(), timeout=10)
        _assert_stopped(client, proc, reader)
        assert not (asyncio.all_tasks() - existing_tasks - {asyncio.current_task()})


def _assert_stopped(client: LspClient, proc: asyncio.subprocess.Process, reader: asyncio.Task[None]) -> None:
    assert proc.returncode is not None
    assert reader.done()
    assert not client.running
    assert client.status == "stopped"
    assert client._proc is None
    assert client._reader_task is None
    assert client._pending == {}
    assert client._start_task is not None and client._start_task.done()
    assert client._stop_task is not None and client._stop_task.done()


def _record_sends(client: LspClient, monkeypatch: pytest.MonkeyPatch) -> asyncio.Queue[tuple[int, asyncio.Future[Any]]]:
    sent: asyncio.Queue[tuple[int, asyncio.Future[Any]]] = asyncio.Queue()
    original = client._send

    async def send(payload: dict[str, Any]) -> None:
        await original(payload)
        if "id" in payload and "method" in payload:
            rid = payload["id"]
            sent.put_nowait((rid, client._pending[rid]))

    monkeypatch.setattr(client, "_send", send)
    return sent


async def _interrupted(task: asyncio.Task[Any], cancel: bool) -> None:
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
    else:
        with pytest.raises(LspClientError, match="timed out"):
            await asyncio.wait_for(task, timeout=10)


async def _checkpoint() -> None:
    resumed = asyncio.Event()
    asyncio.get_running_loop().call_soon(resumed.set)
    await asyncio.wait_for(resumed.wait(), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_request_lock_interruption_preserves_live_request(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    client = live_client
    sent = _record_sends(client, monkeypatch)
    live = asyncio.create_task(client.request("test/live", timeout=10))
    live_id, live_future = await asyncio.wait_for(sent.get(), timeout=10)
    lock = _ObservedLock()
    await lock.acquire()
    lock.entered.clear()
    monkeypatch.setattr(client, "_lock", lock)
    blocked = asyncio.create_task(client.request("test/locked", timeout=10 if cancel else 0.1))
    try:
        await asyncio.wait_for(lock.entered.wait(), timeout=10)
        await _interrupted(blocked, cancel)
        assert client._pending == {live_id: live_future}
        assert not live_future.done()
        assert client.running
        assert client._stop_task is None
    finally:
        lock.release()
        await client._dispatch({"id": live_id, "result": "healthy"})
        assert await asyncio.wait_for(live, timeout=10) == "healthy"
    assert client._pending == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_request_response_interruption_cleans_only_its_future(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    client = live_client
    sent = _record_sends(client, monkeypatch)
    live = asyncio.create_task(client.request("test/live", timeout=10))
    live_id, live_future = await asyncio.wait_for(sent.get(), timeout=10)
    interrupted = asyncio.create_task(client.request("test/no-reply", timeout=10 if cancel else 0.1))
    rid, future = await asyncio.wait_for(sent.get(), timeout=10)
    try:
        await _interrupted(interrupted, cancel)
        assert future.cancelled()
        assert client._pending == {live_id: live_future}
        await client._dispatch({"id": rid, "result": "too late"})
        await client._dispatch({"id": rid, "error": {"code": -1, "message": "also late"}})
        assert client._pending == {live_id: live_future}
        assert client.running
        assert client._stop_task is None
    finally:
        await client._dispatch({"id": live_id, "result": "healthy"})
        assert await asyncio.wait_for(live, timeout=10) == "healthy"
    assert client._pending == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_request_drain_interruption_reaps_transport_and_fails_other_request(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and reader is not None
    sent = _record_sends(client, monkeypatch)
    live = asyncio.create_task(client.request("test/live", timeout=10))
    _, live_future = await asyncio.wait_for(sent.get(), timeout=10)
    gate = _DrainGate()
    monkeypatch.setattr(proc.stdin, "drain", gate)
    blocked = asyncio.create_task(client.request("test/blocked", timeout=10 if cancel else 0.1))
    await asyncio.wait_for(gate.entered.wait(), timeout=10)
    blocked_id = client._request_id
    blocked_future = client._pending[blocked_id]
    await _interrupted(blocked, cancel)
    with pytest.raises(LspClientError):
        await asyncio.wait_for(live, timeout=10)
    assert gate.cancelled.is_set()
    assert blocked_future.done()
    assert live_future.done()
    await client._dispatch({"id": blocked_id, "result": "late"})
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_timeout", [False, True], ids=["shared-send-timeout", "notification-timeout"])
async def test_notification_drain_deadline_reaps_transport(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, explicit_timeout: bool
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and reader is not None
    gate = _DrainGate()
    monkeypatch.setattr(proc.stdin, "drain", gate)
    if explicit_timeout:
        operation = client.notify("test/blocked", timeout=0.1)
    else:
        monkeypatch.setattr(client_module, "_SEND_TIMEOUT_SECONDS", 0.1)
        operation = client.notify("test/blocked")
    task = asyncio.create_task(operation)
    await asyncio.wait_for(gate.entered.wait(), timeout=10)
    with pytest.raises(LspClientError):
        await asyncio.wait_for(task, timeout=10)
    assert gate.cancelled.is_set()
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
async def test_notification_cancellation_during_drain_propagates_after_cleanup(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and reader is not None
    gate = _DrainGate()
    monkeypatch.setattr(proc.stdin, "drain", gate)
    task = asyncio.create_task(client.notify("test/blocked"))
    await asyncio.wait_for(gate.entered.wait(), timeout=10)
    await _interrupted(task, cancel=True)
    assert gate.cancelled.is_set()
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True], ids=["method-not-found", "handler-result"])
async def test_server_response_drain_deadline_does_not_send_second_response(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, handled: bool
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and reader is not None
    gate = _DrainGate()
    monkeypatch.setattr(proc.stdin, "drain", gate)
    monkeypatch.setattr(client_module, "_SEND_TIMEOUT_SECONDS", 0.1)
    writes: list[bytes] = []
    monkeypatch.setattr(proc.stdin, "write", writes.append)
    if handled:
        client.on_request("test/config", lambda params: {"configured": True})
    task = asyncio.create_task(client._dispatch({"id": 77, "method": "test/config", "params": {}}))
    await asyncio.wait_for(gate.entered.wait(), timeout=10)
    with pytest.raises(LspClientError):
        await asyncio.wait_for(task, timeout=10)
    assert len(writes) == 1
    assert b'"id": 77' in writes[0]
    assert gate.cancelled.is_set()
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
async def test_server_response_transport_failure_is_not_a_handler_error(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    responses: list[dict[str, Any]] = []

    async def failed_send(payload: dict[str, Any]) -> None:
        responses.append(payload)
        raise LspClientError("broken transport")

    def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"configured": True}

    client.on_request("test/config", handler)
    monkeypatch.setattr(client, "_send", failed_send)
    with pytest.raises(LspClientError, match="broken transport"):
        await client._dispatch({"id": 77, "method": "test/config"})
    assert responses == [{"jsonrpc": "2.0", "id": 77, "result": {"configured": True}}]


@pytest.mark.asyncio
async def test_reader_originated_response_send_timeout_settles_without_self_wait(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and proc.stdout is not None and reader is not None
    sent = _record_sends(client, monkeypatch)
    request = asyncio.create_task(client.request("test/pending", timeout=10))
    _, future = await asyncio.wait_for(sent.get(), timeout=10)
    gate = _DrainGate()
    monkeypatch.setattr(proc.stdin, "drain", gate)
    monkeypatch.setattr(client_module, "_SEND_TIMEOUT_SECONDS", 0.1)
    writes: list[bytes] = []
    monkeypatch.setattr(proc.stdin, "write", writes.append)
    body = json.dumps({"jsonrpc": "2.0", "id": 77, "method": "test/unknown"}).encode()
    proc.stdout.feed_data(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    await asyncio.wait_for(gate.entered.wait(), timeout=10)
    with pytest.raises(LspClientError):
        await asyncio.wait_for(request, timeout=10)
    assert client._stop_task is not None
    await asyncio.wait_for(asyncio.shield(client._stop_task), timeout=10)
    assert future.done()
    assert len(writes) == 1
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["write", "drain"])
@pytest.mark.parametrize("error_type", [BrokenPipeError, ConnectionResetError])
async def test_send_connection_failure_is_normalized_and_reaped(
    live_client: LspClient,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
    error_type: type[OSError],
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and reader is not None
    futures: list[asyncio.Future[Any]] = []

    def fail_write(data: bytes) -> None:
        futures.extend(client._pending.values())
        raise error_type("pipe closed")

    async def fail_drain() -> None:
        fail_write(b"")

    monkeypatch.setattr(proc.stdin, failure_at, fail_write if failure_at == "write" else fail_drain)
    with pytest.raises(LspClientError):
        await asyncio.wait_for(client.request("test/broken", timeout=10), timeout=10)
    assert len(futures) == 1 and futures[0].done()
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
async def test_unexpected_stdout_eof_fails_pending_request_and_reaps_child(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdout is not None and reader is not None
    sent = _record_sends(client, monkeypatch)
    request = asyncio.create_task(client.request("test/pending", timeout=10))
    _, future = await asyncio.wait_for(sent.get(), timeout=10)
    proc.stdout.feed_eof()
    with pytest.raises(LspClientError):
        await asyncio.wait_for(request, timeout=10)
    assert client._stop_task is not None
    await asyncio.wait_for(asyncio.shield(client._stop_task), timeout=10)
    assert future.done()
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
async def test_stop_is_shared_repeated_and_cancellation_resistant(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and reader is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original_wait = proc.wait
    wait_calls = 0

    async def wait() -> int:
        nonlocal wait_calls
        wait_calls += 1
        entered.set()
        await release.wait()
        return await original_wait()

    monkeypatch.setattr(proc, "wait", wait)
    first = asyncio.create_task(client.stop())
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        retained = client._stop_task
        second = asyncio.create_task(client.stop())
        first.cancel()
        await _checkpoint()
        assert not first.done()
        assert not second.done()
        assert client._stop_task is retained
        assert client._proc is proc
        first.cancel()
        await _checkpoint()
        assert not first.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=10)
        if second is not None:
            await asyncio.wait_for(second, timeout=10)
    await asyncio.wait_for(client.stop(), timeout=10)
    assert wait_calls == 1
    assert client._stop_task is retained
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_stale_start_waiter_retains_newer_cleanup(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, eager: bool
) -> None:
    eager_factory = getattr(asyncio, "eager_task_factory", None)
    if eager and eager_factory is None:
        pytest.skip("Eager tasks require Python 3.12")
    client = live_client
    await client.stop()
    initial_stop = client._stop_task
    original_settle = client_module._settle
    original_stop = client._stop_process
    original_create = asyncio.create_subprocess_exec
    stale_entered, release_stale = asyncio.Event(), asyncio.Event()
    stop_entered, release_stop = asyncio.Event(), asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []

    async def settle(task: asyncio.Task[None], *, cancelled: bool = False) -> None:
        await original_settle(task, cancelled=cancelled)
        if task is initial_stop and not stale_entered.is_set():
            stale_entered.set()
            await release_stale.wait()

    async def stop_process() -> None:
        stop_entered.set()
        await release_stop.wait()
        await original_stop()

    async def create(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await original_create(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(client_module, "_settle", settle)
    monkeypatch.setattr(client, "_stop_process", stop_process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    if eager:
        loop.set_task_factory(eager_factory)
    stale = asyncio.create_task(client.start())
    stopping: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(stale_entered.wait(), timeout=10)
        await asyncio.wait_for(client.start(), timeout=10)
        first_reader = client._reader_task
        assert first_reader is not None
        stopping = asyncio.create_task(client.stop())
        await asyncio.wait_for(stop_entered.wait(), timeout=10)
        retained = client._stop_task
        assert retained is not None and retained is not initial_stop
        release_stale.set()
        await _checkpoint()
        assert client._stop_task is retained
        assert not stale.done()
        assert len(processes) == 1

        release_stop.set()
        await asyncio.wait_for(asyncio.gather(stopping, stale), timeout=10)
        assert len(processes) == 2
        assert processes[0].returncode is not None and first_reader.done()
        reader = client._reader_task
        assert reader is not None
        await asyncio.wait_for(client.stop(), timeout=10)
        assert all(proc.returncode is not None for proc in processes)
        _assert_stopped(client, processes[-1], reader)
    finally:
        release_stale.set()
        release_stop.set()
        try:
            tasks = [stale] if stopping is None else [stale, stopping]
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)
            await asyncio.wait_for(client.stop(), timeout=10)
        finally:
            for proc in processes:
                if proc.returncode is None:
                    proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=10)
            loop.set_task_factory(original_factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["request", "notify", "start", "stop"])
async def test_cancellation_propagates_even_when_cleanup_fails(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    client = live_client
    entered, release = asyncio.Event(), asyncio.Event()
    release_start = asyncio.Event()
    original_stop = client._stop_process

    async def fail_cleanup() -> None:
        entered.set()
        await release.wait()
        raise LspClientError("cleanup failed")

    monkeypatch.setattr(client, "_stop_process", fail_cleanup)
    if operation == "start":
        monkeypatch.setattr(client, "_stop_process", original_stop)
        await client.stop()
        monkeypatch.setattr(client, "_stop_process", fail_cleanup)
        original_start = client._start_process
        starting = asyncio.Event()

        async def start_process() -> None:
            await original_start()
            starting.set()
            await release_start.wait()

        monkeypatch.setattr(client, "_start_process", start_process)
        task = asyncio.create_task(client.start())
        await asyncio.wait_for(starting.wait(), timeout=10)
    elif operation == "stop":
        task = asyncio.create_task(client.stop())
        await asyncio.wait_for(entered.wait(), timeout=10)
    else:
        assert client._proc is not None and client._proc.stdin is not None
        gate = _DrainGate()
        monkeypatch.setattr(client._proc.stdin, "drain", gate)
        task = asyncio.create_task(
            client.request("test/cancel") if operation == "request" else client.notify("test/cancel")
        )
        await asyncio.wait_for(gate.entered.wait(), timeout=10)

    try:
        task.cancel()
        release_start.set()
        await asyncio.wait_for(entered.wait(), timeout=10)
        await _checkpoint()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert client._proc is not None
        assert client._stop_task is not None and client._stop_task.done()
    finally:
        release.set()
        release_start.set()
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(client, "_stop_process", original_stop)
        await asyncio.wait_for(client.stop(), timeout=10)


@pytest.mark.asyncio
async def test_forced_reap_deadline_reports_failure_and_retains_cleanup_ownership(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    proc, reader = client._proc, client._reader_task
    assert proc is not None and reader is not None
    original_wait = proc.wait
    release = asyncio.Event()
    cancelled_waits: list[int] = []
    waits = 0

    async def wait() -> int:
        nonlocal waits
        waits += 1
        attempt = waits
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_waits.append(attempt)
            raise
        return await original_wait()

    def ignore_term() -> None:
        pass

    monkeypatch.setattr(proc, "wait", wait)
    monkeypatch.setattr(proc, "terminate", ignore_term)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(client_module, "_KILL_TIMEOUT_SECONDS", 0.1)
    stop = asyncio.create_task(client.stop())
    try:
        done, _ = await asyncio.wait({stop}, timeout=10)
        assert stop in done
        with pytest.raises(LspClientError, match="cleanup failed"):
            stop.result()
        assert cancelled_waits == [1, 2]
        assert client._proc is proc
        assert client._reader_task is reader and reader.done()
        assert client._stop_task is not None and client._stop_task.done()
        assert client._pending == {}
        assert client.status == "error"
        assert client.last_error is not None and "cleanup failed" in client.last_error
        assert await asyncio.wait_for(original_wait(), timeout=10) == -signal.SIGKILL
    finally:
        release.set()
        await asyncio.gather(stop, return_exceptions=True)
        monkeypatch.setattr(proc, "wait", original_wait)
        await asyncio.wait_for(client.stop(), timeout=10)
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
async def test_stop_during_real_blocked_write_settles_sender(
    live_client: LspClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = live_client
    await client.stop()
    client._command = [sys.executable, "-c", "import signal; signal.pause()"]
    await asyncio.wait_for(client.start(), timeout=10)
    proc, reader = client._proc, client._reader_task
    assert proc is not None and proc.stdin is not None and reader is not None
    entered = asyncio.Event()
    original_drain = proc.stdin.drain

    async def drain() -> None:
        entered.set()
        await original_drain()

    monkeypatch.setattr(proc.stdin, "drain", drain)
    sender = asyncio.create_task(client.request("test/large", {"text": "x" * (4 * 1024 * 1024)}, timeout=10))
    await asyncio.wait_for(entered.wait(), timeout=10)
    assert not sender.done()
    assert client._pending
    await asyncio.wait_for(client.stop(), timeout=10)
    with pytest.raises(LspClientError):
        await asyncio.wait_for(sender, timeout=10)
    _assert_stopped(client, proc, reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_start", [False, True], ids=["stop-during-create", "cancel-during-create"])
async def test_subprocess_creation_keeps_ownership_until_cleanup(
    monkeypatch: pytest.MonkeyPatch, cancel_start: bool
) -> None:
    client = LspClient([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"])
    entered, release = asyncio.Event(), asyncio.Event()
    original_create = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    readers: list[asyncio.Task[None]] = []
    original_start = client._start_process

    async def create(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await original_create(*args, **kwargs)
        processes.append(proc)
        entered.set()
        await release.wait()
        return proc

    async def start_process() -> None:
        await original_start()
        assert client._reader_task is not None
        readers.append(client._reader_task)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(client, "_start_process", start_process)
    start = asyncio.create_task(client.start())
    stop: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        if cancel_start:
            start.cancel()
        else:
            stop = asyncio.create_task(client.stop())
        await _checkpoint()
        assert client._start_task is not None and not client._start_task.done()
        assert not start.done()
        assert processes[0].returncode is None
    finally:
        release.set()
        if cancel_start:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(start, timeout=10)
        else:
            await asyncio.wait_for(start, timeout=10)
            assert stop is not None
            await asyncio.wait_for(stop, timeout=10)
        await asyncio.wait_for(client.stop(), timeout=10)
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert len(readers) == 1
    assert all(task.done() for task in readers)
    assert client._pending == {}
    assert client._proc is None
    assert client._reader_task is None
    assert client._start_task is not None and client._start_task.done()
    assert client._stop_task is not None and client._stop_task.done()
    assert not client.running


_CHILD_READY = (
    "import json, os, signal, sys\n"
    "def ready(padding=0):\n"
    "    body = json.dumps({'method': 'test/ready'})\n"
    "    sys.stdout.write(f'Content-Length: {len(body)}\\r\\n\\r\\n{body}' + 'x' * padding)\n"
    "    sys.stdout.flush()\n"
)


@pytest.mark.asyncio
async def test_stop_escalates_term_resistant_child(monkeypatch: pytest.MonkeyPatch) -> None:
    code = _CHILD_READY + "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nready()\nsignal.pause()\n"
    client = LspClient([sys.executable, "-c", code])
    ready = asyncio.Event()
    client.on_notification("test/ready", lambda params: ready.set())
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.1)
    await asyncio.wait_for(client.start(), timeout=10)
    proc, reader = client._proc, client._reader_task
    assert proc is not None and reader is not None
    try:
        await asyncio.wait_for(ready.wait(), timeout=10)
        await asyncio.wait_for(client.stop(), timeout=10)
        assert proc.returncode == -signal.SIGKILL
        _assert_stopped(client, proc, reader)
    finally:
        await asyncio.wait_for(client.stop(), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("ready_padding", [0, 4 * 1024 * 1024], ids=["idle", "blocked-stdout-write"])
async def test_stop_drains_stdout_while_child_handles_term(ready_padding: int) -> None:
    code = (
        _CHILD_READY + "def terminate(signum, frame):\n"
        "    block = memoryview(b'x' * (4 * 1024 * 1024))\n"
        "    while block:\n"
        "        block = block[os.write(sys.stdout.fileno(), block):]\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, terminate)\n"
        f"ready({ready_padding})\n"
        "signal.pause()\n"
    )
    client = LspClient([sys.executable, "-c", code])
    ready = asyncio.Event()
    client.on_notification("test/ready", lambda params: ready.set())
    await asyncio.wait_for(client.start(), timeout=10)
    proc, reader = client._proc, client._reader_task
    assert proc is not None and reader is not None
    try:
        await asyncio.wait_for(ready.wait(), timeout=10)
        await asyncio.wait_for(client.stop(), timeout=10)
        assert proc.returncode == 0
        _assert_stopped(client, proc, reader)
    finally:
        await asyncio.wait_for(client.stop(), timeout=10)
