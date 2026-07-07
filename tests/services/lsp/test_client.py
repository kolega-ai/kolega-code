"""Unit tests for LSP JSON-RPC client (message framing, request/response, notifications)."""

import asyncio

import pytest

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
