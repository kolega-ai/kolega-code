"""Unit tests for LSP JSON-RPC client (message framing, request/response, notifications)."""

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
