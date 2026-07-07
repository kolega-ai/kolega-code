"""Fake LSP server for integration testing.

Implements a minimal stdio JSON-RPC server that handles the LSP methods we test:
initialize, initialized, didOpen, didChange, didSave, textDocument/diagnostic
(pull), textDocument/definition, /references, /hover, /documentSymbol,
workspace/symbol, and client/registerCapability.

Also sends a ``workspace/configuration`` request during init to verify the
client's server→client request handling.

Configurable via environment variables:
- ``FAKE_LSP_DELAY``: seconds to delay before publishing diagnostics (default 0).
- ``FAKE_LSP_PULL_DIAGS``: if ``"0"``, disable pull diagnostics (force push-only).
- ``FAKE_LSP_STRICT_OPEN``: if ``"1"``, ignore didChange for unopened docs.
- ``FAKE_LSP_SOURCE``: diagnostic/source label (default ``"fake-lsp"``).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
import time


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_exact(n: int) -> bytes:
    """Read exactly *n* bytes from stdin."""
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            raise EOFError("stdin closed")
        data += chunk
    return data


def read_message() -> dict | None:
    """Read a single Content-Length-prefixed JSON-RPC message from stdin."""
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        stripped = line.strip()
        if not stripped:
            break
        key, _, value = stripped.decode("utf-8", errors="replace").partition(":")
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length == 0:
        return None

    body = _read_exact(content_length)
    return json.loads(body.decode("utf-8"))


def write_message(msg: dict) -> None:
    """Write a Content-Length-prefixed JSON-RPC message to stdout."""
    body = json.dumps(msg)
    data = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body.encode("utf-8")
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

# Documents: uri → {"text": str, "version": int}
_open_docs: dict[str, dict] = {}

# Config
_DELAY = float(os.environ.get("FAKE_LSP_DELAY", "0"))
_PULL_DIAGS = os.environ.get("FAKE_LSP_PULL_DIAGS", "1") != "0"
_STRICT_OPEN = os.environ.get("FAKE_LSP_STRICT_OPEN", "0") == "1"
_SOURCE = os.environ.get("FAKE_LSP_SOURCE", "fake-lsp")
# When set, always publish diagnostics with this fixed document version
# (simulates a server that does not track per-edit versions — used to test
# the stale-diagnostics suppression branch).
_FIXED_VERSION = os.environ.get("FAKE_LSP_FIXED_VERSION")
_CONFIGURATION_RESPONSES: list[Any] = []

# Capabilities we advertise
_CAPABILITIES = {
    "textDocumentSync": 1,  # Full sync
    "diagnosticProvider": {"interFileDependencies": False, "workspaceDiagnostics": False},
    "definitionProvider": True,
    "typeDefinitionProvider": True,
    "implementationProvider": True,
    "referencesProvider": True,
    "hoverProvider": True,
    "documentSymbolProvider": True,
    "workspaceSymbolProvider": True,
    "codeActionProvider": True,
    "callHierarchyProvider": True,
    "publishDiagnosticsProvider": True,
}


def _make_diagnostic(uri: str, line: int, message: str, severity: int = 1, source: str | None = None) -> dict:
    return {
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": 80},
        },
        "severity": severity,
        "message": message,
        "source": source or _SOURCE,
    }


def _diagnostics_for(uri: str) -> list[dict]:
    """Return canned diagnostics based on document content."""
    doc = _open_docs.get(uri)
    if doc is None:
        return []
    text = doc.get("text", "")
    diags: list[dict] = []
    for i, line in enumerate(text.split("\n")):
        if "undefined_var" in line:
            diags.append(_make_diagnostic(uri, i, "'undefined_var' is not defined", severity=1))
        if "unused" in line.lower():
            diags.append(_make_diagnostic(uri, i, "Unused import", severity=2))
    return diags


def _publish_diagnostics(uri: str, version: int | None = None) -> None:
    """Publish diagnostics for a URI."""
    if _DELAY > 0:
        time.sleep(_DELAY)
    diags = _diagnostics_for(uri)
    if _FIXED_VERSION is not None:
        version = int(_FIXED_VERSION)
    params: dict = {"uri": uri, "diagnostics": diags}
    if version is not None:
        params["version"] = version
    write_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": params,
        }
    )


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


def handle_initialize(msg: dict) -> dict:
    return {
        "capabilities": _CAPABILITIES,
        "serverInfo": {"name": "fake-lsp", "version": "1.0.0"},
    }


def handle_initialized(msg: dict) -> None:
    # Send a workspace/configuration request to test server→client request handling
    write_message(
        {
            "jsonrpc": "2.0",
            "id": 100,  # server-generated request id
            "method": "workspace/configuration",
            "params": {"items": [{"section": "fake-lsp"}]},
        }
    )


def handle_did_open(msg: dict) -> None:
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    text = td.get("text", "")
    version = td.get("version", 1)
    _open_docs[uri] = {"text": text, "version": version}
    _publish_diagnostics(uri, version)


def handle_did_change(msg: dict) -> None:
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    version = td.get("version", 1)
    changes = params.get("contentChanges", [])
    if _STRICT_OPEN and uri not in _open_docs:
        return
    if changes:
        _open_docs[uri] = {"text": changes[0].get("text", ""), "version": version}
    _publish_diagnostics(uri, version)


def handle_did_save(msg: dict) -> None:
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    doc = _open_docs.get(uri, {})
    _publish_diagnostics(uri, doc.get("version"))


def handle_diagnostic(msg: dict) -> dict:
    """Pull diagnostics."""
    if not _PULL_DIAGS:
        # Simulate method not found
        return {"error": {"code": -32601, "message": "pull diagnostics not supported"}}
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    diags = _diagnostics_for(uri)
    return {"kind": "full", "items": diags}


def handle_definition(msg: dict) -> Any:
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    return [
        {
            "uri": uri,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 10},
            },
        }
    ]


def handle_references(msg: dict) -> Any:
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    return [
        {"uri": uri, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}}},
        {"uri": uri, "range": {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 9}}},
    ]


def handle_hover(msg: dict) -> dict:
    return {
        "contents": {
            "kind": "markdown",
            "value": "```python\n(fake) def example() -> None\n```",
        }
    }


def handle_document_symbol(msg: dict) -> Any:
    params = msg.get("params", {})
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    doc = _open_docs.get(uri, {})
    text = doc.get("text", "")
    symbols = []
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if stripped.startswith("def "):
            name = stripped[4:].split("(")[0]
            symbols.append(
                {
                    "name": name,
                    "kind": 12,  # Function
                    "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": len(line)}},
                    "selectionRange": {
                        "start": {"line": i, "character": 4},
                        "end": {"line": i, "character": 4 + len(name)},
                    },
                }
            )
        elif stripped.startswith("class "):
            name = stripped[6:].split("(")[0].split(":")[0]
            symbols.append(
                {
                    "name": name,
                    "kind": 5,  # Class
                    "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": len(line)}},
                    "selectionRange": {
                        "start": {"line": i, "character": 6},
                        "end": {"line": i, "character": 6 + len(name)},
                    },
                }
            )
    return symbols


def handle_workspace_symbol(msg: dict) -> Any:
    params = msg.get("params", {})
    query = params.get("query", "")
    return [
        {
            "name": f"match_{query}",
            "kind": 12,
            "location": {
                "uri": "file:///fake/path.py",
                "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 10}},
            },
        }
    ]


def handle_code_action(msg: dict) -> Any:
    return [
        {
            "title": "Replace undefined_var with defined_var",
            "kind": "quickfix",
            "diagnostics": [],
            "edit": {
                "changes": {
                    msg.get("params", {}).get("textDocument", {}).get("uri", "file:///fake/path.py"): [
                        {
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 13},
                            },
                            "newText": "defined_var",
                        }
                    ]
                }
            },
        },
        {"title": "Organize imports", "kind": "source.organizeImports", "command": "fake.organizeImports"},
    ]


def _call_hierarchy_item(uri: str = "file:///fake/path.py") -> dict:
    return {
        "name": "example",
        "kind": 12,
        "uri": uri,
        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 2, "character": 0}},
        "selectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 11}},
    }


def handle_prepare_call_hierarchy(msg: dict) -> Any:
    params = msg.get("params", {})
    uri = params.get("textDocument", {}).get("uri", "file:///fake/path.py")
    return [_call_hierarchy_item(uri)]


def handle_incoming_calls(msg: dict) -> Any:
    return [{"from": {**_call_hierarchy_item(), "name": "caller"}, "fromRanges": []}]


def handle_outgoing_calls(msg: dict) -> Any:
    return [{"to": {**_call_hierarchy_item(), "name": "callee"}, "fromRanges": []}]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    while True:
        try:
            msg = read_message()
        except (EOFError, KeyboardInterrupt):
            break
        if msg is None:
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")

        # Response to a server-generated request such as workspace/configuration.
        if msg_id is not None and not method:
            _CONFIGURATION_RESPONSES.append(msg.get("result"))
            continue

        # Notifications (no id → no response)
        if msg_id is None:
            if method == "initialized":
                handle_initialized(msg)
            elif method == "textDocument/didOpen":
                handle_did_open(msg)
            elif method == "textDocument/didChange":
                handle_did_change(msg)
            elif method == "textDocument/didSave":
                handle_did_save(msg)
            elif method == "exit":
                break
            continue

        # Requests (have id → must respond)
        handlers = {
            "initialize": handle_initialize,
            "textDocument/diagnostic": handle_diagnostic,
            "textDocument/definition": handle_definition,
            "textDocument/typeDefinition": handle_definition,
            "textDocument/implementation": handle_definition,
            "textDocument/references": handle_references,
            "textDocument/hover": handle_hover,
            "textDocument/documentSymbol": handle_document_symbol,
            "workspace/symbol": handle_workspace_symbol,
            "textDocument/codeAction": handle_code_action,
            "textDocument/prepareCallHierarchy": handle_prepare_call_hierarchy,
            "callHierarchy/incomingCalls": handle_incoming_calls,
            "callHierarchy/outgoingCalls": handle_outgoing_calls,
            "shutdown": lambda m: None,
        }

        handler = handlers.get(method)
        if handler is None:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )
            continue

        try:
            result = handler(msg)
            if isinstance(result, dict) and "error" in result:
                write_message({"jsonrpc": "2.0", "id": msg_id, "error": result["error"]})
            else:
                write_message({"jsonrpc": "2.0", "id": msg_id, "result": result if result is not None else {}})
        except Exception as exc:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32603, "message": str(exc)},
                }
            )


if __name__ == "__main__":
    main()
