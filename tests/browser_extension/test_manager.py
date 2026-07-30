from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kolega_code.browser_extension.manager import (
    CHROME_EXTENSION_SUPPORTED_TOOLS,
    ChromeExtensionBrowserManager,
    ChromeExtensionProtocolError,
    ChromeExtensionUnavailableError,
)
from kolega_code.browser_extension.multiplex import MultiplexedPeer
from kolega_code.browser_extension.protocol import Envelope, MessageDirection
from kolega_code.browser_extension.registry import RuntimeDescriptor
from kolega_code.browser_extension.runtime import RuntimeServer

ORIGIN = f"chrome-extension://{'a' * 32}/"


class FakePeer:
    def __init__(self) -> None:
        self.closed = False
        self.closed_event = asyncio.Event()
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.active = 0
        self.max_active = 0

    async def request(self, operation: str, params: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.requests.append((operation, params))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.005)
        self.active -= 1
        return {"url": "https://example.com", "title": "Example", "result": {"operation": operation}}

    async def close(self, reason: str = "") -> None:
        self.closed = True
        self.closed_event.set()

    async def wait_closed(self) -> None:
        await self.closed_event.wait()


class FakeServer:
    runtime_id = "runtime_1"

    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


def manager(tmp_path: Path) -> ChromeExtensionBrowserManager:
    return ChromeExtensionBrowserManager(
        state_dir=tmp_path,
        kolega_session_id="session_1",
        extension_origin=ORIGIN,
        connection_timeout=0.02,
        operation_timeout=1,
    )


def ready_event(name: str = "browser.session_ready") -> Envelope:
    return Envelope.event(
        direction=MessageDirection.EXTENSION_TO_RUNTIME,
        request_id="event_1",
        runtime_id="runtime_1",
        session_id="session_1",
        deadline_ms=int(time.time() * 1000) + 1_000,
        event=name,
        data={},
    )


@pytest.mark.asyncio
async def test_manager_waits_for_session_ready_then_serializes_calls(tmp_path: Path) -> None:
    browser = manager(tmp_path)
    server = FakeServer()
    peer = FakePeer()
    browser._server = cast(RuntimeServer, server)
    await browser._handle_peer(cast(MultiplexedPeer, peer))

    with pytest.raises(ChromeExtensionUnavailableError, match="connected but has not confirmed a session"):
        await browser.navigate("https://example.com")
    await browser._handle_event(ready_event("browser.other"))
    assert browser.session_id is None
    await browser._handle_event(ready_event())
    assert browser.session_id == "chrome:runtime_1"

    first, second = await asyncio.gather(
        browser.navigate("https://example.com"),
        browser.press_key("A"),
    )
    assert first["result"] == {"operation": "browser.navigate"}
    assert second["result"] == {"operation": "browser.press_key"}
    assert first["session_id"] == "chrome:runtime_1"
    assert peer.max_active == 1
    assert peer.requests[:2] == [
        ("browser.navigate", {"url": "https://example.com/"}),
        ("browser.press_key", {"key": "A"}),
    ]

    closed_id = await browser.close()
    assert closed_id == "chrome:runtime_1"
    assert peer.requests[-1] == ("browser.detach", {})
    assert browser.session_id is None
    await browser.cleanup_all_browsers()
    await browser.cleanup_all_browsers()
    assert server.close_count == 1


@pytest.mark.asyncio
async def test_manager_validates_locally_and_fails_fast_for_unsupported_methods(tmp_path: Path) -> None:
    browser = manager(tmp_path)
    assert browser.browser_target == "chrome"
    assert browser.supported_tools == CHROME_EXTENSION_SUPPORTED_TOOLS
    with pytest.raises(ChromeExtensionProtocolError, match="index is required"):
        await browser.tabs("select")
    with pytest.raises(ChromeExtensionProtocolError, match="exactly one"):
        await browser.find(text="x", regex="x")

    unsupported = [
        browser.resize(800, 600),
        browser.drop("e1", data={"text/plain": "x"}),
        browser.handle_dialog(True),
        browser.file_upload([]),
        browser.console_messages(),
        browser.network_request(1),
        browser.evaluate("() => 1"),
    ]
    for call in unsupported:
        with pytest.raises(ChromeExtensionProtocolError):
            await call
    assert browser._server is None


@pytest.mark.asyncio
async def test_manager_preserves_browser_result_shapes(tmp_path: Path) -> None:
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    peer = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, peer))
    await browser._handle_event(ready_event())
    result = await browser.screenshot(target=None, image_type="png", full_page=True, scale="device")
    assert result["url"] == "https://example.com"
    assert result["result"] == {"operation": "browser.screenshot"}
    assert peer.requests[-1] == (
        "browser.screenshot",
        {"target": None, "image_type": "png", "full_page": True, "scale": "device"},
    )
    # Inapplicable scroll fields travel as explicit nulls, because the fixed
    # schema requires every key and rejects 0 or "" as a stand-in for unset.
    await browser.scroll(by_pages=1.5)
    assert peer.requests[-1] == (
        "browser.scroll",
        {"by_pages": 1.5, "target": None, "x": None, "y": None},
    )
    await browser.scroll(target="#main")
    assert peer.requests[-1] == (
        "browser.scroll",
        {"by_pages": None, "target": "#main", "x": None, "y": None},
    )
    await browser.cleanup_all_browsers()


def test_scroll_is_part_of_the_supported_chrome_tool_surface() -> None:
    assert "browser_scroll" in CHROME_EXTENSION_SUPPORTED_TOOLS


def _descriptor(runtime_id: str, session_id: str, pid: int = 4321) -> RuntimeDescriptor:
    """Build an advertised runtime, as a second Kolega session would publish."""
    now_ms = int(time.time() * 1000)
    return RuntimeDescriptor(
        runtime_id=runtime_id,
        session_id=session_id,
        transport="unix",
        endpoint=f"/tmp/{runtime_id}.sock",
        token="t" * 43,
        pid=pid,
        created_at_ms=now_ms,
        expires_at_ms=now_ms + 60_000,
        extension_origin=ORIGIN,
    )


def _advertise(
    browser: ChromeExtensionBrowserManager,
    monkeypatch: pytest.MonkeyPatch,
    descriptors: list[RuntimeDescriptor],
) -> None:
    """Inject the advertised runtime list.

    The registry's own liveness rules (TTL, live pid, owner-private socket) are
    covered by the registry tests; here we only care about how the manager
    reports pairing state.
    """
    monkeypatch.setattr(browser, "_live_runtimes", lambda: descriptors)


@pytest.mark.asyncio
async def test_readiness_survives_a_relay_peer_reconnecting(tmp_path: Path) -> None:
    """A reconnecting relay peer must not strand the session unconfirmed.

    The extension announces browser.session_ready once per native connection, so
    clearing readiness whenever the relay peer churned left the runtime stuck at
    "connected but has not confirmed a session" with no way to recover short of
    re-selecting in the popup. The native host only dials a runtime's socket to
    relay a message the extension addressed to that runtime, so a peer existing at
    all already proves this runtime is selected.
    """
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    first = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, first))
    await browser._handle_event(ready_event())
    assert browser.session_id == "chrome:runtime_1"

    # The relay drops; work must block, but the confirmation must not be lost.
    await first.close()
    await asyncio.sleep(0)
    assert browser._peer is None
    assert browser._ready is True

    # A fresh relay peer, with no repeated session_ready, must be usable at once.
    second = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, second))
    result = await browser.navigate("https://example.com")
    assert result["session_id"] == "chrome:runtime_1"
    assert second.requests == [("browser.navigate", {"url": "https://example.com/"})]


@pytest.mark.asyncio
async def test_detach_still_clears_readiness(tmp_path: Path) -> None:
    """Holding readiness across peer churn must not survive an explicit detach."""
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    await browser._handle_peer(cast(MultiplexedPeer, FakePeer()))
    await browser._handle_event(ready_event())

    await browser.close()

    assert browser.session_id is None
    assert browser._ready is False


@pytest.mark.asyncio
async def test_probe_reports_unreachable_without_a_connection(tmp_path: Path) -> None:
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())

    result = await browser.probe()

    assert result["state"] == "unreachable"
    assert result["connected"] is False
    assert result["ready"] is False
    assert "did not connect" in result["detail"]


@pytest.mark.asyncio
async def test_probe_reports_awaiting_selection_and_names_competing_runtimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connected-but-unselected companion is a different problem from an absent
    one, and the operator needs to know which picker entry is theirs."""
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    await browser._handle_peer(cast(MultiplexedPeer, FakePeer()))
    _advertise(
        browser,
        monkeypatch,
        [_descriptor("runtime_1", "session_1"), _descriptor("runtime_2", "session_2")],
    )

    result = await browser.probe()

    assert result["state"] == "awaiting_selection"
    assert result["connected"] is True
    assert result["ready"] is False
    assert {entry["runtime_id"] for entry in result["runtimes"]} == {"runtime_1", "runtime_2"}
    assert [entry for entry in result["runtimes"] if entry["current"]][0]["runtime_id"] == "runtime_1"
    assert "waiting for you to choose" in result["detail"]
    assert "session_2" in result["detail"]
    assert "this session" in result["detail"]


@pytest.mark.asyncio
async def test_probe_reports_awaiting_selection_without_any_peer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realistic pending-choice case has no peer at all.

    The extension only dials a runtime's socket once that runtime is selected, so
    keying "awaiting selection" off a connected peer made the state unreachable in
    practice and every pending choice looked like a dead connection.
    """
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    _advertise(
        browser,
        monkeypatch,
        [_descriptor("runtime_1", "session_1"), _descriptor("runtime_2", "session_2")],
    )

    result = await browser.probe()

    assert result["state"] == "awaiting_selection"
    assert result["connected"] is False
    assert "waiting for you to choose" in result["detail"]


@pytest.mark.asyncio
async def test_probe_reports_paired_once_the_session_is_ready(tmp_path: Path) -> None:
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    await browser._handle_peer(cast(MultiplexedPeer, FakePeer()))
    await browser._handle_event(ready_event())

    result = await browser.probe()

    assert result["state"] == "paired"
    assert result["connected"] is True
    assert result["ready"] is True
    assert result["runtime_id"] == "runtime_1"


@pytest.mark.asyncio
async def test_unavailable_error_names_competing_runtimes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The operation error, not just doctor, must explain a blocked selection."""
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    await browser._handle_peer(cast(MultiplexedPeer, FakePeer()))
    _advertise(
        browser,
        monkeypatch,
        [_descriptor("runtime_1", "session_1"), _descriptor("runtime_2", "session_2")],
    )

    with pytest.raises(ChromeExtensionUnavailableError, match="waiting for you to choose"):
        await browser.navigate("https://example.com")
