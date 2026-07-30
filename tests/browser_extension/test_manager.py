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
from kolega_code.browser_extension.multiplex import MultiplexedPeer, RemoteRequestError
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
        self.error: RemoteRequestError | None = None

    async def request(self, operation: str, params: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.requests.append((operation, params))
        if self.error is not None:
            raise self.error
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
        self.published = True
        self.publish_count = 0

    def publish(self) -> None:
        self.published = True
        self.publish_count += 1

    def withdraw(self) -> None:
        self.published = False

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
async def test_manager_attaches_on_a_live_peer_then_serializes_calls(tmp_path: Path) -> None:
    browser = manager(tmp_path)
    server = FakeServer()
    peer = FakePeer()
    browser._server = cast(RuntimeServer, server)

    with pytest.raises(ChromeExtensionUnavailableError, match="did not connect"):
        await browser.navigate("https://example.com")

    # A live authenticated peer is the attachment: the native host dials a
    # runtime's socket only to relay a message the extension addressed to it, and
    # the extension only ever addresses the runtime the operator selected.
    await browser._handle_peer(cast(MultiplexedPeer, peer))
    assert browser.session_id == "chrome:runtime_1"
    await browser._handle_event(ready_event("browser.other"))
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


@pytest.mark.asyncio
async def test_remote_error_codes_survive_and_coverage_codes_gain_a_remedy(tmp_path: Path) -> None:
    """The remote code was being discarded, leaving callers to string-match prose."""
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    peer = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, peer))
    await browser._handle_event(ready_event())

    peer.error = RemoteRequestError(
        "search_truncated",
        "Page text search covered only the first 499998 characters.",
        retryable=True,
    )
    with pytest.raises(ChromeExtensionUnavailableError) as truncated:
        await browser.wait_for(text="anything")
    assert truncated.value.code == "search_truncated"
    assert "Scope the search" in str(truncated.value)

    # A code with no coverage remedy keeps its message unchanged.
    peer.error = RemoteRequestError("tab_closed", "The selected tab was closed", retryable=False)
    with pytest.raises(ChromeExtensionUnavailableError) as closed:
        await browser.snapshot()
    assert closed.value.code == "tab_closed"
    assert str(closed.value) == "The selected tab was closed"
    await browser.cleanup_all_browsers()


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
async def test_a_reconnecting_relay_peer_is_usable_without_a_repeated_announcement(tmp_path: Path) -> None:
    """A reconnecting relay peer must not strand the session unconfirmed.

    The extension announces browser.session_ready once per discovery, so anything
    that requires a *fresh* announcement to work leaves the runtime stuck with no
    way to recover short of re-selecting in the popup.
    """
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    first = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, first))
    await browser._handle_event(ready_event())
    assert browser.session_id == "chrome:runtime_1"

    # The relay drops; work must block while nothing is connected.
    await first.close()
    await asyncio.sleep(0)
    assert browser._peer is None
    assert browser._ready is False

    # A fresh relay peer, with no repeated session_ready, must be usable at once.
    second = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, second))
    result = await browser.navigate("https://example.com")
    assert result["session_id"] == "chrome:runtime_1"
    assert second.requests == [("browser.navigate", {"url": "https://example.com/"})]


@pytest.mark.asyncio
async def test_detach_leaves_the_session_able_to_drive_the_browser_again(tmp_path: Path) -> None:
    """browser_close must not brick Chrome for the rest of the Kolega session.

    Detaching means "stop driving the user's Chrome", not "shut it down" — we never
    owned it — and every browser sub-agent detaches as cleanup at the end of its
    dispatch. Latching attachment on an announcement the extension makes only once
    per discovery meant the first detach was terminal: every later operation waited
    out the connection timeout and then told the operator to reopen the popup.
    """
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    peer = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, peer))
    await browser._handle_event(ready_event())

    assert await browser.close() == "chrome:runtime_1"
    assert peer.requests[-1] == ("browser.detach", {})
    # The browsing session is over, so the next operation reports a fresh launch.
    assert browser.session_id is None
    # An advertisement is a claim on the browser, so a finished session stops making
    # one. Otherwise every other Kolega session had to break the tie by hand in the
    # extension even though nothing was competing for the browser any more.
    server = cast(FakeServer, browser._server)
    assert server.published is False

    result = await browser.navigate("https://example.com")

    assert result["session_id"] == "chrome:runtime_1"
    assert peer.requests[-1] == ("browser.navigate", {"url": "https://example.com/"})
    # Browsing again re-asserts the claim, which is how a detached session asks for
    # the browser back: the native host turns the change into a fresh discovery.
    assert server.published is True
    assert server.publish_count == 1


@pytest.mark.asyncio
async def test_a_selection_refused_by_the_extension_is_answered_with_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extension is the authority on whether this session may drive Chrome.

    Holding a live peer no longer implies the grant cannot have moved: the operator
    can switch to another Kolega session while our relay stays open. That refusal
    arrives immediately and by code, which is strictly better than inferring it from
    an advertised runtime count and waiting out a timeout — but it must still carry
    the actionable remedy.
    """
    browser = manager(tmp_path)
    browser._server = cast(RuntimeServer, FakeServer())
    peer = FakePeer()
    await browser._handle_peer(cast(MultiplexedPeer, peer))
    _advertise(
        browser,
        monkeypatch,
        [_descriptor("runtime_1", "session_1"), _descriptor("runtime_2", "session_2")],
    )
    peer.error = RemoteRequestError(
        "session_not_selected",
        "The request is not routed to the selected Kolega session",
        retryable=False,
    )

    with pytest.raises(ChromeExtensionUnavailableError) as refused:
        await browser.navigate("https://example.com")

    assert refused.value.code == "session_not_selected"
    assert "waiting for you to choose" in str(refused.value)
    assert "session_2" in str(refused.value)
    assert "this session" in str(refused.value)


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
async def test_probe_names_competing_runtimes_while_a_choice_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator needs to know which picker entry is theirs."""
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
    _advertise(
        browser,
        monkeypatch,
        [_descriptor("runtime_1", "session_1"), _descriptor("runtime_2", "session_2")],
    )

    with pytest.raises(ChromeExtensionUnavailableError, match="waiting for you to choose"):
        await browser.navigate("https://example.com")
