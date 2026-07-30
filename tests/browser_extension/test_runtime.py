from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from kolega_code.browser_extension.multiplex import ConnectionClosedError, RequestTimeoutError
from kolega_code.browser_extension.protocol import JSONValue, Envelope
from kolega_code.browser_extension.registry import (
    DescriptorSecurityError,
    RuntimeDescriptor,
    RuntimeDescriptorRegistry,
)
from kolega_code.browser_extension.runtime import (
    RuntimeAuthenticationError,
    RuntimeServer,
    UnsupportedRuntimeTransportError,
    connect_runtime_channel,
    connect_runtime_peer,
    ensure_runtime_transport_supported,
    selected_runtime_transport,
)

ORIGIN = f"chrome-extension://{'a' * 32}/"
OTHER_ORIGIN = f"chrome-extension://{'b' * 32}/"


def test_runtime_descriptor_is_exact_and_macos_only() -> None:
    now = int(time.time() * 1000)
    descriptor = RuntimeDescriptor(
        runtime_id="runtime_1",
        session_id="session_1",
        transport="unix",
        endpoint="/tmp/runtime_1.sock",
        token="x" * 32,
        pid=os.getpid(),
        created_at_ms=now,
        expires_at_ms=now + 1_000,
        extension_origin=ORIGIN,
    )
    assert descriptor.endpoint == "/tmp/runtime_1.sock"
    assert set(descriptor.to_mapping()) == {
        "protocol_version",
        "runtime_id",
        "session_id",
        "transport",
        "endpoint",
        "token",
        "pid",
        "created_at_ms",
        "expires_at_ms",
        "extension_origin",
    }
    assert selected_runtime_transport(platform="darwin") == "unix"
    for platform in ("linux", "win32"):
        with pytest.raises(UnsupportedRuntimeTransportError, match="only on macOS"):
            ensure_runtime_transport_supported(platform=platform)


@pytest.mark.asyncio
async def test_unix_runtime_auth_multiplex_timeout_disconnect_and_cleanup(tmp_path: Path) -> None:
    registry = RuntimeDescriptorRegistry(tmp_path / "runtimes")
    cancelled = asyncio.Event()
    blocked = asyncio.Event()

    async def handle(envelope: Envelope) -> JSONValue:
        operation = envelope.payload["operation"]
        if operation == "browser.snapshot":
            return {"url": "https://example.com", "snapshot": "- button: Save"}
        if operation == "browser.navigate_back":
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        await blocked.wait()
        return {}

    server = RuntimeServer(
        registry,
        session_id="session_1",
        extension_origin=ORIGIN,
        request_handler=None,
        platform="darwin",
    )
    descriptor = await server.start()
    extension_peer = await connect_runtime_peer(
        descriptor,
        extension_origin=ORIGIN,
        request_handler=handle,
        platform="darwin",
    )
    runtime_peer = await server.wait_for_connection(timeout=1)
    assert await runtime_peer.request(
        "browser.snapshot",
        {"target": None, "depth": None},
        timeout=1,
    ) == {"url": "https://example.com", "snapshot": "- button: Save"}

    with pytest.raises(RequestTimeoutError):
        await runtime_peer.request("browser.navigate_back", {}, timeout=0.02)
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert runtime_peer.pending_count == 0

    pending = asyncio.create_task(runtime_peer.request("browser.press_key", {"key": "A"}, timeout=1))
    while extension_peer.inflight_count == 0:
        await asyncio.sleep(0)
    await extension_peer.close()
    with pytest.raises(ConnectionClosedError):
        await pending

    await server.close()
    await server.close()
    assert not Path(descriptor.endpoint).exists()
    assert not (registry.root / f"{descriptor.runtime_id}.json").exists()


@pytest.mark.asyncio
async def test_withdrawing_drops_the_claim_but_keeps_the_socket_and_its_connections(tmp_path: Path) -> None:
    """An advertisement is a claim on the browser, not a fact about the process.

    The extension refuses to guess between several claims, so a session that has
    finished browsing must stop making one — otherwise every other Kolega session
    has to break the tie by hand in the extension even when nothing is competing.
    Withdrawing must not cost the live connection though: a session that detaches
    and then browses again has to resume without waiting out a fresh discovery.
    """
    registry = RuntimeDescriptorRegistry(tmp_path / "runtimes")
    server = RuntimeServer(registry, session_id="session_1", extension_origin=ORIGIN, platform="darwin")
    descriptor = await server.start()
    assert server.published is True
    assert [entry.runtime_id for entry in registry.list_active()] == [descriptor.runtime_id]

    extension_peer = await connect_runtime_peer(descriptor, extension_origin=ORIGIN, platform="darwin")
    runtime_peer = await server.wait_for_connection(timeout=1)

    server.withdraw()
    assert server.published is False
    assert registry.list_active() == []
    assert Path(descriptor.endpoint).exists()
    assert runtime_peer.closed is False

    # Re-advertising is how a detached session asks for the browser back, and the
    # runtime id is unchanged so an operator's existing choice still matches.
    server.publish()
    assert server.published is True
    republished = registry.list_active()
    assert [entry.runtime_id for entry in republished] == [descriptor.runtime_id]
    assert republished[0].created_at_ms == descriptor.created_at_ms
    assert republished[0].expires_at_ms >= descriptor.expires_at_ms

    await extension_peer.close()
    await server.close()
    assert registry.list_active() == []


@pytest.mark.asyncio
async def test_the_lease_refresh_never_resurrects_a_withdrawn_claim(tmp_path: Path) -> None:
    """The refresh rewrites the descriptor to extend the lease, so left unguarded it
    would reinstate a claim the session explicitly gave up."""
    registry = RuntimeDescriptorRegistry(tmp_path / "runtimes")
    # ttl_ms/3000 is the refresh interval, floored at 50ms.
    server = RuntimeServer(
        registry,
        session_id="session_1",
        extension_origin=ORIGIN,
        ttl_ms=60_000,
        platform="darwin",
    )
    await server.start()
    server.withdraw()

    for _ in range(6):
        await asyncio.sleep(0.02)
        assert registry.list_active() == []

    server.publish()
    assert len(registry.list_active()) == 1
    await server.close()


@pytest.mark.asyncio
async def test_runtime_rejects_wrong_origin_and_registry_permissions(tmp_path: Path) -> None:
    registry = RuntimeDescriptorRegistry(tmp_path / "runtimes")
    server = RuntimeServer(registry, session_id="session_1", extension_origin=ORIGIN, platform="darwin")
    descriptor = await server.start()
    with pytest.raises(RuntimeAuthenticationError):
        await connect_runtime_channel(descriptor, extension_origin=OTHER_ORIGIN, platform="darwin")
    descriptor_path = registry.root / f"{descriptor.runtime_id}.json"
    if hasattr(os, "getuid"):
        descriptor_path.chmod(0o644)
        with pytest.raises(DescriptorSecurityError):
            registry.read(descriptor.runtime_id)
        descriptor_path.chmod(0o600)
    assert registry.unregister(descriptor.runtime_id, token="wrong") is False
    assert registry.read(descriptor.runtime_id) == descriptor
    await server.close()
