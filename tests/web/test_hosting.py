"""The share server: an ASGI server on the host's event loop.

Sharing from inside a running TUI means the server cannot own the process, pick
a fixed port, or install signal handlers. These cover the parts of that which are
easy to regress silently.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.web import hosting
from kolega_code.web.hosting import ALL_INTERFACES, LOCAL_NETWORK, LOOPBACK, ShareServer, ShareServerError


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "cli", {}, session_id="a" * 32, title="shared")
    return store


def _status(url: str) -> int:
    return urllib.request.urlopen(url, timeout=5).getcode()


async def _get(url: str) -> int:
    """Fetch off the loop; a blocking call here would deadlock against the server."""
    return await asyncio.to_thread(_status, url)


@pytest.mark.asyncio
async def test_binds_a_free_port_and_serves_behind_its_token(store: SessionStore) -> None:
    server = ShareServer(store)
    handle = await server.start()
    try:
        assert handle.port > 0, "port 0 must be resolved to the port actually bound"
        assert handle.token and handle.token in server.session_url("a" * 32)
        assert not handle.exposed

        assert await _get(server.session_url("a" * 32)) == 200
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            await _get(f"{handle.url}/api/sessions")
        assert unauthorized.value.code == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_stop_releases_the_port(store: SessionStore) -> None:
    server = ShareServer(store)
    handle = await server.start()
    await server.stop()

    assert not server.running
    assert server.handle is None
    with pytest.raises(Exception):
        await _get(handle.url)

    # A second share is a fresh server, not a resurrected one.
    other = ShareServer(store)
    await other.start()
    try:
        assert other.running
    finally:
        await other.stop()


@pytest.mark.asyncio
async def test_starting_twice_reuses_the_running_server(store: SessionStore) -> None:
    server = ShareServer(store)
    first = await server.start()
    try:
        assert await server.start() is first
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_request_stop_is_safe_without_a_loop_turn(store: SessionStore) -> None:
    """The sync teardown path Textual's on_unmount uses."""
    server = ShareServer(store)
    await server.start()
    server.request_stop()

    assert server.handle is None
    await asyncio.sleep(0.05)
    assert not server.running


@pytest.mark.asyncio
async def test_a_busy_port_is_an_ordinary_error(store: SessionStore) -> None:
    """Not a SystemExit, which asyncio re-raises into the host's event loop.

    Left to itself uvicorn binds during startup and reports failure by calling
    sys.exit. Raised inside a task that does not merely fail the task: asyncio
    re-raises SystemExit into the loop, so a port collision would take down the
    application that is only hosting this server.
    """
    import socket

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]

        server = ShareServer(store, port=port)
        with pytest.raises(ShareServerError) as failure:
            await server.start()

        assert str(port) in str(failure.value)
        assert not server.running
        # The loop is still healthy and the next share still works.
        await asyncio.sleep(0)

    recovered = ShareServer(store)
    await recovered.start()
    await recovered.stop()


def test_exposure_is_known_before_starting(store: SessionStore) -> None:
    assert not ShareServer(store, bind=LOOPBACK).exposed
    assert ShareServer(store, bind=ALL_INTERFACES).exposed


@pytest.mark.asyncio
async def test_does_not_touch_the_host_signal_handlers(store: SessionStore) -> None:
    """uvicorn's own handlers would take Ctrl-C away from the TUI."""
    import signal

    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    server = ShareServer(store)
    await server.start()
    try:
        assert {sig: signal.getsignal(sig) for sig in before} == before
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_reaching_the_lan_binds_one_address_not_every_interface(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Reachable on my local network" must not mean "reachable everywhere".

    Binding the wildcard listens on every interface the machine happens to have
    — a VPN, a tether, a cloud NIC — which is more than the request asks for and
    more than the link handed out advertises. The narrowest listener that
    satisfies it is the one address that link points at.
    """
    monkeypatch.setattr(hosting, "local_network_address", lambda: LOOPBACK)
    server = ShareServer(store, bind=ALL_INTERFACES)

    handle = await server.start()
    try:
        bound = server._socket.getsockname()  # pyright: ignore[reportOptionalMemberAccess]
        assert bound[0] != ALL_INTERFACES, "the share server still listens on every interface"
        assert bound[0] == LOOPBACK
        # The advertised link points at exactly what was bound.
        assert handle.host == LOOPBACK and f"{LOOPBACK}:{handle.port}" in handle.url
        assert handle.exposed, "the caller still asked for a reachable share"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_reaching_the_lan_without_a_network_fails_instead_of_opening_up(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no local address there is nothing to narrow to, and falling back to
    the wildcard would quietly expose more than was asked for."""
    monkeypatch.setattr(hosting, "local_network_address", lambda: None)

    with pytest.raises(ShareServerError) as failure:
        await ShareServer(store, bind=ALL_INTERFACES).start()

    assert "local network address" in str(failure.value)


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", "::0"])
def test_an_unspecified_bind_address_is_refused_outright(wildcard: str) -> None:
    """The last gate before bind, wherever the address came from.

    ``LOCAL_NETWORK`` is how a caller *asks* for reach and is resolved to a real
    address long before this point, so a wildcard arriving at the socket is a
    mistake rather than a choice — and listening on every interface is never
    what a share link means.

    Exercised against the guard rather than through ``start()`` on purpose. A
    test that hands a wildcard to the server gives dataflow analysis a path from
    a literal all the way to ``bind()``; it cannot see the guard raise in
    between, so the negative test would report itself as the vulnerability it
    exists to disprove.
    """
    with pytest.raises(ShareServerError) as refused:
        hosting._specific_address(wildcard)  # pyright: ignore[reportPrivateUsage]

    assert "every interface" in str(refused.value)
    assert "LOCAL_NETWORK" in str(refused.value), "the refusal should name the right way to ask"


def test_a_real_address_passes_the_guard_unchanged() -> None:
    for host in (LOOPBACK, "192.168.1.20", "example.internal"):
        assert hosting._specific_address(host) == host  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_asking_for_all_interfaces_is_a_request_not_an_address(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented spelling still works, and resolves rather than binding."""
    monkeypatch.setattr(hosting, "local_network_address", lambda: LOOPBACK)
    server = ShareServer(store, bind=ALL_INTERFACES)

    handle = await server.start()
    try:
        assert handle.host == LOOPBACK
    finally:
        await server.stop()


def test_the_wildcard_is_no_longer_a_way_to_ask_for_reach() -> None:
    """The reach request is a request, not an address.

    It used to be the wildcard itself, which both misdescribed the behaviour and
    left a literal that any caller could route straight to bind().
    """
    assert "0.0.0.0" not in {LOCAL_NETWORK, ALL_INTERFACES}
    assert ALL_INTERFACES == LOCAL_NETWORK, "the exported name has to keep working"
