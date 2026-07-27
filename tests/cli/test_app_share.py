"""Sharing a live link from inside the session.

The point of /share is that handing someone a link should not mean opening a
second terminal, finding the session id, and assembling a URL. So the assertions
here are about the link actually working, and about the server not outliving the
session that started it.
"""

from __future__ import annotations

import asyncio
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kolega_code.cli import messages
from kolega_code.web import hosting
from kolega_code.web.hosting import ALL_INTERFACES, LOOPBACK

from ._app_test_utils import _build_sub_agent_test_app


def _status(url: str) -> int:
    return urllib.request.urlopen(url, timeout=5).getcode()


async def _get(url: str) -> int:
    return await asyncio.to_thread(_status, url)


async def _wait_until_closed(url: str, *, timeout: float = 5.0) -> None:
    """Poll until the port stops answering; teardown is not instantaneous."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            await _get(url)
        except Exception:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{url} is still being served")


def _system_text(app) -> str:
    return "\n".join(entry.content for entry in app.conversation_entries if entry.kind == "system")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def share_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    app._clipboard: list[str] = []  # type: ignore[attr-defined]
    monkeypatch.setattr(type(app), "copy_to_clipboard", lambda self, text: self._clipboard.append(text))
    # Take an ephemeral port instead of the real default: a test suite has no
    # business seizing a well-known port on the machine running it.
    monkeypatch.setattr("kolega_code.web.hosting.DEFAULT_PORT", 0)
    return app


@pytest.mark.asyncio
async def test_share_hands_back_a_working_link(share_app) -> None:
    async with share_app.run_test():
        try:
            await share_app._command_share("")

            url = share_app._clipboard[-1]
            assert share_app.session.session_id in url
            assert "token=" in url, "a link that is meant to be given away has to carry its own grant"
            assert await _get(url) == 200

            # The same server without the link's token stays shut.
            with pytest.raises(urllib.error.HTTPError) as blocked:
                await _get(f"{share_app._share_server.handle.url}/api/sessions")
            assert blocked.value.code == 401

            assert url in _system_text(share_app), "the link belongs in the transcript, not only the clipboard"
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
async def test_share_twice_reuses_the_same_link(share_app) -> None:
    async with share_app.run_test():
        try:
            await share_app._command_share("")
            first = share_app._clipboard[-1]
            server = share_app._share_server

            await share_app._command_share("")

            assert share_app._share_server is server, "re-sharing must not open a second port"
            assert share_app._clipboard[-1] == first
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
async def test_share_stop_closes_the_port(share_app) -> None:
    async with share_app.run_test():
        await share_app._command_share("")
        url = share_app._clipboard[-1]

        await share_app._command_share("stop")

        assert share_app._share_server is None
        await _wait_until_closed(url)


@pytest.mark.asyncio
async def test_share_stop_without_sharing_says_so(share_app) -> None:
    notices: list[str] = []
    async with share_app.run_test():
        share_app._notify_user = lambda message, **kwargs: notices.append(message)  # type: ignore[method-assign]
        await share_app._command_share("stop")

    assert notices == [messages.SHARE_NOT_RUNNING]
    assert share_app._share_server is None


@pytest.mark.asyncio
async def test_share_lan_binds_wider_and_warns(share_app) -> None:
    async with share_app.run_test():
        try:
            await share_app._command_share("lan")

            assert share_app._share_server is not None
            assert share_app._share_server.exposed
            # The warning is the point: this is reachable by other machines now.
            assert messages.SHARE_LAN_WARNING in _system_text(share_app)
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
async def test_switching_reach_replaces_the_server(share_app) -> None:
    """Going from loopback to the network must not leave the narrow one running."""
    async with share_app.run_test():
        try:
            await share_app._command_share("")
            loopback = share_app._share_server
            assert loopback is not None and not loopback.exposed
            first_url = share_app._clipboard[-1]

            await share_app._command_share("lan")

            assert share_app._share_server is not loopback
            assert share_app._share_server.exposed
            assert not loopback.running
            await _wait_until_closed(first_url)
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
async def test_unknown_argument_explains_itself(share_app) -> None:
    notices: list[str] = []
    async with share_app.run_test():
        share_app._notify_user = lambda message, **kwargs: notices.append(message)  # type: ignore[method-assign]
        await share_app._command_share("public")

    assert notices == [messages.SHARE_USAGE]
    assert share_app._share_server is None, "an unrecognized argument must not open a port"


@pytest.mark.asyncio
async def test_quitting_stops_sharing(share_app) -> None:
    """A port left listening after the session ends is the worst failure here."""
    async with share_app.run_test():
        await share_app._command_share("")
        url = share_app._clipboard[-1]
        assert await _get(url) == 200

        share_app.on_unmount()
        await _wait_until_closed(url)

    assert share_app._share_server is None


def test_bind_addresses_are_distinct() -> None:
    assert LOOPBACK != ALL_INTERFACES


@pytest.mark.asyncio
async def test_share_uses_the_default_port_so_a_tunnel_rule_keeps_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the OS picks a new port per share and any tunnel goes stale."""
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(type(app), "copy_to_clipboard", lambda self, text: None)
    requested: list[object] = []

    real = hosting.ShareServer

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, store, **kwargs):
            requested.append(kwargs.get("port"))
            super().__init__(store, **{**kwargs, "port": 0})

    monkeypatch.setattr(hosting, "ShareServer", Recording)
    async with app.run_test():
        try:
            await app._command_share("")
        finally:
            await app._stop_share_server()

    assert requested == [hosting.DEFAULT_PORT]


@pytest.mark.asyncio
async def test_an_explicit_port_is_honoured(share_app) -> None:
    port = _free_port()
    async with share_app.run_test():
        try:
            await share_app._command_share(str(port))

            server = share_app._share_server
            assert server is not None and server.handle is not None
            assert server.handle.port == port
            assert f":{port}/" in share_app._clipboard[-1]
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
async def test_a_busy_default_port_moves_out_of_the_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is a convenience, so another share holding it is not fatal."""
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    app._clipboard: list[str] = []  # type: ignore[attr-defined]
    monkeypatch.setattr(type(app), "copy_to_clipboard", lambda self, text: self._clipboard.append(text))

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        monkeypatch.setattr("kolega_code.web.hosting.DEFAULT_PORT", taken.getsockname()[1])

        async with app.run_test():
            try:
                await app._command_share("")

                server = app._share_server
                assert server is not None and server.handle is not None
                assert server.handle.port != taken.getsockname()[1]
                assert await _get(app._clipboard[-1]) == 200
                assert messages.SHARE_PORT_TAKEN.format(port=taken.getsockname()[1]) in _system_text(app)
            finally:
                await app._stop_share_server()


@pytest.mark.asyncio
async def test_a_busy_explicit_port_is_an_error_not_a_silent_move(share_app) -> None:
    """A named port is a requirement: a tunnel is probably pointed at it."""
    notices: list[str] = []
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]

        async with share_app.run_test():
            share_app._notify_user = lambda message, **kwargs: notices.append(message)  # type: ignore[method-assign]
            await share_app._command_share(str(port))

    assert share_app._share_server is None
    assert notices and notices[-1].startswith("Could not start sharing")


@pytest.mark.asyncio
async def test_lan_accepts_a_port_too(share_app) -> None:
    port = _free_port()
    async with share_app.run_test():
        try:
            await share_app._command_share(f"lan {port}")

            server = share_app._share_server
            assert server is not None and server.handle is not None
            assert server.exposed
            assert server.handle.port == port
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["public", "70000", "lan lan", "1 2", "stop now"])
async def test_malformed_arguments_explain_themselves(share_app, bad: str) -> None:
    notices: list[str] = []
    async with share_app.run_test():
        share_app._notify_user = lambda message, **kwargs: notices.append(message)  # type: ignore[method-assign]
        await share_app._command_share(bad)

    assert notices == [messages.SHARE_USAGE], bad
    assert share_app._share_server is None


@pytest.mark.asyncio
async def test_a_share_link_reaches_only_the_session_it_was_made_for(share_app) -> None:
    """Sharing one session must not hand over every session on the machine.

    The token gates routes rather than sessions, so before the server was scoped
    the link given to one person read any other session in the same store — a
    different project's transcript, in the clear, over the same token.
    """
    other = share_app.store.create(share_app.project_path, "code", {"model": "test"}, title="not shared")

    async with share_app.run_test():
        try:
            await share_app._command_share("")
            url = share_app._clipboard[-1]
            base, _, query = url.partition("?")
            root = base.rsplit("/s/", 1)[0]

            assert await _get(url) == 200

            for path in (
                f"/s/{other.session_id}",
                f"/api/sessions/{other.session_id}",
                f"/api/sessions/{other.session_id}/events",
            ):
                with pytest.raises(urllib.error.HTTPError) as blocked:
                    await _get(f"{root}{path}?{query}")
                assert blocked.value.code == 404, f"{path} was reachable from another session's link"
        finally:
            await share_app._stop_share_server()


@pytest.mark.asyncio
async def test_share_says_plainly_that_a_live_link_is_not_redacted(share_app) -> None:
    """A live view serves raw events; only an export is scrubbed.

    The read-only note used to be the whole story, which reads as "safe to hand
    out". It is not: whatever the agent printed is visible to whoever holds it.
    """
    async with share_app.run_test():
        try:
            await share_app._command_share("")

            transcript = _system_text(share_app)
            assert messages.SHARE_UNREDACTED_NOTE in transcript
            assert "share export" in messages.SHARE_UNREDACTED_NOTE, (
                "the warning has to point at the alternative, not just state the risk"
            )
        finally:
            await share_app._stop_share_server()
