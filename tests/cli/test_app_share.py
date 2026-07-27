"""Sharing a live link from inside the session.

The point of /share is that handing someone a link should not mean opening a
second terminal, finding the session id, and assembling a URL. So the assertions
here are about the link actually working, and about the server not outliving the
session that started it.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kolega_code.cli import messages
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


@pytest.fixture
def share_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    app._clipboard: list[str] = []  # type: ignore[attr-defined]
    monkeypatch.setattr(type(app), "copy_to_clipboard", lambda self, text: self._clipboard.append(text))
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
