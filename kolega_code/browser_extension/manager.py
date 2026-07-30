"""BrowserManager backed by a selected Chrome extension runtime."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from kolega_code.services.base import BrowserManager

from .multiplex import ConnectionClosedError, MultiplexedPeer, RemoteRequestError, RequestTimeoutError
from .protocol import JSONValue, Envelope, ProtocolValidationError, validate_operation_request
from .registry import RuntimeDescriptor, RuntimeDescriptorRegistry, validate_extension_origin
from .runtime import RuntimeServer, RuntimeTransportError, UnsupportedRuntimeTransportError

CHROME_EXTENSION_SUPPORTED_TOOLS = frozenset(
    "browser_navigate browser_navigate_back browser_snapshot browser_find browser_wait_for browser_click "
    "browser_type browser_fill_form browser_select_option browser_hover browser_drag browser_press_key "
    "browser_scroll browser_tabs browser_network_requests browser_take_screenshot browser_close".split()
)
CHROME_EXTENSION_CAPABILITIES = CHROME_EXTENSION_SUPPORTED_TOOLS
DEFAULT_EXTENSION_CONNECTION_TIMEOUT_SECONDS = 12.0
DEFAULT_BROWSER_OPERATION_TIMEOUT_SECONDS = 30.0


# Codes that mean "this page is larger than one call can cover", as opposed to
# something being broken. Each one has a concrete next step, and saying so is what
# stops an agent rediscovering the same wall from six directions.
_COVERAGE_REMEDIES = {
    "search_truncated": (
        "Scope the search: snapshot a subtree with browser_snapshot target=<selector>, or check a "
        "string that appears earlier in the page."
    ),
    "result_too_large": (
        "The result did not fit its bound. Narrow it: pass a target to capture one element, or "
        "browser_scroll and capture the region you need."
    ),
    "page_too_large": (
        "The page exceeds what one call can cover. Pass a target to browser_snapshot to scope it, "
        "or browser_scroll and snapshot again."
    ),
}

# The extension refuses a request whenever the operator has not granted this
# runtime the browser, or has since granted it to another session. It is the
# authority on that, so these codes are answered with selection guidance rather
# than inferred from anything on this side.
_SELECTION_CODES = frozenset({"session_selection_required", "session_not_selected", "runtime_unavailable"})


class ChromeExtensionUnavailableError(RuntimeError):
    """Chrome or the selected extension runtime is unavailable."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class ChromeExtensionProtocolError(RuntimeError):
    """A request or result violates the fixed browser contract."""


class ChromeExtensionBrowserManager(BrowserManager):
    """Drive Chrome through one extension-selected Kolega runtime."""

    browser_target = "chrome"
    supported_tools = CHROME_EXTENSION_SUPPORTED_TOOLS
    capabilities = CHROME_EXTENSION_CAPABILITIES

    def __init__(
        self,
        *,
        state_dir: Path,
        kolega_session_id: str,
        extension_origin: str,
        connection_timeout: float = DEFAULT_EXTENSION_CONNECTION_TIMEOUT_SECONDS,
        operation_timeout: float = DEFAULT_BROWSER_OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        if connection_timeout <= 0 or operation_timeout <= 0:
            raise ValueError("Chrome extension timeouts must be positive.")
        self.state_dir = state_dir.expanduser().resolve()
        self.kolega_session_id = kolega_session_id
        self.extension_origin = validate_extension_origin(extension_origin)
        self.connection_timeout = connection_timeout
        self.operation_timeout = operation_timeout
        self._server: RuntimeServer | None = None
        self._peer: MultiplexedPeer | None = None
        self._state_changed = asyncio.Event()
        self._browser_session_id: str | None = None
        self._route_epoch = 0
        self._peer_watch_tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._closed = False

    @property
    def session_id(self) -> Optional[str]:
        return self._browser_session_id

    @property
    def _ready(self) -> bool:
        """Whether this runtime may drive the browser right now.

        Derived from having a live authenticated peer rather than latched by the
        extension's ``browser.session_ready`` announcement. The native host dials
        a runtime's socket only to relay a message the extension addressed to that
        runtime, and the extension only ever addresses the runtime the operator
        selected — so a live peer *is* the attachment, and nothing else needs to
        agree. A separate latch was strictly worse: only an announcement the
        extension makes once per discovery could set it, so every code path that
        cleared it stranded the session at "connected but has not confirmed a
        session" until the operator reopened the popup.
        """
        peer = self._peer
        return peer is not None and not peer.closed

    @property
    def runtime_id(self) -> Optional[str]:
        return self._server.runtime_id if self._server is not None else None

    async def start(self) -> None:
        """Publish this runtime for the extension picker."""
        await self._ensure_server()

    async def _ensure_server(self) -> RuntimeServer:
        async with self._lifecycle_lock:
            if self._closed:
                raise ChromeExtensionUnavailableError("Chrome integration is closed for this agent session.")
            if self._server is not None:
                return self._server
            registry = RuntimeDescriptorRegistry(self.state_dir / "browser-extension" / "runtimes")
            try:
                server = RuntimeServer(
                    registry,
                    session_id=self.kolega_session_id,
                    extension_origin=self.extension_origin,
                    event_handler=self._handle_event,
                    on_peer=self._handle_peer,
                )
                await server.start()
            except UnsupportedRuntimeTransportError as exc:
                raise ChromeExtensionUnavailableError(str(exc)) from None
            except RuntimeTransportError as exc:
                if "server" in locals():
                    with contextlib.suppress(Exception):
                        await server.close()
                raise ChromeExtensionUnavailableError(str(exc)) from None
            except BaseException:
                if "server" in locals():
                    with contextlib.suppress(Exception):
                        await server.close()
                raise
            self._server = server
            return server

    async def _handle_peer(self, peer: MultiplexedPeer) -> None:
        if self._closed:
            await peer.close("Chrome integration is closed for this agent session.")
            return
        previous = self._peer
        self._peer = peer
        # A peer proves selection (see _ready), so adopt the browsing session here
        # rather than waiting for an announcement that only accompanies the *first*
        # peer of a native connection.
        self._adopt_browser_session()
        self._state_changed.set()
        watcher = asyncio.create_task(self._watch_peer(peer), name="chrome-extension-peer-lifecycle")
        self._peer_watch_tasks.add(watcher)
        watcher.add_done_callback(self._peer_watch_tasks.discard)
        if previous is not None and previous is not peer:
            await previous.close("A newer Chrome extension connection replaced this connection.")

    async def _watch_peer(self, peer: MultiplexedPeer) -> None:
        await peer.wait_closed()
        if self._peer is peer:
            self._peer = None
            self._state_changed.set()

    async def _handle_event(self, envelope: Envelope) -> None:
        if envelope.payload["event"] != "browser.session_ready":
            return
        # Deliberately independent of peer state: this event is what makes the
        # native host dial the socket in the first place, so requiring a peer here
        # only reintroduces an ordering dependency between two views of the same
        # fact. Counting announcements lets a request that raced a rediscovery wait
        # for the next one instead of failing (see _await_route_confirmation).
        self._route_epoch += 1
        self._adopt_browser_session()
        self._state_changed.set()

    def _adopt_browser_session(self) -> None:
        server = self._server
        if self._browser_session_id is None and server is not None:
            self._browser_session_id = f"chrome:{server.runtime_id}"

    def _live_runtimes(self) -> list[RuntimeDescriptor]:
        """List every runtime currently advertised to the extension picker."""
        registry = RuntimeDescriptorRegistry(self.state_dir / "browser-extension" / "runtimes")
        try:
            return registry.list_active()
        except OSError:
            return []

    def _selection_detail(self) -> str:
        """Say what is actually blocking the browser, and only then involve the operator.

        Two completely different situations produce a selection refusal, and telling
        the operator to click the extension is right in only one of them:

        - **Contested.** Another Kolega session is claiming the browser at the same
          time. The extension will not guess which local runtime may drive it,
          because selecting one grants a local process access to the operator's real
          Chrome profile, so only the operator can break the tie.
        - **Uncontested.** This session holds the only claim. Then there is nothing
          to choose and nothing to click: the extension auto-selects a lone runtime
          on its next enumeration, which the native host triggers within about a
          second of the claim appearing. Sending the operator to the popup here is
          simply wrong, and it is what a claim that outlived its usefulness used to
          make unavoidable.
        """
        mine = self.runtime_id
        runtimes = self._live_runtimes()
        rivals = [descriptor for descriptor in runtimes if descriptor.runtime_id != mine]
        if not rivals:
            return (
                "Chrome has not finished picking this session up yet — no other Kolega session is "
                "competing for the browser, so there is nothing to select and nothing to click. The "
                "companion enumerates sessions about a second after one asks for the browser, and it "
                "selects a lone session automatically. Retry the operation."
            )
        listed = ", ".join(
            f"{descriptor.session_id} (runtime {descriptor.runtime_id}, pid {descriptor.pid})" for descriptor in rivals
        )
        plural = "sessions are" if len(rivals) > 1 else "session is"
        return (
            f"Another Kolega {plural} currently using the browser, so Kolega Browser Companion needs you "
            f"to choose which one may control it: {listed}. Click the extension and select this session"
            + (f" (runtime {mine})" if mine else "")
            + ". The extension badge shows '!' while a choice is required. A session releases its claim "
            "when it finishes browsing, so waiting for the other one to finish also clears this."
        )

    def _unavailable_detail(self) -> str:
        """Explain why no companion connection exists, naming competing runtimes.

        Only reached with no live peer, so this is "nothing reached us" — a peer
        proves selection (see ``_ready``). A selection that is pending or has moved
        to another session is reported by the extension itself when we ask, which is
        both authoritative and immediate; guessing it from the advertised runtime
        count while holding a live connection used to blame the operator for a state
        the session was not actually in.
        """
        mine = self.runtime_id
        if any(descriptor.runtime_id != mine for descriptor in self._live_runtimes()):
            return self._selection_detail()
        return (
            "Kolega Browser Companion did not connect. Confirm the extension is installed and enabled in "
            "Chrome, that Chrome is running, and that `kolega-code browser status` reports a valid native "
            "host, then retry. Opening the extension refreshes the connection."
        )

    async def _connected_peer(self) -> MultiplexedPeer:
        server = await self._ensure_server()
        # Driving the browser is a claim on it, so re-advertise if a previous
        # detach withdrew ours. The native host notices the change and tells Chrome
        # to re-enumerate, which is how a detached session asks for the browser back.
        if not server.published:
            server.publish()
        loop = asyncio.get_running_loop()
        expires = loop.time() + self.connection_timeout
        while True:
            peer = self._peer
            if peer is not None and not peer.closed:
                # Driving the browser starts a browsing session, so a detach that
                # ended the previous one is followed by a fresh browser_launched.
                self._adopt_browser_session()
                return peer
            self._state_changed.clear()
            peer = self._peer
            if peer is not None and not peer.closed:
                self._adopt_browser_session()
                return peer
            remaining = expires - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(self._state_changed.wait(), timeout=remaining)
            except TimeoutError:
                break
        raise ChromeExtensionUnavailableError(self._unavailable_detail())

    async def probe(self) -> dict[str, Any]:
        """Attempt an attach and report the pairing state without raising.

        Used by `kolega-code browser doctor`, which must distinguish a valid
        native-host manifest from an extension that is actually reachable.
        """
        result: dict[str, Any] = {
            "state": "unreachable",
            "connected": False,
            "ready": False,
            "runtime_id": None,
            "runtimes": [],
            "detail": "",
        }
        try:
            await self._ensure_server()
        except ChromeExtensionUnavailableError as exc:
            result["detail"] = str(exc)
            return result
        result["runtime_id"] = self.runtime_id
        try:
            await self._connected_peer()
        except ChromeExtensionUnavailableError as exc:
            result["detail"] = str(exc)
        else:
            result["state"] = "paired"
            result["detail"] = "Kolega Browser Companion is connected and this session is selected."
        result["connected"] = self._ready
        result["ready"] = self._ready
        result["runtimes"] = [
            {
                "runtime_id": descriptor.runtime_id,
                "session_id": descriptor.session_id,
                "pid": descriptor.pid,
                "current": descriptor.runtime_id == self.runtime_id,
            }
            for descriptor in self._live_runtimes()
        ]
        if result["state"] != "paired" and len(result["runtimes"]) > 1:
            result["state"] = "awaiting_selection"
        return result

    async def _await_route_confirmation(self, since: int) -> bool:
        """Wait for the extension to re-confirm that this runtime may drive Chrome.

        Re-publishing a withdrawn claim reaches Chrome only through the native
        host's registry watcher, so an operation issued right after a detach can
        arrive before the rediscovery that re-selects us. The extension refuses it,
        but announces ``browser.session_ready`` as soon as the rediscovery lands, so
        that refusal is transient and worth waiting out once. Compares a
        confirmation counter rather than clearing a flag, so an announcement that
        races the refusal still counts.
        """
        loop = asyncio.get_running_loop()
        expires = loop.time() + self.connection_timeout
        while self._route_epoch == since:
            self._state_changed.clear()
            if self._route_epoch != since:
                break
            remaining = expires - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._state_changed.wait(), timeout=remaining)
            except TimeoutError:
                return False
        return True

    async def _request(self, operation: str, params: Mapping[str, JSONValue]) -> dict[str, Any]:
        async with self._operation_lock:
            try:
                validated = validate_operation_request(operation, dict(params))
            except ProtocolValidationError as exc:
                raise ChromeExtensionProtocolError(exc.message) from None
            result = await self._send(operation, validated, allow_route_retry=True)
            if not isinstance(result, dict):
                raise ChromeExtensionProtocolError("Chrome extension returned a non-object browser result.")
            response = cast(dict[str, Any], result)
            response.setdefault("session_id", self._browser_session_id)
            return response

    async def _send(
        self,
        operation: str,
        validated: dict[str, JSONValue],
        *,
        allow_route_retry: bool,
    ) -> object:
        epoch = self._route_epoch
        peer = await self._connected_peer()
        try:
            return await peer.request(operation, validated, timeout=self.operation_timeout)
        except (ConnectionClosedError, RequestTimeoutError) as exc:
            if self._peer is peer and peer.closed:
                self._peer = None
                self._state_changed.set()
            raise ChromeExtensionUnavailableError(str(exc)) from None
        except RemoteRequestError as exc:
            # A selection refusal is checked before the extension touches the page,
            # so nothing happened and retrying once is safe.
            if allow_route_retry and exc.code in _SELECTION_CODES and await self._await_route_confirmation(epoch):
                return await self._send(operation, validated, allow_route_retry=False)
            # The remote error code was previously discarded, leaving callers to
            # string-match prose. Keep it, and append the concrete next step for the
            # codes that mean "too big to cover in one call".
            if exc.code in _SELECTION_CODES:
                # Replace rather than append here: the extension cannot see how many
                # sessions are competing, so its prose tells the operator to make a
                # choice that usually is not theirs to make. Only this side knows
                # whether anything is actually contending.
                raise ChromeExtensionUnavailableError(self._selection_detail(), code=exc.code) from None
            remedy = _COVERAGE_REMEDIES.get(exc.code or "")
            message = exc.message
            if remedy:
                message = f"{message.rstrip('.')}. {remedy}"
            raise ChromeExtensionUnavailableError(message, code=exc.code) from None

    async def navigate(self, url: str) -> dict[str, Any]:
        return await self._request("browser.navigate", {"url": url})

    async def snapshot(self, target: Optional[str] = None, depth: Optional[int] = None) -> dict[str, Any]:
        return await self._request("browser.snapshot", {"target": target, "depth": depth})

    async def find(self, *, text: Optional[str] = None, regex: Optional[str] = None) -> dict[str, Any]:
        return await self._request("browser.find", {"text": text, "regex": regex})

    async def click(
        self,
        target: str,
        *,
        double_click: bool = False,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        return await self._request(
            "browser.click",
            {
                "target": target,
                "double_click": double_click,
                "button": button,
                "modifiers": cast(list[JSONValue], modifiers or []),
            },
        )

    async def type_text(
        self,
        target: str,
        text: str,
        *,
        submit: bool = False,
        slowly: bool = False,
    ) -> dict[str, Any]:
        return await self._request(
            "browser.type",
            {"target": target, "text": text, "submit": submit, "slowly": slowly},
        )

    async def fill_form(self, fields: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._request("browser.fill_form", {"fields": cast(list[JSONValue], fields)})

    async def select_option(self, target: str, values: list[str]) -> dict[str, Any]:
        return await self._request(
            "browser.select_option",
            {"target": target, "values": cast(list[JSONValue], values)},
        )

    async def hover(self, target: str) -> dict[str, Any]:
        return await self._request("browser.hover", {"target": target})

    async def drag(self, start_target: str, end_target: str) -> dict[str, Any]:
        return await self._request(
            "browser.drag",
            {"start_target": start_target, "end_target": end_target},
        )

    async def drop(
        self,
        target: str,
        *,
        files: Optional[list[dict[str, Any]]] = None,
        data: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        raise ChromeExtensionProtocolError("File and data drop are not supported by the Chrome extension backend.")

    async def press_key(self, key: str) -> dict[str, Any]:
        return await self._request("browser.press_key", {"key": key})

    async def scroll(
        self,
        *,
        target: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        by_pages: Optional[float] = None,
    ) -> dict[str, Any]:
        return await self._request(
            "browser.scroll",
            {"by_pages": by_pages, "target": target, "x": x, "y": y},
        )

    async def navigate_back(self) -> dict[str, Any]:
        return await self._request("browser.navigate_back", {})

    async def wait_for(
        self,
        *,
        time: Optional[float] = None,
        text: Optional[str] = None,
        text_gone: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(
            "browser.wait_for",
            {"time": time, "text": text, "text_gone": text_gone},
        )

    async def resize(self, width: int, height: int) -> dict[str, Any]:
        raise ChromeExtensionProtocolError("Viewport resize is not supported by the Chrome extension backend.")

    async def tabs(
        self,
        action: str,
        *,
        index: Optional[int] = None,
        url: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(
            "browser.tabs",
            {"action": action, "index": index, "url": url},
        )

    async def handle_dialog(self, accept: bool, prompt_text: Optional[str] = None) -> dict[str, Any]:
        raise ChromeExtensionProtocolError("Dialog automation is not supported by the Chrome extension backend.")

    async def file_upload(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        raise ChromeExtensionProtocolError("File upload is not supported by the Chrome extension backend.")

    async def console_messages(self, level: str = "info", *, all_messages: bool = False) -> dict[str, Any]:
        raise ChromeExtensionProtocolError("Console messages are not supported by the Chrome extension backend.")

    async def network_requests(
        self,
        *,
        include_static: bool = False,
        filter_pattern: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(
            "browser.network_requests",
            {"include_static": include_static, "filter_pattern": filter_pattern},
        )

    async def network_request(self, index: int, part: Optional[str] = None) -> dict[str, Any]:
        raise ChromeExtensionProtocolError(
            "Detailed network request inspection is not supported by the Chrome extension backend."
        )

    async def screenshot(
        self,
        *,
        target: Optional[str] = None,
        image_type: str = "png",
        full_page: bool = False,
        scale: str = "css",
    ) -> dict[str, Any]:
        return await self._request(
            "browser.screenshot",
            {
                "target": target,
                "image_type": image_type,
                "full_page": full_page,
                "scale": scale,
            },
        )

    async def evaluate(self, function: str, target: Optional[str] = None) -> dict[str, Any]:
        raise ChromeExtensionProtocolError("Arbitrary JavaScript is not supported by the Chrome extension backend.")

    async def close(self) -> Optional[str]:
        """End this browsing session without giving up the attachment.

        ``browser_close`` on this backend means "stop driving the user's Chrome",
        not "shut the browser down" — we never owned it. So it must stay possible
        to drive it again afterwards: a browser sub-agent closes as a matter of
        hygiene at the end of every dispatch, and making that irreversible bricked
        Chrome for the rest of the Kolega session (every later operation waited out
        the connection timeout and then told the operator to reopen the popup).
        Only the browsing session id is dropped, so the next operation reports a
        fresh browser_launched.

        The advertisement *is* dropped, because it is a claim on the browser rather
        than a fact about the process. A finished session that kept claiming forced
        every other Kolega session to break the tie by hand in the extension, even
        though nothing was competing for the browser any more.
        """
        session_id = self._browser_session_id
        peer = self._peer
        if peer is not None and not peer.closed:
            with contextlib.suppress(ChromeExtensionUnavailableError, ChromeExtensionProtocolError):
                await self._request("browser.detach", {})
        self._browser_session_id = None
        server = self._server
        if server is not None:
            server.withdraw()
        self._state_changed.set()
        return session_id

    async def cleanup_all_browsers(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            server = self._server
            peer = self._peer
            self._server = None
            self._peer = None
            self._browser_session_id = None
            self._state_changed.set()

        async with self._operation_lock:
            if peer is not None and not peer.closed:
                with contextlib.suppress(Exception):
                    await peer.request("browser.detach", {}, timeout=self.operation_timeout)
            if server is not None:
                await server.close()
            watchers = tuple(self._peer_watch_tasks)
            for watcher in watchers:
                watcher.cancel()
            if watchers:
                await asyncio.gather(*watchers, return_exceptions=True)
