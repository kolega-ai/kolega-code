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
    "browser_tabs browser_network_requests browser_take_screenshot browser_close".split()
)
CHROME_EXTENSION_CAPABILITIES = CHROME_EXTENSION_SUPPORTED_TOOLS
DEFAULT_EXTENSION_CONNECTION_TIMEOUT_SECONDS = 12.0
DEFAULT_BROWSER_OPERATION_TIMEOUT_SECONDS = 30.0


class ChromeExtensionUnavailableError(RuntimeError):
    """Chrome or the selected extension runtime is unavailable."""


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
        self._ready = False
        self._browser_session_id: str | None = None
        self._peer_watch_tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._closed = False

    @property
    def session_id(self) -> Optional[str]:
        return self._browser_session_id

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
        self._clear_ready()
        self._peer = peer
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
            self._clear_ready()
            self._state_changed.set()

    async def _handle_event(self, envelope: Envelope) -> None:
        if envelope.payload["event"] != "browser.session_ready":
            return
        peer = self._peer
        server = self._server
        if peer is None or peer.closed or server is None:
            return
        self._ready = True
        self._browser_session_id = f"chrome:{server.runtime_id}"
        self._state_changed.set()

    def _clear_ready(self) -> None:
        self._ready = False
        self._browser_session_id = None

    def _live_runtimes(self) -> list[RuntimeDescriptor]:
        """List every runtime currently advertised to the extension picker."""
        registry = RuntimeDescriptorRegistry(self.state_dir / "browser-extension" / "runtimes")
        try:
            return registry.list_active()
        except OSError:
            return []

    def _unavailable_detail(self) -> str:
        """Explain *why* the companion is not ready, naming competing runtimes.

        The extension connects automatically, but it will not guess which local
        runtime may drive the browser when several are advertised. Distinguish
        "nothing connected" from "awaiting your choice", because the remedies are
        completely different. Note the choice is detected from the advertised
        runtime count, not from having a peer: the extension only dials a
        runtime's socket *after* it is selected, so a pending choice never has a
        peer to observe.
        """
        connected = self._peer is not None and not self._peer.closed
        mine = self.runtime_id
        runtimes = self._live_runtimes()
        if len(runtimes) > 1:
            listed = ", ".join(
                f"{descriptor.session_id} (runtime {descriptor.runtime_id}, pid {descriptor.pid})"
                + (" <- this session" if descriptor.runtime_id == mine else "")
                for descriptor in runtimes
            )
            return (
                "Kolega Browser Companion is connected but waiting for you to choose which Kolega session "
                f"may control the browser. {len(runtimes)} sessions are advertised: {listed}. Click the "
                "extension and select this session"
                + (f" (runtime {mine})" if mine else "")
                + ". The extension badge shows '!' while a choice is required."
            )
        if connected:
            return (
                "Kolega Browser Companion is connected but has not confirmed a session. Open the extension "
                "and select this Kolega session, then retry."
            )
        return (
            "Kolega Browser Companion did not connect. Confirm the extension is installed and enabled in "
            "Chrome, that Chrome is running, and that `kolega-code browser status` reports a valid native "
            "host, then retry."
        )

    async def _connected_peer(self) -> MultiplexedPeer:
        await self._ensure_server()
        loop = asyncio.get_running_loop()
        expires = loop.time() + self.connection_timeout
        while True:
            peer = self._peer
            if peer is not None and not peer.closed and self._ready:
                return peer
            self._state_changed.clear()
            peer = self._peer
            if peer is not None and not peer.closed and self._ready:
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
        result["connected"] = self._peer is not None and not self._peer.closed
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
        if result["state"] != "paired":
            if len(result["runtimes"]) > 1:
                result["state"] = "awaiting_selection"
            elif result["connected"]:
                result["state"] = "connected_not_selected"
        return result

    async def _request(self, operation: str, params: Mapping[str, JSONValue]) -> dict[str, Any]:
        async with self._operation_lock:
            try:
                validated = validate_operation_request(operation, dict(params))
            except ProtocolValidationError as exc:
                raise ChromeExtensionProtocolError(exc.message) from None
            peer = await self._connected_peer()
            try:
                result = await peer.request(operation, validated, timeout=self.operation_timeout)
            except (ConnectionClosedError, RequestTimeoutError) as exc:
                if self._peer is peer and peer.closed:
                    self._peer = None
                    self._clear_ready()
                    self._state_changed.set()
                raise ChromeExtensionUnavailableError(str(exc)) from None
            except RemoteRequestError as exc:
                raise ChromeExtensionUnavailableError(exc.message) from None
            if not isinstance(result, dict):
                raise ChromeExtensionProtocolError("Chrome extension returned a non-object browser result.")
            response = cast(dict[str, Any], result)
            response.setdefault("session_id", self._browser_session_id)
            return response

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
        session_id = self._browser_session_id
        peer = self._peer
        if peer is not None and not peer.closed and self._ready:
            with contextlib.suppress(ChromeExtensionUnavailableError, ChromeExtensionProtocolError):
                await self._request("browser.detach", {})
        self._clear_ready()
        self._state_changed.set()
        return session_id

    async def cleanup_all_browsers(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            server = self._server
            peer = self._peer
            ready = self._ready
            self._server = None
            self._peer = None
            self._clear_ready()
            self._state_changed.set()

        async with self._operation_lock:
            if peer is not None and not peer.closed and ready:
                with contextlib.suppress(Exception):
                    await peer.request("browser.detach", {}, timeout=self.operation_timeout)
            if server is not None:
                await server.close()
            watchers = tuple(self._peer_watch_tasks)
            for watcher in watchers:
                watcher.cancel()
            if watchers:
                await asyncio.gather(*watchers, return_exceptions=True)
