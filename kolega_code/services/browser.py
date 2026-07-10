import asyncio
import base64
import contextlib
import datetime
import json
import mimetypes
import os
import re
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .base import BrowserManager


_BACKENDS = {"local", "browserless", "browserstack"}
_BROWSERLESS_REGIONS = {"sfo", "lon", "ams"}
_REF_PATTERN = re.compile(r"^(?:f\d+)?e\d+$")
_CONSOLE_LEVELS = {"error": 0, "warning": 1, "info": 2, "debug": 3}
_STATIC_RESOURCE_TYPES = {"font", "image", "media", "script", "stylesheet"}
_MAX_CONSOLE_MESSAGES = 200
_MAX_NETWORK_REQUESTS = 500
_MAX_NETWORK_BODY_CHARS = 20_000


def _positive_int(value: Optional[str | int], *, name: str, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


@dataclass
class _PageState:
    page: Any
    console_all: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_CONSOLE_MESSAGES))
    console_recent: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_CONSOLE_MESSAGES))
    network: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_NETWORK_REQUESTS))
    last_title: str = ""


@dataclass
class _BrowserSession:
    session_id: str
    playwright: Any
    browser: Any
    context: Any
    current_page: Any = None
    pages: dict[int, _PageState] = field(default_factory=dict)
    dialog: Any = None
    file_chooser: Any = None
    pending_action: Optional[asyncio.Task] = None
    modal_event: asyncio.Event = field(default_factory=asyncio.Event)
    keepalive_task: Optional[asyncio.Task] = None


class PlaywrightBrowserManager(BrowserManager):
    """Single-session Playwright manager using accessibility snapshot refs."""

    def __init__(
        self,
        browser_backend: str = "local",
        *,
        browserless_endpoint: Optional[str] = None,
        browserless_api_key: Optional[str] = None,
        browserless_timeout_ms: Optional[int] = None,
        browserless_protocol: Optional[str] = None,
    ) -> None:
        self.viewport = {"width": 1280, "height": 720}
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        self.headless = False
        self.action_timeout = 5000
        self.navigation_timeout = 30000
        self.connection_timeout = _positive_int(
            os.environ.get("BROWSER_CONNECT_TIMEOUT_MS"), name="BROWSER_CONNECT_TIMEOUT_MS", default=30000
        )
        self.keepalive_interval = 30
        self.browser_backend = browser_backend.lower()
        self._session: Optional[_BrowserSession] = None

        if self.browser_backend not in _BACKENDS:
            supported = ", ".join(sorted(_BACKENDS))
            raise ValueError(f"Unknown browser backend '{browser_backend}'. Expected one of: {supported}.")

        if self.browser_backend == "browserstack":
            self.browserstack_username = os.environ.get("BROWSERSTACK_USERNAME")
            self.browserstack_access_key = os.environ.get("BROWSERSTACK_ACCESS_KEY")
            if not self.browserstack_username or not self.browserstack_access_key:
                raise ValueError(
                    "BrowserStack credentials not found. Set BROWSERSTACK_USERNAME and BROWSERSTACK_ACCESS_KEY."
                )

        if self.browser_backend == "browserless":
            self.browserless_api_key = browserless_api_key or os.environ.get("BROWSERLESS_API_KEY")
            self.browserless_endpoint = (
                browserless_endpoint
                or os.environ.get("BROWSERLESS_WS_ENDPOINT")
                or os.environ.get("BROWSERLESS_ENDPOINT")
            )
            configured_protocol = browserless_protocol or os.environ.get("BROWSERLESS_PROTOCOL")
            if configured_protocol is None and self.browserless_endpoint:
                configured_protocol = (
                    "playwright" if self.browserless_endpoint.rstrip("/").endswith("/playwright") else "cdp"
                )
            self.browserless_protocol = (configured_protocol or "cdp").lower()
            if self.browserless_protocol not in {"cdp", "playwright"}:
                raise ValueError("BROWSERLESS_PROTOCOL must be either 'cdp' or 'playwright'.")
            self.browserless_timeout_ms = _positive_int(
                browserless_timeout_ms
                if browserless_timeout_ms is not None
                else os.environ.get("BROWSERLESS_TIMEOUT_MS"),
                name="BROWSERLESS_TIMEOUT_MS",
            )
            if self.browserless_endpoint is None:
                self.browserless_endpoint = self._default_browserless_endpoint()
            self._validate_browserless_auth()

    @property
    def session_id(self) -> Optional[str]:
        return self._session.session_id if self._session else None

    def _default_browserless_endpoint(self) -> str:
        region = os.environ.get("BROWSERLESS_REGION", "sfo").lower()
        if region not in _BROWSERLESS_REGIONS:
            supported = ", ".join(sorted(_BROWSERLESS_REGIONS))
            raise ValueError(f"Unknown BROWSERLESS_REGION '{region}'. Expected one of: {supported}.")
        path = "/chromium/playwright" if self.browserless_protocol == "playwright" else ""
        return f"wss://production-{region}.browserless.io{path}"

    def _validate_browserless_auth(self) -> None:
        endpoint = self.browserless_endpoint or ""
        parsed = urllib.parse.urlsplit(endpoint.replace("{token}", "placeholder"))
        embedded_token = "{token}" in endpoint or "token" in urllib.parse.parse_qs(parsed.query)
        is_cloud = bool(parsed.hostname and parsed.hostname.endswith("browserless.io"))
        if is_cloud and not embedded_token and not self.browserless_api_key:
            raise ValueError(
                "Browserless credentials not found. Set BROWSERLESS_API_KEY or include token= in "
                "BROWSERLESS_WS_ENDPOINT."
            )

    def _browserless_url(self) -> str:
        endpoint = self.browserless_endpoint or ""
        if "{token}" in endpoint:
            if not self.browserless_api_key:
                raise ValueError("BROWSERLESS_WS_ENDPOINT uses {token}, but BROWSERLESS_API_KEY is not set.")
            endpoint = endpoint.replace("{token}", urllib.parse.quote(self.browserless_api_key, safe=""))
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError("BROWSERLESS_WS_ENDPOINT must use ws:// or wss://.")
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        keys = {key for key, _ in query}
        if self.browserless_api_key and "token" not in keys:
            query.append(("token", self.browserless_api_key))
        if self.browserless_timeout_ms is not None and "timeout" not in keys:
            query.append(("timeout", str(self.browserless_timeout_ms)))
        return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))

    def _browserstack_url(self) -> str:
        capabilities = {
            "browser": "chrome",
            "browser_version": "latest",
            "os": "Windows",
            "os_version": "11",
            "name": "Kolega Browser Agent",
            "build": "kolega-platform",
            "browserstack.username": self.browserstack_username,
            "browserstack.accessKey": self.browserstack_access_key,
            "browserstack.console": "verbose",
            "browserstack.networkLogs": "true",
        }
        return "wss://cdp.browserstack.com/playwright?caps=" + urllib.parse.quote(json.dumps(capabilities))

    @staticmethod
    def _normalize_url(url: str) -> str:
        normalized = url.strip()
        if normalized == "about:blank":
            return normalized
        if "://" not in normalized:
            prefix = "http://" if normalized.startswith(("localhost", "127.0.0.1", "[::1]")) else "https://"
            normalized = prefix + normalized
        parsed = urllib.parse.urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Browser navigation only supports http:// and https:// URLs.")
        return normalized

    async def _connect(self, playwright: Any) -> Any:
        if self.browser_backend == "browserstack":
            return await playwright.chromium.connect(endpoint=self._browserstack_url(), timeout=self.connection_timeout)
        if self.browser_backend == "browserless":
            endpoint = self._browserless_url()
            if self.browserless_protocol == "playwright":
                return await playwright.chromium.connect(endpoint, timeout=self.connection_timeout)
            return await playwright.chromium.connect_over_cdp(endpoint, timeout=self.connection_timeout)
        return await playwright.chromium.launch(headless=self.headless)

    def _redact_connection_error(self, error: Exception) -> str:
        message = str(error)
        for secret_name in ("browserless_api_key", "browserstack_access_key"):
            secret = getattr(self, secret_name, None)
            if secret:
                message = message.replace(secret, "***")
        return re.sub(r"([?&]token=)[^&\s]+", r"\1***", message)

    async def _ensure_session(self) -> _BrowserSession:
        if self._session and self._session.browser.is_connected():
            return self._session

        from playwright.async_api import async_playwright

        playwright = browser = context = None
        try:
            playwright = await async_playwright().start()
            browser = await self._connect(playwright)
            context = await browser.new_context(viewport=self.viewport, user_agent=self.user_agent)
            session = _BrowserSession(str(uuid.uuid4()), playwright, browser, context)
            self._session = session
            context.on("page", lambda page: self._attach_page(session, page, make_current=True))
            page = await context.new_page()
            self._attach_page(session, page, make_current=True)
            if self.browser_backend in {"browserless", "browserstack"}:
                session.keepalive_task = asyncio.create_task(self._keepalive_loop(session))
            return session
        except Exception as exc:
            if context is not None:
                with contextlib.suppress(Exception):
                    await context.close()
            if browser is not None:
                with contextlib.suppress(Exception):
                    await browser.close()
            if playwright is not None:
                with contextlib.suppress(Exception):
                    await playwright.stop()
            self._session = None
            raise RuntimeError(self._redact_connection_error(exc)) from None

    async def _keepalive_loop(self, session: _BrowserSession) -> None:
        while self._session is session:
            try:
                page = self._current_page(session)
                await page.evaluate("() => undefined")
                await asyncio.sleep(self.keepalive_interval)
            except asyncio.CancelledError:
                return
            except Exception:
                await asyncio.sleep(5)

    def _attach_page(self, session: _BrowserSession, page: Any, *, make_current: bool) -> _PageState:
        existing = session.pages.get(id(page))
        if existing:
            if make_current:
                session.current_page = page
            return existing

        state = _PageState(page)
        session.pages[id(page)] = state
        if make_current:
            session.current_page = page

        def on_console(message: Any) -> None:
            entry = {
                "type": message.type,
                "text": message.text,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "location": message.location,
            }
            state.console_all.append(entry)
            state.console_recent.append(entry)

        def on_page_error(error: Any) -> None:
            entry = {
                "type": "error",
                "text": f"Page Error: {error}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "location": None,
            }
            state.console_all.append(entry)
            state.console_recent.append(entry)

        def on_request(request: Any) -> None:
            state.network.append(
                {
                    "request": request,
                    "response": None,
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                    "status": None,
                    "status_text": None,
                    "failure": None,
                }
            )

        def on_response(response: Any) -> None:
            record = self._request_record(state, response.request)
            if record is not None:
                record["response"] = response
                record["status"] = response.status
                record["status_text"] = response.status_text

        def on_request_failed(request: Any) -> None:
            record = self._request_record(state, request)
            if record is not None:
                record["failure"] = request.failure

        def on_dialog(dialog: Any) -> None:
            session.dialog = dialog
            session.modal_event.set()

        def on_file_chooser(file_chooser: Any) -> None:
            session.file_chooser = file_chooser
            session.modal_event.set()

        def on_close() -> None:
            session.pages.pop(id(page), None)
            if session.current_page is page:
                open_pages = self._open_pages(session)
                session.current_page = open_pages[-1] if open_pages else None

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)
        page.on("dialog", on_dialog)
        page.on("filechooser", on_file_chooser)
        page.on("close", on_close)
        return state

    @staticmethod
    def _request_record(state: _PageState, request: Any) -> Optional[dict[str, Any]]:
        return next((record for record in reversed(state.network) if record["request"] is request), None)

    @staticmethod
    def _open_pages(session: _BrowserSession) -> list[Any]:
        return [state.page for state in session.pages.values() if not state.page.is_closed()]

    def _current_page(self, session: _BrowserSession) -> Any:
        if session.current_page is not None and not session.current_page.is_closed():
            return session.current_page
        pages = self._open_pages(session)
        if not pages:
            raise RuntimeError("The browser session has no open tabs.")
        session.current_page = pages[-1]
        return session.current_page

    def _page_state(self, session: _BrowserSession, page: Optional[Any] = None) -> _PageState:
        page = page or self._current_page(session)
        return session.pages[id(page)]

    @staticmethod
    def _modal_description(session: _BrowserSession) -> Optional[dict[str, Any]]:
        if session.dialog is not None:
            return {
                "type": "dialog",
                "dialog_type": session.dialog.type,
                "message": session.dialog.message,
            }
        if session.file_chooser is not None:
            return {"type": "file_chooser", "multiple": session.file_chooser.is_multiple()}
        return None

    def _assert_no_modal(self, session: _BrowserSession) -> None:
        modal = self._modal_description(session)
        if modal:
            handler = "browser_handle_dialog" if modal["type"] == "dialog" else "browser_file_upload"
            raise RuntimeError(f"A {modal['type']} is waiting. Call {handler} before using another browser action.")

    async def _target_locator(self, page: Any, target: str) -> Any:
        locator = page.locator(f"aria-ref={target}" if _REF_PATTERN.fullmatch(target) else target)
        count = await locator.count()
        if count == 0:
            if _REF_PATTERN.fullmatch(target):
                raise ValueError(f"Ref {target} is stale or missing. Capture a fresh browser_snapshot and retry.")
            raise ValueError(f'"{target}" does not match any elements.')
        if count > 1:
            raise ValueError(f'"{target}" matches {count} elements. Use a snapshot ref or a unique selector.')
        return locator

    async def _snapshot(self, page: Any, *, target: Optional[str] = None, depth: Optional[int] = None) -> str:
        if target:
            locator = await self._target_locator(page, target)
            return await locator.aria_snapshot(mode="ai", depth=depth, timeout=self.action_timeout)
        return await page.aria_snapshot(mode="ai", depth=depth, timeout=self.action_timeout)

    async def _state_result(
        self,
        session: _BrowserSession,
        *,
        include_snapshot: bool = True,
        target: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> dict[str, Any]:
        page = self._current_page(session)
        state = self._page_state(session, page)
        modal = self._modal_description(session)
        if modal:
            return {
                "session_id": session.session_id,
                "url": page.url,
                "title": state.last_title,
                "modal": modal,
            }
        state.last_title = await page.title()
        result: dict[str, Any] = {
            "session_id": session.session_id,
            "url": page.url,
            "title": state.last_title,
        }
        if include_snapshot:
            result["snapshot"] = await self._snapshot(page, target=target, depth=depth)
        return result

    async def _perform_action(
        self, session: _BrowserSession, action: Callable[[Any], Awaitable[Any]]
    ) -> dict[str, Any]:
        self._assert_no_modal(session)
        page = self._current_page(session)
        existing_pages = {id(candidate) for candidate in self._open_pages(session)}
        session.modal_event.clear()
        action_task = asyncio.ensure_future(action(page))
        modal_task = asyncio.create_task(session.modal_event.wait())
        done, _ = await asyncio.wait({action_task, modal_task}, return_when=asyncio.FIRST_COMPLETED)

        if action_task in done and not session.modal_event.is_set():
            # Playwright may resolve a click just before dispatching its
            # filechooser callback. Give modal delivery a short grace period so
            # the caller reliably receives modal state instead of a snapshot.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(modal_task), timeout=0.05)

        if session.modal_event.is_set():
            modal_task.cancel()
            if not action_task.done():
                session.pending_action = action_task
            else:
                await action_task
            return await self._state_result(session, include_snapshot=False)

        modal_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await modal_task
        action_result = await action_task
        new_pages = [candidate for candidate in self._open_pages(session) if id(candidate) not in existing_pages]
        if new_pages:
            session.current_page = new_pages[-1]
            with contextlib.suppress(Exception):
                await session.current_page.wait_for_load_state("domcontentloaded", timeout=5000)
        result = await self._state_result(session)
        if action_result is not None:
            result["result"] = action_result
        return result

    async def navigate(self, url: str) -> dict[str, Any]:
        session = await self._ensure_session()
        self._assert_no_modal(session)
        page = self._current_page(session)
        state = self._page_state(session, page)
        state.console_recent.clear()
        state.network.clear()
        await page.goto(self._normalize_url(url), wait_until="domcontentloaded", timeout=self.navigation_timeout)
        return await self._state_result(session)

    async def snapshot(self, target: Optional[str] = None, depth: Optional[int] = None) -> dict[str, Any]:
        session = await self._ensure_session()
        return await self._state_result(session, target=target, depth=depth)

    async def find(self, *, text: Optional[str] = None, regex: Optional[str] = None) -> dict[str, Any]:
        if bool(text) == bool(regex):
            raise ValueError("Provide exactly one of text or regex.")
        session = await self._ensure_session()
        page = self._current_page(session)
        lines = (await self._snapshot(page)).splitlines()
        if regex:
            literal = re.fullmatch(r"/(.*)/([a-zA-Z]*)", regex)
            pattern = literal.group(1) if literal else regex
            flags = re.IGNORECASE if literal and "i" in literal.group(2) else 0
            compiled = re.compile(pattern, flags)
            indexes = [index for index, line in enumerate(lines) if compiled.search(line)]
            query = regex
        else:
            needle = (text or "").lower()
            indexes = [index for index, line in enumerate(lines) if needle in line.lower()]
            query = text
        snippets: list[str] = []
        for index in indexes:
            snippet = "\n".join(lines[max(0, index - 3) : min(len(lines), index + 4)])
            if snippet not in snippets:
                snippets.append(snippet)
        return {"query": query, "match_count": len(indexes), "matches": snippets}

    async def click(
        self,
        target: str,
        *,
        double_click: bool = False,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            locator = await self._target_locator(page, target)
            options = {"button": button, "modifiers": modifiers or [], "timeout": self.action_timeout}
            if double_click:
                await locator.dblclick(**options)
            else:
                await locator.click(**options)

        return await self._perform_action(session, action)

    async def type_text(self, target: str, text: str, *, submit: bool = False, slowly: bool = False) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            locator = await self._target_locator(page, target)
            if slowly:
                await locator.press_sequentially(text, timeout=self.action_timeout)
            else:
                await locator.fill(text, timeout=self.action_timeout)
            if submit:
                await locator.press("Enter", timeout=self.action_timeout)

        return await self._perform_action(session, action)

    async def fill_form(self, fields: list[dict[str, Any]]) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            for form_field in fields:
                locator = await self._target_locator(page, str(form_field["target"]))
                field_type = str(form_field["type"])
                value = str(form_field["value"])
                if field_type in {"textbox", "slider"}:
                    await locator.fill(value, timeout=self.action_timeout)
                elif field_type in {"checkbox", "radio"}:
                    if value.lower() not in {"true", "false"}:
                        raise ValueError(f"{field_type} value must be 'true' or 'false'.")
                    await locator.set_checked(value.lower() == "true", timeout=self.action_timeout)
                elif field_type == "combobox":
                    await locator.select_option(label=value, timeout=self.action_timeout)
                else:
                    raise ValueError(f"Unsupported form field type: {field_type}")

        return await self._perform_action(session, action)

    async def select_option(self, target: str, values: list[str]) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> list[str]:
            return await (await self._target_locator(page, target)).select_option(values, timeout=self.action_timeout)

        return await self._perform_action(session, action)

    async def hover(self, target: str) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            await (await self._target_locator(page, target)).hover(timeout=self.action_timeout)

        return await self._perform_action(session, action)

    async def drag(self, start_target: str, end_target: str) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            start, end = await asyncio.gather(
                self._target_locator(page, start_target), self._target_locator(page, end_target)
            )
            await start.drag_to(end, timeout=self.action_timeout)

        return await self._perform_action(session, action)

    async def drop(
        self, target: str, *, files: Optional[list[dict[str, Any]]] = None, data: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        if not files and not data:
            raise ValueError("Provide files or data to drop.")
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            locator = await self._target_locator(page, target)
            encoded_files = [
                {
                    "name": item["name"],
                    "mime_type": item["mime_type"],
                    "base64": base64.b64encode(item["buffer"]).decode("ascii"),
                }
                for item in files or []
            ]
            await locator.evaluate(
                """(element, payload) => {
                    const transfer = new DataTransfer();
                    for (const [type, value] of Object.entries(payload.data || {}))
                        transfer.setData(type, value);
                    for (const file of payload.files || []) {
                        const binary = atob(file.base64);
                        const bytes = Uint8Array.from(binary, ch => ch.charCodeAt(0));
                        transfer.items.add(new File([bytes], file.name, { type: file.mime_type }));
                    }
                    for (const type of ['dragenter', 'dragover', 'drop'])
                        element.dispatchEvent(new DragEvent(type, { bubbles: true, dataTransfer: transfer }));
                }""",
                {"files": encoded_files, "data": data or {}},
            )

        return await self._perform_action(session, action)

    async def press_key(self, key: str) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            await page.keyboard.press(key)

        return await self._perform_action(session, action)

    async def navigate_back(self) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            await page.go_back(wait_until="domcontentloaded", timeout=self.navigation_timeout)

        return await self._perform_action(session, action)

    async def wait_for(
        self, *, time: Optional[float] = None, text: Optional[str] = None, text_gone: Optional[str] = None
    ) -> dict[str, Any]:
        if time is None and text is None and text_gone is None:
            raise ValueError("Provide time, text, or text_gone.")
        session = await self._ensure_session()
        self._assert_no_modal(session)
        page = self._current_page(session)
        if time is not None:
            await asyncio.sleep(min(max(time, 0), 30))
        if text_gone is not None:
            await page.get_by_text(text_gone).first.wait_for(state="hidden", timeout=self.action_timeout)
        if text is not None:
            await page.get_by_text(text).first.wait_for(state="visible", timeout=self.action_timeout)
        return await self._state_result(session)

    async def resize(self, width: int, height: int) -> dict[str, Any]:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        session = await self._ensure_session()

        async def action(page: Any) -> None:
            await page.set_viewport_size({"width": width, "height": height})

        return await self._perform_action(session, action)

    async def tabs(self, action: str, *, index: Optional[int] = None, url: Optional[str] = None) -> dict[str, Any]:
        if action not in {"list", "new", "close", "select"}:
            raise ValueError("action must be list, new, close, or select.")
        session = await self._ensure_session()
        self._assert_no_modal(session)
        pages = self._open_pages(session)
        if action == "new":
            page = await session.context.new_page()
            self._attach_page(session, page, make_current=True)
            if url:
                state = self._page_state(session, page)
                state.console_recent.clear()
                state.network.clear()
                await page.goto(
                    self._normalize_url(url), wait_until="domcontentloaded", timeout=self.navigation_timeout
                )
        elif action in {"close", "select"}:
            if action == "select" and index is None:
                raise ValueError("index is required when selecting a tab.")
            selected_index = index if index is not None else pages.index(self._current_page(session))
            if selected_index < 0 or selected_index >= len(pages):
                raise ValueError(f"Tab index {selected_index} is out of range.")
            if action == "select":
                session.current_page = pages[selected_index]
                await session.current_page.bring_to_front()
            else:
                await pages[selected_index].close()
                remaining = self._open_pages(session)
                if not remaining:
                    closed_id = session.session_id
                    await self.close()
                    return {"session_id": closed_id, "tabs": []}
                session.current_page = remaining[min(selected_index, len(remaining) - 1)]

        pages = self._open_pages(session)
        tab_results = []
        for tab_index, page in enumerate(pages):
            state = self._page_state(session, page)
            if not self._modal_description(session) or page is not session.current_page:
                with contextlib.suppress(Exception):
                    state.last_title = await page.title()
            tab_results.append(
                {
                    "index": tab_index,
                    "title": state.last_title,
                    "url": page.url,
                    "current": page is session.current_page,
                }
            )
        result: dict[str, Any] = {"session_id": session.session_id, "tabs": tab_results}
        if action != "list":
            result.update(await self._state_result(session))
        return result

    async def handle_dialog(self, accept: bool, prompt_text: Optional[str] = None) -> dict[str, Any]:
        session = await self._ensure_session()
        dialog = session.dialog
        if dialog is None:
            raise RuntimeError("There is no dialog to handle.")
        try:
            if accept:
                await dialog.accept(prompt_text)
            else:
                await dialog.dismiss()
        finally:
            session.dialog = None
            session.modal_event.clear()
        await self._finish_pending_action(session)
        return await self._state_result(session)

    async def file_upload(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        session = await self._ensure_session()
        chooser = session.file_chooser
        if chooser is None:
            raise RuntimeError("There is no file chooser to handle.")
        payloads = [{"name": item["name"], "mimeType": item["mime_type"], "buffer": item["buffer"]} for item in files]
        try:
            await chooser.set_files(payloads)
        finally:
            session.file_chooser = None
            session.modal_event.clear()
        await self._finish_pending_action(session)
        return await self._state_result(session)

    async def _finish_pending_action(self, session: _BrowserSession) -> None:
        task = session.pending_action
        session.pending_action = None
        if task is not None:
            await asyncio.wait_for(task, timeout=self.navigation_timeout / 1000)

    async def console_messages(self, level: str = "info", *, all_messages: bool = False) -> dict[str, Any]:
        if level not in _CONSOLE_LEVELS:
            raise ValueError("level must be error, warning, info, or debug.")
        session = await self._ensure_session()
        state = self._page_state(session)
        source = state.console_all if all_messages else state.console_recent

        def severity(entry: dict[str, Any]) -> str:
            kind = entry["type"]
            if kind in {"assert", "error"}:
                return "error"
            if kind == "warning":
                return "warning"
            if kind == "debug":
                return "debug"
            return "info"

        messages = [entry for entry in source if _CONSOLE_LEVELS[severity(entry)] <= _CONSOLE_LEVELS[level]]
        return {
            "messages": messages,
            "total": len(source),
            "errors": sum(severity(entry) == "error" for entry in source),
            "warnings": sum(severity(entry) == "warning" for entry in source),
        }

    async def network_requests(
        self, *, include_static: bool = False, filter_pattern: Optional[str] = None
    ) -> dict[str, Any]:
        session = await self._ensure_session()
        records = list(self._page_state(session).network)
        compiled = re.compile(filter_pattern) if filter_pattern else None
        results = []
        for index, record in enumerate(records, 1):
            if not include_static and record["resource_type"] in _STATIC_RESOURCE_TYPES:
                continue
            if compiled and not compiled.search(record["url"]):
                continue
            results.append(
                {
                    "index": index,
                    "method": record["method"],
                    "url": record["url"],
                    "resource_type": record["resource_type"],
                    "status": record["status"],
                    "status_text": record["status_text"],
                    "failure": record["failure"],
                }
            )
        return {"requests": results, "total": len(results)}

    async def network_request(self, index: int, part: Optional[str] = None) -> dict[str, Any]:
        session = await self._ensure_session()
        records = list(self._page_state(session).network)
        if index <= 0 or index > len(records):
            raise ValueError(f"Network request index {index} is out of range.")
        record = records[index - 1]
        request = record["request"]
        response = record["response"]
        result: dict[str, Any] = {
            "index": index,
            "method": record["method"],
            "url": record["url"],
            "status": record["status"],
            "failure": record["failure"],
        }
        valid_parts = {"request_headers", "request_body", "response_headers", "response_body"}
        if part is not None and part not in valid_parts:
            raise ValueError(f"part must be one of: {', '.join(sorted(valid_parts))}.")
        if part in {None, "request_headers"}:
            result["request_headers"] = await request.all_headers()
        if part in {None, "request_body"}:
            result["request_body"] = request.post_data
        if response is not None and part in {None, "response_headers"}:
            result["response_headers"] = await response.all_headers()
        if response is not None and part in {None, "response_body"}:
            try:
                body = (await response.body()).decode("utf-8", errors="replace")
                result["response_body"] = body[:_MAX_NETWORK_BODY_CHARS]
                result["response_body_truncated"] = len(body) > _MAX_NETWORK_BODY_CHARS
            except Exception as exc:
                result["response_body_error"] = str(exc)
        return result

    async def screenshot(
        self,
        *,
        target: Optional[str] = None,
        image_type: str = "png",
        full_page: bool = False,
        scale: str = "css",
    ) -> dict[str, Any]:
        if image_type not in {"png", "jpeg"}:
            raise ValueError("image_type must be png or jpeg.")
        if scale not in {"css", "device"}:
            raise ValueError("scale must be css or device.")
        session = await self._ensure_session()
        self._assert_no_modal(session)
        page = self._current_page(session)
        if target and full_page:
            raise ValueError("full_page cannot be combined with an element target.")
        if target:
            image = await (await self._target_locator(page, target)).screenshot(type=image_type, scale=scale)
        else:
            image = await page.screenshot(type=image_type, full_page=full_page, scale=scale)
        state = self._page_state(session)
        state.last_title = await page.title()
        return {
            "url": page.url,
            "title": state.last_title,
            "image": base64.b64encode(image).decode("ascii"),
            "media_type": f"image/{image_type}",
        }

    async def evaluate(self, function: str, target: Optional[str] = None) -> dict[str, Any]:
        session = await self._ensure_session()

        async def action(page: Any) -> Any:
            if target:
                return await (await self._target_locator(page, target)).evaluate(function)
            return await page.evaluate(function)

        result = await self._perform_action(session, action)
        if "result" in result:
            serialized = json.dumps(result["result"], ensure_ascii=False, default=str)
            if len(serialized) > _MAX_NETWORK_BODY_CHARS:
                result["result"] = serialized[:_MAX_NETWORK_BODY_CHARS]
                result["result_truncated"] = True
        return result

    async def close(self) -> Optional[str]:
        session = self._session
        if session is None:
            return None
        self._session = None
        if session.pending_action and not session.pending_action.done():
            session.pending_action.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.pending_action
        if session.keepalive_task:
            session.keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.keepalive_task
        with contextlib.suppress(Exception):
            await session.context.close()
        with contextlib.suppress(Exception):
            await session.browser.close()
        with contextlib.suppress(Exception):
            await session.playwright.stop()
        return session.session_id

    async def cleanup_all_browsers(self) -> None:
        await self.close()


def file_payload(path: str, content: bytes) -> dict[str, Any]:
    """Build the transport-neutral file payload consumed by upload/drop operations."""
    return {
        "name": os.path.basename(path),
        "mime_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
        "buffer": content,
    }
