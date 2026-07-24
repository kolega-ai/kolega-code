import json
import urllib.parse
from pathlib import Path
from typing import Any, Optional, Union

from kolega_code.config import AgentConfig
from kolega_code.events import AgentEvent
from kolega_code.services.browser import PlaywrightBrowserManager, file_payload
from .base_tool import BaseTool


_TARGET = {
    "type": "string",
    "description": "Exact element ref from browser_snapshot (for example e12), or a unique selector.",
}

_LOOPBACK_REFUSED_HINT = (
    "The connection was refused — no server is listening on that port. The browser runs on the "
    "same machine as the terminal; if the server was started backgrounded (`&`) in an earlier "
    "terminal command it has since exited. Restart it with exec_command background=true (or "
    "`nohup`), confirm it answers with curl, then retry."
)


def _is_loopback_url(url: str) -> bool:
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    host = (urllib.parse.urlsplit(candidate).hostname or "").lower()
    return host == "localhost" or host.endswith(".localhost") or host == "::1" or host.startswith("127.")


def _augment_loopback_refused(url: Optional[str], exc: Exception) -> Exception:
    """Append localhost troubleshooting guidance to loopback connection refusals."""
    if url and "ERR_CONNECTION_REFUSED" in str(exc) and _is_loopback_url(url):
        return RuntimeError(f"{exc}\n\n{_LOOPBACK_REFUSED_HINT}")
    return exc


def _schema(properties: dict[str, Any], required: Optional[list[str]] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


BROWSER_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "browser_navigate": _schema(
        {"url": {"type": "string", "description": "HTTP or HTTPS URL to navigate to."}}, ["url"]
    ),
    "browser_navigate_back": _schema({}),
    "browser_snapshot": _schema(
        {
            "target": _TARGET,
            "depth": {"type": "integer", "description": "Optional maximum accessibility-tree depth."},
        }
    ),
    "browser_find": _schema(
        {
            "text": {"type": "string", "description": "Case-insensitive text to find in the snapshot."},
            "regex": {"type": "string", "description": "Regular expression to find in the snapshot."},
        }
    ),
    "browser_wait_for": _schema(
        {
            "time": {"type": "number", "description": "Seconds to wait, capped at 30."},
            "text": {"type": "string", "description": "Text to wait for until visible."},
            "text_gone": {"type": "string", "description": "Text to wait for until hidden."},
        }
    ),
    "browser_resize": _schema(
        {
            "width": {"type": "integer", "description": "Viewport width in CSS pixels."},
            "height": {"type": "integer", "description": "Viewport height in CSS pixels."},
        },
        ["width", "height"],
    ),
    "browser_click": _schema(
        {
            "target": _TARGET,
            "double_click": {"type": "boolean", "description": "Perform a double click."},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "modifiers": {
                "type": "array",
                "items": {"type": "string", "enum": ["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]},
            },
        },
        ["target"],
    ),
    "browser_type": _schema(
        {
            "target": _TARGET,
            "text": {"type": "string", "description": "Text to enter."},
            "submit": {"type": "boolean", "description": "Press Enter after entering text."},
            "slowly": {"type": "boolean", "description": "Type character by character instead of filling."},
        },
        ["target", "text"],
    ),
    "browser_fill_form": _schema(
        {
            "fields": {
                "type": "array",
                "description": "Form fields to fill.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Human-readable field name."},
                        "target": _TARGET,
                        "type": {
                            "type": "string",
                            "enum": ["textbox", "checkbox", "radio", "combobox", "slider"],
                        },
                        "value": {"type": "string", "description": "Value to set."},
                    },
                    "required": ["name", "target", "type", "value"],
                },
            }
        },
        ["fields"],
    ),
    "browser_select_option": _schema(
        {
            "target": _TARGET,
            "values": {"type": "array", "items": {"type": "string"}, "description": "Option values to select."},
        },
        ["target", "values"],
    ),
    "browser_hover": _schema({"target": _TARGET}, ["target"]),
    "browser_drag": _schema({"start_target": _TARGET, "end_target": _TARGET}, ["start_target", "end_target"]),
    "browser_drop": _schema(
        {
            "target": _TARGET,
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace file paths to drop.",
            },
            "data": {
                "type": "object",
                "description": "MIME type to string data, such as text/plain or text/uri-list.",
                "additionalProperties": {"type": "string"},
            },
        },
        ["target"],
    ),
    "browser_press_key": _schema(
        {"key": {"type": "string", "description": "Key name or character, such as ArrowLeft or a."}}, ["key"]
    ),
    "browser_tabs": _schema(
        {
            "action": {"type": "string", "enum": ["list", "new", "close", "select"]},
            "index": {"type": "integer", "description": "Tab index for close or select."},
            "url": {"type": "string", "description": "Optional URL for a new tab."},
        },
        ["action"],
    ),
    "browser_handle_dialog": _schema(
        {
            "accept": {"type": "boolean", "description": "Accept rather than dismiss the dialog."},
            "prompt_text": {"type": "string", "description": "Text for a prompt dialog."},
        },
        ["accept"],
    ),
    "browser_file_upload": _schema(
        {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace file paths to upload. Use an empty list to cancel.",
            }
        },
        ["paths"],
    ),
    "browser_console_messages": _schema(
        {
            "level": {"type": "string", "enum": ["error", "warning", "info", "debug"]},
            "all_messages": {
                "type": "boolean",
                "description": "Include the whole session rather than only messages since navigation.",
            },
        }
    ),
    "browser_network_requests": _schema(
        {
            "include_static": {"type": "boolean", "description": "Include images, fonts, scripts, and styles."},
            "filter_pattern": {"type": "string", "description": "Regular expression matched against request URLs."},
        }
    ),
    "browser_network_request": _schema(
        {
            "index": {"type": "integer", "description": "1-based index from browser_network_requests."},
            "part": {
                "type": "string",
                "enum": ["request_headers", "request_body", "response_headers", "response_body"],
            },
        },
        ["index"],
    ),
    "browser_take_screenshot": _schema(
        {
            "target": _TARGET,
            "image_type": {"type": "string", "enum": ["png", "jpeg"]},
            "full_page": {"type": "boolean", "description": "Capture the full scrollable page."},
            "scale": {"type": "string", "enum": ["css", "device"]},
        }
    ),
    "browser_evaluate": _schema(
        {
            "function": {
                "type": "string",
                "description": "JavaScript function evaluated in the page, or with the target element as its argument.",
            },
            "target": _TARGET,
        },
        ["function"],
    ),
    "browser_close": _schema({}),
}


class BrowserTool(BaseTool):
    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        filesystem=None,
        browser_manager=None,
    ) -> None:
        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            config,
            caller,
            filesystem,
            browser_manager=browser_manager,
        )
        if self.browser_manager is None:
            self.browser_manager = PlaywrightBrowserManager()

    async def _broadcast_launched(self, previous_session_id: Optional[str], result: dict[str, Any]) -> None:
        session_id = result.get("session_id")
        if session_id and session_id != previous_session_id:
            event = AgentEvent(
                event_type="browser_launched", sender=self.caller.agent_name, content={"browser_id": session_id}
            )
            await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

    @staticmethod
    def _format_page(result: dict[str, Any]) -> str:
        parts = ["## Page", f"- URL: {result.get('url', 'about:blank')}", f"- Title: {result.get('title', '')}"]
        if result.get("modal"):
            parts.extend(["", "## Modal state", "```json", json.dumps(result["modal"], indent=2), "```"])
        if "result" in result:
            parts.extend(["", "## Result", "```json", json.dumps(result["result"], indent=2, default=str), "```"])
        if result.get("result_truncated"):
            parts.append("Result truncated by size.")
        if result.get("snapshot") is not None:
            parts.extend(["", "## Snapshot", "```yaml", result["snapshot"], "```"])
        return "\n".join(parts)

    def _file_payloads(self, paths: list[str]) -> list[dict[str, Any]]:
        payloads = []
        for path in paths:
            candidate = Path(path)
            resolved = candidate.resolve() if candidate.is_absolute() else (self.project_path / candidate).resolve()
            try:
                resolved.relative_to(self.project_path.resolve())
            except ValueError as exc:
                raise ValueError(f"File path is outside the allowed root: {path}") from exc
            payloads.append(file_payload(path, self.filesystem.read_bytes(path)))
        return payloads

    async def browser_navigate(self, url: str) -> str:
        previous = self.browser_manager.session_id
        try:
            result = await self.browser_manager.navigate(url)
        except Exception as exc:
            raise _augment_loopback_refused(url, exc) from exc
        await self._broadcast_launched(previous, result)
        return self._format_page(result)

    async def browser_navigate_back(self) -> str:
        return self._format_page(await self.browser_manager.navigate_back())

    async def browser_snapshot(self, target: Optional[str] = None, depth: Optional[int] = None) -> str:
        previous = self.browser_manager.session_id
        result = await self.browser_manager.snapshot(target=target, depth=depth)
        await self._broadcast_launched(previous, result)
        return self._format_page(result)

    async def browser_find(self, text: Optional[str] = None, regex: Optional[str] = None) -> str:
        result = await self.browser_manager.find(text=text, regex=regex)
        if not result["matches"]:
            return f"No matches found for {result['query']!r}."
        return f"Found {result['match_count']} matches for {result['query']!r}:\n\n" + "\n\n---\n\n".join(
            result["matches"]
        )

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None, text_gone: Optional[str] = None
    ) -> str:
        return self._format_page(await self.browser_manager.wait_for(time=time, text=text, text_gone=text_gone))

    async def browser_resize(self, width: int, height: int) -> str:
        return self._format_page(await self.browser_manager.resize(width, height))

    async def browser_click(
        self,
        target: str,
        double_click: bool = False,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> str:
        return self._format_page(
            await self.browser_manager.click(target, double_click=double_click, button=button, modifiers=modifiers)
        )

    async def browser_type(self, target: str, text: str, submit: bool = False, slowly: bool = False) -> str:
        return self._format_page(await self.browser_manager.type_text(target, text, submit=submit, slowly=slowly))

    async def browser_fill_form(self, fields: list[dict[str, Any]]) -> str:
        return self._format_page(await self.browser_manager.fill_form(fields))

    async def browser_select_option(self, target: str, values: list[str]) -> str:
        return self._format_page(await self.browser_manager.select_option(target, values))

    async def browser_hover(self, target: str) -> str:
        return self._format_page(await self.browser_manager.hover(target))

    async def browser_drag(self, start_target: str, end_target: str) -> str:
        return self._format_page(await self.browser_manager.drag(start_target, end_target))

    async def browser_drop(
        self, target: str, paths: Optional[list[str]] = None, data: Optional[dict[str, str]] = None
    ) -> str:
        return self._format_page(
            await self.browser_manager.drop(target, files=self._file_payloads(paths or []), data=data)
        )

    async def browser_press_key(self, key: str) -> str:
        return self._format_page(await self.browser_manager.press_key(key))

    async def browser_tabs(self, action: str, index: Optional[int] = None, url: Optional[str] = None) -> str:
        previous = self.browser_manager.session_id
        try:
            result = await self.browser_manager.tabs(action, index=index, url=url)
        except Exception as exc:
            raise _augment_loopback_refused(url if action == "new" else None, exc) from exc
        await self._broadcast_launched(previous, result)
        lines = ["## Open tabs"]
        for tab in result["tabs"]:
            marker = " (current)" if tab["current"] else ""
            lines.append(f"- {tab['index']}: {tab['title']} — {tab['url']}{marker}")
        if result.get("snapshot") is not None or result.get("modal"):
            lines.extend(["", self._format_page(result)])
        return "\n".join(lines)

    async def browser_handle_dialog(self, accept: bool, prompt_text: Optional[str] = None) -> str:
        return self._format_page(await self.browser_manager.handle_dialog(accept, prompt_text))

    async def browser_file_upload(self, paths: list[str]) -> str:
        return self._format_page(await self.browser_manager.file_upload(self._file_payloads(paths)))

    async def browser_console_messages(self, level: str = "info", all_messages: bool = False) -> str:
        result = await self.browser_manager.console_messages(level, all_messages=all_messages)
        header = f"Total messages: {result['total']} (Errors: {result['errors']}, Warnings: {result['warnings']})"
        messages = []
        for message in result["messages"]:
            location = message.get("location") or {}
            location_text = f" @ {location.get('url')}:{location.get('lineNumber')}" if location.get("url") else ""
            messages.append(f"[{message['type'].upper()}] {message['text']}{location_text}")
        return "\n".join([header, "", *messages])

    async def browser_network_requests(self, include_static: bool = False, filter_pattern: Optional[str] = None) -> str:
        result = await self.browser_manager.network_requests(
            include_static=include_static, filter_pattern=filter_pattern
        )
        if not result["requests"]:
            return "No matching network requests."
        lines = ["## Network requests"]
        for request in result["requests"]:
            status = request["status"] if request["status"] is not None else request["failure"] or "pending"
            lines.append(f"- {request['index']}: {request['method']} {request['url']} => {status}")
        return "\n".join(lines)

    async def browser_network_request(self, index: int, part: Optional[str] = None) -> str:
        return (
            "```json\n"
            + json.dumps(await self.browser_manager.network_request(index, part), indent=2, default=str)
            + "\n```"
        )

    async def browser_take_screenshot(
        self,
        target: Optional[str] = None,
        image_type: str = "png",
        full_page: bool = False,
        scale: str = "css",
    ) -> dict[str, Any]:
        return await self.browser_manager.screenshot(
            target=target, image_type=image_type, full_page=full_page, scale=scale
        )

    async def browser_evaluate(self, function: str, target: Optional[str] = None) -> str:
        return self._format_page(await self.browser_manager.evaluate(function, target))

    async def browser_close(self) -> str:
        session_id = await self.browser_manager.close()
        if session_id is None:
            return "No browser session is running."
        event = AgentEvent(
            event_type="browser_closed", sender=self.caller.agent_name, content={"browser_id": session_id}
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)
        return "Browser session closed."

    async def cleanup(self) -> None:
        await self.browser_manager.cleanup_all_browsers()
