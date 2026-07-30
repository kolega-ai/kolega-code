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
    "terminal command it has since exited. Restart it with exec_command background=true, "
    "confirm it answers with curl, then retry."
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


# Several models fill in *every* property a tool schema declares rather than
# omitting the ones they are not using, so an unused string arrives as "" and an
# unused number as 0. Neither can express a request, and the backends rightly
# reject them — but rejecting is a dead end: the model has no way to stop padding,
# so it retries the identical call until it gives up. (Observed: 13 consecutive
# `browser_tabs {"action":"list","index":0,"url":""}` calls rejected with "url is
# invalid", which ended the run at step one.) Treat a value that cannot express a
# request as the omission it was meant to be, at this agent-facing boundary only:
# the wire protocol below stays exact.
def _text_or_none(value: Optional[str]) -> Optional[str]:
    """Drop a string that carries no request, such as "" or "   "."""
    if value is None or not value.strip():
        return None
    return value


def _choice_or_default(value: Optional[str], default: str) -> str:
    """Fall back to the documented default for an unset enum-valued argument."""
    return value if value is not None and value.strip() else default


def _resolve_scroll(
    target: Optional[str],
    x: Optional[int],
    y: Optional[int],
    by_pages: Optional[float],
) -> tuple[Optional[str], Optional[int], Optional[int], Optional[float]]:
    """Pick the one movement the caller asked for, ignoring padded-out arguments.

    Both backends require exactly one of target, by_pages, or x/y, because three
    different movements in one call cannot be resolved. Padding is not a fourth
    movement though: a zero page count moves nothing, and zero offsets alongside a
    real target say nothing the target does not already say. Only a genuine
    conflict — two movements that would each go somewhere — is still rejected, and
    the rejection names what arrived so it can be corrected.

    ``x``/``y`` of 0 stay meaningful on their own: scrolling to the top of a page
    is an ordinary request.
    """
    target = _text_or_none(target)
    wants_target = target is not None
    wants_pages = by_pages is not None and by_pages != 0
    wants_offset = (x is not None and x != 0) or (y is not None and y != 0)
    requested = sum((wants_target, wants_pages, wants_offset))
    if requested > 1:
        supplied = ", ".join(
            part
            for part in (
                f"target={target!r}" if wants_target else "",
                f"by_pages={by_pages}" if wants_pages else "",
                f"x={x}, y={y}" if wants_offset else "",
            )
            if part
        )
        raise ValueError(
            f"Provide exactly one of target, by_pages, or x/y; received {supplied}. "
            "Pass only the movement you want and omit the others."
        )
    if requested == 1:
        if wants_target:
            return target, None, None, None
        if wants_pages:
            return None, None, None, by_pages
        return None, x, y, None
    # Nothing but zeros and blanks arrived. An explicit x/y still means "go to the
    # top"; a lone by_pages=0 is a no-op the backends accept. Otherwise there is
    # genuinely no movement to perform.
    if x is not None or y is not None:
        return None, x, y, None
    if by_pages is not None:
        return None, None, None, by_pages
    raise ValueError(
        "Provide exactly one of target, by_pages, or x/y: a selector or ref to scroll into view, "
        "a signed number of viewport heights, or an absolute x/y offset in CSS pixels."
    )


BROWSER_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "browser_navigate": _schema(
        {"url": {"type": "string", "description": "HTTP or HTTPS URL to navigate to."}}, ["url"]
    ),
    "browser_navigate_back": _schema({}),
    "browser_snapshot": _schema(
        {
            "target": _TARGET,
            "depth": {
                "type": "integer",
                "description": (
                    "Optional maximum accessibility-tree depth. This counts emitted nodes, not raw DOM "
                    "nesting, so ordinary deeply nested markup is not pruned."
                ),
            },
        }
    ),
    "browser_find": _schema(
        {
            "text": {"type": "string", "description": "Case-insensitive text to find in the snapshot."},
            "regex": {
                "type": "string",
                "description": (
                    "Regular expression to find in the snapshot. On the Chrome backend a restricted, "
                    "linear-time subset applies: literals, '.', character classes, anchors, escapes, and the "
                    "quantifiers ?, *, + and {n,m} on a single character or class. Groups '(' ')', "
                    "alternation '|', and backreferences are rejected; at most 4 quantifiers and repetition "
                    "counts up to 1000."
                ),
            },
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
        {
            "key": {
                "type": "string",
                "description": (
                    "Key name or character, such as ArrowLeft or a. PageDown, PageUp, Home, End, "
                    "ArrowDown, ArrowUp and Space scroll the page unless focus is in a text field or "
                    "the page handles the key itself."
                ),
            }
        },
        ["key"],
    ),
    "browser_scroll": _schema(
        {
            "target": {
                "type": "string",
                "description": "Scroll this element into view. Exact ref from browser_snapshot, or a unique selector.",
            },
            "x": {"type": "integer", "description": "Absolute horizontal offset in CSS pixels."},
            "y": {"type": "integer", "description": "Absolute vertical offset in CSS pixels."},
            "by_pages": {
                "type": "number",
                "description": (
                    "Scroll by this many viewport heights; negative scrolls up. Fractions are allowed, range -10 to 10."
                ),
            },
        }
    ),
    "browser_tabs": _schema(
        {
            "action": {"type": "string", "enum": ["list", "new", "close", "select"]},
            "index": {
                "type": "integer",
                "description": (
                    "Tab index, required for select and optional for close, where omitting it means the "
                    "current tab. Ignored by list and new. 0 is a real tab index."
                ),
            },
            "url": {
                "type": "string",
                "description": "URL for a new tab. Ignored by every other action; omit it for a blank tab.",
            },
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
            "filter_pattern": {
                "type": "string",
                "description": (
                    "Regular expression matched against request URLs. On the Chrome backend the same "
                    "restricted subset as browser_find applies: no groups, no alternation, at most 4 "
                    "quantifiers."
                ),
            },
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
            "full_page": {
                "type": "boolean",
                "description": (
                    "Capture the full scrollable page. A page too tall to capture legibly is clipped "
                    "from the current scroll position and reports what it left out, so scroll and "
                    "capture again to see the rest."
                ),
            },
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
        scroll_y = result.get("scroll_y")
        if isinstance(scroll_y, (int, float)):
            # Where the viewport ended up, so a scroll-then-snapshot loop can tell
            # progress from a scroll that hit the end of the page.
            content_height = result.get("content_height")
            extent = f" of {int(content_height)}" if isinstance(content_height, (int, float)) else ""
            parts.append(f"- Scroll position: y={int(scroll_y)}{extent} px")
        if result.get("modal"):
            parts.extend(["", "## Modal state", "```json", json.dumps(result["modal"], indent=2), "```"])
        if "result" in result:
            parts.extend(["", "## Result", "```json", json.dumps(result["result"], indent=2, default=str), "```"])
        if result.get("result_truncated"):
            parts.append("Result truncated by size.")
        if result.get("snapshot") is not None:
            parts.extend(["", "## Snapshot", "```yaml", result["snapshot"], "```"])
        coverage = BrowserTool._format_coverage(result.get("coverage"))
        if coverage:
            parts.extend(["", coverage])
        return "\n".join(parts)

    @staticmethod
    def _format_coverage(coverage: Any) -> str:
        """Say what a partial snapshot left out, and what to do about it.

        Only emitted when coverage is genuinely incomplete: a truncated snapshot
        that looks complete is what made an agent conclude a page was unreadable.
        """
        if not isinstance(coverage, dict) or coverage.get("complete") is not False:
            return ""
        emitted = coverage.get("emitted")
        candidates = coverage.get("candidates")
        reason = coverage.get("reason") or "a size bound"
        scope = f"Showing {emitted} of {candidates} page nodes" if emitted and candidates else "Snapshot truncated"
        position = ""
        scroll_y = coverage.get("scroll_y")
        content_height = coverage.get("content_height")
        if isinstance(scroll_y, (int, float)) and isinstance(content_height, (int, float)) and content_height:
            position = f", at y={int(scroll_y)} of {int(content_height)} px"
        return (
            f"Coverage: {scope} ({reason}){position}. Nodes nearest the viewport are shown first. "
            "Narrow the scope with browser_snapshot target=<selector>, or browser_scroll and snapshot again."
        )

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
        # depth is a positive maximum, so 0 can only be padding.
        target, depth = _text_or_none(target), depth or None
        previous = self.browser_manager.session_id
        result = await self.browser_manager.snapshot(target=target, depth=depth)
        await self._broadcast_launched(previous, result)
        return self._format_page(result)

    async def browser_find(self, text: Optional[str] = None, regex: Optional[str] = None) -> str:
        text, regex = _text_or_none(text), _text_or_none(regex)
        result = await self.browser_manager.find(text=text, regex=regex)
        query = result["query"]
        coverage = self._format_coverage(result.get("snapshot_coverage"))
        if not result["matches"]:
            # A bounded search must never be rendered as an absence. The three
            # cases are: genuinely absent, present in the page but outside the
            # region the snapshot covered, and undetermined because the search
            # itself was truncated.
            if result.get("page_text_match") is True:
                return (
                    f"No snapshot matches for {query!r}, but the page's rendered text does contain it, "
                    "so it lies outside the region this snapshot covered." + (f"\n\n{coverage}" if coverage else "")
                )
            if result.get("page_text_match") is False:
                return f"No matches found for {query!r}, and the page's rendered text does not contain it either."
            return f"No matches found for {query!r} in the covered region; coverage was incomplete, so this is " + (
                "not a reliable absence." + (f"\n\n{coverage}" if coverage else "")
            )
        found = f"Found {result['match_count']} matches for {query!r}:\n\n" + "\n\n---\n\n".join(result["matches"])
        return f"{found}\n\n{coverage}" if coverage else found

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None, text_gone: Optional[str] = None
    ) -> str:
        text, text_gone = _text_or_none(text), _text_or_none(text_gone)
        if time == 0 and (text is not None or text_gone is not None):
            # A zero-second wait alongside a real condition is padding; on its own
            # it stays a legitimate no-op wait.
            time = None
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
            await self.browser_manager.click(
                target,
                double_click=double_click,
                button=_choice_or_default(button, "left"),
                modifiers=modifiers,
            )
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

    async def browser_scroll(
        self,
        target: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        by_pages: Optional[float] = None,
    ) -> str:
        target, x, y, by_pages = _resolve_scroll(target, x, y, by_pages)
        return self._format_page(await self.browser_manager.scroll(target=target, x=x, y=y, by_pages=by_pages))

    async def browser_tabs(self, action: str, index: Optional[int] = None, url: Optional[str] = None) -> str:
        # The action alone decides which of index and url applies, so anything
        # supplied for the other one is padding and is dropped rather than
        # rejected. list takes neither; new takes url ("" means a blank tab);
        # close and select take index, where 0 is a real tab.
        if action in {"list", "new"}:
            index = None
        url = _text_or_none(url) if action == "new" else None
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
        result = await self.browser_manager.console_messages(
            _choice_or_default(level, "info"), all_messages=all_messages
        )
        header = f"Total messages: {result['total']} (Errors: {result['errors']}, Warnings: {result['warnings']})"
        messages = []
        for message in result["messages"]:
            location = message.get("location") or {}
            location_text = f" @ {location.get('url')}:{location.get('lineNumber')}" if location.get("url") else ""
            messages.append(f"[{message['type'].upper()}] {message['text']}{location_text}")
        return "\n".join([header, "", *messages])

    async def browser_network_requests(self, include_static: bool = False, filter_pattern: Optional[str] = None) -> str:
        result = await self.browser_manager.network_requests(
            include_static=include_static, filter_pattern=_text_or_none(filter_pattern)
        )
        if not result["requests"]:
            return "No matching network requests."
        lines = ["## Network requests"]
        for request in result["requests"]:
            status = request["status"] if request["status"] is not None else request["failure"] or "pending"
            resource_type = request.get("resource_type")
            kind = f" [{resource_type}]" if resource_type else ""
            lines.append(f"- {request['index']}: {request['method']} {request['url']}{kind} => {status}")
        return "\n".join(lines)

    async def browser_network_request(self, index: int, part: Optional[str] = None) -> str:
        return (
            "```json\n"
            + json.dumps(await self.browser_manager.network_request(index, _text_or_none(part)), indent=2, default=str)
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
            target=_text_or_none(target),
            image_type=_choice_or_default(image_type, "png"),
            full_page=full_page,
            scale=_choice_or_default(scale, "css"),
        )

    async def browser_evaluate(self, function: str, target: Optional[str] = None) -> str:
        return self._format_page(await self.browser_manager.evaluate(function, _text_or_none(target)))

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
