import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.tool_backend.browser_tool import BROWSER_TOOL_SCHEMAS, BrowserTool
from kolega_code.config import AgentConfig
from kolega_code.llm.models import ToolDefinition
from kolega_code.services.file_system import LocalFileSystem


@pytest.fixture
def browser_manager():
    manager = MagicMock()
    manager.session_id = None
    manager.navigate = AsyncMock(
        return_value={
            "session_id": "session-1",
            "url": "https://example.com",
            "title": "Example",
            "snapshot": '- heading "Example" [ref=e2]',
        }
    )
    manager.click = AsyncMock()
    manager.screenshot = AsyncMock()
    manager.close = AsyncMock()
    manager.cleanup_all_browsers = AsyncMock()
    return manager


@pytest.fixture
def browser_tool(tmp_path, browser_manager):
    caller = MagicMock()
    caller.agent_name = "test-agent"
    return BrowserTool(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(),
        config=MagicMock(spec=AgentConfig),
        caller=caller,
        filesystem=LocalFileSystem(root_path=tmp_path),
        browser_manager=browser_manager,
    )


@pytest.mark.asyncio
async def test_navigate_formats_snapshot_and_broadcasts_launch(browser_tool, browser_manager):
    result = await browser_tool.browser_navigate("https://example.com")

    browser_manager.navigate.assert_awaited_once_with("https://example.com")
    assert result == "\n".join(
        [
            "## Page",
            "- URL: https://example.com",
            "- Title: Example",
            "",
            "## Snapshot",
            "```yaml",
            '- heading "Example" [ref=e2]',
            "```",
        ]
    )
    browser_tool.connection_manager.broadcast_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_click_forwards_snake_case_options(browser_tool, browser_manager):
    browser_manager.click.return_value = {
        "session_id": "session-1",
        "url": "https://example.com",
        "title": "Example",
        "snapshot": '- button "Saved" [ref=e3]',
    }

    await browser_tool.browser_click("e2", double_click=True, button="right", modifiers=["Shift"])

    browser_manager.click.assert_awaited_once_with("e2", double_click=True, button="right", modifiers=["Shift"])


@pytest.mark.asyncio
async def test_scroll_forwards_one_movement_and_reports_the_new_position(browser_tool, browser_manager):
    browser_manager.scroll = AsyncMock(
        return_value={
            "session_id": "session-1",
            "url": "https://example.com",
            "title": "Example",
            "snapshot": '- heading "Example" [ref=e2]',
            "scroll_x": 0,
            "scroll_y": 4_800,
            "content_height": 20_176,
            "viewport_height": 767,
        }
    )

    result = await browser_tool.browser_scroll(by_pages=3)

    browser_manager.scroll.assert_awaited_once_with(target=None, x=None, y=None, by_pages=3)
    # Reporting where the viewport landed is what lets a scroll-then-snapshot
    # loop tell progress from having reached the end of the page.
    assert "- Scroll position: y=4800 of 20176 px" in result
    assert '- heading "Example" [ref=e2]' in result


@pytest.mark.asyncio
async def test_page_format_omits_scroll_position_when_absent(browser_tool, browser_manager):
    result = await browser_tool.browser_navigate("https://example.com")

    assert "Scroll position" not in result


def _coverage(**overrides):
    coverage = {
        "candidates": 39_501,
        "complete": False,
        "content_height": 20_176,
        "emitted": 639,
        "reason": "char_cap",
        "scroll_y": 588,
        "viewport_height": 767,
        "visited": 39_501,
    }
    coverage.update(overrides)
    return coverage


@pytest.mark.asyncio
async def test_incomplete_coverage_is_reported_with_a_remedy(browser_tool, browser_manager):
    browser_manager.navigate.return_value = {
        "session_id": "session-1",
        "url": "https://example.com",
        "title": "Example",
        "snapshot": '- heading "Example" [ref=e2]',
        "coverage": _coverage(),
    }

    result = await browser_tool.browser_navigate("https://example.com")

    assert "Coverage: Showing 639 of 39501 page nodes (char_cap), at y=588 of 20176 px" in result
    assert "browser_scroll and snapshot again" in result


@pytest.mark.asyncio
async def test_complete_coverage_is_not_reported_at_all(browser_tool, browser_manager):
    browser_manager.navigate.return_value = {
        "session_id": "session-1",
        "url": "https://example.com",
        "title": "Example",
        "snapshot": '- heading "Example" [ref=e2]',
        "coverage": _coverage(complete=True, reason=None),
    }

    result = await browser_tool.browser_navigate("https://example.com")

    assert "Coverage:" not in result


@pytest.mark.asyncio
async def test_find_distinguishes_absence_from_incomplete_coverage(browser_tool, browser_manager):
    """A bounded search must never render as a flat absence.

    Reporting "No matches found" for text plainly on the page is the wrong answer,
    and it cost an earlier session most of its tool calls.
    """
    browser_manager.find = AsyncMock()

    browser_manager.find.return_value = {
        "query": "See more",
        "match_count": 0,
        "matches": [],
        "page_text_match": True,
        "snapshot_coverage": _coverage(),
    }
    outside = await browser_tool.browser_find(text="See more")
    assert "outside the region this snapshot covered" in outside
    assert "Coverage:" in outside

    browser_manager.find.return_value = {
        "query": "Nowhere",
        "match_count": 0,
        "matches": [],
        "page_text_match": False,
        "snapshot_coverage": _coverage(complete=True, reason=None),
    }
    absent = await browser_tool.browser_find(text="Nowhere")
    assert "does not contain it either" in absent
    assert "Coverage:" not in absent

    browser_manager.find.return_value = {
        "query": "Unknown",
        "match_count": 0,
        "matches": [],
        "page_text_match": None,
        "snapshot_coverage": _coverage(),
    }
    undetermined = await browser_tool.browser_find(text="Unknown")
    assert "not a reliable absence" in undetermined

    browser_manager.find.return_value = {
        "query": "Save",
        "match_count": 2,
        "matches": ['- button "Save" [ref=e4]'],
        "page_text_match": None,
        "snapshot_coverage": _coverage(),
    }
    hit = await browser_tool.browser_find(text="Save")
    assert "Found 2 matches" in hit
    # A hit under partial coverage still warns, because there may be more.
    assert "Coverage:" in hit


@pytest.mark.asyncio
async def test_file_upload_reads_only_through_workspace_filesystem(browser_tool, browser_manager, tmp_path):
    upload = tmp_path / "avatar.txt"
    upload.write_text("hello", encoding="utf-8")
    browser_manager.file_upload = AsyncMock(
        return_value={
            "session_id": "session-1",
            "url": "about:blank",
            "title": "",
            "snapshot": "- document",
        }
    )

    await browser_tool.browser_file_upload(["avatar.txt"])

    call = browser_manager.file_upload.await_args
    assert call is not None
    payload = call.args[0][0]
    assert payload["name"] == "avatar.txt"
    assert payload["buffer"] == b"hello"


@pytest.mark.asyncio
async def test_file_upload_rejects_path_outside_workspace(browser_tool, browser_manager, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the allowed root"):
        await browser_tool.browser_file_upload([str(outside)])

    browser_manager.file_upload.assert_not_called()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_broadcasts_only_for_live_session(browser_tool, browser_manager):
    browser_manager.close.return_value = None
    assert await browser_tool.browser_close() == "No browser session is running."
    browser_tool.connection_manager.broadcast_event.assert_not_awaited()

    browser_manager.close.return_value = "session-1"
    assert await browser_tool.browser_close() == "Browser session closed."
    browser_tool.connection_manager.broadcast_event.assert_awaited_once()


def test_browser_tool_schemas_use_snake_case_and_exclude_legacy_contract():
    assert "double_click" in BROWSER_TOOL_SCHEMAS["browser_click"]["properties"]
    assert "full_page" in BROWSER_TOOL_SCHEMAS["browser_take_screenshot"]["properties"]
    assert "text_gone" in BROWSER_TOOL_SCHEMAS["browser_wait_for"]["properties"]
    assert "doubleClick" not in BROWSER_TOOL_SCHEMAS["browser_click"]["properties"]


class TestLoopbackRefusedHint:
    """Loopback connection refusals get guidance instead of a bare net error."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:9999/",
            "http://127.0.0.1:9999/",
            "http://[::1]:9999/",
            "http://app.localhost:9999/",
            "localhost:9999",
        ],
    )
    async def test_refused_loopback_navigate_includes_hint(self, browser_tool, browser_manager, url):
        browser_manager.navigate = AsyncMock(
            side_effect=RuntimeError(f"Page.goto: net::ERR_CONNECTION_REFUSED at {url}")
        )

        with pytest.raises(RuntimeError) as exc_info:
            await browser_tool.browser_navigate(url)

        message = str(exc_info.value)
        assert "ERR_CONNECTION_REFUSED" in message
        assert "no server is listening on that port" in message
        assert "same machine as the terminal" in message

    @pytest.mark.asyncio
    async def test_refused_remote_navigate_passes_through(self, browser_tool, browser_manager):
        error = RuntimeError("Page.goto: net::ERR_CONNECTION_REFUSED at https://example.com/")
        browser_manager.navigate = AsyncMock(side_effect=error)

        with pytest.raises(RuntimeError) as exc_info:
            await browser_tool.browser_navigate("https://example.com/")

        assert exc_info.value is error
        assert "no server is listening" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_loopback_other_errors_pass_through(self, browser_tool, browser_manager):
        error = RuntimeError("Page.goto: Timeout 30000ms exceeded")
        browser_manager.navigate = AsyncMock(side_effect=error)

        with pytest.raises(RuntimeError) as exc_info:
            await browser_tool.browser_navigate("http://localhost:9999/")

        assert exc_info.value is error
        assert "no server is listening" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_refused_loopback_new_tab_includes_hint(self, browser_tool, browser_manager):
        browser_manager.tabs = AsyncMock(
            side_effect=RuntimeError("Page.goto: net::ERR_CONNECTION_REFUSED at http://127.0.0.1:9999/")
        )

        with pytest.raises(RuntimeError, match="no server is listening on that port"):
            await browser_tool.browser_tabs("new", url="http://127.0.0.1:9999/")

    legacy = {
        "launch_browser",
        "list_browsers",
        "get_browser_content",
        "get_browser_interactive_elements",
        "interact_with_browser",
        "set_browser_select_value",
        "close_browser",
    }
    assert legacy.isdisjoint(BROWSER_TOOL_SCHEMAS)


def test_browser_schema_enums_serialize_for_google():
    definition = ToolDefinition(
        name="browser_click",
        description="Click",
        parameters=[],
        input_schema=BROWSER_TOOL_SCHEMAS["browser_click"],
    )

    declarations = definition.to_google().function_declarations
    assert declarations is not None
    parameters = declarations[0].parameters
    assert parameters is not None
    assert parameters.properties is not None
    button = parameters.properties["button"]
    modifiers = parameters.properties["modifiers"]
    assert modifiers.items is not None

    assert button.enum == ["left", "right", "middle"]
    assert modifiers.items.enum == [
        "Alt",
        "Control",
        "ControlOrMeta",
        "Meta",
        "Shift",
    ]


@pytest.mark.asyncio
async def test_network_requests_report_resource_type(browser_tool, browser_manager):
    """The Chrome and Playwright backends both supply resource_type; the shared
    formatter used to drop it, so callers could not tell a fetch from a stylesheet."""
    browser_manager.network_requests = AsyncMock(
        return_value={
            "requests": [
                {
                    "index": 1,
                    "method": "GET",
                    "url": "http://127.0.0.1:8931/api/ping",
                    "status": 200,
                    "failure": None,
                    "resource_type": "xmlhttprequest",
                },
                {
                    "index": 2,
                    "method": "GET",
                    "url": "http://127.0.0.1:8931/fixture.css",
                    "status": 200,
                    "failure": None,
                    "resource_type": "stylesheet",
                },
            ]
        }
    )

    output = await browser_tool.browser_network_requests(include_static=True)

    assert "1: GET http://127.0.0.1:8931/api/ping [xmlhttprequest] => 200" in output
    assert "2: GET http://127.0.0.1:8931/fixture.css [stylesheet] => 200" in output


@pytest.mark.asyncio
async def test_network_requests_omit_missing_resource_type(browser_tool, browser_manager):
    """A backend that omits the field must still produce a clean line."""
    browser_manager.network_requests = AsyncMock(
        return_value={
            "requests": [
                {"index": 1, "method": "POST", "url": "https://example.com/x", "status": None, "failure": "blocked"}
            ]
        }
    )

    output = await browser_tool.browser_network_requests()

    assert "1: POST https://example.com/x => blocked" in output
    assert "[" not in output.split("\n")[-1]


def test_regex_schema_documents_its_constraints():
    find_regex = BROWSER_TOOL_SCHEMAS["browser_find"]["properties"]["regex"]["description"]
    assert "alternation" in find_regex
    assert "{n,m}" in find_regex


class TestPaddedArgumentsAreTreatedAsOmissions:
    """Models that fill in every declared property must still be able to work.

    Documenting "pass null, never 0 or an empty string" was tried and did not work:
    a model that pads has no way to stop, so it retries the identical rejected call
    until it gives up. One run lost its whole first phase to thirteen consecutive
    `browser_tabs {"action":"list","index":0,"url":""}` calls answered with "url is
    invalid". Padding that cannot express a request is therefore treated as the
    omission it was meant to be, here at the agent-facing boundary; the wire
    protocol stays exact.
    """

    @pytest.mark.asyncio
    async def test_listing_tabs_ignores_padded_index_and_url(self, browser_tool, browser_manager):
        browser_manager.tabs = AsyncMock(return_value={"tabs": []})

        await browser_tool.browser_tabs("list", index=0, url="")

        browser_manager.tabs.assert_awaited_once_with("list", index=None, url=None)

    @pytest.mark.asyncio
    async def test_a_new_tab_ignores_padded_index_and_reads_an_empty_url_as_blank(self, browser_tool, browser_manager):
        browser_manager.tabs = AsyncMock(return_value={"tabs": []})

        await browser_tool.browser_tabs("new", index=0, url="   ")

        browser_manager.tabs.assert_awaited_once_with("new", index=None, url=None)

    @pytest.mark.asyncio
    async def test_selecting_a_tab_keeps_index_zero_and_drops_the_inapplicable_url(self, browser_tool, browser_manager):
        """0 is a real tab index, so only url is inapplicable to select."""
        browser_manager.tabs = AsyncMock(return_value={"tabs": []})

        await browser_tool.browser_tabs("select", index=0, url="")

        browser_manager.tabs.assert_awaited_once_with("select", index=0, url=None)

    @pytest.mark.asyncio
    async def test_scroll_by_pages_ignores_padded_offsets(self, browser_tool, browser_manager):
        browser_manager.scroll = AsyncMock(return_value={"url": "https://example.com", "title": "Example"})

        await browser_tool.browser_scroll(target="", x=0, y=0, by_pages=2)

        browser_manager.scroll.assert_awaited_once_with(target=None, x=None, y=None, by_pages=2)

    @pytest.mark.asyncio
    async def test_scroll_to_a_target_ignores_padded_offsets_and_page_count(self, browser_tool, browser_manager):
        browser_manager.scroll = AsyncMock(return_value={"url": "https://example.com", "title": "Example"})

        await browser_tool.browser_scroll(target="#footer", x=0, y=0, by_pages=0)

        browser_manager.scroll.assert_awaited_once_with(target="#footer", x=None, y=None, by_pages=None)

    @pytest.mark.asyncio
    async def test_scrolling_to_the_top_stays_a_real_request(self, browser_tool, browser_manager):
        """x/y of 0 alone means the top of the page, so it must not be dropped."""
        browser_manager.scroll = AsyncMock(return_value={"url": "https://example.com", "title": "Example"})

        await browser_tool.browser_scroll(x=0, y=0)

        browser_manager.scroll.assert_awaited_once_with(target=None, x=0, y=0, by_pages=None)

    @pytest.mark.asyncio
    async def test_two_real_movements_are_still_rejected_and_named(self, browser_tool, browser_manager):
        browser_manager.scroll = AsyncMock()

        with pytest.raises(ValueError, match=r"received target='#footer', by_pages=2"):
            await browser_tool.browser_scroll(target="#footer", by_pages=2)
        browser_manager.scroll.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_movement_at_all_explains_the_three_shapes(self, browser_tool, browser_manager):
        browser_manager.scroll = AsyncMock()

        with pytest.raises(ValueError, match="Provide exactly one of target, by_pages, or x/y"):
            await browser_tool.browser_scroll(target="")
        browser_manager.scroll.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_find_and_snapshot_drop_padded_strings_and_zero_depth(self, browser_tool, browser_manager):
        browser_manager.find = AsyncMock(return_value={"query": "Sign in", "matches": [], "match_count": 0})
        browser_manager.snapshot = AsyncMock(return_value={"url": "https://example.com", "title": "Example"})

        await browser_tool.browser_find(text="Sign in", regex="")
        await browser_tool.browser_snapshot(target="", depth=0)

        browser_manager.find.assert_awaited_once_with(text="Sign in", regex=None)
        browser_manager.snapshot.assert_awaited_once_with(target=None, depth=None)

    @pytest.mark.asyncio
    async def test_wait_for_drops_a_zero_timeout_beside_a_real_condition(self, browser_tool, browser_manager):
        browser_manager.wait_for = AsyncMock(return_value={"url": "https://example.com", "title": "Example"})

        await browser_tool.browser_wait_for(time=0, text="Ready", text_gone="")
        browser_manager.wait_for.assert_awaited_once_with(time=None, text="Ready", text_gone=None)

        # On its own a zero wait is a legitimate no-op and stays a request.
        browser_manager.wait_for.reset_mock()
        await browser_tool.browser_wait_for(time=0)
        browser_manager.wait_for.assert_awaited_once_with(time=0, text=None, text_gone=None)

    @pytest.mark.asyncio
    async def test_screenshot_and_click_fall_back_to_documented_defaults(self, browser_tool, browser_manager):
        browser_manager.click = AsyncMock(return_value={"url": "https://example.com", "title": "Example"})

        await browser_tool.browser_take_screenshot(target="", image_type="", scale="")
        await browser_tool.browser_click("e2", button="")

        browser_manager.screenshot.assert_awaited_once_with(target=None, image_type="png", full_page=False, scale="css")
        browser_manager.click.assert_awaited_once_with("e2", double_click=False, button="left", modifiers=None)
