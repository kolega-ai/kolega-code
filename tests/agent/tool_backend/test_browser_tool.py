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
