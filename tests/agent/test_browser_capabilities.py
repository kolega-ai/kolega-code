from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig
from kolega_code.browser_extension.manager import (
    CHROME_EXTENSION_SUPPORTED_TOOLS,
    ChromeExtensionBrowserManager,
)
from kolega_code.cli.browser_backend import build_browser_manager
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.services.base import browser_manager_agent_lock
from kolega_code.services.browser import PlaywrightBrowserManager

_EXTENSION_ORIGIN = "chrome-extension://edihigldhbmimflgjkohkgnjefmhngdn/"


def _agent_config() -> AgentConfig:
    # A real vision-capable model: the browser agent requires image support, and
    # dispatch_browser_agent is now gated on the browser-agent model's vision.
    model = ModelConfig(
        provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-5-20250929", rate_limits=RateLimitConfig()
    )
    return AgentConfig(
        anthropic_api_key="test-key",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
    )


def _caller() -> Mock:
    caller = Mock()
    caller.agent_name = "test-agent"
    caller.sub_agent = False
    caller.gigacode_enabled = False
    caller.supports_vision = True
    caller.custom_agent_catalog = None
    caller.sub_agent_context = None
    return caller


def _collection(
    tmp_path: Path,
    browser_manager: PlaywrightBrowserManager | ChromeExtensionBrowserManager,
    *,
    browser_only: bool = False,
    dispatch: bool = False,
) -> ToolCollection:
    return ToolCollection(
        tmp_path,
        "workspace",
        str(uuid.uuid4()),
        AsyncMock(),
        _agent_config(),
        _caller(),
        browser_manager=browser_manager,
        tool_config=ToolCollectionConfig(
            browser_only=browser_only,
            include_agent_dispatch_tools=dispatch,
        ),
    )


def test_playwright_retains_the_complete_browser_tool_inventory(tmp_path: Path) -> None:
    collection = _collection(tmp_path, PlaywrightBrowserManager(), browser_only=True)

    assert set(collection.registry().names()) == set(ToolCollection.browser_tools) | {"read_image"}


def test_chrome_exposes_only_its_fixed_browser_tool_inventory(tmp_path: Path) -> None:
    manager = ChromeExtensionBrowserManager(
        state_dir=tmp_path,
        kolega_session_id="session-1",
        extension_origin=_EXTENSION_ORIGIN,
    )
    collection = _collection(tmp_path, manager, browser_only=True)

    assert set(collection.registry().names()) == CHROME_EXTENSION_SUPPORTED_TOOLS | {"read_image"}


def test_configured_chrome_target_is_offered_on_browser_dispatch(tmp_path: Path) -> None:
    with patch("kolega_code.cli.browser_backend._configured_extension_origin", return_value=_EXTENSION_ORIGIN):
        manager = build_browser_manager(tmp_path, "session-1")
        collection = _collection(tmp_path, manager, dispatch=True)

    dispatch = collection.registry().get("dispatch_browser_agent")
    assert dispatch.definition.input_schema is not None
    assert dispatch.definition.input_schema["required"] == ["task"]
    assert dispatch.definition.input_schema["properties"]["browser_target"]["enum"] == [
        "playwright",
        "chrome",
    ]
    assert dispatch.parallel_safe is False
    assert collection.registry().get("dispatch_investigation_agent").parallel_safe is True


@pytest.mark.asyncio
async def test_browser_dispatch_injects_the_selected_concrete_manager(tmp_path: Path) -> None:
    manager = build_browser_manager(tmp_path, "session-1")
    selected = PlaywrightBrowserManager()
    manager.resolve_browser_target = Mock(return_value=selected)  # type: ignore[method-assign]
    collection = _collection(tmp_path, manager, dispatch=True)
    dispatch = AsyncMock(return_value="done")
    collection.agent_tool._dispatch_agent = dispatch  # type: ignore[method-assign]

    assert await collection.agent_tool.dispatch_browser_agent("Use Chrome", browser_target="chrome") == "done"
    manager.resolve_browser_target.assert_called_once_with("chrome")
    assert dispatch.await_args is not None
    assert dispatch.await_args.kwargs["browser_manager_override"] is selected


@pytest.mark.asyncio
async def test_browser_agent_runs_are_serialized_per_concrete_manager(tmp_path: Path) -> None:
    manager = build_browser_manager(tmp_path, "session-1")
    selected = PlaywrightBrowserManager()
    manager.resolve_browser_target = Mock(return_value=selected)  # type: ignore[method-assign]
    collection = _collection(tmp_path, manager, dispatch=True)
    active = 0
    maximum_active = 0

    async def dispatch(**_: object) -> str:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return "done"

    collection.agent_tool._dispatch_agent = dispatch  # type: ignore[method-assign]

    await asyncio.gather(
        collection.agent_tool.dispatch_browser_agent("first", browser_target="chrome"),
        collection.agent_tool.dispatch_browser_agent("second", browser_target="chrome"),
    )

    assert maximum_active == 1
    assert browser_manager_agent_lock(selected) is browser_manager_agent_lock(selected)
    assert browser_manager_agent_lock(selected) is not browser_manager_agent_lock(manager)


@pytest.mark.asyncio
async def test_legacy_browser_manager_rejects_an_explicit_target(tmp_path: Path) -> None:
    collection = _collection(tmp_path, PlaywrightBrowserManager(), dispatch=True)

    with pytest.raises(ValueError, match="browser_target"):
        await collection.agent_tool.dispatch_browser_agent("task", browser_target="chrome")
