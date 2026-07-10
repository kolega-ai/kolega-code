import os
import re

import pytest

from kolega_code.services.browser import PlaywrightBrowserManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_browserless_snapshot_action_and_screenshot():
    if not os.getenv("BROWSERLESS_API_KEY"):
        pytest.skip("BROWSERLESS_API_KEY is not set")

    manager = PlaywrightBrowserManager("browserless")
    try:
        state = await manager.navigate("https://example.com")
        assert state["url"].startswith("https://example.com")
        assert "Example Domain" in state["snapshot"]

        match = re.search(r"link .*?\[ref=((?:f\d+)?e\d+)\]", state["snapshot"])
        assert match is not None
        link_ref = match.group(1)
        clicked = await manager.click(link_ref)
        assert clicked["url"].startswith("https://www.iana.org")

        screenshot = await manager.screenshot()
        assert screenshot["media_type"] == "image/png"
        assert screenshot["image"]
    finally:
        await manager.close()
