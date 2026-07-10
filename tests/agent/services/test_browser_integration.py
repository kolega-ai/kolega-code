import os
import re
import urllib.parse

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
        current_url = urllib.parse.urlsplit(state["url"])
        assert current_url.scheme == "https"
        assert current_url.hostname == "example.com"
        assert "Example Domain" in state["snapshot"]

        match = re.search(r"link .*?\[ref=((?:f\d+)?e\d+)\]", state["snapshot"])
        assert match is not None
        link_ref = match.group(1)
        clicked = await manager.click(link_ref)
        clicked_url = urllib.parse.urlsplit(clicked["url"])
        assert clicked_url.scheme == "https"
        assert clicked_url.hostname == "www.iana.org"

        screenshot = await manager.screenshot()
        assert screenshot["media_type"] == "image/png"
        assert screenshot["image"]
    finally:
        await manager.close()
