import re
import time

import pytest
import pytest_asyncio

from kolega_code.services.browser import PlaywrightBrowserManager


@pytest_asyncio.fixture
async def browser_manager():
    manager = PlaywrightBrowserManager()
    manager.headless = True
    try:
        yield manager
    finally:
        await manager.close()


async def _set_content(manager: PlaywrightBrowserManager, html: str) -> None:
    await manager.evaluate(f"() => {{ document.body.innerHTML = {html!r}; }}")


def _ref(snapshot: str, role: str) -> str:
    match = re.search(rf"{role}.*?\[ref=((?:f\d+)?e\d+)\]", snapshot)
    assert match, snapshot
    return match.group(1)


@pytest.mark.asyncio
async def test_snapshot_refs_drive_atomic_actions(browser_manager):
    await _set_content(
        browser_manager,
        """<main><h1>Profile</h1>
        <button onclick="this.textContent='Saved'">Save</button>
        <input aria-label="Name">
        <select aria-label="Color"><option value="red">Red</option><option value="blue">Blue</option></select>
        </main>""",
    )
    snapshot = (await browser_manager.snapshot())["snapshot"]

    clicked = await browser_manager.click(_ref(snapshot, "button"))
    assert 'button "Saved"' in clicked["snapshot"]

    await browser_manager.type_text(_ref(snapshot, "textbox"), "Ada")
    selected = await browser_manager.select_option(_ref(snapshot, "combobox"), ["blue"])
    assert selected["result"] == ["blue"]


@pytest.mark.asyncio
async def test_fill_form_and_data_drop(browser_manager):
    await _set_content(
        browser_manager,
        """<input aria-label="Email"><input type="checkbox" aria-label="Subscribe">
        <div role="button" aria-label="Drop target"
             ondragover="event.preventDefault()"
             ondrop="event.preventDefault(); this.textContent=event.dataTransfer.getData('text/plain')">Drop</div>""",
    )
    snapshot = (await browser_manager.snapshot())["snapshot"]
    textbox = _ref(snapshot, "textbox")
    checkbox = _ref(snapshot, "checkbox")
    button = _ref(snapshot, "button")

    filled = await browser_manager.fill_form(
        [
            {"name": "Email", "target": textbox, "type": "textbox", "value": "ada@example.com"},
            {"name": "Subscribe", "target": checkbox, "type": "checkbox", "value": "true"},
        ]
    )
    assert 'textbox "Email"' in filled["snapshot"]
    assert "ada@example.com" in filled["snapshot"]
    assert "[checked]" in filled["snapshot"]

    dropped = await browser_manager.drop(button, data={"text/plain": "Dropped value"})
    assert "Dropped value" in dropped["snapshot"]


@pytest.mark.asyncio
async def test_stale_ref_error_requests_fresh_snapshot(browser_manager):
    await _set_content(browser_manager, '<button id="old">Old</button>')
    stale_ref = _ref((await browser_manager.snapshot())["snapshot"], "button")
    await _set_content(browser_manager, '<button id="new">New</button>')

    with pytest.raises(ValueError, match="fresh browser_snapshot"):
        await browser_manager.click(stale_ref)


@pytest.mark.asyncio
async def test_ambiguous_selector_is_rejected(browser_manager):
    await _set_content(browser_manager, "<button>One</button><button>Two</button>")

    with pytest.raises(ValueError, match="matches 2 elements"):
        await browser_manager.click("button")


@pytest.mark.asyncio
async def test_click_does_not_wait_for_network_idle(browser_manager):
    await _set_content(
        browser_manager,
        """<button onclick="window.timer = setInterval(() => fetch('data:text/plain,ping'), 25); this.textContent='Running'">Start</button>""",
    )
    target = _ref((await browser_manager.snapshot())["snapshot"], "button")

    started = time.monotonic()
    result = await browser_manager.click(target)

    assert time.monotonic() - started < 2
    assert 'button "Running"' in result["snapshot"]


@pytest.mark.asyncio
async def test_tabs_use_implicit_current_session(browser_manager):
    initial = await browser_manager.tabs("list")
    assert initial["tabs"][0]["current"] is True

    created = await browser_manager.tabs("new")
    assert len(created["tabs"]) == 2
    assert created["tabs"][1]["current"] is True

    selected = await browser_manager.tabs("select", index=0)
    assert selected["tabs"][0]["current"] is True

    closed = await browser_manager.tabs("close", index=1)
    assert len(closed["tabs"]) == 1


@pytest.mark.asyncio
async def test_navigation_rejects_file_urls(browser_manager):
    with pytest.raises(ValueError, match="only supports http"):
        await browser_manager.navigate("file:///tmp/secret")
