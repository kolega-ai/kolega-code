import asyncio
import re

import pytest
import pytest_asyncio

from kolega_code.services.browser import PlaywrightBrowserManager, file_payload


@pytest_asyncio.fixture
async def browser_manager():
    manager = PlaywrightBrowserManager()
    manager.headless = True
    try:
        yield manager
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_console_messages_filter_by_severity(browser_manager):
    await browser_manager.evaluate("() => { console.log('hello'); console.warn('careful'); console.error('broken'); }")

    warnings = await browser_manager.console_messages("warning")
    assert warnings["total"] == 3
    assert warnings["errors"] == 1
    assert warnings["warnings"] == 1
    assert [message["text"] for message in warnings["messages"]] == ["careful", "broken"]


@pytest.mark.asyncio
async def test_console_recent_is_cleared_by_navigation(browser_manager, unused_tcp_port):
    await browser_manager.evaluate("() => console.error('before navigation')")

    # A failed navigation still starts a new navigation scope before Playwright reports the connection error.
    with pytest.raises(Exception):
        await browser_manager.navigate(f"http://127.0.0.1:{unused_tcp_port}")

    recent = await browser_manager.console_messages("debug")
    assert all(message["text"] != "before navigation" for message in recent["messages"])


@pytest.mark.asyncio
async def test_dialog_returns_modal_state_and_can_be_handled(browser_manager):
    await browser_manager.evaluate(
        "() => { document.body.innerHTML = `<button onclick=\"alert('hello')\">Open</button>`; }"
    )
    snapshot = (await browser_manager.snapshot())["snapshot"]
    match = re.search(r"button.*?\[ref=(e\d+)\]", snapshot)
    assert match is not None
    target = match.group(1)

    modal = await browser_manager.click(target)
    assert modal["modal"] == {"type": "dialog", "dialog_type": "alert", "message": "hello"}

    result = await browser_manager.handle_dialog(True)
    assert "snapshot" in result
    assert result.get("modal") is None


@pytest.mark.asyncio
async def test_action_is_blocked_until_modal_is_handled(browser_manager):
    await browser_manager.evaluate(
        "() => { document.body.innerHTML = `<button onclick=\"confirm('continue?')\">Open</button>`; }"
    )
    snapshot = (await browser_manager.snapshot())["snapshot"]
    match = re.search(r"button.*?\[ref=(e\d+)\]", snapshot)
    assert match is not None
    target = match.group(1)
    await browser_manager.click(target)

    with pytest.raises(RuntimeError, match="browser_handle_dialog"):
        await browser_manager.press_key("Enter")

    await browser_manager.handle_dialog(False)


@pytest.mark.asyncio
async def test_file_chooser_returns_modal_state_and_accepts_payload(browser_manager):
    await browser_manager.evaluate("() => { document.body.innerHTML = `<input type=file aria-label='Upload'>`; }")
    snapshot = (await browser_manager.snapshot())["snapshot"]
    match = re.search(r"button.*?\[ref=(e\d+)\]", snapshot)
    assert match is not None
    target = match.group(1)

    modal = await browser_manager.click(target)
    assert modal["modal"] == {"type": "file_chooser", "multiple": False}

    result = await browser_manager.file_upload([file_payload("avatar.txt", b"hello")])
    assert "snapshot" in result
    uploaded = await browser_manager.evaluate("() => document.querySelector('input').files[0].name")
    assert uploaded["result"] == "avatar.txt"


@pytest.mark.asyncio
async def test_evaluate_result_is_size_capped(browser_manager):
    result = await browser_manager.evaluate("() => 'x'.repeat(25000)")

    assert result["result_truncated"] is True
    assert len(result["result"]) == 20_000


@pytest.mark.asyncio
async def test_network_requests_are_indexed_and_details_are_retrievable(browser_manager):
    async def handle_request(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        body = b"hello from server"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        await browser_manager.navigate(f"http://127.0.0.1:{port}/hello")
        requests = await browser_manager.network_requests()
        document = next(request for request in requests["requests"] if request["resource_type"] == "document")
        details = await browser_manager.network_request(document["index"], "response_body")
    finally:
        server.close()
        await server.wait_closed()

    assert details["status"] == 200
    assert details["response_body"] == "hello from server"
