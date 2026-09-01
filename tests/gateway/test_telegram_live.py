"""Live Telegram round-trip against a real bot (integration).

Requires a test bot and chat in the repo `.env`:

    KOLEGA_GATEWAY_TEST_TOKEN=123456:AAH...
    KOLEGA_GATEWAY_TEST_CHAT_ID=<your telegram user id>

The test messages itself and waits for the echo to come back through the
adapter's inbound queue. Skipped everywhere else, including CI.
"""

import asyncio
import os
import uuid

import pytest

from kolega_code.gateway.adapters.telegram import TelegramAdapter

pytestmark = pytest.mark.integration

TEST_TOKEN = os.getenv("KOLEGA_GATEWAY_TEST_TOKEN")
TEST_CHAT_ID = os.getenv("KOLEGA_GATEWAY_TEST_CHAT_ID")

pytestmark = pytest.mark.skipif(
    not (TEST_TOKEN and TEST_CHAT_ID),
    reason="set KOLEGA_GATEWAY_TEST_TOKEN and KOLEGA_GATEWAY_TEST_CHAT_ID in .env for the live telegram test",
)


@pytest.mark.asyncio
async def test_send_and_receive_round_trip() -> None:
    assert TEST_TOKEN is not None
    marker = uuid.uuid4().hex[:8]
    adapter = TelegramAdapter(token=TEST_TOKEN)
    await adapter.start()
    try:
        await adapter.send_text(str(TEST_CHAT_ID), f"gateway-test {marker}")
        received = None
        for _ in range(60):
            try:
                inbound = await asyncio.wait_for(adapter.inbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if inbound.chat_id == str(TEST_CHAT_ID) and f"gateway-test {marker}" in inbound.text:
                received = inbound
                break
        assert received is not None
        assert received.sender_id != "unknown"
    finally:
        await adapter.stop()
