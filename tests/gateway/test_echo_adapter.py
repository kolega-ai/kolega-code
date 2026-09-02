"""EchoAdapter: stdin->envelope and outbound printing through the contract."""

import io

import pytest

from kolega_code.gateway.adapters.base import STREAMING_EDIT_IN_PLACE, InboundMessage
from kolega_code.gateway.adapters.echo import EchoAdapter


def make_adapter(stdin_text: str = "") -> tuple[EchoAdapter, io.StringIO]:
    stdout = io.StringIO()
    adapter = EchoAdapter(stdin=io.StringIO(stdin_text), stdout=stdout)
    return adapter, stdout


def test_capabilities_advertise_edits() -> None:
    adapter, _ = make_adapter()
    assert adapter.capabilities.supports_edits is True
    assert adapter.capabilities.supports_delete is True
    assert adapter.capabilities.streaming_mode == STREAMING_EDIT_IN_PLACE
    assert adapter.capabilities.text_chunk_limit == 4096


def test_health_reflects_lifecycle() -> None:
    adapter, _ = make_adapter()
    assert adapter.health()["state"] == "stopped"


@pytest.mark.asyncio
async def test_read_loop_publishes_envelopes_until_eof() -> None:
    adapter, _ = make_adapter(stdin_text="hello\n\nworld\n")
    await adapter.start()
    first = await adapter.inbound.get()
    second = await adapter.inbound.get()
    await adapter.stop()

    assert isinstance(first, InboundMessage)
    assert first.channel == "echo"
    assert first.chat_id == "console"
    assert first.sender_id == "owner"
    assert first.text == "hello"
    assert first.message_id == "echo-1"
    # Blank lines are skipped.
    assert second.text == "world"
    assert second.message_id == "echo-2"
    assert adapter.health()["state"] == "stopped"


@pytest.mark.asyncio
async def test_publish_is_directly_usable_by_tests() -> None:
    adapter, _ = make_adapter()
    adapter.publish("direct")
    message = await adapter.inbound.get()
    assert message.text == "direct"
    assert message.chat_id == "console"


@pytest.mark.asyncio
async def test_outbound_operations_print_contract_lines() -> None:
    adapter, stdout = make_adapter()
    reply_id = await adapter.send_text("console", "hello there")
    assert reply_id
    await adapter.edit_text("console", reply_id, "hello, edited")
    await adapter.delete_message("console", reply_id)

    lines = stdout.getvalue().splitlines()
    assert lines == [
        f"[gateway:reply] {reply_id} hello there",
        f"[gateway:edit] {reply_id} hello, edited",
        f"[gateway:delete] {reply_id} ",
    ]
