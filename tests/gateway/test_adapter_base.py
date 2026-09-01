"""Adapter contract: envelope records, routing keys, and capability defaults."""

import pytest

from kolega_code.gateway.adapters.base import (
    AdapterCapabilities,
    Attachment,
    ButtonOption,
    ChatRef,
    GatewayAdapter,
    InboundMessage,
    ReplyContext,
    UnsupportedCapability,
)


class MinimalAdapter(GatewayAdapter):
    name = "minimal"

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str:
        return "minimal-1"


def test_chat_ref_key_includes_channel_and_topic() -> None:
    assert ChatRef("telegram", "42").key == "telegram:42"
    assert ChatRef("telegram", "42", account_id="bot2", topic_id="7").key == "telegram:42:7"
    # Same chat id on another channel must never collide.
    assert ChatRef("discord", "42").key != ChatRef("telegram", "42").key


def test_chat_ref_from_message_maps_envelope_fields() -> None:
    message = InboundMessage(
        channel="telegram",
        chat_id="9",
        sender_id="1",
        message_id="m-1",
        account_id="bot1",
        topic_id="3",
    )
    ref = ChatRef.from_message(message)
    assert ref.key == "telegram:9:3"


def test_inbound_message_defaults() -> None:
    message = InboundMessage(channel="echo", chat_id="console", sender_id="owner", message_id="m-1")
    assert message.text == ""
    assert message.attachments == ()
    assert message.reply_to is None
    assert message.callback_token is None


def test_envelope_records_are_frozen() -> None:
    message = InboundMessage(channel="echo", chat_id="c", sender_id="s", message_id="m", text="hi")
    with pytest.raises(Exception):
        message.text = "changed"  # type: ignore[misc]
    attachment = Attachment(kind="image", source="/tmp/a.png")
    with pytest.raises(Exception):
        attachment.kind = "audio"  # type: ignore[misc]


def test_reply_context_defaults() -> None:
    reply = ReplyContext(message_id="m-1")
    assert reply.text == ""
    assert reply.sender_id == ""


def test_capability_defaults_are_conservative() -> None:
    capabilities = AdapterCapabilities()
    assert capabilities.supports_edits is False
    assert capabilities.supports_inline_buttons is False
    assert capabilities.text_chunk_limit == 4096
    assert capabilities.streaming_mode == "final_only"


@pytest.mark.asyncio
async def test_unsupported_outbound_operations_raise() -> None:
    adapter = MinimalAdapter()
    with pytest.raises(UnsupportedCapability):
        await adapter.edit_text("c", "m", "text")
    with pytest.raises(UnsupportedCapability):
        await adapter.delete_message("c", "m")
    with pytest.raises(UnsupportedCapability):
        await adapter.send_media("c", [Attachment("image", "/tmp/a.png")])
    with pytest.raises(UnsupportedCapability):
        await adapter.send_buttons("c", "pick one", [ButtonOption("a", "A")])


@pytest.mark.asyncio
async def test_adapter_inbound_queue_round_trip() -> None:
    adapter = MinimalAdapter()
    message = InboundMessage(
        channel="telegram",
        chat_id="1",
        sender_id="2",
        message_id="m-1",
        text="hello",
        attachments=(Attachment(kind="image", source="/tmp/a.png", mime="image/png"),),
    )
    await adapter.inbound.put(message)
    received = await adapter.inbound.get()
    assert received is message
    assert received.attachments[0].kind == "image"


@pytest.mark.asyncio
async def test_adapter_lifecycle_defaults() -> None:
    adapter = MinimalAdapter()
    assert adapter.health()["state"] == "stopped"
    await adapter.start()  # no-op default
    await adapter.stop()  # no-op default
