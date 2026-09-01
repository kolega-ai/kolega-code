"""TelegramAdapter envelope conversion and capability contract (no network)."""

from datetime import datetime, timezone

import pytest
from aiogram.types import Chat, Message, User
from aiogram.enums import ChatType

from kolega_code.gateway.adapters.telegram import TelegramAdapter


def make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(token="123:fake")
    adapter._bot_id = "123"  # type: ignore[assignment] — set post-init for offline tests
    return adapter


def make_message(
    text: str | None = "hello",
    *,
    chat_id: int = 42,
    user_id: int = 7,
    message_id: int = 100,
    reply: Message | None = None,
    thread_id: int | None = None,
    caption: str | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        date=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
        chat=Chat(id=chat_id, type=ChatType.PRIVATE),
        from_user=User(id=user_id, is_bot=False, first_name="Test User"),
        text=text,
        caption=caption,
        message_thread_id=thread_id,
        reply_to_message=reply,
    )


def test_capabilities() -> None:
    adapter = make_adapter()
    assert adapter.capabilities.supports_edits is True
    assert adapter.capabilities.supports_typing is True
    assert adapter.capabilities.supports_groups is True
    assert adapter.capabilities.text_chunk_limit == 4000
    assert adapter.capabilities.streaming_mode == "edit_in_place"


def test_to_inbound_maps_plain_message() -> None:
    adapter = make_adapter()
    inbound = adapter._to_inbound(make_message("hi there"))
    assert inbound is not None
    assert inbound.channel == "telegram"
    assert inbound.chat_id == "42"
    assert inbound.sender_id == "7"
    assert inbound.sender_name == "Test User"
    assert inbound.message_id == "100"
    assert inbound.text == "hi there"
    assert inbound.topic_id is None
    assert inbound.reply_to is None
    assert inbound.account_id == "123"


def test_to_inbound_maps_forum_topic() -> None:
    adapter = make_adapter()
    inbound = adapter._to_inbound(make_message("in thread", thread_id=99))
    assert inbound is not None
    assert inbound.topic_id == "99"


def test_to_inbound_captures_quoted_reply() -> None:
    adapter = make_adapter()
    quoted = make_message("original text", chat_id=42, user_id=3, message_id=55)
    inbound = adapter._to_inbound(make_message("see above", reply=quoted))
    assert inbound is not None
    assert inbound.reply_to is not None
    assert inbound.reply_to.message_id == "55"
    assert inbound.reply_to.text == "original text"
    assert inbound.reply_to.sender_id == "3"


def test_to_inbound_uses_caption_when_no_text() -> None:
    adapter = make_adapter()
    inbound = adapter._to_inbound(make_message(None, caption="image caption"))
    assert inbound is not None
    assert inbound.text == "image caption"


def test_media_without_caption_needs_notice() -> None:
    adapter = make_adapter()
    assert adapter._handle_message is not None
    # A media-only message (no text, no caption) must not produce an envelope.
    message = make_message(None)
    assert message.text is None and message.caption is None


def test_health_reports_stopped_before_start() -> None:
    adapter = make_adapter()
    assert adapter.health()["state"] == "stopped"


@pytest.mark.asyncio
async def test_outbound_requires_running_bot() -> None:
    adapter = make_adapter()
    with pytest.raises(RuntimeError):
        await adapter.send_text("42", "hi")
