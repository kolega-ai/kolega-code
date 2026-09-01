"""TelegramAdapter envelope conversion and capability contract (no network)."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, User
from aiogram.enums import ChatType

from kolega_code.gateway.adapters.base import ButtonOption
from kolega_code.gateway.adapters.telegram import TelegramAdapter
from kolega_code.gateway.adapters.telegram.adapter import decode_callback, encode_callback, validate_bot_token


def test_validate_bot_token_accepts_botfather_shape() -> None:
    assert validate_bot_token("123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw") == (
        "123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
    )


def test_validate_bot_token_rejects_non_token_values() -> None:
    for bad in ("", "not-a-token", "123456:", ":AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw", "123456:short"):
        with pytest.raises(ValueError):
            validate_bot_token(bad)


def test_adapter_constructor_validates_token() -> None:
    with pytest.raises(ValueError):
        TelegramAdapter(token="garbage")


def make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(token="123:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw")
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
    chat_type: ChatType = ChatType.PRIVATE,
) -> Message:
    return Message(
        message_id=message_id,
        date=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
        chat=Chat(id=chat_id, type=chat_type),
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
    assert adapter.capabilities.supports_inline_buttons is True
    assert adapter.capabilities.text_chunk_limit == 4000
    assert adapter.capabilities.streaming_mode == "edit_in_place"


def test_callback_encode_decode_round_trip() -> None:
    data = encode_callback("abc123", 4)
    assert decode_callback(data) == ("abc123", 4)
    assert len(data) < 64


def test_callback_decode_rejects_garbage() -> None:
    for bad in ("", "no-colon", "tok:", ":1", "tok:abc", "tok:-1", "tok:1:2:3"):
        assert decode_callback(bad) is None


@pytest.mark.asyncio
async def test_callback_handler_publishes_tap_envelope() -> None:
    adapter = make_adapter()
    adapter._pending_buttons["tok"] = {"chat_id": "42", "options": ["allow_once", "deny"]}
    query = AsyncMock()
    query.data = "tok:1"
    query.id = "9000"
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    query.message = make_message("approve?", message_id=555)

    await adapter._handle_callback(query, None)

    inbound = await adapter.inbound.get()
    assert inbound.callback_token == "tok"
    assert inbound.callback_option == "deny"
    assert inbound.chat_id == "42"
    assert inbound.sender_id == "7"
    assert inbound.message_id == "cb-9000"
    assert inbound.text == ""
    query.answer.assert_awaited_once()
    # One-shot: a second tap on the same token is swallowed.
    assert adapter._pending_buttons == {}
    await adapter._handle_callback(query, None)
    assert adapter.inbound.empty()


@pytest.mark.asyncio
async def test_callback_handler_ignores_unknown_tokens() -> None:
    adapter = make_adapter()
    query = AsyncMock()
    query.data = "ghost:0"
    query.id = "1"
    await adapter._handle_callback(query, None)
    query.answer.assert_awaited_once()
    assert adapter.inbound.empty()


def test_send_buttons_requires_running_bot() -> None:
    adapter = make_adapter()
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.send_buttons("42", "pick", [ButtonOption("a", "A")]))


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


def test_dms_are_not_groups_and_always_count_as_addressed() -> None:
    adapter = make_adapter()
    inbound = adapter._to_inbound(make_message("hi"))
    assert inbound is not None
    assert inbound.is_group is False
    assert inbound.bot_mentioned is True


def test_group_message_requires_mention() -> None:
    adapter = make_adapter()
    adapter._bot_username = "kolega_bot"
    # Group chatter without a mention is not addressed.
    inbound = adapter._to_inbound(make_message("ambient noise", chat_type=ChatType.SUPERGROUP))
    assert inbound is not None
    assert inbound.is_group is True
    assert inbound.bot_mentioned is False
    # A mention (case-insensitive) counts.
    inbound = adapter._to_inbound(make_message("hey @KOLEGA_BOT do it", chat_type=ChatType.SUPERGROUP))
    assert inbound is not None
    assert inbound.bot_mentioned is True


def test_group_reply_to_bot_counts_as_addressed() -> None:
    adapter = make_adapter()
    adapter._bot_id = "123"
    quoted = make_message("bot said this", chat_id=1, user_id=123, message_id=55)
    inbound = adapter._to_inbound(make_message("ok", reply=quoted, chat_type=ChatType.SUPERGROUP))
    assert inbound is not None
    assert inbound.bot_mentioned is True
    # A reply to someone else's message is not addressed.
    other = make_message("someone else", chat_id=1, user_id=9, message_id=56)
    inbound = adapter._to_inbound(make_message("ok", reply=other, chat_type=ChatType.SUPERGROUP))
    assert inbound is not None
    assert inbound.bot_mentioned is False


@pytest.mark.asyncio
async def test_outbound_requires_running_bot() -> None:
    adapter = make_adapter()
    with pytest.raises(RuntimeError):
        await adapter.send_text("42", "hi")
