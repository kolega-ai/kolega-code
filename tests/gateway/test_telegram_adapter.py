"""TelegramAdapter envelope conversion and capability contract (no network)."""

import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.types import Chat, Document, InaccessibleMessage, Message, PhotoSize, User, Voice
from aiogram.enums import ChatType

from kolega_code.gateway.adapters.base import ButtonOption, InboundMessage
from kolega_code.gateway.adapters.telegram import TelegramAdapter
from kolega_code.gateway.adapters.telegram.adapter import (
    MAX_DOWNLOAD_BYTES,
    MEDIA_UNSUPPORTED_REPLY,
    decode_callback,
    encode_callback,
    validate_bot_token,
)


def test_validate_bot_token_accepts_botfather_shape() -> None:
    assert validate_bot_token("123456:fake-bot-token-for-tests-only") == ("123456:fake-bot-token-for-tests-only")


def test_validate_bot_token_rejects_non_token_values() -> None:
    for bad in ("", "not-a-token", "123456:", ":fake-bot-token-for-tests-only", "123456:short"):
        with pytest.raises(ValueError):
            validate_bot_token(bad)


def test_adapter_constructor_validates_token() -> None:
    with pytest.raises(ValueError):
        TelegramAdapter(token="garbage")


def make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(token="123:fake-bot-token-for-tests-only")
    adapter.set_inbound_authorizer(lambda message: message.sender_id == "7")
    adapter._bot_id = "123"  # type: ignore[assignment] — set post-init for offline tests
    return adapter


def make_photo_message(message_id: int = 100, caption: str | None = None) -> Message:
    return Message(
        message_id=message_id,
        date=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
        chat=Chat(id=42, type=ChatType.PRIVATE),
        from_user=User(id=7, is_bot=False, first_name="Test User"),
        caption=caption,
        photo=[PhotoSize(file_id="PHOTO-1", file_unique_id="u1", width=100, height=100)],
    )


def make_voice_message(message_id: int = 100) -> Message:
    return Message(
        message_id=message_id,
        date=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
        chat=Chat(id=42, type=ChatType.PRIVATE),
        from_user=User(id=7, is_bot=False, first_name="Test User"),
        voice=Voice(file_id="VOICE-1", file_unique_id="u1", duration=3),
    )


def make_document_message(message_id: int = 100, *, file_size: int = 1000, file_name: str | None = None) -> Message:
    return Message(
        message_id=message_id,
        date=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
        chat=Chat(id=42, type=ChatType.PRIVATE),
        from_user=User(id=7, is_bot=False, first_name="Test User"),
        document=Document(file_id="DOC-1", file_unique_id="u1", file_size=file_size, file_name=file_name),
    )


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


def make_fake_bot() -> AsyncMock:
    bot = AsyncMock()
    bot.download = AsyncMock()
    bot.send_message = AsyncMock()
    bot.answer_callback_query = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_start_requires_authorizer_before_creating_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = TelegramAdapter(token="123:fake-bot-token-for-tests-only")
    bot_constructor = MagicMock(side_effect=AssertionError("must not create a Telegram client"))
    monkeypatch.setattr("kolega_code.gateway.adapters.telegram.adapter.Bot", bot_constructor)

    with pytest.raises(RuntimeError, match="inbound authorizer"):
        await adapter.start()

    bot_constructor.assert_not_called()
    assert adapter.health()["state"] == "stopped"
    assert adapter._bot is None
    assert adapter._dispatcher is None
    assert adapter._poll_task is None


@pytest.mark.asyncio
async def test_start_with_explicit_policy_uses_only_fake_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = make_adapter()
    bot = make_fake_bot()
    bot.me.return_value = User(id=123, is_bot=True, first_name="Test Bot", username="test_bot")
    monkeypatch.setattr("kolega_code.gateway.adapters.telegram.adapter.Bot", MagicMock(return_value=bot))
    monkeypatch.setattr("kolega_code.gateway.adapters.telegram.adapter.Dispatcher", MagicMock())
    supervisor = AsyncMock()
    monkeypatch.setattr(adapter, "_poll_supervisor", supervisor)

    await adapter.start()
    try:
        assert adapter.health() == {"state": "running", "bot_id": "123"}
        bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=True)
    finally:
        await adapter.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["photo", "voice", "document", "oversize", "unsupported", "text"])
@pytest.mark.parametrize("rejection", ["sender", "group", "missing_hook"])
async def test_rejected_messages_queue_only_metadata_without_media_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media_kind: str, rejection: str
) -> None:
    adapter = TelegramAdapter(token="123:fake-bot-token-for-tests-only", media_dir=tmp_path / "media")
    adapter._bot_username = "test_bot"
    if rejection == "sender":
        adapter.set_inbound_authorizer(lambda message: message.sender_id == "8")
    elif rejection == "group":
        adapter.set_inbound_authorizer(lambda message: message.sender_id == "7" and not message.is_group)
    messages = {
        "photo": make_photo_message(caption="@test_bot private caption"),
        "voice": make_voice_message(),
        "document": make_document_message(file_name="private.txt"),
        "oversize": make_document_message(file_size=MAX_DOWNLOAD_BYTES + 1),
        "unsupported": make_message(None),
        "text": make_message("/permissions auto"),
    }
    message = messages[media_kind].model_copy(
        update={
            "chat": Chat(id=-42, type=ChatType.SUPERGROUP),
            "message_thread_id": 99,
            "reply_to_message": make_message("private quote", user_id=123),
        }
    )
    bot = make_fake_bot()
    adapter._bot = bot
    download_media = AsyncMock(side_effect=AssertionError("unauthorized media processing"))
    monkeypatch.setattr(adapter, "_download_media", download_media)

    await adapter._handle_message(message, bot)

    inbound = adapter.inbound.get_nowait()
    assert inbound.channel == "telegram"
    assert inbound.sender_id == "7"
    assert inbound.sender_name == "Test User"
    assert inbound.chat_id == "-42"
    assert inbound.message_id == "100"
    assert inbound.is_group is True
    assert inbound.topic_id == "99"
    assert inbound.text == ""
    assert inbound.reply_to is None
    assert inbound.attachments == ()
    download_media.assert_not_awaited()
    bot.download.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    assert not (tmp_path / "media").exists()


@pytest.mark.asyncio
async def test_missing_message_sender_is_not_a_synthetic_identity() -> None:
    adapter = make_adapter()
    message = make_photo_message().model_copy(update={"from_user": None})
    bot = make_fake_bot()

    await adapter._handle_message(message, bot)

    inbound = adapter.inbound.get_nowait()
    assert inbound.sender_id == ""
    assert inbound.attachments == ()
    bot.download.assert_not_awaited()
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorized_message_downloads_after_metadata_admission(tmp_path: Path) -> None:
    adapter = make_adapter()
    adapter._media_dir = tmp_path
    bot = make_fake_bot()
    adapter._bot = bot
    checked: list[InboundMessage] = []

    def authorize(message: InboundMessage) -> bool:
        bot.download.assert_not_awaited()
        assert message.attachments == ()
        checked.append(message)
        return message.sender_id == "7"

    adapter.set_inbound_authorizer(authorize)
    await adapter._handle_message(make_photo_message(caption="look"), bot)

    inbound = adapter.inbound.get_nowait()
    assert len(checked) == 1
    assert inbound.text == "look"
    assert len(inbound.attachments) == 1
    assert inbound.attachments[0].kind == "image"
    bot.download.assert_awaited_once()


def test_handler_signatures_inject_the_bot_by_name() -> None:
    # aiogram injects the Bot instance by parameter NAME ("bot"), not
    # position — a renamed parameter silently breaks every update.
    adapter = make_adapter()
    message_params = inspect.signature(adapter._handle_message).parameters
    callback_params = inspect.signature(adapter._handle_callback).parameters
    assert "message" in message_params and "bot" in message_params
    assert "query" in callback_params and "bot" in callback_params


@pytest.mark.asyncio
async def test_callback_handler_publishes_tap_envelope() -> None:
    adapter = make_adapter()
    adapter._pending_buttons["tok"] = {"chat_id": "42", "options": ["allow_once", "deny"]}
    bot = make_fake_bot()
    query = AsyncMock()
    query.data = "tok:1"
    query.id = "9000"
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    query.message = make_message("approve?", message_id=555)

    await adapter._handle_callback(query, bot)

    inbound = await adapter.inbound.get()
    assert inbound.callback_token == "tok"
    assert inbound.callback_option == "deny"
    assert inbound.chat_id == "42"
    assert inbound.sender_id == "7"
    assert inbound.message_id == "cb-9000"
    assert inbound.text == ""
    bot.answer_callback_query.assert_awaited_once_with("9000")
    # One-shot: a second tap on the same token is swallowed.
    assert adapter._pending_buttons == {}
    await adapter._handle_callback(query, bot)
    assert adapter.inbound.empty()


@pytest.mark.asyncio
async def test_callback_handler_ignores_unknown_tokens() -> None:
    adapter = make_adapter()
    bot = make_fake_bot()
    query = AsyncMock()
    query.data = "ghost:0"
    query.id = "1"
    await adapter._handle_callback(query, bot)
    bot.answer_callback_query.assert_awaited_once()
    assert adapter.inbound.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection",
    [
        "sender",
        "wrong_chat",
        "invalid_option",
        "negative_option",
        "malformed",
        "missing_sender",
        "missing_context",
        "inaccessible_context",
        "group",
        "missing_hook",
    ],
)
async def test_rejected_callback_preserves_prompt_for_authorized_tap(rejection: str) -> None:
    adapter = TelegramAdapter(token="123:fake-bot-token-for-tests-only")
    if rejection != "missing_hook":
        adapter.set_inbound_authorizer(lambda message: message.sender_id == "7" and not message.is_group)
    pending = {"chat_id": "42", "options": ["allow_once", "deny"]}
    adapter._pending_buttons["tok"] = pending
    bot = make_fake_bot()
    query = AsyncMock()
    query.data = "tok:0"
    query.id = "9000"
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    query.message = make_message("approve?", message_id=555)
    if rejection == "sender":
        query.from_user = User(id=8, is_bot=False, first_name="Outsider")
    elif rejection == "wrong_chat":
        query.message = make_message("approve?", chat_id=43)
    elif rejection == "invalid_option":
        query.data = "tok:2"
    elif rejection == "negative_option":
        query.data = "tok:-1"
    elif rejection == "malformed":
        query.data = "tok:not-an-index"
    elif rejection == "missing_sender":
        query.from_user = None
    elif rejection == "missing_context":
        query.message = None
    elif rejection == "inaccessible_context":
        query.message = InaccessibleMessage(chat=Chat(id=42, type=ChatType.PRIVATE), message_id=555, date=0)
    elif rejection == "group":
        query.message = make_message("approve?", chat_type=ChatType.SUPERGROUP)

    await adapter._handle_callback(query, bot)

    bot.answer_callback_query.assert_awaited_once()
    bot.send_message.assert_not_awaited()
    assert adapter.inbound.empty()
    assert adapter._pending_buttons["tok"] is pending

    adapter.set_inbound_authorizer(lambda message: message.sender_id == "7" and not message.is_group)
    query.data = "tok:0"
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    query.message = make_message("approve?", message_id=555)
    await adapter._handle_callback(query, bot)
    assert adapter.inbound.get_nowait().callback_option == "allow_once"
    assert adapter._pending_buttons == {}


@pytest.mark.asyncio
async def test_group_callback_retains_topic_and_must_pass_sender_and_group_policy() -> None:
    adapter = make_adapter()
    allowed_groups: set[str] = set()
    adapter.set_inbound_authorizer(
        lambda message: (
            message.sender_id == "7"
            and message.is_group
            and message.chat_id in allowed_groups
            and message.bot_mentioned
        )
    )
    adapter._pending_buttons["tok"] = {"chat_id": "-42", "options": ["allow_once"]}
    bot = make_fake_bot()
    query = AsyncMock()
    query.data = "tok:0"
    query.id = "9000"
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    query.message = make_message("approve?", chat_id=-42, thread_id=99, chat_type=ChatType.SUPERGROUP)

    await adapter._handle_callback(query, bot)
    assert adapter.inbound.empty()
    assert "tok" in adapter._pending_buttons
    allowed_groups.add("-42")
    query.from_user = User(id=8, is_bot=False, first_name="Outsider")
    await adapter._handle_callback(query, bot)
    assert adapter.inbound.empty()
    assert "tok" in adapter._pending_buttons
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    await adapter._handle_callback(query, bot)

    inbound = adapter.inbound.get_nowait()
    assert inbound.chat_id == "-42"
    assert inbound.topic_id == "99"
    assert inbound.is_group is True
    assert inbound.bot_mentioned is True
    assert inbound.sender_id == "7"
    assert inbound.callback_option == "allow_once"
    assert adapter._pending_buttons == {}


@pytest.mark.asyncio
async def test_concurrent_authorized_callbacks_only_publish_once() -> None:
    adapter = make_adapter()
    adapter._pending_buttons["tok"] = {"chat_id": "42", "options": ["allow_once"]}
    bot = make_fake_bot()
    query = AsyncMock()
    query.data = "tok:0"
    query.id = "9000"
    query.from_user = User(id=7, is_bot=False, first_name="Tapper")
    query.message = make_message("approve?")

    await asyncio.gather(adapter._handle_callback(query, bot), adapter._handle_callback(query, bot))

    assert adapter.inbound.qsize() == 1
    assert adapter._pending_buttons == {}


def test_send_buttons_requires_running_bot() -> None:
    adapter = make_adapter()
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.send_buttons("42", "pick", [ButtonOption("a", "A")]))


def test_media_kinds_detection() -> None:
    adapter = make_adapter()
    assert adapter._media_kinds(make_photo_message()) == ["image"]
    assert adapter._media_kinds(make_voice_message()) == ["voice"]
    assert adapter._media_kinds(make_document_message()) == ["document"]
    assert adapter._media_kinds(make_message("plain text")) == []


@pytest.mark.asyncio
async def test_photo_downloads_as_image_attachment(tmp_path: Path) -> None:
    adapter = make_adapter()
    adapter._media_dir = tmp_path
    adapter._bot = MagicMock()
    adapter._bot.download = AsyncMock()
    attachments = await adapter._download_media(make_photo_message(message_id=77, caption="look!"), make_fake_bot())
    assert len(attachments) == 1
    assert attachments[0].kind == "image"
    assert attachments[0].source.endswith("image-77.jpg")
    adapter._bot.download.assert_awaited_once_with("PHOTO-1", destination=tmp_path / "image-77.jpg")


@pytest.mark.asyncio
async def test_voice_downloads_as_voice_attachment(tmp_path: Path) -> None:
    adapter = make_adapter()
    adapter._media_dir = tmp_path
    adapter._bot = MagicMock()
    adapter._bot.download = AsyncMock()
    attachments = await adapter._download_media(make_voice_message(message_id=3), make_fake_bot())
    assert len(attachments) == 1
    assert attachments[0].kind == "voice"
    assert attachments[0].source.endswith("voice-3.ogg")


@pytest.mark.asyncio
async def test_document_downloads_with_sanitized_name(tmp_path: Path) -> None:
    adapter = make_adapter()
    adapter._media_dir = tmp_path
    adapter._bot = MagicMock()
    adapter._bot.download = AsyncMock()
    attachments = await adapter._download_media(
        make_document_message(message_id=9, file_name="../../report:final.pdf"), make_fake_bot()
    )
    assert len(attachments) == 1
    assert attachments[0].kind == "document"
    assert attachments[0].file_name == "../../report:final.pdf"
    assert attachments[0].source == str(tmp_path / "report_final.pdf")


@pytest.mark.asyncio
async def test_oversized_document_is_not_downloaded(tmp_path: Path) -> None:
    adapter = make_adapter()
    adapter._media_dir = tmp_path
    adapter._bot = MagicMock()
    adapter._bot.download = AsyncMock()
    attachments = await adapter._download_media(
        make_document_message(message_id=5, file_size=MAX_DOWNLOAD_BYTES + 1), make_fake_bot()
    )
    assert attachments == ()
    adapter._bot.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_text_treats_not_modified_as_a_noop() -> None:
    from aiogram.exceptions import TelegramBadRequest

    adapter = make_adapter()
    adapter._bot = MagicMock()
    not_modified = TelegramBadRequest(  # type: ignore[arg-type] — aiogram over-constrains `method`
        method=cast(Any, "editMessageText"), message="Bad Request: message is not modified"
    )
    adapter._bot.edit_message_text = AsyncMock(side_effect=not_modified)
    # No retry, no raise: editing to the shown content is a semantic no-op.
    await adapter.edit_text("42", "77", "same text")
    adapter._bot.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_text_retries_plain_on_html_failure() -> None:
    from aiogram.exceptions import TelegramBadRequest

    parse_error = TelegramBadRequest(  # type: ignore[arg-type] — aiogram over-constrains `method`
        method=cast(Any, "editMessageText"), message="can't parse entities"
    )
    adapter = make_adapter()
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock(side_effect=[parse_error, None])
    await adapter.edit_text("42", "77", "**bold**")
    assert adapter._bot.edit_message_text.await_count == 2
    # The retry carries the raw text without the HTML parse mode.
    second_call = adapter._bot.edit_message_text.await_args
    assert second_call.kwargs.get("text") == "**bold**"
    assert "parse_mode" not in second_call.kwargs


def test_safe_file_name_falls_back_for_missing_names() -> None:
    adapter = make_adapter()
    assert adapter._safe_file_name(make_document_message(message_id=11, file_name=None)) == "document-11"


@pytest.mark.asyncio
async def test_stop_cancels_a_stuck_polling_loop() -> None:
    """A stalled getUpdates must not wedge Ctrl-C: stop() cancels the polling
    task instead of awaiting dispatcher.stop_polling() (which waits for the
    in-flight network request)."""
    adapter = make_adapter()
    adapter._state = "running"
    release = asyncio.Event()
    start_calls: list[dict] = []

    async def never_returns(*args, **kwargs) -> None:
        start_calls.append(dict(kwargs))
        try:
            await release.wait()
        except asyncio.CancelledError:
            raise

    dispatcher = MagicMock()
    dispatcher.start_polling = never_returns
    adapter._dispatcher = dispatcher
    adapter._bot = MagicMock()
    adapter._bot.session = AsyncMock()
    adapter._poll_task = asyncio.create_task(adapter._poll_supervisor())
    deadline = asyncio.get_running_loop().time() + 2
    while not start_calls and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    await asyncio.wait_for(adapter.stop(), timeout=5)
    assert adapter._poll_task is None
    assert adapter._bot is None
    assert adapter._dispatcher is None
    # The gateway owns signals; aiogram's own handlers stay off.
    assert start_calls and start_calls[0].get("handle_signals") is False


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
    assert inbound.reply_to.text == "original text"
    assert inbound.reply_to.sender_id == "3"


def test_to_inbound_uses_caption_when_no_text() -> None:
    adapter = make_adapter()
    inbound = adapter._to_inbound(make_message(None, caption="image caption"))
    assert inbound is not None
    assert inbound.text == "image caption"


@pytest.mark.asyncio
async def test_authorized_unsupported_media_without_caption_gets_notice() -> None:
    adapter = make_adapter()
    bot = make_fake_bot()

    await adapter._handle_message(make_message(None), bot)

    bot.send_message.assert_awaited_once_with(chat_id=42, text=MEDIA_UNSUPPORTED_REPLY)
    assert adapter.inbound.empty()


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
