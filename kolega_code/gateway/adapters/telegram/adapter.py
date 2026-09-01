"""Telegram adapter: the official Bot API via aiogram long-polling.

The gateway owns the event loop; the adapter runs aiogram's polling loop as a
task and publishes normalized inbound envelopes on ``self.inbound``. Outbound
operations render through the Markdown subset in :mod:`.formatting` with a
plain-text fallback when Telegram rejects the HTML payload.

Inbound quirks handled here:

- media without a caption is answered with a short notice (media support
  lands in a later phase) instead of silently vanishing;
- forum topics become the envelope ``topic_id``, so each thread gets its own
  session;
- quoted replies carry the quoted message in ``reply_to`` so the session host
  can hand the model its context.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from typing import Any, Optional, Sequence

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ChatType, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.types import Message as TelegramMessage

from kolega_code.gateway.adapters.base import (
    STREAMING_EDIT_IN_PLACE,
    AdapterCapabilities,
    ButtonOption,
    GatewayAdapter,
    InboundMessage,
    ReplyContext,
)
from kolega_code.gateway.adapters.telegram.formatting import telegram_html

logger = logging.getLogger(__name__)

#: Notice for media messages until the media phase lands.
MEDIA_UNSUPPORTED_REPLY = "📎 Media messages aren't supported yet (coming soon)."
#: Polling shutdown grace period before the task is cancelled outright.
_POLL_STOP_TIMEOUT_SECONDS = 5.0
#: BotFather issues tokens as "<bot-id>:<secret>" (digits, colon, 35 chars of
#: base64url-ish alphabet). Accept a generous window so future token shapes
#: don't break, but catch obviously wrong values (userbot session strings,
#: pasted commands) before the first poll.
_BOT_TOKEN_RE = re.compile(r"^\d{1,15}:[A-Za-z0-9_-]{20,64}$")
#: Cap on remembered button prompts; the oldest unanswered prompt is evicted.
_MAX_PENDING_BUTTONS = 64
#: Telegram caps callback_data at 64 bytes; labels are truncated the same way.
_MAX_BUTTON_LABEL_CHARS = 64


def validate_bot_token(token: str) -> str:
    """Return the token if it looks like a BotFather bot token, else raise.

    The gateway only talks to bots created with @BotFather on the official
    Bot API — never to user accounts.
    """
    if not _BOT_TOKEN_RE.match(token):
        raise ValueError(
            "KOLEGA_GATEWAY_TELEGRAM_TOKEN does not look like a BotFather bot token "
            "(expected <bot-id>:<secret>). Create a bot with @BotFather and export its token."
        )
    return token


def encode_callback(token: str, index: int) -> str:
    """Wire format for one inline button's callback_data: ``token:index``.

    Indices, not option ids, so untrusted option ids (which the gateway
    chooses freely) can never blow the 64-byte callback_data limit.
    """
    return f"{token}:{index}"


def decode_callback(data: str) -> Optional[tuple[str, int]]:
    """Parse ``token:index``, returning None for anything malformed."""
    token, separator, raw_index = data.partition(":")
    if not separator or not token:
        return None
    try:
        index = int(raw_index)
    except ValueError:
        return None
    if index < 0:
        return None
    return token, index


class TelegramAdapter(GatewayAdapter):
    """Long-polling Telegram bot exposing the gateway envelope."""

    name = "telegram"
    capabilities = AdapterCapabilities(
        supports_edits=True,
        supports_delete=True,
        supports_typing=True,
        supports_groups=True,
        supports_inline_buttons=True,
        streaming_mode=STREAMING_EDIT_IN_PLACE,
        text_chunk_limit=4000,
        # Inbound download cap; outbound media is capped lower by the bridge.
        max_media_mb=50.0,
    )

    def __init__(self, token: str, *, proxy: Optional[str] = None) -> None:
        super().__init__()
        self._token = validate_bot_token(token)
        self._proxy = proxy
        self._bot: Optional[Bot] = None
        self._dispatcher: Optional[Dispatcher] = None
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._state = "stopped"
        self._bot_id: Optional[str] = None
        self._bot_username: Optional[str] = None
        #: callback token -> {"chat_id": str, "options": [option_id, ...]}, for
        #: resolving taps back to the option the gateway chose.
        self._pending_buttons: dict[str, dict[str, Any]] = {}

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        session = None
        if self._proxy:
            from aiogram.client.session.aiohttp import AiohttpSession

            session = AiohttpSession(proxy=self._proxy)
        self._bot = Bot(token=self._token, session=session, default=DefaultBotProperties())
        self._dispatcher = Dispatcher()
        self._dispatcher.message.register(self._handle_message)
        self._dispatcher.callback_query.register(self._handle_callback)
        self._state = "starting"
        # Long-polling must be the only consumer of updates.
        await self._bot.delete_webhook(drop_pending_updates=True)
        self._poll_task = asyncio.create_task(
            self._dispatcher.start_polling(self._bot),
            name="telegram-adapter-polling",
        )
        me = await self._bot.me()
        self._bot_id = str(me.id)
        self._bot_username = me.username
        self._state = "running"

    async def stop(self) -> None:
        self._state = "stopped"
        dispatcher, task = self._dispatcher, self._poll_task
        if dispatcher is not None:
            try:
                await dispatcher.stop_polling()
            except Exception:  # noqa: BLE001 — shutdown is best-effort
                logger.debug("telegram: stop_polling failed", exc_info=True)
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=_POLL_STOP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                task.cancel()
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._bot is not None:
            try:
                await self._bot.session.close()
            except Exception:  # noqa: BLE001
                logger.debug("telegram: bot session close failed", exc_info=True)
        self._bot = None
        self._dispatcher = None

    def health(self) -> dict[str, Any]:
        return {"state": self._state, "bot_id": self._bot_id}

    # -- Inbound -----------------------------------------------------------

    async def _handle_message(self, message: TelegramMessage, _bot: Bot) -> None:
        if message.text is None and message.caption is None:
            await self._media_notice(message)
            return
        inbound = self._to_inbound(message)
        if inbound is not None:
            await self.inbound.put(inbound)

    def _to_inbound(self, message: TelegramMessage) -> Optional[InboundMessage]:
        sender = message.from_user
        reply_to = None
        if message.reply_to_message is not None:
            quoted = message.reply_to_message
            reply_to = ReplyContext(
                message_id=str(quoted.message_id),
                text=quoted.text or quoted.caption or "",
                sender_id=str(quoted.from_user.id) if quoted.from_user else "",
            )
        is_group = message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        return InboundMessage(
            channel=self.name,
            account_id=self._bot_id or "",
            chat_id=str(message.chat.id),
            topic_id=str(message.message_thread_id) if message.message_thread_id else None,
            sender_id=str(sender.id) if sender else "unknown",
            sender_name=sender.full_name if sender else "",
            message_id=str(message.message_id),
            text=message.text or message.caption or "",
            reply_to=reply_to,
            timestamp=message.date.isoformat() if message.date else None,
            is_group=is_group,
            bot_mentioned=self._bot_mentioned(message, is_group),
        )

    def _bot_mentioned(self, message: TelegramMessage, is_group: bool) -> bool:
        """Whether the bot was addressed, for the gateway's group policy.

        A group message counts when it @mentions the bot or replies to a bot
        message. DMs always count. Note Telegram privacy mode already filters
        group messages to mentions for non-admin bots, so this only matters
        when the bot sees everything.
        """
        if not is_group:
            return True
        if self._bot_username is not None:
            mention = f"@{self._bot_username}".lower()
            if mention in (message.text or "").lower():
                return True
        quoted = message.reply_to_message
        if quoted is not None and quoted.from_user is not None and self._bot_id is not None:
            return str(quoted.from_user.id) == self._bot_id
        return False

    async def _media_notice(self, message: TelegramMessage) -> None:
        try:
            await message.answer(MEDIA_UNSUPPORTED_REPLY)
        except Exception:  # noqa: BLE001 — a failed notice must not break polling
            logger.debug("telegram: media notice failed", exc_info=True)

    async def _handle_callback(self, query: CallbackQuery, _bot: Optional[Bot]) -> None:
        """Turn an inline-button tap into an inbound envelope with the token."""
        decoded = decode_callback(query.data or "")
        token = decoded[0] if decoded else None
        pending = self._pending_buttons.pop(token, None) if token else None
        option_id = ""
        if pending is not None and decoded is not None:
            options = pending.get("options") or []
            if decoded[1] < len(options):
                option_id = str(options[decoded[1]])
        try:
            await query.answer()
        except Exception:  # noqa: BLE001 — acks are cosmetic
            logger.debug("telegram: callback ack failed", exc_info=True)
        if pending is None or not option_id or token is None:
            # Unknown/expired token or stale button: swallow the tap.
            return
        message = query.message
        sender = query.from_user
        await self.inbound.put(
            InboundMessage(
                channel=self.name,
                account_id=self._bot_id or "",
                chat_id=str(message.chat.id) if message is not None else str(pending.get("chat_id") or ""),
                sender_id=str(sender.id) if sender else "unknown",
                sender_name=sender.full_name if sender else "",
                message_id=f"cb-{query.id}",
                callback_token=token,
                callback_option=option_id,
            )
        )

    # -- Outbound ----------------------------------------------------------

    def _require_bot(self) -> Bot:
        if self._bot is None:
            raise RuntimeError("telegram adapter is not running")
        return self._bot

    async def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
    ) -> str:
        bot = self._require_bot()
        reply_to = int(reply_to_message_id) if reply_to_message_id else None
        try:
            sent = await bot.send_message(
                chat_id=int(chat_id),
                text=telegram_html(text),
                parse_mode=ParseMode.HTML,
                reply_to_message_id=reply_to,
            )
        except TelegramBadRequest:
            # Malformed HTML (an unmatched tag, a broken fence split across
            # chunks): fall back to plain text so the reply still lands.
            logger.info("telegram: HTML send failed for chat %s; retrying as plain text", chat_id)
            sent = await bot.send_message(chat_id=int(chat_id), text=text, reply_to_message_id=reply_to)
        return str(sent.message_id)

    async def edit_text(self, chat_id: str, message_id: str, text: str) -> None:
        bot = self._require_bot()
        try:
            await bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(message_id),
                text=telegram_html(text),
                parse_mode=ParseMode.HTML,
            )
        except TelegramBadRequest:
            logger.info("telegram: HTML edit failed for chat %s; retrying as plain text", chat_id)
            await bot.edit_message_text(chat_id=int(chat_id), message_id=int(message_id), text=text)

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        bot = self._require_bot()
        try:
            await bot.delete_message(chat_id=int(chat_id), message_id=int(message_id))
        except TelegramBadRequest:
            logger.debug("telegram: delete failed for chat %s message %s", chat_id, message_id)

    async def set_typing(self, chat_id: str, active: bool) -> None:
        if not active:
            return
        bot = self._require_bot()
        try:
            await bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)
        except Exception:  # noqa: BLE001 — typing indicators are cosmetic
            logger.debug("telegram: typing indicator failed", exc_info=True)

    async def send_buttons(self, chat_id: str, prompt: str, options: Sequence[ButtonOption]) -> str:
        """Send a prompt with one row of inline buttons; returns the callback token.

        A tap arrives as an inbound message carrying the token and the tapped
        option id (resolved from the button index, so option ids never ride in
        the 64-byte callback_data).
        """
        bot = self._require_bot()
        token = secrets.token_hex(8)
        buttons = [
            InlineKeyboardButton(
                text=option.label[:_MAX_BUTTON_LABEL_CHARS] or "\u00a0",
                callback_data=encode_callback(token, index),
            )
            for index, option in enumerate(options)
        ]
        try:
            await bot.send_message(
                chat_id=int(chat_id),
                text=telegram_html(prompt),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[buttons]),
            )
        except TelegramBadRequest:
            logger.info("telegram: HTML button prompt failed for chat %s; retrying as plain text", chat_id)
            await bot.send_message(
                chat_id=int(chat_id),
                text=prompt,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[buttons]),
            )
        if len(self._pending_buttons) >= _MAX_PENDING_BUTTONS:
            self._pending_buttons.pop(next(iter(self._pending_buttons)))
        self._pending_buttons[token] = {"chat_id": chat_id, "options": [option.option_id for option in options]}
        return token
