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
from typing import Any, Optional

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message as TelegramMessage

from kolega_code.gateway.adapters.base import (
    STREAMING_EDIT_IN_PLACE,
    AdapterCapabilities,
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


class TelegramAdapter(GatewayAdapter):
    """Long-polling Telegram bot exposing the gateway envelope."""

    name = "telegram"
    capabilities = AdapterCapabilities(
        supports_edits=True,
        supports_delete=True,
        supports_typing=True,
        supports_groups=True,
        streaming_mode=STREAMING_EDIT_IN_PLACE,
        text_chunk_limit=4000,
        # Inbound download cap; outbound media is capped lower by the bridge.
        max_media_mb=50.0,
    )

    def __init__(self, token: str, *, proxy: Optional[str] = None) -> None:
        super().__init__()
        self._token = token
        self._proxy = proxy
        self._bot: Optional[Bot] = None
        self._dispatcher: Optional[Dispatcher] = None
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._state = "stopped"
        self._bot_id: Optional[str] = None

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        session = None
        if self._proxy:
            from aiogram.client.session.aiohttp import AiohttpSession

            session = AiohttpSession(proxy=self._proxy)
        self._bot = Bot(token=self._token, session=session, default=DefaultBotProperties())
        self._dispatcher = Dispatcher()
        self._dispatcher.message.register(self._handle_message)
        self._state = "starting"
        # Long-polling must be the only consumer of updates.
        await self._bot.delete_webhook(drop_pending_updates=True)
        self._poll_task = asyncio.create_task(
            self._dispatcher.start_polling(self._bot),
            name="telegram-adapter-polling",
        )
        me = await self._bot.me()
        self._bot_id = str(me.id)
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
        )

    async def _media_notice(self, message: TelegramMessage) -> None:
        try:
            await message.answer(MEDIA_UNSUPPORTED_REPLY)
        except Exception:  # noqa: BLE001 — a failed notice must not break polling
            logger.debug("telegram: media notice failed", exc_info=True)

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
