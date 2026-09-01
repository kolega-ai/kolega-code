"""The Telegram platform adapter."""

from kolega_code.gateway.adapters.telegram.adapter import TelegramAdapter
from kolega_code.gateway.adapters.telegram.formatting import chunk_text, telegram_html

__all__ = ["TelegramAdapter", "chunk_text", "telegram_html"]
