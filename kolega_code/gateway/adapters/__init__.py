"""Platform adapters: normalize chat platforms onto the gateway envelope."""

from __future__ import annotations

import importlib
from typing import Any

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
from kolega_code.gateway.config import GatewayConfig, GatewayConfigError

__all__ = [
    "AdapterCapabilities",
    "Attachment",
    "ButtonOption",
    "ChatRef",
    "GatewayAdapter",
    "GatewayConfigError",
    "InboundMessage",
    "ReplyContext",
    "UnsupportedCapability",
    "build_adapter",
    "adapter_names",
]

#: Adapter id -> (module path, class name). Loaded lazily so an adapter's
#: third-party SDK (aiogram, …) is only imported when that adapter is used, and
#: so the registry can reference adapters that ship in other modules.
_ADAPTER_CLASSES: dict[str, tuple[str, str]] = {
    "echo": ("kolega_code.gateway.adapters.echo", "EchoAdapter"),
    "telegram": ("kolega_code.gateway.adapters.telegram", "TelegramAdapter"),
}


def adapter_names() -> list[str]:
    """Adapter ids accepted by ``build_adapter``, sorted for CLI help text."""
    return sorted(_ADAPTER_CLASSES)


def _adapter_kwargs(config: GatewayConfig, name: str) -> dict[str, Any]:
    """Constructor arguments an adapter reads from the gateway config."""
    if name == "telegram":
        if not config.telegram_token:
            raise GatewayConfigError(
                "No Telegram bot token is configured. Create a bot with @BotFather, "
                "then run `kolega-code gateway telegram setup` to save its token."
            )
        return {
            "token": config.telegram_token,
            "proxy": config.telegram_proxy,
            "media_dir": config.state_dir / "gateway_media",
        }
    return {}


def build_adapter(config: GatewayConfig) -> GatewayAdapter:
    """Instantiate the configured adapter, validating the adapter id."""
    if config.adapter not in _ADAPTER_CLASSES:
        valid = ", ".join(adapter_names())
        raise GatewayConfigError(f"Unknown gateway adapter {config.adapter!r}. Valid adapters: {valid}")
    module_path, class_name = _ADAPTER_CLASSES[config.adapter]
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    return adapter_class(**_adapter_kwargs(config, config.adapter))
