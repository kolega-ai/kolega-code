"""Gateway configuration, resolved from the persisted settings surface.

Gateway settings live in ``settings.json`` under the ``gateway`` section
(plus the top-level ``telegram_bot_token``), alongside every other user
setting. Every key has a built-in default — the stored section only needs
what the user explicitly configured, which ``kolega-code gateway telegram
setup`` writes for them. CLI flags (``--adapter``/``--project``) override the
persisted values; provider/model selection is *not* configured here — the
gateway builds agents with the standard ``build_agent_config`` chain.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kolega_code.cli.session_store import default_state_dir
from kolega_code.cli.settings import CliSettings, SettingsStore, SettingsStoreError
from kolega_code.gateway.stt import (
    DEFAULT_STT_PROVIDER,
    get_stt_provider_class,
    stt_provider_names,
)
from kolega_code.permissions import PermissionMode, normalize_permission_mode

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER = "echo"
#: Where gateway sessions work when nothing is configured. Never the
#: daemon's cwd: services start with "/" or a temp dir, and the gateway must
#: not silently anchor a chat-driven agent there.
DEFAULT_GATEWAY_WORKSPACE_NAME = "kolega-code-workspace"
#: Long enough for a human decision, short enough that a phone user who never
#: answers does not stall a turn for 15 minutes (the control-channel default).
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_SESSIONS = 50
DEFAULT_SESSION_IDLE_TTL_SECONDS = 3600.0
DEFAULT_EDIT_THROTTLE_SECONDS = 1.0
DEFAULT_PAIRING_TTL_SECONDS = 3600.0


class GatewayConfigError(ValueError):
    """Raised when gateway configuration is missing or invalid."""


def default_gateway_project() -> Path:
    """The gateway's fallback working directory: ``~/kolega-code-workspace``."""
    return Path.home() / DEFAULT_GATEWAY_WORKSPACE_NAME


@dataclass(frozen=True)
class GatewayConfig:
    """One gateway daemon's configuration, immutable after load."""

    adapter: str
    project_path: Path
    state_dir: Path
    allowed_users: tuple[str, ...] = ()
    #: Group chats allowed to reach the gateway (empty = all groups,
    #: mention-gated). Group ids are Telegram chat ids.
    group_ids: tuple[str, ...] = ()
    #: Whether unknown senders get a pairing code instead of silence. Only
    #: meaningful when an allowlist is configured.
    pairing_enabled: bool = False
    pairing_code_ttl_seconds: float = DEFAULT_PAIRING_TTL_SECONDS
    permission_mode: str = PermissionMode.ASK.value
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_sessions: int = DEFAULT_MAX_SESSIONS
    #: None disables idle eviction (useful in tests and short-lived runs).
    session_idle_ttl_seconds: Optional[float] = DEFAULT_SESSION_IDLE_TTL_SECONDS
    edit_throttle_seconds: float = DEFAULT_EDIT_THROTTLE_SECONDS
    stt_enabled: bool = False
    #: Pluggable remote speech-to-text provider (currently "groq" hosted
    #: Whisper). Configured under Tools → Voice transcription, like the
    #: web-search backend.
    stt_provider: str = DEFAULT_STT_PROVIDER
    #: Provider-specific model override; None applies the provider's default
    #: ("whisper-large-v3-turbo" for groq).
    stt_model: Optional[str] = None
    #: API key for the selected STT provider, resolved from the environment
    #: or the stored provider keys. Only groq needs one today.
    stt_api_key: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_proxy: Optional[str] = None


def _gateway_settings(state_dir: Path) -> tuple[dict[str, Any], Optional[CliSettings]]:
    """The stored gateway section (and token), tolerating hand-edits.

    A gateway daemon must not refuse to start because settings.json was
    touched; missing pieces fall back to defaults and the adapter's own
    startup error explains what to fix. The full ``CliSettings`` object rides
    along because top-level STT settings and provider API keys live outside
    the gateway section.
    """
    try:
        settings = SettingsStore(root=state_dir).load()
        return {"_telegram_bot_token": settings.telegram_bot_token, **settings.gateway}, settings
    except SettingsStoreError as exc:
        logger.warning("gateway: could not read settings (%s); using defaults", exc)
        return {}, None
    except Exception as exc:  # noqa: BLE001 — config lookup must be best-effort
        logger.warning("gateway: could not read gateway settings (%s); using defaults", exc)
        return {}, None


def _resolve_state_dir(param: Optional[Path]) -> Path:
    if param is not None:
        return param.expanduser().resolve()
    shared = os.environ.get("KOLEGA_CODE_STATE_DIR")
    if shared:
        return Path(shared).expanduser().resolve()
    return default_state_dir()


def _resolve_project(param: Optional[Path], stored: dict[str, Any]) -> Path:
    """``--project`` beats the stored value beats ``~/kolega-code-workspace``.

    The daemon's launch cwd is deliberately never consulted.
    """
    if param is not None:
        return param.expanduser().resolve()
    raw = stored.get("project")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return default_gateway_project()


def _resolve_stt(settings: Optional[CliSettings]) -> tuple[bool, str, Optional[str], Optional[str]]:
    """Resolve (enabled, provider, model, api_key) for voice transcription.

    All three knobs are top-level settings (Tools → Voice transcription).
    The API key follows the provider's ``key_env_var`` environment variable
    first, then the stored provider key (``api_keys``).
    """
    stt_enabled = bool(settings.stt_enabled) if settings is not None else False
    provider = (settings.stt_provider if settings is not None else None) or DEFAULT_STT_PROVIDER
    if provider not in stt_provider_names():
        logger.warning("gateway: unknown stt_provider %r; using %r", provider, DEFAULT_STT_PROVIDER)
        provider = DEFAULT_STT_PROVIDER
    model = settings.stt_model if settings is not None else None
    provider_cls = get_stt_provider_class(provider)
    env_name = provider_cls.key_env_var
    api_key = (os.environ.get(env_name) if env_name else None) or (
        settings.api_keys.get(provider) if settings is not None else None
    )
    return stt_enabled, provider, model, api_key


def load_gateway_config(
    *,
    state_dir: Optional[Path] = None,
    project: Optional[Path] = None,
    adapter: Optional[str] = None,
) -> GatewayConfig:
    """Build the gateway config from settings.json with CLI-flag overrides."""
    resolved_state_dir = _resolve_state_dir(state_dir)
    stored, settings = _gateway_settings(resolved_state_dir)
    adapter = adapter or stored.get("adapter") or DEFAULT_ADAPTER
    try:
        permission_mode = normalize_permission_mode(
            stored.get("permission_mode"),
            default=PermissionMode.ASK,
        ).value
    except ValueError:
        permission_mode = PermissionMode.ASK.value
    stt_enabled, stt_provider, stt_model, stt_api_key = _resolve_stt(settings)
    return GatewayConfig(
        adapter=str(adapter),
        project_path=_resolve_project(project, stored),
        state_dir=resolved_state_dir,
        allowed_users=tuple(str(item) for item in (stored.get("allowed_users") or ())),
        group_ids=tuple(str(item) for item in (stored.get("group_ids") or ())),
        pairing_enabled=bool(stored.get("pairing_enabled", False)),
        pairing_code_ttl_seconds=float(stored.get("pairing_code_ttl_seconds", DEFAULT_PAIRING_TTL_SECONDS)),
        permission_mode=permission_mode,
        request_timeout_seconds=float(stored.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
        max_sessions=int(stored.get("max_sessions", DEFAULT_MAX_SESSIONS)),
        session_idle_ttl_seconds=stored.get("session_idle_ttl_seconds", DEFAULT_SESSION_IDLE_TTL_SECONDS),
        edit_throttle_seconds=float(stored.get("edit_throttle_seconds", DEFAULT_EDIT_THROTTLE_SECONDS)),
        stt_enabled=stt_enabled,
        stt_provider=stt_provider,
        stt_model=stt_model,
        stt_api_key=stt_api_key,
        telegram_token=stored.get("_telegram_bot_token"),
        telegram_proxy=stored.get("telegram_proxy"),
    )
