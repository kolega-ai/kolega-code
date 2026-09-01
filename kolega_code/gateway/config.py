"""Gateway configuration, resolved from ``KOLEGA_GATEWAY_*`` environment variables.

v1 config is environment-only on purpose: the Telegram bot token and allowlists
are secrets/access-control that must not land in settings.json, and a daemon
reads its config once at startup. Provider/model selection is *not* configured
here — the gateway builds agents with the standard ``build_agent_config`` chain,
so ``KOLEGA_CODE_PROVIDER``/``KOLEGA_CODE_MODEL``/settings.json apply unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from kolega_code.cli.session_store import default_state_dir
from kolega_code.permissions import PermissionMode, normalize_permission_mode

ENV_ADAPTER = "KOLEGA_GATEWAY_ADAPTER"
ENV_PROJECT = "KOLEGA_GATEWAY_PROJECT"
ENV_STATE_DIR = "KOLEGA_GATEWAY_STATE_DIR"
ENV_ALLOWED_USERS = "KOLEGA_GATEWAY_ALLOWED_USERS"
ENV_PERMISSION_MODE = "KOLEGA_GATEWAY_PERMISSION_MODE"
ENV_REQUEST_TIMEOUT_SECONDS = "KOLEGA_GATEWAY_REQUEST_TIMEOUT_SECONDS"
ENV_MAX_SESSIONS = "KOLEGA_GATEWAY_MAX_SESSIONS"
ENV_SESSION_IDLE_TTL_SECONDS = "KOLEGA_GATEWAY_SESSION_IDLE_TTL_SECONDS"
ENV_EDIT_THROTTLE_SECONDS = "KOLEGA_GATEWAY_EDIT_THROTTLE_SECONDS"
ENV_STT = "KOLEGA_GATEWAY_STT"
ENV_TELEGRAM_TOKEN = "KOLEGA_GATEWAY_TELEGRAM_TOKEN"
ENV_TELEGRAM_PROXY = "KOLEGA_GATEWAY_TELEGRAM_PROXY"

DEFAULT_ADAPTER = "echo"
#: Where gateway sessions work when no --project/env is given. Never the
#: daemon's cwd: services start with "/" or a temp dir, and the gateway must
#: not silently anchor a chat-driven agent there.
DEFAULT_GATEWAY_WORKSPACE_NAME = "kolega-code-workspace"
#: Long enough for a human decision, short enough that a phone user who never
#: answers does not stall a turn for 15 minutes (the control-channel default).
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_SESSIONS = 50
DEFAULT_SESSION_IDLE_TTL_SECONDS = 3600.0
DEFAULT_EDIT_THROTTLE_SECONDS = 1.0

_TRUE_VALUES = {"1", "true", "yes", "on"}


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
    permission_mode: str = PermissionMode.ASK.value
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_sessions: int = DEFAULT_MAX_SESSIONS
    #: None disables idle eviction (useful in tests and short-lived runs).
    session_idle_ttl_seconds: Optional[float] = DEFAULT_SESSION_IDLE_TTL_SECONDS
    edit_throttle_seconds: float = DEFAULT_EDIT_THROTTLE_SECONDS
    stt_enabled: bool = False
    telegram_token: Optional[str] = None
    telegram_proxy: Optional[str] = None


def _env_str(env: Mapping[str, str], key: str) -> Optional[str]:
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _env_str(env, key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise GatewayConfigError(f"{key} must be a number, got {raw!r}.") from exc
    if value <= 0:
        raise GatewayConfigError(f"{key} must be positive, got {raw!r}.")
    return value


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _env_str(env, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise GatewayConfigError(f"{key} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise GatewayConfigError(f"{key} must be positive, got {raw!r}.")
    return value


def _env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = _env_str(env, key)
    if raw is None:
        return default
    return raw.lower() in _TRUE_VALUES


def _env_users(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = _env_str(env, ENV_ALLOWED_USERS)
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _resolve_state_dir(param: Optional[Path], env: Mapping[str, str]) -> Path:
    if param is not None:
        return param.expanduser().resolve()
    raw = _env_str(env, ENV_STATE_DIR) or env.get("KOLEGA_CODE_STATE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return default_state_dir(env=dict(env))


def _resolve_project(param: Optional[Path], env: Mapping[str, str]) -> Path:
    """Resolve the working directory for new gateway sessions.

    ``--project`` beats the environment variable beats ``~/kolega-code-workspace``.
    The daemon's launch cwd is deliberately never consulted.
    """
    if param is not None:
        return param.expanduser().resolve()
    raw = _env_str(env, ENV_PROJECT)
    if raw:
        return Path(raw).expanduser().resolve()
    return default_gateway_project()


def load_gateway_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    state_dir: Optional[Path] = None,
    project: Optional[Path] = None,
) -> GatewayConfig:
    """Build the gateway config from the process environment and explicit overrides."""
    loaded_env = dict(env if env is not None else os.environ)
    adapter = _env_str(loaded_env, ENV_ADAPTER) or DEFAULT_ADAPTER
    permission_mode = normalize_permission_mode(
        _env_str(loaded_env, ENV_PERMISSION_MODE),
        default=PermissionMode.ASK,
    ).value
    return GatewayConfig(
        adapter=adapter,
        project_path=_resolve_project(project, loaded_env),
        state_dir=_resolve_state_dir(state_dir, loaded_env),
        allowed_users=_env_users(loaded_env),
        permission_mode=permission_mode,
        request_timeout_seconds=_env_float(loaded_env, ENV_REQUEST_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        max_sessions=_env_int(loaded_env, ENV_MAX_SESSIONS, DEFAULT_MAX_SESSIONS),
        session_idle_ttl_seconds=_env_float(loaded_env, ENV_SESSION_IDLE_TTL_SECONDS, DEFAULT_SESSION_IDLE_TTL_SECONDS),
        edit_throttle_seconds=_env_float(loaded_env, ENV_EDIT_THROTTLE_SECONDS, DEFAULT_EDIT_THROTTLE_SECONDS),
        stt_enabled=_env_bool(loaded_env, ENV_STT),
        telegram_token=_env_str(loaded_env, ENV_TELEGRAM_TOKEN),
        telegram_proxy=_env_str(loaded_env, ENV_TELEGRAM_PROXY),
    )
