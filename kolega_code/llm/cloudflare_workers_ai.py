"""Connection details for the Cloudflare Workers AI provider (no vendor SDK imports)."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

PROVIDER = "cloudflare_workers_ai"
API_TOKEN_ENV = "CLOUDFLARE_API_TOKEN"
ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
GATEWAY_ID_ENV = "CLOUDFLARE_AI_GATEWAY_ID"
API_ROOT = "https://api.cloudflare.com/client/v4"
SESSION_AFFINITY_HEADER = "x-session-affinity"
GATEWAY_HEADER = "cf-aig-gateway-id"

_ACCOUNT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_GATEWAY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_account_id(value: Any) -> Optional[str]:
    """Return the canonical account ID, ``None`` when blank, or raise ``ValueError``.

    The ID is interpolated into the request path, so anything other than
    Cloudflare's 32-character hex form is rejected outright.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not _ACCOUNT_ID_PATTERN.fullmatch(text):
        raise ValueError(
            "The Cloudflare account ID must be the 32-character hex ID shown in the Cloudflare dashboard "
            "(Workers AI → Use REST API)."
        )
    return text


def normalize_gateway_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _GATEWAY_ID_PATTERN.fullmatch(text):
        raise ValueError(
            "The Cloudflare AI Gateway ID may only contain letters, digits, hyphens, and underscores "
            "(at most 64 characters)."
        )
    return text


def base_url(account_id: str) -> str:
    return f"{API_ROOT}/accounts/{account_id}/ai/v1"


def models_search_url(account_id: str) -> str:
    return f"{API_ROOT}/accounts/{account_id}/ai/models/search"


def gateway_headers(gateway_id: Optional[str]) -> Dict[str, str]:
    return {GATEWAY_HEADER: gateway_id} if gateway_id else {}


def connection_kwargs(account_id: Optional[str], gateway_id: Optional[str] = None) -> Dict[str, Any]:
    """``LLMClient`` keyword arguments for one account (and optional gateway)."""
    if not account_id:
        raise ValueError(
            f"Cloudflare Workers AI needs an account ID: set {ACCOUNT_ID_ENV} or enter it in Settings → Providers."
        )
    kwargs: Dict[str, Any] = {"base_url": base_url(account_id)}
    headers = gateway_headers(gateway_id)
    if headers:
        kwargs["default_headers"] = headers
    return kwargs


def env_connection_kwargs(environ: Mapping[str, str]) -> Dict[str, Any]:
    return connection_kwargs(
        normalize_account_id(environ.get(ACCOUNT_ID_ENV)),
        normalize_gateway_id(environ.get(GATEWAY_ID_ENV)),
    )
