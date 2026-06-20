"""OAuth token model and refresh manager for ChatGPT-subscription auth.

This module is intentionally free of CLI/settings imports: persistence is
injected as a callback so the core stays decoupled from where tokens live on
disk.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel, Field

from . import constants
from ._jwt import extract_chatgpt_claims

# Scope sent on a refresh-token grant (narrower than the initial authorize scope,
# matching Codex).
REFRESH_SCOPE = "openid profile email"


class OAuthError(RuntimeError):
    """Base error for the ChatGPT OAuth flow (token endpoint / transport)."""


class TokenRefreshError(OAuthError):
    """Raised when the refresh-token grant fails (expired/revoked refresh token)."""


async def request_token(
    data: dict[str, str],
    *,
    token_url: str = constants.TOKEN_URL,
    http_client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """POST to an OAuth token endpoint (form-encoded) and return the JSON body.

    Used by both the authorization-code exchange and the refresh-token grant.
    Raises :class:`OAuthError` on any transport/HTTP error.
    """
    try:
        if http_client is not None:
            response = await http_client.post(token_url, data=data)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(token_url, data=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise OAuthError(f"token endpoint returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise OAuthError(f"token request failed: {exc}") from exc


class OAuthTokens(BaseModel):
    """Persisted ChatGPT OAuth credentials for a single account."""

    access_token: str
    refresh_token: str
    id_token: str = ""
    expires_at: float = Field(default=0.0, description="Unix epoch seconds at which access_token expires")
    account_id: Optional[str] = None
    plan_type: Optional[str] = None
    email: Optional[str] = None

    def is_expired(self, leeway: float = constants.REFRESH_LEEWAY_SECONDS, now: Optional[float] = None) -> bool:
        """True when the access token is within ``leeway`` seconds of expiry."""
        current = time.time() if now is None else now
        return current >= (self.expires_at - leeway)

    @classmethod
    def from_token_response(
        cls,
        payload: dict[str, Any],
        *,
        previous: Optional["OAuthTokens"] = None,
        now: Optional[float] = None,
    ) -> "OAuthTokens":
        """Build tokens from an OAuth token-endpoint JSON response.

        Tolerant of partial refresh responses: a missing ``refresh_token`` keeps
        the previous one, and claims (account id, plan, email) are re-derived from
        a fresh ``id_token`` when present, else carried over from ``previous``.
        """
        issued_at = time.time() if now is None else now
        expires_in = float(payload.get("expires_in", 0) or 0)
        id_token = payload.get("id_token") or (previous.id_token if previous else "")
        claims = extract_chatgpt_claims(id_token) if id_token else {}
        refresh_token = payload.get("refresh_token") or (previous.refresh_token if previous else "")
        return cls(
            access_token=payload["access_token"],
            refresh_token=refresh_token,
            id_token=id_token,
            expires_at=issued_at + expires_in if expires_in else (previous.expires_at if previous else issued_at),
            account_id=claims.get("account_id") or (previous.account_id if previous else None),
            plan_type=claims.get("plan_type") or (previous.plan_type if previous else None),
            email=claims.get("email") or (previous.email if previous else None),
        )


PersistCallback = Callable[[OAuthTokens], None]


class ChatGPTTokenManager:
    """Holds the current tokens and refreshes them lazily / on demand.

    ``authorization()`` returns a *fresh* bearer token + account id, refreshing
    when within the leeway window. ``refresh()`` forces a refresh (used after a
    401). Refreshed tokens are pushed to the injected ``persist`` callback so
    they survive restarts.
    """

    def __init__(
        self,
        tokens: OAuthTokens,
        *,
        client_id: Optional[str] = None,
        persist: Optional[PersistCallback] = None,
        token_url: str = constants.TOKEN_URL,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._tokens = tokens
        self._client_id = client_id or constants.client_id()
        self._persist = persist
        self._token_url = token_url
        self._http_client = http_client
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> OAuthTokens:
        return self._tokens

    @property
    def account_id(self) -> Optional[str]:
        return self._tokens.account_id

    async def authorization(self) -> tuple[str, str]:
        """Return ``(access_token, account_id)``, refreshing if near expiry."""
        if self._tokens.is_expired():
            await self._refresh()
        return self._tokens.access_token, self._tokens.account_id or ""

    async def refresh(self) -> None:
        """Force a refresh regardless of expiry (e.g. after a 401)."""
        await self._refresh(force=True)

    async def _refresh(self, force: bool = False) -> None:
        async with self._lock:
            # Another coroutine may have refreshed while we waited for the lock.
            if not force and not self._tokens.is_expired():
                return
            data = {
                "client_id": self._client_id,
                "grant_type": "refresh_token",
                "refresh_token": self._tokens.refresh_token,
                "scope": REFRESH_SCOPE,
            }
            try:
                payload = await request_token(data, token_url=self._token_url, http_client=self._http_client)
            except OAuthError as exc:
                raise TokenRefreshError(
                    f"ChatGPT token refresh failed: {exc}; sign in again with /login chatgpt."
                ) from exc
            new_tokens = OAuthTokens.from_token_response(payload, previous=self._tokens)
            self._tokens = new_tokens
            if self._persist is not None:
                self._persist(new_tokens)
