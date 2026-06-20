import base64
import json

import httpx
import pytest

from kolega_code.auth import constants
from kolega_code.auth._jwt import decode_jwt_payload, extract_chatgpt_claims
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens, TokenRefreshError


def make_id_token(*, account_id="acct_123", plan_type="pro", email="user@example.com") -> str:
    """Build an unsigned JWT whose payload carries the ChatGPT auth claims."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    payload_obj = {
        "email": email,
        constants.AUTH_CLAIMS_NAMESPACE: {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": plan_type,
            "chatgpt_user_id": "user_1",
        },
    }
    payload = base64.urlsafe_b64encode(json.dumps(payload_obj).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


def test_decode_jwt_payload_handles_padding_and_garbage() -> None:
    token = make_id_token()
    assert decode_jwt_payload(token)["email"] == "user@example.com"
    assert decode_jwt_payload("not-a-jwt") == {}
    assert decode_jwt_payload("a.b") == {}  # middle segment not valid base64 json


def test_extract_chatgpt_claims() -> None:
    claims = extract_chatgpt_claims(make_id_token(account_id="acct_x", plan_type="plus"))
    assert claims["account_id"] == "acct_x"
    assert claims["plan_type"] == "plus"
    assert claims["email"] == "user@example.com"


def test_is_expired_respects_leeway() -> None:
    tokens = OAuthTokens(access_token="a", refresh_token="r", expires_at=1000.0)
    assert tokens.is_expired(leeway=300, now=701)  # within 300s window
    assert not tokens.is_expired(leeway=300, now=699)
    assert tokens.is_expired(leeway=0, now=1000)


def test_from_token_response_computes_expiry_and_claims() -> None:
    payload = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "id_token": make_id_token(account_id="acct_9", plan_type="business"),
        "expires_in": 3600,
    }
    tokens = OAuthTokens.from_token_response(payload, now=1000.0)
    assert tokens.access_token == "new-access"
    assert tokens.refresh_token == "new-refresh"
    assert tokens.expires_at == 4600.0
    assert tokens.account_id == "acct_9"
    assert tokens.plan_type == "business"


def test_from_token_response_carries_over_missing_refresh_token() -> None:
    previous = OAuthTokens(
        access_token="old", refresh_token="keep-me", id_token=make_id_token(), account_id="acct_123"
    )
    payload = {"access_token": "rotated", "expires_in": 60}  # no refresh_token, no id_token
    tokens = OAuthTokens.from_token_response(payload, previous=previous, now=0.0)
    assert tokens.refresh_token == "keep-me"
    assert tokens.account_id == "acct_123"  # carried from previous


@pytest.mark.asyncio
async def test_authorization_refreshes_when_expired_and_persists() -> None:
    persisted: list[OAuthTokens] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        assert request.url == httpx.URL(constants.TOKEN_URL)
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "refresh_token=the-refresh" in body
        return httpx.Response(
            200,
            json={
                "access_token": "fresh-access",
                "refresh_token": "the-refresh",
                "id_token": make_id_token(account_id="acct_42"),
                "expires_in": 3600,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    expired = OAuthTokens(access_token="stale", refresh_token="the-refresh", expires_at=0.0, account_id="acct_42")
    manager = ChatGPTTokenManager(expired, persist=persisted.append, http_client=client)

    access, account_id = await manager.authorization()

    assert access == "fresh-access"
    assert account_id == "acct_42"
    assert calls["count"] == 1
    assert persisted and persisted[-1].access_token == "fresh-access"
    await client.aclose()


@pytest.mark.asyncio
async def test_authorization_skips_refresh_when_token_valid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be hit
        raise AssertionError("refresh should not be called for a valid token")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    valid = OAuthTokens(access_token="good", refresh_token="r", expires_at=10**12, account_id="acct_1")
    manager = ChatGPTTokenManager(valid, http_client=client)

    access, account_id = await manager.authorization()

    assert access == "good"
    assert account_id == "acct_1"
    await client.aclose()


@pytest.mark.asyncio
async def test_refresh_failure_raises_token_refresh_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    expired = OAuthTokens(access_token="stale", refresh_token="revoked", expires_at=0.0)
    manager = ChatGPTTokenManager(expired, http_client=client)

    with pytest.raises(TokenRefreshError):
        await manager.authorization()
    await client.aclose()
