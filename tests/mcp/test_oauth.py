from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from mcp.shared.auth import OAuthToken
from rich.console import Console

from kolega_code.mcp.config import MCPOAuthConfig, MCPServerConfig
from kolega_code.mcp.oauth import (
    LocalOAuthCallbackServer,
    MCPFileTokenStorage,
    MCPOAuthError,
    OAuthInteraction,
    build_oauth_provider,
    default_redirect_handler,
)
from kolega_code.mcp.state import MCPOAuthTokenStore


def _server(**oauth: Any) -> MCPServerConfig:
    return MCPServerConfig(
        id="example",
        url="https://mcp.example/mcp",
        oauth=MCPOAuthConfig(client_id="fake-client", **oauth),
    )


def test_explicitly_disabled_oauth_survives_round_trip() -> None:
    server = _server(enabled=False, client_secret="fake-secret")
    assert not server.oauth.enabled
    assert not MCPServerConfig.model_validate_json(server.model_dump_json()).oauth.enabled


@pytest.mark.parametrize(
    "oauth",
    [
        {"client_secret": "fake-secret"},
        {"client_secret_env": "FAKE_OAUTH_SECRET"},
        {"client_id": "fake-client", "token_endpoint_auth_method": "client_secret_basic"},
        {"client_id": "fake-client", "token_endpoint_auth_method": "client_secret_post"},
    ],
)
def test_incomplete_credentials_fail_without_exposing_secrets(oauth: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="client_id") as exc:
        MCPServerConfig.model_validate({"id": "example", "url": "https://mcp.example/mcp", "oauth": oauth})
    assert "fake-secret" not in str(exc.value)


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:bad/callback",
        "http://localhost:65536/callback",
        "http://fake-user:fake-password@localhost:1234/callback",
        "http://localhost:1234/callback#fragment",
    ],
)
def test_invalid_callback_uri_rejected_on_configuration(uri: str) -> None:
    with pytest.raises(ValueError):
        MCPOAuthConfig(redirect_uri=uri)


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
@pytest.mark.parametrize("suffix", ["", "/callback", "/callback?tenant=example"])
async def test_fixed_callback_uri_receives_code(host: str, suffix: str, unused_tcp_port: int) -> None:
    uri = f"http://{host}:{unused_tcp_port}{suffix}"
    async with LocalOAuthCallbackServer(uri, timeout_seconds=2) as callback:
        assert callback.redirect_uri == uri
        separator = "&" if "?" in uri else "?"
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(uri + separator + "code=fake-code&state=fake-state")
        assert response.status_code == 200
        assert await callback.wait_for_callback() == ("fake-code", "fake-state")


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
async def test_ephemeral_callback_preserves_query(host: str) -> None:
    async with LocalOAuthCallbackServer(f"http://{host}:0/callback?tenant=example") as callback:
        assert urlparse(callback.redirect_uri).query == "tenant=example"
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(callback.redirect_uri + "&code=fake-code&state=fake-state")
        assert response.status_code == 200
        assert await callback.wait_for_callback() == ("fake-code", "fake-state")


@pytest.mark.asyncio
async def test_occupied_callback_port_has_actionable_error(unused_tcp_port: int) -> None:
    async def ignore_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    async with await asyncio.start_server(ignore_client, "127.0.0.1", unused_tcp_port):
        with pytest.raises(MCPOAuthError, match="Could not bind OAuth callback"):
            async with LocalOAuthCallbackServer(f"http://127.0.0.1:{unused_tcp_port}/callback"):
                pytest.fail("An occupied port must not silently become an ephemeral callback")


@pytest.mark.asyncio
@pytest.mark.parametrize("rich_console", [False, True])
async def test_authorization_url_output_supports_cli_and_tui(rich_console: bool) -> None:
    stream = io.StringIO()
    output = Console(file=stream, width=40) if rich_console else stream
    url = "https://auth.example/authorize?state=fake-state&client_id=fake-client"
    await default_redirect_handler(url, open_browser=False, output=output)
    assert url in stream.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["client_id", "scope", "redirect_uri", "url", "secret_env"])
async def test_cached_tokens_follow_oauth_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    monkeypatch.setenv("FAKE_OAUTH_SECRET", "fake-old-secret")
    server = _server(client_secret_env="FAKE_OAUTH_SECRET")
    store = MCPOAuthTokenStore(tmp_path)
    tokens = OAuthToken(access_token="fake-access-token", token_type="Bearer")
    await MCPFileTokenStorage(server, store).set_tokens(tokens)
    assert await MCPFileTokenStorage(server, store).get_tokens() == tokens
    if changed == "url":
        server = server.model_copy(update={"url": "https://other.example/mcp"})
    elif changed == "secret_env":
        monkeypatch.setenv("FAKE_OAUTH_SECRET", "fake-new-secret")
    else:
        new_value = "http://127.0.0.1:1234/callback" if changed == "redirect_uri" else "new-value"
        server = server.model_copy(update={"oauth": server.oauth.model_copy(update={changed: new_value})})
    assert await MCPFileTokenStorage(server, store).get_tokens() is None
    persisted = store.path.read_text()
    assert "fake-old-secret" not in persisted
    assert "fake-new-secret" not in persisted


@pytest.mark.asyncio
async def test_cached_token_fingerprints_are_salted_and_derived_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(client_secret="fake-client-secret")
    store = MCPOAuthTokenStore(tmp_path)
    storage = MCPFileTokenStorage(server, store)
    tokens = OAuthToken(access_token="fake-access-token", token_type="Bearer")
    event_loop_thread = threading.get_ident()
    derivation_threads: list[int] = []
    pbkdf2_hmac = hashlib.pbkdf2_hmac

    def tracked_pbkdf2(algorithm: str, data: bytes, salt: bytes, iterations: int) -> bytes:
        derivation_threads.append(threading.get_ident())
        return pbkdf2_hmac(algorithm, data, salt, iterations)

    monkeypatch.setattr(hashlib, "pbkdf2_hmac", tracked_pbkdf2)
    await storage.set_tokens(tokens)
    first = store.get(server.id).config_fingerprint
    assert first is not None
    assert first.startswith("pbkdf2-sha256$600000$")
    assert await MCPFileTokenStorage(server, store).get_tokens() == tokens

    await storage.set_tokens(tokens)
    second = store.get(server.id).config_fingerprint
    assert second is not None
    assert first.split("$")[2] != second.split("$")[2]
    assert await MCPFileTokenStorage(server, store).get_tokens() == tokens
    assert derivation_threads and event_loop_thread not in derivation_threads
    assert "fake-client-secret" not in store.path.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fingerprint",
    [
        "a" * 64,
        "pbkdf2-sha256$1$" + "aa" * 16 + "$" + "bb" * 32,
        "pbkdf2-sha256$600000$not-hex$" + "bb" * 32,
        "pbkdf2-sha256$600000$aa$" + "bb" * 32,
    ],
)
async def test_legacy_or_invalid_fingerprints_require_new_authorization(tmp_path: Path, fingerprint: str) -> None:
    store = MCPOAuthTokenStore(tmp_path)
    store.set_tokens(
        "example",
        {"access_token": "fake-access-token", "token_type": "Bearer"},
        config_fingerprint=fingerprint,
    )
    assert await MCPFileTokenStorage(_server(), store).get_tokens() is None


@pytest.mark.asyncio
async def test_legacy_tokens_are_not_reused_for_pre_registered_client(tmp_path: Path) -> None:
    store = MCPOAuthTokenStore(tmp_path)
    store.set_tokens("example", {"access_token": "fake-legacy-token", "token_type": "Bearer"})
    assert await MCPFileTokenStorage(_server(), store).get_tokens() is None
    dcr_server = MCPServerConfig(id="example", url="https://mcp.example/mcp", oauth=MCPOAuthConfig(enabled=True))
    assert await MCPFileTokenStorage(dcr_server, store).get_tokens() is not None


@pytest.mark.asyncio
async def test_explicit_public_auth_method_is_used_for_dynamic_registration(tmp_path: Path) -> None:
    server = MCPServerConfig(
        id="example",
        url="https://mcp.example/mcp",
        oauth=MCPOAuthConfig(enabled=True, token_endpoint_auth_method="none"),
    )
    provider = await build_oauth_provider(server, MCPOAuthTokenStore(tmp_path))
    assert provider is not None
    from mcp.client.auth.utils import create_client_registration_request

    request = create_client_registration_request(None, provider.context.client_metadata, "https://mcp.example")
    assert json.loads(request.content)["token_endpoint_auth_method"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["client_secret_post", "client_secret_basic", "none"])
async def test_pre_registered_oauth_authorization_refresh_and_reconnect(
    tmp_path: Path, method: Literal["client_secret_post", "client_secret_basic", "none"]
) -> None:
    server = _server(
        client_secret="fake-secret" if method != "none" else None,
        token_endpoint_auth_method=method,
        redirect_uri="http://127.0.0.1:1234/callback",
        scope="read",
    )
    store = MCPOAuthTokenStore(tmp_path)
    authorization: dict[str, list[str]] = {}
    grants: list[str] = []
    redirects: list[str] = []

    async def redirect_handler(url: str) -> None:
        redirects.append(url)
        authorization.update(parse_qs(urlparse(url).query))

    async def callback_handler() -> tuple[str, str | None]:
        return "fake-code", authorization["state"][0]

    async def close() -> None:
        return None

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == server.url:
            if request.headers.get("Authorization") == "Bearer fake-access-token":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(
                401, headers={"WWW-Authenticate": 'Bearer resource_metadata="https://mcp.example/prm"'}
            )
        if url == "https://mcp.example/prm":
            return httpx.Response(
                200,
                json={
                    "resource": server.url,
                    "authorization_servers": ["https://auth.example"],
                    "scopes_supported": ["read", "write"],
                },
            )
        if url == "https://auth.example/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example",
                    "authorization_endpoint": "https://auth.example/authorize",
                    "token_endpoint": "https://auth.example/token",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                },
            )
        if url == "https://auth.example/token":
            data = parse_qs(request.content.decode())
            grants.append(data["grant_type"][0])
            assert data["client_id"] == ["fake-client"]
            if method == "client_secret_post":
                assert data["client_secret"] == ["fake-secret"]
                assert "Authorization" not in request.headers
            elif method == "client_secret_basic":
                assert "client_secret" not in data
                assert (
                    base64.b64decode(request.headers["Authorization"].removeprefix("Basic "))
                    == b"fake-client:fake-secret"
                )
            else:
                assert "client_secret" not in data
                assert "Authorization" not in request.headers
            if grants[-1] == "authorization_code":
                assert data["redirect_uri"] == [server.oauth.redirect_uri]
                digest = hashlib.sha256(data["code_verifier"][0].encode()).digest()
                assert authorization["code_challenge"] == [base64.urlsafe_b64encode(digest).decode().rstrip("=")]
                assert authorization["code_challenge_method"] == ["S256"]
                assert authorization["scope"] == ["read"]
            else:
                assert data["refresh_token"] == ["fake-refresh-token"]
            return httpx.Response(
                200,
                json={
                    "access_token": "fake-access-token",
                    "refresh_token": "fake-refresh-token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        pytest.fail(f"Unexpected OAuth request (including any dynamic registration): {request.method} {url}")

    interaction = OAuthInteraction(server.oauth.redirect_uri or "", redirect_handler, callback_handler, close)
    provider = await build_oauth_provider(server, store, interaction=interaction)
    assert provider is not None
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), auth=provider) as client:
        assert (await client.get(server.url or "")).status_code == 200
        provider.context.token_expiry_time = 1.0
        assert (await client.get(server.url or "")).status_code == 200
    assert grants == ["authorization_code", "refresh_token"]

    # Expiration after a restart must refresh at the discovered auth server,
    # without needing another interactive browser session.
    saved = store.load()
    saved_context = saved.servers[server.id].oauth_context
    assert saved_context is not None
    saved_context["token_expiry_time"] = 1.0
    store.save(saved)
    provider = await build_oauth_provider(server, store)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), auth=provider) as client:
        assert (await client.get(server.url or "")).status_code == 200
    assert grants == ["authorization_code", "refresh_token", "refresh_token"]
    assert len(redirects) == 1
    assert store.get(server.id).client_info is None

    # A new non-interactive connection uses the saved token without registration.
    provider = await build_oauth_provider(server, store)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), auth=provider) as client:
        assert (await client.get(server.url or "")).status_code == 200
    assert grants == ["authorization_code", "refresh_token", "refresh_token"]


@pytest.mark.asyncio
async def test_oauth_rejects_wrong_callback_state(tmp_path: Path) -> None:
    async def redirect_handler(url: str) -> None:
        return None

    async def callback_handler() -> tuple[str, str | None]:
        return "fake-code", "wrong-state"

    async def close() -> None:
        return None

    interaction = OAuthInteraction("http://127.0.0.1:1234/callback", redirect_handler, callback_handler, close)
    provider = await build_oauth_provider(_server(), MCPOAuthTokenStore(tmp_path), interaction=interaction)
    assert provider is not None
    await provider._initialize()
    with pytest.raises(Exception, match="State parameter mismatch"):
        await provider._perform_authorization_code_grant()
