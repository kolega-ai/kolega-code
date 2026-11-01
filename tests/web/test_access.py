from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.types import Message, Scope
from starlette.websockets import WebSocketDisconnect

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.web.server import ServerConfig, create_app

BASE_URL = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765"
TOKEN = "fake-access-token-for-regression-tests"
WRONG_TOKEN = "fake-wrong-access-token"
TOKEN_COOKIE = "kc_session_token"
API_PATH = "/api/sessions"
REJECTED_STREAM = f"{WS_URL}/api/sessions/nonexistent/stream"
PROXY_ORIGIN = "https://viewer.example.invalid:9443"

BAD_HOST_HEADERS = [
    pytest.param([("host", host)], id=name)
    for name, host in (
        ("empty", ""),
        ("foreign", "attacker.invalid:8765"),
        ("localhost-suffix", "localhost.attacker.invalid:8765"),
        ("ipv4-suffix", "127.0.0.1.attacker.invalid:8765"),
        ("other-loopback", "127.0.0.2:8765"),
        ("userinfo-local", "attacker.invalid@localhost:8765"),
        ("userinfo-foreign", "localhost@attacker.invalid:8765"),
        ("path", "localhost:8765/path"),
        ("scheme", "http://localhost:8765"),
        ("query", "localhost:8765?redirect=attacker.invalid"),
        ("fragment", "localhost:8765#fragment"),
        ("non-numeric-port", "localhost:not-a-port"),
        ("overflow-port", "localhost:65536"),
        ("negative-port", "localhost:-1"),
        ("empty-port", "localhost:"),
        ("multiple-ports", "localhost:8765:80"),
        ("unbracketed-ipv6", "::1:8765"),
        ("unclosed-ipv6", "[::1"),
        ("ipv6-bad-port", "[::1]:bad"),
        ("other-ipv6", "[::2]:8765"),
        ("comma-separated", "localhost:8765,attacker.invalid"),
        ("leading-space", " localhost:8765"),
        ("trailing-space", "localhost:8765 "),
    )
] + [
    pytest.param([("host", "localhost:8765"), ("host", "localhost:8765")], id="duplicate-identical"),
    pytest.param([("host", "localhost:8765"), ("host", "attacker.invalid")], id="duplicate-foreign-last"),
    pytest.param([("host", "attacker.invalid"), ("host", "localhost:8765")], id="duplicate-foreign-first"),
]

BAD_ORIGIN_HEADERS = [
    pytest.param([("origin", origin)], id=name)
    for name, origin in (
        ("empty", ""),
        ("null", "null"),
        ("foreign", "http://attacker.invalid:8765"),
        ("localhost-suffix", "http://localhost.attacker.invalid:8765"),
        ("different-loopback-host", "http://localhost:8765"),
        ("different-port", "http://127.0.0.1:8766"),
        ("omitted-port", "http://127.0.0.1"),
        ("different-scheme", "https://127.0.0.1:8765"),
        ("websocket-scheme", "ws://127.0.0.1:8765"),
        ("file-scheme", "file://127.0.0.1:8765"),
        ("no-scheme", "127.0.0.1:8765"),
        ("userinfo", "http://attacker.invalid@127.0.0.1:8765"),
        ("password", "http://fake-user:fake-password@127.0.0.1:8765"),
        ("path", f"{BASE_URL}/path"),
        ("trailing-slash", f"{BASE_URL}/"),
        ("query", f"{BASE_URL}?token=fake"),
        ("fragment", f"{BASE_URL}#fragment"),
        ("non-numeric-port", "http://127.0.0.1:bad"),
        ("overflow-port", "http://127.0.0.1:65536"),
        ("empty-port", "http://127.0.0.1:"),
        ("unbracketed-ipv6", "http://::1:8765"),
        ("unclosed-ipv6", "http://[::1:8765"),
        ("wildcard", "http://*.localhost:8765"),
        ("leading-space", f" {BASE_URL}"),
        ("trailing-space", f"{BASE_URL} "),
        ("space-separated", f"{BASE_URL} http://attacker.invalid"),
        ("comma-separated", f"{BASE_URL},http://attacker.invalid"),
    )
] + [
    pytest.param([("origin", BASE_URL), ("origin", BASE_URL)], id="duplicate-identical"),
    pytest.param([("origin", BASE_URL), ("origin", "http://attacker.invalid")], id="duplicate-foreign-last"),
    pytest.param([("origin", "http://attacker.invalid"), ("origin", BASE_URL)], id="duplicate-foreign-first"),
]

AUTH_CASES = [
    pytest.param(None, None, None, False, id="missing"),
    pytest.param(f"Bearer {TOKEN}", None, None, True, id="header"),
    pytest.param(None, TOKEN, None, True, id="query"),
    pytest.param(None, None, TOKEN, True, id="cookie"),
    pytest.param(f"Bearer {WRONG_TOKEN}", None, None, False, id="wrong-header"),
    pytest.param(None, WRONG_TOKEN, None, False, id="wrong-query"),
    pytest.param(None, None, WRONG_TOKEN, False, id="wrong-cookie"),
    pytest.param(f"Bearer {TOKEN}", WRONG_TOKEN, WRONG_TOKEN, True, id="header-beats-query-and-cookie"),
    pytest.param(f"Bearer {WRONG_TOKEN}", TOKEN, TOKEN, False, id="wrong-header-blocks-fallback"),
    pytest.param("", TOKEN, TOKEN, False, id="empty-header-blocks-fallback"),
    pytest.param(f"Basic {TOKEN}", TOKEN, TOKEN, False, id="wrong-header-scheme-blocks-fallback"),
    pytest.param(None, TOKEN, WRONG_TOKEN, True, id="query-beats-cookie"),
    pytest.param(None, WRONG_TOKEN, TOKEN, False, id="wrong-query-blocks-cookie"),
    pytest.param(None, "", TOKEN, False, id="empty-query-blocks-cookie"),
]


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(root=tmp_path / "access-state")


def _client(store: SessionStore) -> TestClient:
    return TestClient(create_app(ServerConfig(store=store, token=TOKEN)), base_url=BASE_URL)


async def _seed(store: SessionStore, project: Path) -> str:
    record = store.create(project, "code", {"model": "test"}, title="Access regression")
    events = FileSessionEventStore(store.journal(record.session_id))
    await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.TURN_STARTED,
            content={"turn_id": "access-turn", "user_text": "Exercise the real session stream"},
        )
    )
    return record.session_id


def _assert_stream(socket: WebSocketTestSession, session_id: str) -> None:
    event = json.loads(socket.receive_text())
    assert event["session_id"] == session_id
    assert event["event_type"] == KnownEventType.TURN_STARTED
    assert event["content"]["turn_id"] == "access-turn"
    assert isinstance(event["seq"], int) and event["seq"] > 0


def test_token_is_a_required_configuration_argument() -> None:
    assert inspect.signature(ServerConfig).parameters["token"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "token", [None, "", " ", "fake token", "fake\ttoken", "fake\ntoken", "\x00fake", "\x7f", "faké"]
)
def test_create_app_rejects_invalid_tokens_eagerly(store: SessionStore, token: Any) -> None:
    config = ServerConfig(store=store, token=token)
    with pytest.raises(ValueError):
        create_app(config)


@pytest.mark.parametrize("token", [TOKEN, "fake!#$%&*+-._~", "".join(chr(value) for value in range(33, 127))])
def test_create_app_accepts_visible_ascii_tokens(store: SessionStore, token: str) -> None:
    assert create_app(ServerConfig(store=store, token=token)) is not None


@pytest.mark.parametrize(
    "host",
    [
        "",
        "*",
        "*.example.invalid",
        "http://localhost",
        "localhost:8765",
        "127.0.0.1:8765",
        "[::1]:8765",
        "[::1",
        "localhost/path",
        "fake-user@localhost",
        "localhost?query",
        "localhost#fragment",
        "bad host",
        "localhost,attacker.invalid",
    ],
)
def test_create_app_rejects_invalid_host_allowlist_eagerly(store: SessionStore, host: str) -> None:
    config = ServerConfig(store=store, token=TOKEN, allowed_hosts=frozenset({host}))
    with pytest.raises(ValueError):
        create_app(config)


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "null",
        "*",
        "https://*.example.invalid",
        "viewer.example.invalid",
        "ws://viewer.example.invalid",
        "file://viewer.example.invalid",
        "https://fake-user@viewer.example.invalid",
        "https://fake-user:fake-password@viewer.example.invalid",
        f"{PROXY_ORIGIN}/",
        f"{PROXY_ORIGIN}/path",
        f"{PROXY_ORIGIN}?query",
        f"{PROXY_ORIGIN}#fragment",
        "https://viewer.example.invalid:bad",
        "https://viewer.example.invalid:65536",
        "https://viewer.example.invalid:",
        "https://[::1",
        f" {PROXY_ORIGIN}",
        f"{PROXY_ORIGIN} ",
        f"{PROXY_ORIGIN},https://attacker.invalid",
    ],
)
def test_create_app_rejects_invalid_origin_allowlist_eagerly(store: SessionStore, origin: str) -> None:
    config = ServerConfig(store=store, token=TOKEN, allowed_origins=frozenset({origin}))
    with pytest.raises(ValueError):
        create_app(config)


@pytest.mark.parametrize("host_headers", BAD_HOST_HEADERS)
@pytest.mark.parametrize("token", [TOKEN, WRONG_TOKEN, None], ids=["valid-token", "wrong-token", "no-token"])
def test_http_rejects_untrusted_or_malformed_host_before_auth(
    store: SessionStore, host_headers: list[tuple[str, str]], token: str | None
) -> None:
    with _client(store) as client:
        response = client.get(
            API_PATH, headers=httpx.Headers(host_headers), params={"token": token} if token is not None else {}
        )
    assert response.status_code == 400
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("host_headers", BAD_HOST_HEADERS)
@pytest.mark.parametrize("token", [TOKEN, WRONG_TOKEN, None], ids=["valid-token", "wrong-token", "no-token"])
def test_websocket_rejects_untrusted_or_malformed_host_before_auth(
    store: SessionStore, host_headers: list[tuple[str, str]], token: str | None
) -> None:
    with _client(store) as client:
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                REJECTED_STREAM,
                headers=httpx.Headers(host_headers),
                params={"token": token} if token is not None else {},
            ):
                pytest.fail("Untrusted Host reached an accepted WebSocket")
    assert caught.value.code == 4400


@pytest.mark.parametrize("origin_headers", BAD_ORIGIN_HEADERS)
@pytest.mark.parametrize("token", [TOKEN, WRONG_TOKEN, None], ids=["valid-token", "wrong-token", "no-token"])
def test_http_rejects_untrusted_or_malformed_origin_before_auth(
    store: SessionStore, origin_headers: list[tuple[str, str]], token: str | None
) -> None:
    with _client(store) as client:
        response = client.get(
            API_PATH, headers=httpx.Headers(origin_headers), params={"token": token} if token is not None else {}
        )
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("origin_headers", BAD_ORIGIN_HEADERS)
@pytest.mark.parametrize("token", [TOKEN, WRONG_TOKEN, None], ids=["valid-token", "wrong-token", "no-token"])
def test_websocket_rejects_untrusted_or_malformed_origin_before_auth(
    store: SessionStore, origin_headers: list[tuple[str, str]], token: str | None
) -> None:
    with _client(store) as client:
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                REJECTED_STREAM,
                headers=httpx.Headers(origin_headers),
                params={"token": token} if token is not None else {},
            ):
                pytest.fail("Untrusted Origin reached an accepted WebSocket")
    assert caught.value.code == 4403


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "websocket"])
async def test_missing_host_is_rejected_without_minting_a_cookie(store: SessionStore, transport: str) -> None:
    app = create_app(ServerConfig(store=store, token=TOKEN))
    scope: Scope = {
        "type": transport,
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http" if transport == "http" else "ws",
        "path": API_PATH if transport == "http" else "/api/sessions/nonexistent/stream",
        "raw_path": b"/api/sessions",
        "root_path": "",
        "query_string": f"token={TOKEN}".encode(),
        "headers": [],
        "server": ("127.0.0.1", 8765),
        "client": ("127.0.0.1", 12345),
        "subprotocols": [],
        "state": {},
    }
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    incoming.put_nowait(
        {"type": "http.request", "body": b"", "more_body": False}
        if transport == "http"
        else {"type": "websocket.connect"}
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return await incoming.get()

    async def send(message: Message) -> None:
        sent.append(message)

    # TestClient synthesizes Host even when it is removed from the httpx request.
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    if transport == "http":
        start = next(message for message in sent if message["type"] == "http.response.start")
        assert start["status"] == 400
        assert not any(name.lower() == b"set-cookie" for name, _ in start.get("headers", []))
    else:
        assert not any(message["type"] == "websocket.accept" for message in sent)
        close = next(message for message in sent if message["type"] == "websocket.close")
        assert close["code"] == 4400


@pytest.mark.parametrize("header,query,cookie,accepted", AUTH_CASES)
def test_http_token_sources_and_precedence(
    store: SessionStore, header: str | None, query: str | None, cookie: str | None, accepted: bool
) -> None:
    with _client(store) as client:
        if cookie is not None:
            client.cookies.set(TOKEN_COOKIE, cookie)
        response = client.get(
            API_PATH,
            headers={"authorization": header} if header is not None else {},
            params={"token": query} if query is not None else {},
        )
    assert response.status_code == (200 if accepted else 401)


@pytest.mark.asyncio
@pytest.mark.parametrize("header,query,cookie,accepted", AUTH_CASES)
async def test_websocket_token_sources_and_precedence(
    store: SessionStore,
    tmp_path: Path,
    header: str | None,
    query: str | None,
    cookie: str | None,
    accepted: bool,
) -> None:
    session_id = await _seed(store, tmp_path)
    with _client(store) as client:
        if cookie is not None:
            client.cookies.set(TOKEN_COOKIE, cookie)
        headers = {"authorization": header} if header is not None else {}
        params = {"token": query} if query is not None else {}
        url = f"{WS_URL}/api/sessions/{session_id}/stream"
        if accepted:
            with client.websocket_connect(url, headers=headers, params=params) as socket:
                _assert_stream(socket, session_id)
        else:
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(url, headers=headers, params=params):
                    pytest.fail("Invalid credential reached an accepted WebSocket")
            assert caught.value.code == 4401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme,host,origin",
    [
        ("http", "127.0.0.1:8765", None),
        ("http", "LOCALHOST:8765", None),
        ("http", "[::1]:8765", None),
        ("http", "127.0.0.1:8765", BASE_URL),
        ("http", "LOCALHOST:8765", "http://localhost:8765"),
        ("http", "[0:0:0:0:0:0:0:1]:8765", "http://[::1]:8765"),
        ("http", "localhost", "http://LOCALHOST:80"),
        ("https", "127.0.0.1:8765", "https://127.0.0.1:8765"),
        ("https", "localhost", "https://localhost:443"),
    ],
)
async def test_loopback_hosts_and_actual_same_origin_are_accepted(
    store: SessionStore, tmp_path: Path, scheme: str, host: str, origin: str | None
) -> None:
    session_id = await _seed(store, tmp_path)
    headers = {"host": host, "authorization": f"Bearer {TOKEN}"}
    if origin is not None:
        headers["origin"] = origin
    with _client(store) as client:
        assert client.get(f"{scheme}://127.0.0.1:8765{API_PATH}", headers=headers).status_code == 200
        ws_scheme = "wss" if scheme == "https" else "ws"
        with client.websocket_connect(
            f"{ws_scheme}://127.0.0.1:8765/api/sessions/{session_id}/stream", headers=headers
        ) as socket:
            _assert_stream(socket, session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host,accepted",
    [
        ("WORKSTATION.LAN:8765", True),
        ("192.0.2.10:8765", True),
        ("[2001:0db8:0:0:0:0:0:1]:8765", True),
        ("sub.workstation.lan:8765", False),
        ("workstation.lan.attacker.invalid:8765", False),
        ("192.0.2.11:8765", False),
        ("[2001:db8::2]:8765", False),
    ],
)
async def test_configured_lan_hosts_match_exactly_and_normalize_ips(
    store: SessionStore, tmp_path: Path, host: str, accepted: bool
) -> None:
    session_id = await _seed(store, tmp_path)
    config = ServerConfig(
        store=store,
        token=TOKEN,
        allowed_hosts=frozenset({"127.0.0.1", "workstation.lan", "192.0.2.10", "2001:db8::1"}),
    )
    headers = {"host": host, "authorization": f"Bearer {TOKEN}"}
    with TestClient(create_app(config), base_url=BASE_URL) as client:
        response = client.get(API_PATH, headers=headers)
        assert response.status_code == (200 if accepted else 400)
        if accepted:
            with client.websocket_connect(f"{WS_URL}/api/sessions/{session_id}/stream", headers=headers) as socket:
                _assert_stream(socket, session_id)
        else:
            assert "set-cookie" not in response.headers
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(REJECTED_STREAM, headers=headers):
                    pytest.fail("A non-allowlisted LAN host was accepted")
            assert caught.value.code == 4400


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize(
    "host,origin,status",
    [
        ("127.0.0.1:8765", PROXY_ORIGIN, 200),
        ("VIEWER.EXAMPLE.INVALID:8765", None, 200),
        ("viewer.example.invalid:9443", PROXY_ORIGIN, 200),
        ("sub.viewer.example.invalid:9443", None, 400),
        ("viewer.example.invalid.attacker.invalid:9443", None, 400),
        ("127.0.0.1:8765", "https://sub.viewer.example.invalid:9443", 403),
        ("127.0.0.1:8765", "http://viewer.example.invalid:9443", 403),
        ("127.0.0.1:8765", "https://viewer.example.invalid:9444", 403),
        ("127.0.0.1:8765", "https://viewer.example.invalid", 403),
        ("127.0.0.1:8765", f"{PROXY_ORIGIN}/path", 403),
        ("viewer.example.invalid:9443", "http://viewer.example.invalid:9443", 403),
        ("viewer.example.invalid:9444", "https://viewer.example.invalid:9444", 403),
        ("viewer.example.invalid:9444", "http://viewer.example.invalid:9444", 403),
        ("viewer.example.invalid", "http://viewer.example.invalid", 403),
    ],
)
async def test_explicit_proxy_origin_is_exact_and_only_its_exact_host_is_allowlisted(
    store: SessionStore, tmp_path: Path, scheme: str, host: str, origin: str | None, status: int
) -> None:
    session_id = await _seed(store, tmp_path)
    config = ServerConfig(store=store, token=TOKEN, allowed_origins=frozenset({PROXY_ORIGIN}))
    headers = {"host": host}
    if origin is not None:
        headers["origin"] = origin
    websocket_base = WS_URL.replace("ws:", "wss:") if scheme == "https" else WS_URL
    with TestClient(create_app(config), base_url=BASE_URL.replace("http:", f"{scheme}:")) as client:
        response = client.get(API_PATH, headers=headers, params={"token": TOKEN})
        assert response.status_code == status
        if status == 200:
            with client.websocket_connect(
                f"{websocket_base}/api/sessions/{session_id}/stream", headers=headers, params={"token": TOKEN}
            ) as socket:
                _assert_stream(socket, session_id)
        else:
            assert "set-cookie" not in response.headers
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(
                    f"{websocket_base}/api/sessions/nonexistent/stream", headers=headers, params={"token": TOKEN}
                ):
                    pytest.fail("Proxy allowlist accepted a nonmatching authority or origin")
            assert caught.value.code == (4400 if status == 400 else 4403)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host,origin,forwarded_host,forwarded_proto,status",
    [
        ("attacker.invalid:8765", None, "127.0.0.1:8765", "http", 400),
        ("127.0.0.1:8765", "https://attacker.invalid:9443", "attacker.invalid:9443", "https", 403),
        ("127.0.0.1:8765", BASE_URL, "attacker.invalid:9443", "https", 200),
    ],
)
async def test_forwarded_headers_do_not_override_actual_host_or_scheme(
    store: SessionStore,
    tmp_path: Path,
    host: str,
    origin: str | None,
    forwarded_host: str,
    forwarded_proto: str,
    status: int,
) -> None:
    session_id = await _seed(store, tmp_path)
    headers = {
        "host": host,
        "authorization": f"Bearer {TOKEN}",
        "forwarded": f'for=192.0.2.99;host="{forwarded_host}";proto={forwarded_proto}',
        "x-forwarded-host": forwarded_host,
        "x-forwarded-proto": forwarded_proto,
        "x-forwarded-port": "9443",
    }
    if origin is not None:
        headers["origin"] = origin
    with _client(store) as client:
        response = client.get(API_PATH, headers=headers)
        assert response.status_code == status
        if status == 200:
            with client.websocket_connect(f"{WS_URL}/api/sessions/{session_id}/stream", headers=headers) as socket:
                _assert_stream(socket, session_id)
        else:
            assert "set-cookie" not in response.headers
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(REJECTED_STREAM, headers=headers):
                    pytest.fail("Forwarded headers bypassed access protection")
            assert caught.value.code == (4400 if status == 400 else 4403)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/",
        API_PATH,
        "/api/sessions/{session_id}",
        "/api/sessions/{session_id}/events",
        "/api/sessions/{session_id}/state",
        "/api/sessions/{session_id}/artifacts/" + "a" * 64,
        "/api/themes",
        "/theme.css",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
        "/assets/player.js",
        "/assets/player.css",
        "/s/{session_id}",
        "/s/{session_id}/manifest.json",
        "/s/{session_id}/events.jsonl",
        "/s/{session_id}/player.html",
        "/s/{session_id}/player.js",
        "/s/{session_id}/player.css",
        "/s/{session_id}/fold.js",
        "/nonexistent-route",
    ],
)
async def test_every_http_surface_is_guarded(store: SessionStore, tmp_path: Path, path: str) -> None:
    session_id = await _seed(store, tmp_path)
    path = path.format(session_id=session_id)
    with _client(store) as client:
        unauthorized = client.get(path)
        assert unauthorized.status_code == 401
        assert "set-cookie" not in unauthorized.headers
        for headers, expected in (
            ({"host": "attacker.invalid"}, 400),
            ({"origin": "http://attacker.invalid"}, 403),
        ):
            rejected = client.get(path, headers=headers, params={"token": TOKEN})
            assert rejected.status_code == expected
            assert "set-cookie" not in rejected.headers
        authorized = client.get(path, headers={"authorization": f"Bearer {TOKEN}"})
        expected = 404 if "/artifacts/" in path or path == "/nonexistent-route" else 200
        assert authorized.status_code == expected


@pytest.mark.asyncio
async def test_valid_query_link_mints_cookie_for_assets_and_real_websocket(store: SessionStore, tmp_path: Path) -> None:
    session_id = await _seed(store, tmp_path)
    with _client(store) as client:
        response = client.get(f"/s/{session_id}", params={"token": TOKEN}, headers={"origin": BASE_URL})
        assert response.status_code == 200
        assert client.cookies.get(TOKEN_COOKIE) == TOKEN
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "path=/" in cookie
        assert "domain=" not in cookie
        for path in ("/assets/player.js", "/theme.css", f"/s/{session_id}/manifest.json", "/docs"):
            assert client.get(path).status_code == 200
        with client.websocket_connect(
            f"{WS_URL}/api/sessions/{session_id}/stream", headers={"origin": BASE_URL}
        ) as socket:
            _assert_stream(socket, session_id)
