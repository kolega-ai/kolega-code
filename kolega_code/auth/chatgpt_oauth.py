"""Browser-loopback "Sign in with ChatGPT" flow (OAuth 2.0 + PKCE).

Replicates Codex's login: open the system browser to OpenAI's authorize
endpoint, catch the redirect on a loopback port, exchange the code (with the
PKCE verifier) for tokens, and read the ChatGPT plan/account claims.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import webbrowser
from typing import Awaitable, Callable, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from . import constants
from .tokens import OAuthError, OAuthTokens, request_token


class LoginError(OAuthError):
    """Raised when the interactive login flow fails or is denied."""


def generate_pkce_pair() -> tuple[str, str]:
    """Return a ``(code_verifier, code_challenge)`` pair using S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url(
    redirect_uri: str,
    code_challenge: str,
    state: str,
    *,
    client_id: Optional[str] = None,
) -> str:
    """Build the OpenAI authorize URL with PKCE + Codex's extra params."""
    params = {
        "response_type": "code",
        "client_id": client_id or constants.client_id(),
        "redirect_uri": redirect_uri,
        "scope": constants.SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        **constants.EXTRA_AUTHORIZE_PARAMS,
    }
    return f"{constants.AUTHORIZE_URL}?{urlencode(params)}"


def is_plan_eligible(plan_type: Optional[str]) -> bool:
    """False for plans that cannot use subscription inference (e.g. ``free``)."""
    return (plan_type or "").lower() not in constants.INELIGIBLE_PLAN_TYPES


async def exchange_code(
    code: str,
    code_verifier: str,
    redirect_uri: str,
    *,
    client_id: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    now: Optional[float] = None,
) -> OAuthTokens:
    """Exchange an authorization code (+ PKCE verifier) for tokens."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id or constants.client_id(),
        "code_verifier": code_verifier,
    }
    try:
        payload = await request_token(data, http_client=http_client)
    except OAuthError as exc:
        raise LoginError(f"Could not exchange the authorization code: {exc}") from exc
    if "access_token" not in payload:
        raise LoginError("Token endpoint did not return an access token.")
    return OAuthTokens.from_token_response(payload, now=now)


_SUCCESS_HTML = (
    b"<!doctype html><html><head><meta charset=utf-8><title>Signed in</title></head>"
    b"<body style='font-family:system-ui;text-align:center;padding-top:4rem'>"
    b"<h2>You're signed in to ChatGPT.</h2><p>You can close this tab and return to kolega-code.</p>"
    b"</body></html>"
)


def _error_html(reason: str) -> bytes:
    safe = reason.replace("<", "&lt;").replace(">", "&gt;")
    return (
        b"<!doctype html><html><head><meta charset=utf-8><title>Sign-in failed</title></head>"
        b"<body style='font-family:system-ui;text-align:center;padding-top:4rem'>"
        b"<h2>Sign-in failed.</h2><p>" + safe.encode("utf-8", "replace") + b"</p>"
        b"<p>Return to kolega-code and try <code>/login chatgpt</code> again.</p></body></html>"
    )


async def _write_response(writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(head + body)
    try:
        await writer.drain()
    except OSError:
        pass


async def _bind_loopback(
    handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]],
    ports: tuple[int, ...],
) -> tuple[asyncio.AbstractServer, int]:
    last_exc: Optional[OSError] = None
    for port in ports:
        try:
            server = await asyncio.start_server(handler, "127.0.0.1", port)
            return server, port
        except OSError as exc:
            last_exc = exc
    raise LoginError(
        f"Could not bind a loopback port {ports} for sign-in (is another login in progress?): {last_exc}"
    )


async def run_login_flow(
    *,
    client_id: Optional[str] = None,
    ports: tuple[int, ...] = (constants.DEFAULT_REDIRECT_PORT, constants.FALLBACK_REDIRECT_PORT),
    open_browser: bool = True,
    on_url: Optional[Callable[[str], None]] = None,
    timeout: float = 300.0,
    require_paid_plan: bool = True,
    http_client: Optional[httpx.AsyncClient] = None,
) -> OAuthTokens:
    """Run the full browser-loopback login and return the resulting tokens.

    ``on_url`` is called with the authorize URL so the caller can display it
    (in case the browser does not open). Raises :class:`LoginError` on denial,
    state mismatch, timeout, or (when ``require_paid_plan``) an ineligible plan.
    """
    resolved_client = client_id or constants.client_id()
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            try:
                _method, target, _rest = request_line.decode("latin-1").split(" ", 2)
            except ValueError:
                await _write_response(writer, 400, _error_html("malformed request"))
                return
            parts = urlsplit(target)
            if parts.path != constants.REDIRECT_PATH:
                await _write_response(writer, 404, _error_html("unexpected path"))
                return
            query = parse_qs(parts.query)
            error = query.get("error", [None])[0]
            code = query.get("code", [None])[0]
            got_state = query.get("state", [None])[0]
            if error:
                await _write_response(writer, 400, _error_html(error))
                if not code_future.done():
                    code_future.set_exception(LoginError(f"Authorization denied: {error}"))
            elif got_state != state:
                await _write_response(writer, 400, _error_html("state mismatch"))
                if not code_future.done():
                    code_future.set_exception(LoginError("OAuth state mismatch; aborting for safety."))
            elif not code:
                await _write_response(writer, 400, _error_html("missing authorization code"))
            else:
                await _write_response(writer, 200, _SUCCESS_HTML)
                if not code_future.done():
                    code_future.set_result(code)
        except Exception as exc:  # pragma: no cover - defensive
            if not code_future.done():
                code_future.set_exception(LoginError(f"sign-in callback failed: {exc}"))
        finally:
            writer.close()

    server, port = await _bind_loopback(handle, ports)
    redirect = constants.redirect_uri(port)
    url = build_authorize_url(redirect, challenge, state, client_id=resolved_client)
    try:
        if on_url is not None:
            on_url(url)
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:  # pragma: no cover - headless / no browser
                pass
        try:
            code = await asyncio.wait_for(asyncio.shield(code_future), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise LoginError("Timed out waiting for the browser sign-in to complete.") from exc
    finally:
        server.close()
        try:
            await server.wait_closed()
        except Exception:  # pragma: no cover
            pass

    tokens = await exchange_code(code, verifier, redirect, client_id=resolved_client, http_client=http_client)
    if require_paid_plan and not is_plan_eligible(tokens.plan_type):
        raise LoginError(
            f"This ChatGPT account is on the '{tokens.plan_type or 'free'}' plan, which can't run models "
            "in third-party tools. A Plus, Pro, or Business plan is required."
        )
    return tokens
