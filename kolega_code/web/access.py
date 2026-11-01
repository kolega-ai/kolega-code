from __future__ import annotations

import ipaddress
import re

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_AUTHORITY = re.compile(r"(?:[a-zA-Z0-9.-]+|\[[a-fA-F0-9:.]+\])(?::[0-9]+)?")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_ORIGIN = re.compile(r"(https?)://([^/?#]+)")


def _hostname(value: str) -> str:
    if not value or "%" in value:
        raise ValueError("Invalid allowed host")
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        hostname = value.lower()
        if len(hostname) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in hostname.split(".")):
            raise ValueError("Invalid allowed host") from None
        return hostname


def _authority(value: str) -> tuple[str, int | None]:
    if not _AUTHORITY.fullmatch(value):
        raise ValueError("Invalid host authority")
    if value.startswith("["):
        hostname, _, suffix = value[1:].partition("]")
        hostname = ipaddress.IPv6Address(hostname).compressed
    else:
        hostname, _, port_text = value.partition(":")
        hostname = _hostname(hostname)
        suffix = f":{port_text}" if port_text else ""
    port = int(suffix[1:]) if suffix else None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid host port")
    return hostname, port


def _origin(value: str) -> tuple[str, str, int]:
    match = _ORIGIN.fullmatch(value)
    if match is None:
        raise ValueError("Expected an http(s) origin without a path, credentials, query or fragment")
    scheme, authority = match.groups()
    hostname, port = _authority(authority)
    return scheme, hostname, port if port is not None else _DEFAULT_PORTS[scheme]


class HostOriginPolicy:
    same_origin_hosts: frozenset[str]
    allowed_hosts: frozenset[str]
    allowed_origins: frozenset[tuple[str, str, int]]

    def __init__(self, allowed_hosts: frozenset[str], allowed_origins: frozenset[str]) -> None:
        self.allowed_origins = frozenset(_origin(origin) for origin in allowed_origins)
        self.same_origin_hosts = frozenset(_hostname(host) for host in allowed_hosts)
        self.allowed_hosts = self.same_origin_hosts | frozenset(host for _, host, _ in self.allowed_origins)
        if not self.allowed_hosts:
            raise ValueError("At least one allowed host is required")

    def rejection(self, connection: HTTPConnection) -> tuple[int, str] | None:
        hosts = connection.headers.getlist("host")
        if len(hosts) != 1:
            return 400, "Invalid Host header"
        try:
            hostname, port = _authority(hosts[0])
        except ValueError:
            return 400, "Invalid Host header"
        if hostname not in self.allowed_hosts:
            return 400, "Host not allowed"

        origins = connection.headers.getlist("origin")
        if not origins:
            return None
        if len(origins) != 1:
            return 403, "Origin not allowed"
        try:
            origin = _origin(origins[0])
        except ValueError:
            return 403, "Origin not allowed"
        scheme = str(connection.scope.get("scheme", ""))
        scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
        same_origin = (scheme, hostname, port if port is not None else _DEFAULT_PORTS.get(scheme))
        if origin not in self.allowed_origins and (hostname not in self.same_origin_hosts or origin != same_origin):
            return 403, "Origin not allowed"
        return None


class HostOriginMiddleware:
    app: ASGIApp
    policy: HostOriginPolicy

    def __init__(self, app: ASGIApp, policy: HostOriginPolicy) -> None:
        self.app = app
        self.policy = policy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in {"http", "websocket"}:
            rejection = self.policy.rejection(HTTPConnection(scope))
            if rejection is not None:
                status, detail = rejection
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 4000 + status, "reason": detail})
                else:
                    await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)
                return
        await self.app(scope, receive, send)
