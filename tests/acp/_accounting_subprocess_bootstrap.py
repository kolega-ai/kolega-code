"""Block external sockets and seed offline preflight counting, never provider inference or accounting."""

from ipaddress import ip_address
import socket
from typing import Any

import tiktoken
import tiktoken.registry


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo


def _check_host(host: Any) -> None:
    if host is None or host in ("localhost", b"localhost"):
        return
    try:
        if ip_address(host.decode() if isinstance(host, bytes) else host).is_loopback:
            return
    except ValueError:
        pass
    raise OSError(f"Hermetic ACP fixture blocked external network destination: {host!r}")


def _local_connect(connection: socket.socket, address: Any) -> None:
    if connection.family != socket.AF_UNIX:
        _check_host(address[0])
    return _connect(connection, address)


def _local_connect_ex(connection: socket.socket, address: Any) -> int:
    if connection.family != socket.AF_UNIX:
        _check_host(address[0])
    return _connect_ex(connection, address)


def _local_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    _check_host(host)
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _local_connect
socket.socket.connect_ex = _local_connect_ex
socket.getaddrinfo = _local_getaddrinfo

for name in ("cl100k_base", "o200k_base"):
    tiktoken.registry.ENCODINGS[name] = tiktoken.Encoding(
        name=name,
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
